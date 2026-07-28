// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

use crate::multimodal::{GemmaMultimodalConfig, IMAGE_MARKER, IMAGE_MARKER_HF};
use crate::multimodal_runner::{GemmaMultimodalRunner, MultimodalWeights};
use crate::{GemmaConfig, GemmaGenerator, encode_prompt_auto, gemma_cfg_from_gguf};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{LmRunner, WeightFormat};
use rlx_core::gguf_support::{
    GgufModelFamily, ResolveWeightsOptions, assert_gguf_family, gguf_f32_bytes_estimate,
    resolve_weights_file_with_options,
};
use rlx_core::pick_lm_device;
use rlx_qwen3::SampleOpts;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

// ────────────────────────────────────────────────────────────────
// Gemma runner — Meta Llama 3.x small LMs (1B / 3B).
// ────────────────────────────────────────────────────────────────

/// Where to load the Gemma config from. Alias of the shared
/// `rlx_runtime::ConfigSource<T>` — same `Embedded | JsonFile | Explicit(T)`
/// shape as the other family `*ConfigSource` enums.
pub type GemmaConfigSource = rlx_runtime::ConfigSource<GemmaConfig>;

#[derive(Debug, Clone, Default)]
pub struct GemmaRunnerBuilder {
    weights: Option<PathBuf>,
    config: Option<GemmaConfigSource>,
    device: Option<Device>,
    max_seq: Option<usize>,
    max_memory_gb: Option<f32>,
    stream: bool,
    sample: Option<SampleOpts>,
    format: Option<WeightFormat>,
    /// `None` = auto (packed GGUF when file ≥ 256 MiB on disk).
    packed_weights: Option<bool>,
    /// Optional llama.cpp-style mmproj GGUF (enables multimodal).
    mmproj: Option<PathBuf>,
    /// Optional HF `config.json` for `GemmaMultimodalConfig` (vision/audio token ids).
    /// When unset, looks next to weights; falls back to Gemma-4 defaults if mmproj is set.
    mm_config: Option<PathBuf>,
}

impl GemmaRunnerBuilder {
    pub fn weights<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.weights = Some(path.into());
        self
    }

    pub fn format(mut self, fmt: WeightFormat) -> Self {
        self.format = Some(fmt);
        self
    }

    pub fn config(mut self, src: GemmaConfigSource) -> Self {
        self.config = Some(src);
        self
    }

    pub fn config_value(self, cfg: GemmaConfig) -> Self {
        self.config(GemmaConfigSource::Explicit(cfg))
    }

    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }

    pub fn max_memory_gb(mut self, gb: f32) -> Self {
        self.max_memory_gb = Some(gb);
        self
    }

    pub fn stream(mut self, on: bool) -> Self {
        self.stream = on;
        self
    }

    pub fn sample(mut self, opts: SampleOpts) -> Self {
        self.sample = Some(opts);
        self
    }

    /// Keep K-quant weights packed in the arena (`Op::DequantMatMul`).
    /// GGUF only. When unset, large GGUF files (≥ 256 MiB) auto-enable packed paths.
    /// Forced off when `.mmproj(...)` is set (multimodal needs the F32 embed path).
    pub fn packed_weights(mut self, on: bool) -> Self {
        self.packed_weights = Some(on);
        self
    }

    /// Attach a llama.cpp-style mmproj GGUF (vision / audio projector).
    pub fn mmproj(mut self, path: impl Into<PathBuf>) -> Self {
        self.mmproj = Some(path.into());
        self
    }

    /// HF `config.json` providing `vision_config` / media token ids.
    pub fn mm_config(mut self, path: impl Into<PathBuf>) -> Self {
        self.mm_config = Some(path.into());
        self
    }

    fn gguf_auto_packed(path: &Path) -> bool {
        std::fs::metadata(path)
            .map(|m| m.len() >= 256 * 1024 * 1024)
            .unwrap_or(false)
    }

    pub fn build(self) -> Result<GemmaRunner> {
        let resolve = ResolveWeightsOptions {
            prefer_gguf_substring: Some(rlx_core::DEFAULT_GGUF_PREFER_SUBSTR),
            ..Default::default()
        };
        let weights_path = resolve_weights_file_with_options(
            self.weights
                .as_ref()
                .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?,
            &resolve,
        )?;
        let format = WeightFormat::resolve(&weights_path, self.format)?;
        let device = self.device.unwrap_or_else(pick_lm_device);
        if self.device.is_none() {
            eprintln!("[gemma-runner] auto device → {device:?}");
        }
        let max_seq = self.max_seq.unwrap_or(128);
        let stream = self.stream;
        let sample = self.sample.unwrap_or_else(SampleOpts::greedy);

        let (cfg, total_bytes_estimate) = match format {
            WeightFormat::Gguf => load_gemma_gguf_config(&weights_path, self.config.as_ref())?,
            WeightFormat::Safetensors => {
                load_gemma_safetensors_config(&weights_path, self.config.as_ref())?
            }
        };

        if let Some(cap_gb) = self.max_memory_gb {
            let est_gb = total_bytes_estimate as f32 / (1024.0 * 1024.0 * 1024.0);
            if est_gb > cap_gb {
                bail!(
                    "weights would dequant to ~{est_gb:.1} GB at F32, exceeds cap {cap_gb:.1} GB"
                );
            }
        }

        let mmproj_path = self.mmproj.clone();
        // Multimodal fuse needs `model.embed_tokens` via the F32 generator.
        let use_packed = if mmproj_path.is_some() {
            false
        } else {
            self.packed_weights.unwrap_or_else(|| {
                matches!(format, WeightFormat::Gguf) && Self::gguf_auto_packed(&weights_path)
            })
        };

        crate::capabilities::validate_device(&cfg, device, use_packed)?;

        let path_str = weights_path
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 weights path"))?;
        let generator = if use_packed {
            None
        } else {
            Some(
                GemmaGenerator::from_path(cfg.clone(), path_str, device)?
                    .with_inference_caches(max_seq),
            )
        };

        let packed = if use_packed {
            if !matches!(format, WeightFormat::Gguf) {
                bail!(
                    "packed_weights(true) requires a .gguf file; got {:?} for {:?}",
                    format,
                    weights_path
                );
            }
            if self.packed_weights.is_none() {
                eprintln!("[gemma-runner] auto packed_weights=true (GGUF ≥ 256 MiB) on {device:?}");
            }
            eprintln!(
                "[gemma-runner] packed_weights=true — Q4 prefill + bucketed decode on {device:?}"
            );
            Some(crate::packed_session::GemmaPackedSession::build(
                cfg.clone(),
                &weights_path,
                max_seq,
                device,
            )?)
        } else {
            None
        };

        let multimodal = match mmproj_path {
            Some(mp) => {
                let mm_weights = MultimodalWeights::from_mmproj_gguf(&mp)
                    .with_context(|| format!("gemma: load mmproj {mp:?}"))?;
                let mm_cfg = resolve_mm_config(
                    self.mm_config.as_deref(),
                    &weights_path,
                    self.config.as_ref(),
                )?;
                if !mm_cfg.has_vision() && !mm_cfg.has_audio() {
                    bail!(
                        "gemma: mmproj attached but no vision_config/audio_config found \
                         (pass .mm_config(config.json) or place config.json next to weights)"
                    );
                }
                let max_soft = mm_cfg
                    .vision
                    .as_ref()
                    .map(|v| v.num_soft_tokens)
                    .unwrap_or(280);
                let mut mm_runner = GemmaMultimodalRunner::new(
                    mm_cfg.clone(),
                    cfg.hidden_size,
                    device,
                    Some(max_soft),
                    None,
                )?;
                // Warm vision graph for default soft-token budget.
                if mm_cfg.has_vision() {
                    mm_runner.compile_vision(max_soft)?;
                    mm_runner.load_vision_weights(&mm_weights)?;
                }
                eprintln!("[gemma-runner] multimodal enabled mmproj={mp:?} soft_tokens={max_soft}");
                Some(GemmaMultimodalState {
                    mm_cfg,
                    mm_weights,
                    mm_runner,
                    max_side_patches: 32,
                })
            }
            None => None,
        };

        Ok(GemmaRunner {
            generator,
            cfg,
            sample,
            stream,
            device,
            packed,
            weights_path,
            multimodal,
        })
    }
}

struct GemmaMultimodalState {
    mm_cfg: GemmaMultimodalConfig,
    mm_weights: MultimodalWeights,
    mm_runner: GemmaMultimodalRunner,
    max_side_patches: usize,
}

pub struct GemmaRunner {
    generator: Option<GemmaGenerator>,
    cfg: GemmaConfig,
    sample: SampleOpts,
    stream: bool,
    device: Device,
    packed: Option<crate::packed_session::GemmaPackedSession>,
    weights_path: PathBuf,
    multimodal: Option<GemmaMultimodalState>,
}

impl GemmaRunner {
    pub fn builder() -> GemmaRunnerBuilder {
        GemmaRunnerBuilder::default()
    }

    pub fn config(&self) -> &GemmaConfig {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Whether an mmproj vision/audio projector was attached at build time.
    pub fn has_mmproj(&self) -> bool {
        self.multimodal.is_some()
    }

    /// Single prefill forward; returns last-position logits `[vocab]`.
    pub fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        if let Some(p) = self.packed.as_mut() {
            return p.predict_logits(prompt_ids);
        }
        let generator = self
            .generator
            .as_mut()
            .ok_or_else(|| anyhow!("F32 generator unavailable in packed_weights mode"))?;
        generator.prefill_get_last_logits(prompt_ids)
    }

    /// Post-`model.norm` hidden for the last prompt token (packed GGUF path only).
    pub fn predict_last_hidden(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.packed
            .as_mut()
            .ok_or_else(|| anyhow!("predict_last_hidden requires packed_weights(true)"))?
            .predict_last_hidden(prompt_ids)
    }

    pub fn generate_packed(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        if self.packed.is_none() {
            bail!("generate_packed() only works in packed_weights(true) mode");
        }
        let sample = self.sample;
        self.packed
            .as_mut()
            .unwrap()
            .generate(prompt_ids, n_new, sample, on_token)
    }

    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        if self.packed.is_some() {
            return self.generate_packed(prompt_ids, n_new, on_token);
        }
        let generator = self
            .generator
            .as_mut()
            .ok_or_else(|| anyhow!("F32 generator unavailable in packed_weights mode"))?;
        generator.prefill(prompt_ids);
        let tokens = if self.stream {
            generator.generate_cached_with(n_new, self.sample, &mut on_token)?
        } else {
            let toks = generator.generate_cached(n_new, self.sample)?;
            for &t in &toks {
                on_token(t);
            }
            toks
        };
        Ok(tokens)
    }

    /// EAGLE3-style generate with a per-step **aux hidden state**
    /// callback. After each decoded token, `on_aux` receives the
    /// per-layer pre-attention-norm hidden states from the layer
    /// indices in `aux_hidden_layer_ids` (one `Vec<f32>` per id,
    /// each of length `hidden_size`).
    ///
    /// This is the entry point the rlx-eagle3 verifier bridge
    /// (`AuxStateBuffer::write`) consumes — wire it like:
    ///
    /// ```ignore
    /// runner.generate_with_aux(prompt, n, vec![2, 30, 57],
    ///     |tok| { ... },
    ///     |aux| { aux_buffer.write(aux); });
    /// ```
    ///
    /// **Not supported in `packed_weights(true)` mode** — packed
    /// builds use a separate compile path that doesn't yet thread
    /// the aux tap.
    pub fn generate_with_aux(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        aux_hidden_layer_ids: Vec<usize>,
        mut on_token: impl FnMut(u32),
        mut on_aux: impl FnMut(Vec<Vec<f32>>),
    ) -> Result<Vec<u32>> {
        if self.packed.is_some() {
            bail!("generate_with_aux is not supported with packed_weights(true)");
        }
        if aux_hidden_layer_ids.is_empty() {
            // No aux ids → falls back to plain generate.
            return self.generate(prompt_ids, n_new, on_token);
        }
        let generator = self
            .generator
            .as_mut()
            .ok_or_else(|| anyhow!("F32 generator unavailable in packed_weights mode"))?;
        generator.set_aux_hidden_layer_ids(aux_hidden_layer_ids);
        generator.prefill(prompt_ids);
        let sample = self.sample;
        let mut emitted: Vec<u32> = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let tok = generator.step_cached(sample)?;
            emitted.push(tok);
            on_token(tok);
            if let Some(aux) = generator.take_last_aux() {
                on_aux(aux);
            }
        }
        // Clear the tap so subsequent generate() calls go through
        // the bucketed fast path again.
        generator.set_aux_hidden_layer_ids(Vec::new());
        Ok(emitted)
    }

    /// Generate after splicing vision/audio rows into pre-scaled text embeddings.
    pub fn generate_from_embeds(
        &mut self,
        prompt_ids: &[u32],
        inputs_embeds: &[f32],
        n_new: usize,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        if self.packed.is_some() {
            bail!("generate_from_embeds is not supported with packed_weights(true)");
        }
        let generator = self
            .generator
            .as_mut()
            .ok_or_else(|| anyhow!("F32 generator unavailable in packed_weights mode"))?;
        let tokens = if self.stream {
            generator.generate_from_embeds_with(
                prompt_ids,
                inputs_embeds,
                n_new,
                self.sample,
                &mut on_token,
            )?
        } else {
            let toks =
                generator.generate_from_embeds(prompt_ids, inputs_embeds, n_new, self.sample)?;
            for &t in &toks {
                on_token(t);
            }
            toks
        };
        Ok(tokens)
    }

    /// Build fused inputs + run generation (text LM weights must include `model.embed_tokens.weight`).
    pub fn generate_multimodal(
        &mut self,
        mm_cfg: &crate::multimodal::GemmaMultimodalConfig,
        token_ids: &[u32],
        image_embeds: &[f32],
        audio_embeds: &[f32],
        video_embeds: &[f32],
        n_new: usize,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let generator = self
            .generator
            .as_ref()
            .ok_or_else(|| anyhow!("F32 generator unavailable in packed_weights mode"))?;
        let embeds = crate::multimodal_embed::build_multimodal_inputs_embeds(
            generator.weights_cache(),
            &self.cfg,
            mm_cfg,
            token_ids,
            image_embeds,
            audio_embeds,
            video_embeds,
        )?;
        let attn_bias = crate::multimodal_mask::build_multimodal_prefill_attn_bias(
            token_ids, &self.cfg, mm_cfg, 1,
        );
        let generator = self
            .generator
            .as_mut()
            .ok_or_else(|| anyhow!("F32 generator unavailable in packed_weights mode"))?;
        let tokens = if self.stream {
            generator.generate_from_embeds_with_bias_and_callback(
                token_ids,
                &embeds,
                attn_bias,
                n_new,
                self.sample,
                &mut on_token,
            )?
        } else {
            let toks = generator.generate_from_embeds_with_bias(
                token_ids,
                &embeds,
                attn_bias,
                n_new,
                self.sample,
            )?;
            for &t in &toks {
                on_token(t);
            }
            toks
        };
        Ok(tokens)
    }

    /// End-to-end multimodal generate from RGB pixels (LmRunner / skill path).
    ///
    /// Accepts prompts containing `<image>`, `<|image|>`, or Qwen-style
    /// `<__media__>` markers. When none are present, appends one `<image>`
    /// marker so a single attached frame still lands in the soft-token span.
    pub fn generate_multimodal_rgb(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        tokenizer: Option<&Path>,
        n_new: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let mm = self
            .multimodal
            .as_mut()
            .ok_or_else(|| anyhow!("gemma: generate_multimodal requires .mmproj(...)"))?;
        if rgb.len() != img_w.saturating_mul(img_h).saturating_mul(3) {
            bail!("gemma: rgb len {} != {img_w}×{img_h}×3", rgb.len());
        }

        let tmp = write_rgb_temp_png(rgb, img_w, img_h)?;
        let soft_count = mm
            .mm_runner
            .image_soft_token_count(&tmp)
            .unwrap_or_else(|_| {
                mm.mm_cfg
                    .vision
                    .as_ref()
                    .map(|v| v.num_soft_tokens)
                    .unwrap_or(280)
            });
        let image_soft =
            mm.mm_runner
                .project_image_file(&tmp, &mm.mm_weights, mm.max_side_patches)?;
        let _ = std::fs::remove_file(&tmp);

        let prompt = normalize_gemma_media_prompt(prompt);
        let weights = self.weights_path.clone();
        let encode = |s: &str| -> Result<Vec<u32>> { encode_prompt_auto(&weights, tokenizer, s) };
        let token_ids = mm
            .mm_runner
            .tokenize_prompt(&prompt, &[soft_count], &[], &[], encode)?;

        let mm_cfg = mm.mm_cfg.clone();
        let mut keep_going = true;
        self.generate_multimodal(&mm_cfg, &token_ids, &image_soft, &[], &[], n_new, |tok| {
            if keep_going {
                keep_going = on_token(tok);
            }
        })
    }
}

impl LmRunner for GemmaRunner {
    fn family(&self) -> &'static str {
        "gemma"
    }
    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }
    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        GemmaRunner::predict_logits(self, prompt_ids)
    }
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        // Inherent generate ignores stop signal — drop the bool.
        GemmaRunner::generate(self, prompt_ids, n_new, |tok| {
            let _ = on_token(tok);
        })
    }

    fn supports_multimodal(&self) -> bool {
        self.has_mmproj()
    }

    fn generate_multimodal(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        tokenizer: Option<&Path>,
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.generate_multimodal_rgb(prompt, rgb, img_w, img_h, tokenizer, n_new, on_token)
    }
}

fn normalize_gemma_media_prompt(prompt: &str) -> String {
    const QWEN_MEDIA: &str = "<__media__>";
    let mut p = prompt.replace(QWEN_MEDIA, IMAGE_MARKER);
    if !p.contains(IMAGE_MARKER) && !p.contains(IMAGE_MARKER_HF) {
        if !p.is_empty() && !p.ends_with(|c: char| c.is_whitespace()) {
            p.push(' ');
        }
        p.push_str(IMAGE_MARKER);
    }
    p
}

fn write_rgb_temp_png(rgb: &[u8], w: usize, h: usize) -> Result<PathBuf> {
    use image::{ImageBuffer, Rgb};
    let img: ImageBuffer<Rgb<u8>, _> = ImageBuffer::from_raw(w as u32, h as u32, rgb.to_vec())
        .ok_or_else(|| anyhow!("gemma: failed to wrap rgb buffer as ImageBuffer"))?;
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "rlx-gemma-mm-{}-{}.png",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    img.save(&path)
        .with_context(|| format!("gemma: write temp png {path:?}"))?;
    Ok(path)
}

/// Resolve multimodal HF config: explicit path → weights sidecar → LM JsonFile → Gemma-4 defaults.
fn resolve_mm_config(
    explicit: Option<&Path>,
    weights_path: &Path,
    lm_config: Option<&GemmaConfigSource>,
) -> Result<GemmaMultimodalConfig> {
    let candidates: Vec<PathBuf> = {
        let mut v = Vec::new();
        if let Some(p) = explicit {
            v.push(p.to_path_buf());
        }
        if let Some(parent) = weights_path.parent() {
            v.push(parent.join("config.json"));
        }
        if let Some(GemmaConfigSource::JsonFile(p)) = lm_config {
            v.push(p.clone());
        }
        v
    };
    for p in &candidates {
        if p.is_file() {
            let cfg = GemmaMultimodalConfig::from_file(p)
                .with_context(|| format!("gemma: read multimodal config {p:?}"))?;
            if cfg.has_vision() || cfg.has_audio() {
                return Ok(cfg);
            }
        }
    }
    // Gemma-4 defaults (token ids from google/gemma-4-E2B-it).
    Ok(GemmaMultimodalConfig {
        vision: Some(crate::multimodal::GemmaVisionConfig::default()),
        audio: None,
        image_token_id: Some(258_880),
        audio_token_id: Some(258_881),
        video_token_id: Some(258_884),
        boi_token_id: Some(255_999),
        eoi_token_id: Some(256_000),
        ..Default::default()
    })
}

fn load_gemma_gguf_config(
    path: &Path,
    override_src: Option<&GemmaConfigSource>,
) -> Result<(GemmaConfig, u64)> {
    let raw = assert_gguf_family(path, GgufModelFamily::Gemma)?;
    let cfg = match override_src {
        Some(GemmaConfigSource::Explicit(c)) => c.clone(),
        Some(GemmaConfigSource::JsonFile(p)) => {
            GemmaConfig::from_file(p).with_context(|| format!("reading override config {p:?}"))?
        }
        Some(GemmaConfigSource::Embedded) | None => gemma_cfg_from_gguf(&raw)?,
    };
    Ok((cfg, gguf_f32_bytes_estimate(&raw)))
}

fn load_gemma_safetensors_config(
    path: &Path,
    override_src: Option<&GemmaConfigSource>,
) -> Result<(GemmaConfig, u64)> {
    let cfg_path = match override_src {
        Some(GemmaConfigSource::Explicit(c)) => {
            return Ok((c.clone(), default_st_size_estimate(path)));
        }
        Some(GemmaConfigSource::JsonFile(p)) => p.clone(),
        Some(GemmaConfigSource::Embedded) => {
            bail!("ConfigSource::Embedded only valid for GGUF; pass JsonFile for safetensors")
        }
        None => path
            .parent()
            .ok_or_else(|| anyhow!("weights path has no parent dir"))?
            .join("config.json"),
    };
    let cfg = GemmaConfig::from_file(&cfg_path)
        .with_context(|| format!("reading config {cfg_path:?}"))?;
    Ok((cfg, default_st_size_estimate(path)))
}

fn default_st_size_estimate(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}
