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

//! Combined Qwen3-VL runner — Qwen3 LM + SigLIP mmproj vision tower.

use crate::multimodal::{
    MEDIA_MARKER, MultimodalPrefill, MultimodalPrompt, VisionEncodeOutput, normalize_media_prompt,
};
use crate::runner::Qwen3VlVisionRunner;
use anyhow::{Context, Result, anyhow, bail, ensure};
use rlx_cli::LmRunner;
use rlx_core::WeightMap;
use rlx_core::autoregressive::{KvCacheState, split_decode_logits_kv};
use rlx_core::flow_util::compile_built;
use rlx_core::weight_loader::{ArcCacheLoader, ArcF32Tensor};
use rlx_qwen3::flow::{
    Qwen3DecodeOpts, Qwen3PrefillOpts, build_qwen3_decode_built, build_qwen3_prefill_embeds_built,
};
use rlx_qwen3::{Qwen3Config, Qwen3Runner};
use rlx_qwen35::encode_prompt_auto;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct Qwen3VlRunnerBuilder {
    weights: Option<PathBuf>,
    mmproj: Option<PathBuf>,
    hf_config: Option<PathBuf>,
    device: Option<Device>,
    max_seq: Option<usize>,
}

impl Qwen3VlRunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights = Some(path.into());
        self
    }
    pub fn mmproj(mut self, path: impl Into<PathBuf>) -> Self {
        self.mmproj = Some(path.into());
        self
    }
    pub fn hf_config(mut self, path: impl Into<PathBuf>) -> Self {
        self.hf_config = Some(path.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }

    pub fn build(self) -> Result<Qwen3VlRunner> {
        let weights = self
            .weights
            .ok_or_else(|| anyhow!("Qwen3VlRunner: .weights(...) required"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        let max_seq = self.max_seq.unwrap_or(4096);

        let mut lm_builder = Qwen3Runner::builder()
            .weights(&weights)
            .device(device)
            .max_seq(max_seq);
        // Prefer packed path for large GGUFs (same heuristic as other runners).
        if weights
            .metadata()
            .map(|m| m.len() >= 256 * 1024 * 1024)
            .unwrap_or(false)
        {
            lm_builder = lm_builder.packed_weights(true);
        }
        let lm = lm_builder
            .build()
            .with_context(|| format!("Qwen3VlRunner: load LM weights {weights:?}"))?;

        let vision = match self.mmproj {
            Some(mp) => {
                let hf = resolve_hf_config(self.hf_config.as_deref(), &weights, &mp)?;
                let mut vb = Qwen3VlVisionRunner::builder().mmproj(&mp).device(device);
                if let Some(cfg_path) = hf {
                    vb = vb.hf_config(cfg_path);
                } else {
                    // Synthesize vision config from LM hidden size when no HF config.
                    let mut cfg = crate::config::Qwen3VlVisionConfig {
                        hidden_size: 1152,
                        num_hidden_layers: 27,
                        num_attention_heads: 16,
                        intermediate_size: 4304,
                        image_size: 384,
                        patch_size: 14,
                        num_channels: 3,
                        layer_norm_eps: 1e-6,
                        projector_output_dim: lm.config().hidden_size,
                    };
                    // Prefer projector dim = LM hidden.
                    cfg.projector_output_dim = lm.config().hidden_size;
                    let _ = cfg;
                    vb = vb.config(crate::config::Qwen3VlVisionConfig {
                        hidden_size: 1152,
                        num_hidden_layers: 27,
                        num_attention_heads: 16,
                        intermediate_size: 4304,
                        image_size: 384,
                        patch_size: 14,
                        num_channels: 3,
                        layer_norm_eps: 1e-6,
                        projector_output_dim: lm.config().hidden_size,
                    });
                }
                Some(
                    vb.build()
                        .with_context(|| format!("Qwen3VlRunner: load mmproj {mp:?}"))?,
                )
            }
            None => None,
        };

        Ok(Qwen3VlRunner {
            lm,
            vision,
            weights_path: weights,
            device,
            max_seq,
            lm_weights_cache: HashMap::new(),
            embed_cached: None,
            decode_cache: None,
        })
    }
}

fn resolve_hf_config(
    explicit: Option<&Path>,
    weights: &Path,
    mmproj: &Path,
) -> Result<Option<PathBuf>> {
    let mut candidates = Vec::new();
    if let Some(p) = explicit {
        candidates.push(p.to_path_buf());
    }
    if let Some(parent) = weights.parent() {
        candidates.push(parent.join("config.json"));
    }
    if let Some(parent) = mmproj.parent() {
        candidates.push(parent.join("config.json"));
    }
    for p in candidates {
        if p.is_file() {
            return Ok(Some(p));
        }
    }
    Ok(None)
}

pub struct Qwen3VlRunner {
    lm: Qwen3Runner,
    vision: Option<Qwen3VlVisionRunner>,
    weights_path: PathBuf,
    device: Device,
    max_seq: usize,
    lm_weights_cache: HashMap<String, ArcF32Tensor>,
    embed_cached: Option<Vec<f32>>,
    decode_cache: Option<KvCacheState>,
}

impl Qwen3VlRunner {
    pub fn builder() -> Qwen3VlRunnerBuilder {
        Qwen3VlRunnerBuilder::default()
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub fn config(&self) -> &Qwen3Config {
        self.lm.config()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.lm.predict_logits(prompt_ids)
    }

    pub fn generate_text(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.lm.generate_stoppable(prompt_ids, n_new, on_token)
    }

    fn encode_rgb(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<VisionEncodeOutput> {
        let enc = self
            .vision
            .as_mut()
            .ok_or_else(|| anyhow!("Qwen3VlRunner: encode requires .mmproj(...)"))?;
        let n_embd = enc.config().projector_output_dim;
        // Write temp PNG for the vision preprocessor.
        let tmp = write_rgb_temp_png(rgb, w, h)?;
        let flat = enc.embed_image_path(&tmp)?;
        let _ = std::fs::remove_file(&tmp);
        VisionEncodeOutput::from_flat(flat, n_embd)
    }

    fn ensure_embed_cached(&mut self) -> Result<Vec<f32>> {
        if let Some(e) = &self.embed_cached {
            return Ok(e.clone());
        }
        self.ensure_lm_weights_cache()?;
        let key_candidates = [
            "token_embd.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ];
        let mut found: Option<(Vec<f32>, Vec<usize>)> = None;
        for k in key_candidates {
            if let Some((data, shape)) = self.lm_weights_cache.get(k) {
                found = Some((data.as_ref().clone(), shape.clone()));
                break;
            }
        }
        let (data, _shape) = found.ok_or_else(|| {
            anyhow!("Qwen3VlRunner: token embedding table not found in LM weights")
        })?;
        self.embed_cached = Some(data.clone());
        Ok(data)
    }

    fn ensure_lm_weights_cache(&mut self) -> Result<()> {
        if !self.lm_weights_cache.is_empty() {
            return Ok(());
        }
        let file = rlx_core::weights::pick_default(&self.weights_path)?;
        let mut loader = rlx_core::open_weight_loader(&file)
            .with_context(|| format!("open LM weights {file:?}"))?;
        let mut map = WeightMap::from_weight_loader_dequant_all(loader.as_mut())
            .with_context(|| format!("dequant LM weights {file:?}"))?;
        let keys: Vec<String> = map.keys().map(str::to_string).collect();
        for key in keys {
            let (data, shape) = map.take(&key)?;
            let canonical =
                rlx_core::weight_loader::gguf_to_hf_name(&key).unwrap_or_else(|| key.clone());
            self.lm_weights_cache
                .insert(canonical, (Arc::new(data), shape));
        }
        Ok(())
    }

    pub fn prefill_multimodal(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        w: usize,
        h: usize,
        tokenizer: Option<&Path>,
    ) -> Result<Vec<f32>> {
        let vision = self.encode_rgb(rgb, w, h)?;
        let n_embd = self.lm.config().hidden_size;
        ensure!(
            vision.embeddings.len() == vision.n_tokens * n_embd
                || vision.embeddings.len()
                    == vision.n_tokens
                        * self
                            .vision
                            .as_ref()
                            .map(|v| v.config().projector_output_dim)
                            .unwrap_or(n_embd),
            "vision embed dim mismatch"
        );
        // If projector dim != LM hidden, we cannot splice — require match.
        let proj_dim = self
            .vision
            .as_ref()
            .map(|v| v.config().projector_output_dim)
            .unwrap_or(n_embd);
        ensure!(
            proj_dim == n_embd,
            "Qwen3VlRunner: projector_output_dim {proj_dim} != LM hidden {n_embd}"
        );

        let embed = self.ensure_embed_cached()?;
        let prompt = normalize_media_prompt(prompt);
        let weights = self.weights_path.clone();
        let mut tok = |s: &str| -> Result<Vec<u32>> { encode_prompt_auto(&weights, tokenizer, s) };
        let mm = MultimodalPrompt {
            prompt: &prompt,
            vision: &vision,
        };
        let prefill = mm.assemble(&mut tok, &embed, n_embd)?;
        self.prefill_from_assembled(prefill)
    }

    fn prefill_from_assembled(&mut self, prefill: MultimodalPrefill) -> Result<Vec<f32>> {
        let seq = prefill.seq.len();
        ensure!(seq > 0, "empty multimodal sequence");
        ensure!(
            seq <= self.max_seq,
            "multimodal seq {seq} exceeds max_seq {}",
            self.max_seq
        );
        self.ensure_lm_weights_cache()?;
        let mut weight_loader = ArcCacheLoader::new(&self.lm_weights_cache);
        let mut opts = Qwen3PrefillOpts::static_prefill(1, seq);
        opts.with_kv_outputs = true;
        opts.with_lm_head = true;
        opts.last_logits_only = true;
        let built = build_qwen3_prefill_embeds_built(self.lm.config(), &mut weight_loader, &opts)?;
        let mut compiled = compile_built(built, self.device)?;
        let outputs = compiled.run(&[("inputs_embeds", prefill.hidden.as_slice())]);
        let n_layers = self.lm.config().num_hidden_layers;
        // Expect: logits (+ optional hidden) + per-layer K/V.
        // Prefer split_decode style: last outputs are K/V pairs.
        ensure!(
            outputs.len() > 2 * n_layers,
            "prefill produced {} outputs, need >= {}",
            outputs.len(),
            1 + 2 * n_layers
        );
        let logits = outputs[0].clone();
        let mut layers_k = Vec::with_capacity(n_layers);
        let mut layers_v = Vec::with_capacity(n_layers);
        // Layout: [logits, k0, v0, k1, v1, ...] or [logits, hidden, k0, v0, ...]
        let kv_start = if outputs.len() == 1 + 2 * n_layers {
            1
        } else {
            2
        };
        for i in 0..n_layers {
            layers_k.push(outputs[kv_start + 2 * i].clone());
            layers_v.push(outputs[kv_start + 2 * i + 1].clone());
        }
        self.decode_cache = Some(KvCacheState {
            past_len: seq,
            layers_kv_base: vec![0; n_layers],
            layers_k,
            layers_v,
        });
        let vocab = self.lm.config().vocab_size;
        Ok(logits[..vocab.min(logits.len())].to_vec())
    }

    fn decode_step(&mut self, token_id: u32) -> Result<Vec<f32>> {
        let past_seq = self
            .decode_cache
            .as_ref()
            .ok_or_else(|| anyhow!("decode_step: call prefill_multimodal first"))?
            .past_len;
        let layers_k = self.decode_cache.as_ref().unwrap().layers_k.clone();
        let layers_v = self.decode_cache.as_ref().unwrap().layers_v.clone();

        self.ensure_lm_weights_cache()?;
        let mut weight_loader = ArcCacheLoader::new(&self.lm_weights_cache);
        let opts = Qwen3DecodeOpts {
            batch: 1,
            past_seq,
            ..Default::default()
        };
        let built = build_qwen3_decode_built(self.lm.config(), &mut weight_loader, &opts)?;
        let mut compiled = compile_built(built, self.device)?;

        let input_ids = [token_id as f32];
        let mut inputs: Vec<(&str, &[f32])> = vec![("input_ids", input_ids.as_slice())];
        let key_strs: Vec<String> = (0..self.lm.config().num_hidden_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        for i in 0..self.lm.config().num_hidden_layers {
            inputs.push((&key_strs[2 * i], layers_k[i].as_slice()));
            inputs.push((&key_strs[2 * i + 1], layers_v[i].as_slice()));
        }
        // RoPE tables may be required as graph inputs — provide empty and let
        // the built graph use param-backed tables when present.
        let outputs = compiled.run(&inputs);
        let (logits, new_k, new_v) =
            split_decode_logits_kv(outputs, self.lm.config().num_hidden_layers)?;
        let cache_mut = self.decode_cache.as_mut().unwrap();
        cache_mut.past_len = past_seq + 1;
        cache_mut.layers_k = new_k;
        cache_mut.layers_v = new_v;
        let vocab = self.lm.config().vocab_size;
        Ok(logits[..vocab.min(logits.len())].to_vec())
    }

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
        if self.vision.is_none() {
            bail!("Qwen3VlRunner: generate_multimodal requires .mmproj(...)");
        }
        let mut logits = self.prefill_multimodal(prompt, rgb, img_w, img_h, tokenizer)?;
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let next = argmax_u32(&logits);
            out.push(next);
            if !on_token(next) {
                break;
            }
            logits = self.decode_step(next)?;
        }
        Ok(out)
    }
}

impl LmRunner for Qwen3VlRunner {
    fn family(&self) -> &'static str {
        "qwen3-vl"
    }
    fn vocab_size(&self) -> usize {
        self.lm.config().vocab_size
    }
    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        Qwen3VlRunner::predict_logits(self, prompt_ids)
    }
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.generate_text(prompt_ids, n_new, on_token)
    }
    fn supports_multimodal(&self) -> bool {
        self.has_vision()
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

fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

fn write_rgb_temp_png(rgb: &[u8], w: usize, h: usize) -> Result<PathBuf> {
    use image::{ImageBuffer, Rgb};
    ensure!(
        rgb.len() == w.saturating_mul(h).saturating_mul(3),
        "rgb len {} != {w}×{h}×3",
        rgb.len()
    );
    let img: ImageBuffer<Rgb<u8>, _> = ImageBuffer::from_raw(w as u32, h as u32, rgb.to_vec())
        .ok_or_else(|| anyhow!("failed to wrap rgb as ImageBuffer"))?;
    let path = std::env::temp_dir().join(format!(
        "rlx-qwen3vl-{}-{}.png",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    img.save(&path)
        .with_context(|| format!("write temp png {path:?}"))?;
    Ok(path)
}

// Silence unused MEDIA_MARKER import warning when only used via normalize.
#[allow(dead_code)]
fn _media_marker() -> &'static str {
    MEDIA_MARKER
}
