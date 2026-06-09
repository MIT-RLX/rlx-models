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

//! End-to-end runner: image → MoonViT → projector → Qwen2.5 LM → tokens.

use crate::compile_support::{lm_gpu_kv_enabled, vision_encode_device};
use crate::config::LocateAnythingConfig;
use crate::embed::fuse_inputs_embeds_from_store;
use crate::generation::{GenerationMode, SampleOpts, TokenIds, sample_token as sample_ar};
use crate::lm_flow::{compute_rope_chunk, compute_rope_slice, qwen3_config};
use crate::load::{LocateAnythingWeightStore, resolve_model_dir};
use crate::mask::mtp_prefill_mask_2d;
use crate::moonvit::MoonVitCache;
use crate::mtp::{decode_bbox_block, handle_pattern};
use crate::preprocess::{PreprocessedImage, preprocess_image, preprocess_path};
use crate::projector::build_projector_built;
use crate::session_cache::{LmSessionCaches, kv_state_from_runner, truncate_kv_state};
use anyhow::{Context, Result, ensure};
use rlx_core::KvCacheState;
use rlx_core::flow_util::compile_built;
use rlx_core::validate_standard_device;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Max KV past length for bucketed decode (matches MTP cap in `generate_from_embeds`).
const LM_MAX_PAST_BUCKETS: usize = 2048;
const LM_MAX_PAST_BUCKETS_WGPU: usize = 1024;

fn lm_max_past_buckets(device: rlx_runtime::Device) -> usize {
    match device {
        rlx_runtime::Device::Gpu | rlx_runtime::Device::Vulkan => LM_MAX_PAST_BUCKETS_WGPU,
        _ => LM_MAX_PAST_BUCKETS,
    }
}

/// Per-phase timings from [`LocateAnythingRunner::generate_with_profile`].
#[derive(Debug, Clone, Default)]
pub struct GenerateProfile {
    pub vision_ms: f64,
    pub fuse_embed_ms: f64,
    pub prefill_ms: f64,
    pub decode_mtp_ms: f64,
    pub prefill_cache_hit: bool,
    pub vision_cache_hit: bool,
    /// Decode used GPU-resident K/V (`bind_gpu_handle` on MLX / Metal).
    pub gpu_kv_resident: bool,
}

#[derive(Clone)]
struct CachedPrefill {
    grid_h: usize,
    grid_w: usize,
    prompt_ids: Vec<u32>,
    kv: KvCacheState,
    prefill_logits: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct LocateAnythingRunnerBuilder {
    weights: Option<PathBuf>,
    config_path: Option<PathBuf>,
    config: Option<LocateAnythingConfig>,
    device: Option<Device>,
    sample: SampleOpts,
}

impl LocateAnythingRunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights = Some(path.into());
        self
    }

    pub fn config_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config_path = Some(path.into());
        self
    }

    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn max_new_tokens(mut self, n: usize) -> Self {
        self.sample.max_new_tokens = n;
        self
    }

    pub fn generation_mode(mut self, mode: GenerationMode) -> Self {
        self.sample.mode = mode;
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.sample.temperature = t;
        self
    }

    pub fn repetition_penalty(mut self, r: f32) -> Self {
        self.sample.repetition_penalty = r;
        self
    }

    pub fn build(self) -> Result<LocateAnythingRunner> {
        let weights_path = self
            .weights
            .ok_or_else(|| anyhow::anyhow!("weights path required"))?;
        let model_dir = resolve_model_dir(&weights_path)?;
        let cfg_path = self
            .config_path
            .clone()
            .unwrap_or_else(|| model_dir.join("config.json"));
        let cfg = match self.config {
            Some(c) => c,
            None => LocateAnythingConfig::from_file(&cfg_path)
                .with_context(|| format!("reading {cfg_path:?}"))?,
        };
        cfg.validate()?;
        let device = self.device.unwrap_or(Device::Cpu);
        validate_standard_device("locateanything", device)?;
        Ok(LocateAnythingRunner {
            cfg,
            device,
            sample: self.sample,
            store: LocateAnythingWeightStore::open(&weights_path)?,
            vit_cache: MoonVitCache::default(),
            lm_caches: LmSessionCaches::new(device, lm_max_past_buckets(device)),
            cached_projected_vision: None,
            prefill_cache: None,
        })
    }
}

impl LocateAnythingRunner {
    /// Drop cached LM prefill (e.g. after image or prompt change).
    pub fn clear_prefill_cache(&mut self) {
        self.prefill_cache = None;
    }
}

pub struct LocateAnythingRunner {
    pub cfg: LocateAnythingConfig,
    device: Device,
    sample: SampleOpts,
    store: LocateAnythingWeightStore,
    vit_cache: MoonVitCache,
    lm_caches: LmSessionCaches,
    /// Projector output for the last `(grid_h, grid_w)` — skips MoonViT+projector on repeat.
    cached_projected_vision: Option<(usize, usize, Vec<f32>)>,
    /// Reuse LM prefill KV for repeated `ground()` on the same image + prompt.
    prefill_cache: Option<CachedPrefill>,
}

impl LocateAnythingRunner {
    pub fn builder() -> LocateAnythingRunnerBuilder {
        LocateAnythingRunnerBuilder::default()
    }

    pub fn model_dir(&self) -> &Path {
        self.store.model_dir()
    }

    pub fn preprocess_image(&self, img: &image::DynamicImage) -> Result<PreprocessedImage> {
        preprocess_image(img, &self.cfg)
    }

    pub fn preprocess_path(&self, path: &Path) -> Result<PreprocessedImage> {
        preprocess_path(path, &self.cfg)
    }

    pub fn encode_vision(&self, img: &PreprocessedImage) -> Result<Vec<f32>> {
        let vit_cfg = self.cfg.vision_config.clone();
        let mut wm = self.store.load_vision_weights()?;
        let mut vit_cache = MoonVitCache::default();
        let merged = vit_cache.encode(&vit_cfg, Some(&mut wm), img, self.device)?;
        let n_tokens = merged.len() / self.cfg.projector_input_dim();
        let mut wm_p = self.store.load_projector_weights()?;
        let proj_built = build_projector_built(&self.cfg, &mut wm_p, 1, n_tokens)?;
        let params = proj_built.model.params().clone();
        let mut proj = compile_built(proj_built.model, self.device)?;
        for (n, d) in &params {
            proj.set_param(n, d);
        }
        proj.run(&[("vision", merged.as_slice())])
            .into_iter()
            .next()
            .context("projector output")
    }

    /// MoonViT + projector on `device` (compiled graphs cached per grid / token count).
    pub fn encode_vision_cached(&mut self, img: &PreprocessedImage) -> Result<Vec<f32>> {
        if let Some((gh, gw, ref out)) = self.cached_projected_vision {
            if gh == img.grid_h && gw == img.grid_w {
                return Ok(out.clone());
            }
        }
        let vit_cfg = self.cfg.vision_config.clone();
        let enc_device = vision_encode_device(self.device);
        let merged = if self.vit_cache.has_graph(img, enc_device) {
            self.vit_cache.encode(&vit_cfg, None, img, enc_device)?
        } else {
            let mut wm = self.store.load_vision_weights()?;
            self.vit_cache
                .encode(&vit_cfg, Some(&mut wm), img, enc_device)?
        };
        let n_tokens = merged.len() / self.cfg.projector_input_dim();
        let cfg = self.cfg.clone();
        let store = self.store.clone();
        let proj = self.lm_caches.projector_graph(n_tokens, || {
            let mut wm_p = store.load_projector_weights()?;
            let built = build_projector_built(&cfg, &mut wm_p, 1, n_tokens)?;
            let params = built.model.params().clone();
            let mut compiled = compile_built(built.model, enc_device)?;
            for (n, d) in &params {
                compiled.set_param(n, d);
            }
            Ok(compiled)
        })?;
        let out = proj
            .run(&[("vision", merged.as_slice())])
            .into_iter()
            .next()
            .context("projector output")?;
        self.cached_projected_vision = Some((img.grid_h, img.grid_w, out.clone()));
        Ok(out)
    }

    fn ensure_lm_weights(&mut self) -> Result<()> {
        self.lm_caches.ensure_lm_store(Arc::new(self.store.clone()));
        Ok(())
    }

    fn prefill_logits_mtp(
        &mut self,
        past_len: usize,
        kv: &mut KvCacheState,
        window_ids: &[u32],
        vision: &[f32],
    ) -> Result<(Vec<f32>, KvCacheState)> {
        self.ensure_lm_weights()?;
        let seq = window_ids.len();
        let q_len = seq.saturating_sub(past_len);
        let block = self.cfg.text_config.block_size;
        ensure!(
            q_len == block,
            "mtp query len {q_len} != block_size {block}"
        );
        let text_mask = crate::generation::TokenIds::from_config(&self.cfg).text_mask;
        let causal = self.cfg.text_config.causal_attn;
        let mask_2d = mtp_prefill_mask_2d(window_ids, text_mask, block, true, causal);

        let q_ids = &window_ids[past_len..];
        // MTP query window is text-only; vision was fused during the initial prompt prefill.
        let inputs_embeds = if q_ids.contains(&self.cfg.image_token_index) {
            fuse_inputs_embeds_from_store(&self.cfg, &self.store, q_ids, vision)?
        } else {
            fuse_inputs_embeds_from_store(&self.cfg, &self.store, q_ids, &[])?
        };

        let qcfg = qwen3_config(&self.cfg);
        let (rope_cos, rope_sin) = compute_rope_chunk(&qcfg, past_len, q_len);
        self.lm_caches.mtp_logits(
            &self.cfg,
            past_len,
            q_len,
            &inputs_embeds,
            &mask_2d,
            seq,
            &rope_cos,
            &rope_sin,
            kv,
        )
    }

    fn prefill_logits(
        &mut self,
        inputs_embeds: &[f32],
        seq: usize,
    ) -> Result<(Vec<f32>, KvCacheState)> {
        self.ensure_lm_weights()?;
        let layers = self.cfg.text_config.num_hidden_layers;
        let (logits, kv_flat) = self
            .lm_caches
            .prefill_with_kv(&self.cfg, seq, inputs_embeds)?;
        let kv_dim = self.cfg.text_config.num_key_value_heads * self.cfg.text_config.head_dim();
        let kv = kv_state_from_runner(seq, &kv_flat, layers, kv_dim)?;
        Ok((logits, kv))
    }

    /// Greedy / sampled continuation from `prompt_ids` (image placeholders required).
    pub fn generate(&mut self, prompt_ids: &[u32], img: &PreprocessedImage) -> Result<Vec<u32>> {
        self.generate_with_profile(prompt_ids, img).map(|(t, _)| t)
    }

    /// Like [`Self::generate`] but returns per-phase timings (vision / prefill / decode).
    pub fn generate_with_profile(
        &mut self,
        prompt_ids: &[u32],
        img: &PreprocessedImage,
    ) -> Result<(Vec<u32>, GenerateProfile)> {
        let mut profile = GenerateProfile::default();
        let t0 = Instant::now();
        let had_vision = self
            .cached_projected_vision
            .as_ref()
            .is_some_and(|(gh, gw, _)| *gh == img.grid_h && *gw == img.grid_w);
        profile.vision_cache_hit = had_vision;
        let vision = self.encode_vision_cached(img)?;
        profile.vision_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let h = self.cfg.text_config.hidden_size;
        let n_image = vision.len() / h;
        let t0 = Instant::now();
        let inputs_embeds =
            fuse_inputs_embeds_from_store(&self.cfg, &self.store, prompt_ids, &vision)?;
        profile.fuse_embed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        ensure!(
            prompt_ids
                .iter()
                .filter(|&&t| t == self.cfg.image_token_index)
                .count()
                == n_image,
            "image token count mismatch"
        );
        let tokens = self.generate_from_embeds_profile(
            prompt_ids,
            &inputs_embeds,
            Some(&vision),
            Some(img),
            &mut profile,
        )?;
        Ok((tokens, profile))
    }

    /// Continue generation from pre-fused `inputs_embeds` (`Slow` uses LM only; `Fast`/`Hybrid` need `vision_for_mtp`).
    pub fn generate_from_embeds(
        &mut self,
        prompt_ids: &[u32],
        inputs_embeds: &[f32],
        vision_for_mtp: Option<&[f32]>,
    ) -> Result<Vec<u32>> {
        let mut profile = GenerateProfile::default();
        self.generate_from_embeds_profile(
            prompt_ids,
            inputs_embeds,
            vision_for_mtp,
            None,
            &mut profile,
        )
    }

    fn generate_from_embeds_profile(
        &mut self,
        prompt_ids: &[u32],
        inputs_embeds: &[f32],
        vision_for_mtp: Option<&[f32]>,
        img: Option<&PreprocessedImage>,
        profile: &mut GenerateProfile,
    ) -> Result<Vec<u32>> {
        let h = self.cfg.text_config.hidden_size;
        let seq = prompt_ids.len();
        ensure!(
            inputs_embeds.len() == seq * h,
            "inputs_embeds len {} != seq * hidden {}",
            inputs_embeds.len(),
            seq * h
        );

        let vision = match (self.sample.mode, vision_for_mtp) {
            (GenerationMode::Slow, _) => None,
            (_, Some(v)) => Some(v),
            _ => anyhow::bail!("Fast/Hybrid generate_from_embeds requires vision_for_mtp"),
        };

        let vocab = self.cfg.text_config.vocab_size;
        let layers = self.cfg.text_config.num_hidden_layers;

        let (mut kv, mut next) = if let (Some(img), Some(c)) = (img, self.prefill_cache.as_ref()) {
            if c.grid_h == img.grid_h && c.grid_w == img.grid_w && c.prompt_ids == prompt_ids {
                profile.prefill_cache_hit = true;
                (
                    c.kv.clone(),
                    sample_ar(&c.prefill_logits, &self.sample, prompt_ids),
                )
            } else {
                self.run_prefill_timed(inputs_embeds, seq, layers, prompt_ids, Some(img), profile)?
            }
        } else {
            self.run_prefill_timed(inputs_embeds, seq, layers, prompt_ids, img, profile)?
        };

        let mut tokens: Vec<u32> = prompt_ids.to_vec();

        let mut past_len = prompt_ids.len();
        let qcfg = qwen3_config(&self.cfg);
        let ids = TokenIds::from_config(&self.cfg);
        let block = self.cfg.text_config.block_size;
        let text_mask = ids.text_mask;
        let kv_dim = self.cfg.text_config.num_key_value_heads * self.cfg.text_config.head_dim();
        let mode_str = match self.sample.mode {
            GenerationMode::Fast => "fast",
            GenerationMode::Slow => "slow",
            GenerationMode::Hybrid => "hybrid",
        };
        let mut use_mtp = matches!(
            self.sample.mode,
            GenerationMode::Fast | GenerationMode::Hybrid
        );
        profile.gpu_kv_resident = lm_gpu_kv_enabled(self.device);
        let decode_t0 = Instant::now();

        for _ in 0..self.sample.max_new_tokens {
            if next == ids.im_end {
                break;
            }

            let mut append: Vec<u32> = Vec::new();
            let mut mtp_kv: Option<KvCacheState> = None;
            let mut mtp_logits_slab: Option<Vec<f32>> = None;

            if use_mtp && tokens.len() < 2048 {
                self.lm_caches
                    .sync_kv_from_gpu(&self.cfg, past_len, &mut kv)?;
                let mut window = tokens.clone();
                let last = *window.last().unwrap_or(&next);
                window.push(last);
                window.extend(std::iter::repeat_n(text_mask, block.saturating_sub(1)));
                let mtp_past = tokens.len();
                let (all_logits, new_kv) = self.prefill_logits_mtp(
                    mtp_past,
                    &mut kv,
                    &window,
                    vision.expect("vision_for_mtp"),
                )?;
                mtp_kv = Some(new_kv);
                mtp_logits_slab = Some(all_logits);
                let slab = &mtp_logits_slab.as_ref().expect("mtp slab")[..block * vocab];
                let mode_name = mode_str;
                if let Some(box_tokens) = decode_bbox_block(slab, vocab, &ids, mode_name) {
                    let pat = handle_pattern(&box_tokens, &ids, mode_name);
                    if pat.terminal {
                        tokens.extend(pat.tokens);
                        break;
                    }
                    append = pat.tokens;
                    if self.sample.mode == GenerationMode::Hybrid && pat.need_ar {
                        use_mtp = false;
                    }
                } else if self.sample.mode == GenerationMode::Hybrid {
                    use_mtp = false;
                    append.push(next);
                    mtp_kv = None;
                } else {
                    append.push(next);
                    mtp_kv = None;
                }
            } else {
                append.push(next);
            }

            let mtp_bulk = mtp_kv.is_some() && append.len() > 1;
            if mtp_bulk {
                let mtp_past = past_len;
                for t in &append {
                    if *t == ids.im_end {
                        tokens.push(*t);
                        return Ok(tokens);
                    }
                    tokens.push(*t);
                }
                let committed = append.len();
                kv =
                    truncate_kv_state(mtp_kv.take().expect("mtp kv"), mtp_past, committed, kv_dim)?;
                self.lm_caches.reset_decode_after_mtp();
                past_len = kv.past_len;
                let row = committed.saturating_sub(1);
                let slab = mtp_logits_slab.as_ref().expect("mtp slab");
                let row_logits = &slab[row * vocab..(row + 1) * vocab];
                next = sample_ar(row_logits, &self.sample, &tokens);
                continue;
            }

            for t in append {
                if t == ids.im_end {
                    tokens.push(t);
                    return Ok(tokens);
                }
                tokens.push(t);

                let (cos, sin) = compute_rope_slice(&qcfg, past_len);
                let mtp_window = if use_mtp {
                    Some((block, past_len))
                } else {
                    None
                };
                let logits = self.lm_caches.decode_step_in_place(
                    &self.cfg, past_len, t, &cos, &sin, mtp_window, &mut kv,
                )?;
                next = sample_ar(&logits, &self.sample, &tokens);
                past_len = kv.past_len;

                if self.sample.mode == GenerationMode::Hybrid && !use_mtp {
                    let out_type = crate::generation::classify_ar_token(next, &ids);
                    if out_type == "box_end_ar" {
                        use_mtp = true;
                    }
                    if out_type == "im_end" {
                        tokens.push(next);
                        return Ok(tokens);
                    }
                }
            }
        }

        profile.decode_mtp_ms = decode_t0.elapsed().as_secs_f64() * 1000.0;

        Ok(tokens)
    }

    fn run_prefill_timed(
        &mut self,
        inputs_embeds: &[f32],
        seq: usize,
        layers: usize,
        prompt_ids: &[u32],
        img: Option<&PreprocessedImage>,
        profile: &mut GenerateProfile,
    ) -> Result<(KvCacheState, u32)> {
        let t0 = Instant::now();
        let (logits, kv_flat) = {
            self.ensure_lm_weights()?;
            self.lm_caches
                .prefill_with_kv(&self.cfg, seq, inputs_embeds)?
        };
        profile.prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let kv_dim = self.cfg.text_config.num_key_value_heads * self.cfg.text_config.head_dim();
        let kv = kv_state_from_runner(seq, &kv_flat, layers, kv_dim)?;
        let vocab = self.cfg.text_config.vocab_size;
        ensure!(logits.len() == vocab);
        let next = sample_ar(&logits, &self.sample, prompt_ids);
        if let Some(img) = img {
            self.prefill_cache = Some(CachedPrefill {
                grid_h: img.grid_h,
                grid_w: img.grid_w,
                prompt_ids: prompt_ids.to_vec(),
                kv: kv.clone(),
                prefill_logits: logits,
            });
        }
        Ok((kv, next))
    }

    pub fn generate_path(&mut self, prompt_ids: &[u32], image_path: &Path) -> Result<Vec<u32>> {
        let img = self.preprocess_path(image_path)?;
        self.generate(prompt_ids, &img)
    }

    /// Load LM weights into compile caches (optional before first query).
    pub fn preload_language_model(&mut self) -> Result<()> {
        self.ensure_lm_weights()
    }

    /// Compile vision + projector + LM prefill for this prompt shape (no generation).
    pub fn warmup_compile(&mut self, prompt_ids: &[u32], img: &PreprocessedImage) -> Result<()> {
        let vision = self.encode_vision_cached(img)?;
        let inputs_embeds =
            fuse_inputs_embeds_from_store(&self.cfg, &self.store, prompt_ids, &vision)?;
        let seq = prompt_ids.len();
        self.prefill_logits(&inputs_embeds, seq)?;
        Ok(())
    }

    /// Build prompt ids from user text + image patch count (requires `tokenizer` feature).
    #[cfg(feature = "tokenizer")]
    pub fn build_prompt_from_text(
        &self,
        user_text: &str,
        img: &PreprocessedImage,
    ) -> Result<Vec<u32>> {
        let tok = crate::tokenizer::load_tokenizer(self.model_dir())?;
        let kh = self.cfg.vision_config.merge_kernel_size[0];
        let kw = self.cfg.vision_config.merge_kernel_size[1];
        let n_image = (img.grid_h / kh) * (img.grid_w / kw);
        crate::tokenizer::build_user_prompt_ids(&self.cfg, &tok, user_text, n_image)
    }

    /// HF `LocateAnythingProcessor` layout (`<image-1>` → `<image 1><img>…</img>`, system message).
    #[cfg(feature = "tokenizer")]
    pub fn build_prompt_processor(
        &self,
        user_text_with_placeholder: &str,
        img: &PreprocessedImage,
    ) -> Result<Vec<u32>> {
        let tok = crate::tokenizer::load_tokenizer(self.model_dir())?;
        let kh = self.cfg.vision_config.merge_kernel_size[0];
        let kw = self.cfg.vision_config.merge_kernel_size[1];
        let n_image = (img.grid_h / kh) * (img.grid_w / kw);
        crate::processor_prompt::build_processor_prompt_ids(
            self.model_dir(),
            &self.cfg,
            &tok,
            user_text_with_placeholder,
            n_image,
        )
    }
}
