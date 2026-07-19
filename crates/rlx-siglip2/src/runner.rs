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

//! SigLIP 2 runner — loads the checkpoint, compiles the vision and text
//! towers, and exposes image/text encoding + sigmoid zero-shot. Supports
//! both the fixed-resolution and NaFlex families (the latter compiled at
//! `seq = max_num_patches`, batch 1).

use crate::config::{Siglip2Config, Variant};
use crate::naflex::{
    NaflexEmbedWeights, NaflexInput, assemble_naflex_hidden, build_key_mask,
    extract_naflex_embed_weights,
};
use crate::preprocess::{
    VisionEmbedWeights, assemble_vision_hidden, extract_pooling_weights,
    extract_vision_embed_weights, siglip_normalize_nchw,
};
use crate::text_embed::{TextEmbedWeights, assemble_text_hidden, extract_text_embed_weights};
use crate::tokenizer::SiglipTokenizer;
use anyhow::{Result, anyhow};
use image::DynamicImage;
use rlx_core::validate_standard_device;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};
use std::path::{Path, PathBuf};

/// Host-side vision stem, per architecture family.
enum VisionPre {
    Fixed(VisionEmbedWeights),
    NaFlex(NaflexEmbedWeights),
}

/// Builder for [`Siglip2Runner`].
#[derive(Debug, Clone, Default)]
pub struct Siglip2RunnerBuilder {
    model_dir: Option<PathBuf>,
    weights: Option<PathBuf>,
    config: Option<Siglip2Config>,
    device: Option<Device>,
    batch: Option<usize>,
}

impl Siglip2RunnerBuilder {
    /// Model directory containing `model.safetensors`, `config.json`, and
    /// `tokenizer.json`.
    pub fn model_dir<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.model_dir = Some(p.into());
        self
    }
    /// Override the weights path (default `<model_dir>/model.safetensors`).
    pub fn weights<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.weights = Some(p.into());
        self
    }
    /// Override the config (default: parsed from `<model_dir>/config.json`).
    pub fn config(mut self, cfg: Siglip2Config) -> Self {
        self.config = Some(cfg);
        self
    }
    /// Target device (default [`Device::Cpu`]).
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    /// Compile both towers for a fixed batch size `n` (default 1). Forced to
    /// 1 for NaFlex (each image carries its own padding mask).
    pub fn batch(mut self, n: usize) -> Self {
        self.batch = Some(n.max(1));
        self
    }

    /// Load the checkpoint, compile both towers for `device`, and resolve the
    /// tokenizer. The architecture family is taken from `config.json`.
    pub fn build(self) -> Result<Siglip2Runner> {
        let model_dir = self
            .model_dir
            .clone()
            .ok_or_else(|| anyhow!("model_dir required (call .model_dir(...))"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        validate_standard_device("siglip2", device)?;

        let cfg = match self.config {
            Some(c) => c,
            None => {
                let cfg_path = model_dir.join("config.json");
                if cfg_path.exists() {
                    Siglip2Config::from_hf_config_json(&cfg_path)?
                } else {
                    Siglip2Config::base_patch16_224()
                }
            }
        };

        let weights_path = self
            .weights
            .unwrap_or_else(|| model_dir.join("model.safetensors"));
        let mut wm = WeightMap::from_file(
            weights_path
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 weights path"))?,
        )?;

        let batch = match cfg.variant {
            Variant::Fixed => self.batch.unwrap_or(1).max(1),
            Variant::NaFlex => 1,
        };

        // Host-side vision stem + MAP-head split (taken before graph build).
        let vision_pre = match cfg.variant {
            Variant::Fixed => VisionPre::Fixed(extract_vision_embed_weights(&mut wm, &cfg)?),
            Variant::NaFlex => VisionPre::NaFlex(extract_naflex_embed_weights(&mut wm, &cfg)?),
        };
        let pooling = extract_pooling_weights(&mut wm, &cfg, batch)?;
        let text_pre = extract_text_embed_weights(&mut wm, &cfg)?;

        // Host-side text head + logit scalars.
        let (text_head_w, thw_shape) = wm.take("text_model.head.weight")?;
        anyhow::ensure!(
            thw_shape == vec![cfg.text.projection, cfg.text.width],
            "text head weight {thw_shape:?} != [{}, {}]",
            cfg.text.projection,
            cfg.text.width
        );
        let (text_head_b, _) = wm.take("text_model.head.bias")?;
        let (logit_scale_raw, _) = wm.take("logit_scale")?;
        let (logit_bias_raw, _) = wm.take("logit_bias")?;
        let logit_scale = *logit_scale_raw
            .first()
            .ok_or_else(|| anyhow!("empty logit_scale"))?;
        let logit_bias = *logit_bias_raw
            .first()
            .ok_or_else(|| anyhow!("empty logit_bias"))?;

        let vision_built = crate::flow::build_vision_flow(&cfg, &mut wm, batch, pooling)?;
        let text_built = crate::flow::build_text_flow(&cfg, &mut wm, batch)?;
        let vision = rlx_core::flow_util::compile_built(vision_built, device)?;
        let text = rlx_core::flow_util::compile_built(text_built, device)?;

        let tokenizer = SiglipTokenizer::from_path(&model_dir, cfg.text.context_length)?;

        Ok(Siglip2Runner {
            cfg,
            device,
            batch,
            vision,
            text,
            vision_pre,
            text_pre,
            tokenizer,
            text_head_w,
            text_head_b,
            logit_scale,
            logit_bias,
        })
    }
}

/// Resolved SigLIP 2 runner.
pub struct Siglip2Runner {
    cfg: Siglip2Config,
    device: Device,
    batch: usize,
    vision: CompiledGraph,
    text: CompiledGraph,
    vision_pre: VisionPre,
    text_pre: TextEmbedWeights,
    tokenizer: SiglipTokenizer,
    text_head_w: Vec<f32>,
    text_head_b: Vec<f32>,
    logit_scale: f32,
    logit_bias: f32,
}

impl Siglip2Runner {
    /// Start a new [`Siglip2RunnerBuilder`].
    pub fn builder() -> Siglip2RunnerBuilder {
        Siglip2RunnerBuilder::default()
    }
    /// The resolved model configuration.
    pub fn config(&self) -> &Siglip2Config {
        &self.cfg
    }
    /// The device both towers were compiled for.
    pub fn device(&self) -> Device {
        self.device
    }
    /// Learned `logit_scale` (raw; apply `.exp()` for the zero-shot temperature).
    pub fn logit_scale(&self) -> f32 {
        self.logit_scale
    }
    /// Learned `logit_bias` (added to scaled cosine logits before the sigmoid).
    pub fn logit_bias(&self) -> f32 {
        self.logit_bias
    }
    /// Compiled batch size (graphs run up to this many items per call).
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Encode an image into a raw (un-normalized) image embedding
    /// `[embed_dim]` (== HF `pooler_output`). Dispatches on the variant.
    pub fn encode_image(&mut self, img: &DynamicImage) -> Result<Vec<f32>> {
        match self.cfg.variant {
            Variant::Fixed => {
                let nchw = self.preprocess_fixed(img);
                self.encode_image_nchw(&nchw)
            }
            Variant::NaFlex => {
                let rgb = img.to_rgb8();
                let (w, h) = (rgb.width() as usize, rgb.height() as usize);
                let input = crate::naflex::preprocess(
                    rgb.as_raw(),
                    h,
                    w,
                    self.cfg.vision.patch_size,
                    self.cfg.vision.num_positions,
                )?;
                self.run_vision_naflex(&input)
            }
        }
    }

    // ---- Fixed-resolution image path ------------------------------------

    /// Encode a SigLIP-normalized NCHW pixel tensor `[3·img·img]`
    /// (fixed-resolution only).
    pub fn encode_image_nchw(&mut self, nchw: &[f32]) -> Result<Vec<f32>> {
        Ok(self
            .encode_images_nchw(&[nchw])?
            .into_iter()
            .next()
            .expect("one input → one output"))
    }

    /// Batched fixed-resolution image encoding.
    pub fn encode_images_nchw(&mut self, items: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        let b = self.batch;
        let mut out = Vec::with_capacity(items.len());
        for chunk in items.chunks(b) {
            out.extend(self.run_vision_chunk(chunk)?);
        }
        Ok(out)
    }

    fn run_vision_chunk(&mut self, items: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        debug_assert!(!items.is_empty() && items.len() <= self.batch);
        let VisionPre::Fixed(pre) = &self.vision_pre else {
            anyhow::bail!(
                "encode_images_nchw is fixed-resolution only; use encode_image for NaFlex"
            );
        };
        let b = self.batch;
        let img = self.cfg.vision.image_size;
        let ps = self.cfg.vision.patch_size;
        let np = self.cfg.vision.num_patches();
        let width = self.cfg.vision.width;
        let mut hidden = Vec::with_capacity(b * np * width);
        for i in 0..b {
            let nchw = items[i.min(items.len() - 1)];
            hidden.extend_from_slice(&assemble_vision_hidden(pre, nchw, 1, ps, img)?);
        }
        let flat = self
            .vision
            .run(&[("hidden", hidden.as_slice())])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("vision forward returned no output"))?;
        let per = flat.len() / b;
        Ok((0..items.len())
            .map(|i| flat[i * per..(i + 1) * per].to_vec())
            .collect())
    }

    fn preprocess_fixed(&self, img: &DynamicImage) -> Vec<f32> {
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        siglip_normalize_nchw(rgb.as_raw(), h, w, self.cfg.vision.image_size)
    }

    // ---- NaFlex image path ----------------------------------------------

    /// Encode NaFlex `pixel_values` `[n_patches·patch_dim]` (already
    /// normalized + padded to `max_num_patches`) with grid `(nph, npw)`.
    /// Used to compare against HF `pixel_values` directly.
    pub fn encode_naflex_patches(
        &mut self,
        pixel_values: &[f32],
        nph: usize,
        npw: usize,
    ) -> Result<Vec<f32>> {
        let max = self.cfg.vision.num_positions;
        let pd = 3 * self.cfg.vision.patch_size * self.cfg.vision.patch_size;
        anyhow::ensure!(
            pixel_values.len() == max * pd,
            "pixel_values len {} != max_patches*patch_dim ({max}*{pd})",
            pixel_values.len()
        );
        let input = NaflexInput {
            pixel_values: pixel_values.to_vec(),
            nph,
            npw,
            n_valid: nph * npw,
            max_patches: max,
        };
        self.run_vision_naflex(&input)
    }

    fn run_vision_naflex(&mut self, input: &NaflexInput) -> Result<Vec<f32>> {
        let VisionPre::NaFlex(pre) = &self.vision_pre else {
            anyhow::bail!("run_vision_naflex called on a fixed-resolution model");
        };
        let seq = self.cfg.vision.num_positions;
        let hidden = assemble_naflex_hidden(pre, input);
        let key_mask = build_key_mask(input.n_valid, seq);
        let flat = self
            .vision
            .run(&[
                ("hidden", hidden.as_slice()),
                ("key_mask", key_mask.as_slice()),
            ])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("naflex vision forward returned no output"))?;
        Ok(flat)
    }

    // ---- Text path (shared) ---------------------------------------------

    /// Encode raw token ids `[ctx]` into a raw text embedding `[embed_dim]`.
    pub fn encode_text_ids(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        Ok(self
            .encode_texts_ids(std::slice::from_ref(&ids))?
            .into_iter()
            .next()
            .expect("one input → one output"))
    }

    /// Batched text encoding. Each row must be `ctx`-length.
    pub fn encode_texts_ids(&mut self, ids: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        let ctx = self.cfg.text.context_length;
        for row in ids {
            anyhow::ensure!(row.len() == ctx, "expected {ctx} ids, got {}", row.len());
        }
        let b = self.batch;
        let mut out = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(b) {
            out.extend(self.run_text_chunk(chunk)?);
        }
        Ok(out)
    }

    fn run_text_chunk(&mut self, ids: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        debug_assert!(!ids.is_empty() && ids.len() <= self.batch);
        let b = self.batch;
        let ctx = self.cfg.text.context_length;
        let width = self.cfg.text.width;
        let pool = self.tokenizer.pool_index();
        let mut hidden = Vec::with_capacity(b * ctx * width);
        for i in 0..b {
            let row = ids[i.min(ids.len() - 1)];
            hidden.extend_from_slice(&assemble_text_hidden(&self.text_pre, row)?);
        }
        let flat = self
            .text
            .run(&[("hidden", hidden.as_slice())])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("text forward returned no output"))?;
        let per = ctx * width;
        let mut out = Vec::with_capacity(ids.len());
        for i in 0..ids.len() {
            let seq_out = &flat[i * per..(i + 1) * per];
            let last = &seq_out[pool * width..(pool + 1) * width];
            // last @ headᵀ + bias  (head.weight [proj, width] row-major).
            let mut feat = self.text_head_b.clone();
            for (j, fj) in feat.iter_mut().enumerate() {
                let wr = &self.text_head_w[j * width..(j + 1) * width];
                let mut acc = *fj;
                for (k, &x) in last.iter().enumerate() {
                    acc += x * wr[k];
                }
                *fj = acc;
            }
            out.push(feat);
        }
        Ok(out)
    }

    /// Encode a text prompt into a raw text embedding.
    pub fn encode_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokenizer.encode(text)?;
        self.encode_text_ids(&ids)
    }

    /// SigLIP zero-shot logits per image: `logit_scale.exp()·⟨î,t̂⟩ +
    /// logit_bias` (== HF `logits_per_image`). Apply `sigmoid` for
    /// independent per-pair probabilities. Returns `[n_images][n_labels]`.
    pub fn zeroshot(
        &mut self,
        images: &[DynamicImage],
        labels: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        let scale = self.logit_scale.exp();
        let bias = self.logit_bias;

        let ids: Vec<Vec<u32>> = labels
            .iter()
            .map(|l| self.tokenizer.encode(l))
            .collect::<Result<_>>()?;
        let id_refs: Vec<&[u32]> = ids.iter().map(|v| v.as_slice()).collect();
        let txt: Vec<Vec<f32>> = self
            .encode_texts_ids(&id_refs)?
            .into_iter()
            .map(l2_normalize)
            .collect();

        let mut imgs: Vec<Vec<f32>> = Vec::with_capacity(images.len());
        for im in images {
            imgs.push(l2_normalize(self.encode_image(im)?));
        }

        Ok(imgs
            .iter()
            .map(|img| txt.iter().map(|t| scale * dot(img, t) + bias).collect())
            .collect())
    }
}

/// Resolve the model directory, erroring with a download hint if absent.
pub fn ensure_model_dir(dest: &Path) -> Result<PathBuf> {
    if dest.join("model.safetensors").exists() {
        return Ok(dest.to_path_buf());
    }
    anyhow::bail!(
        "model files not found in {dest:?}. Download with:\n  \
         hf download google/siglip2-base-patch16-224 --local-dir {dest:?}"
    )
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
