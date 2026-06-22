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

//! BioCLIP-2 runner — loads the OpenCLIP checkpoint, compiles the vision
//! and text towers (batch=1), and exposes image/text encoding + zero-shot
//! classification.

use crate::config::BioClip2Config;
use crate::preprocess::{
    VisionEmbedWeights, assemble_vision_hidden, clip_normalize_nchw, extract_vision_embed_weights,
};
use crate::text_embed::{TextEmbedWeights, assemble_text_hidden, extract_text_embed_weights};
use crate::tokenizer::{ClipTokenizer, eot_index};
use anyhow::{Result, anyhow};
use image::DynamicImage;
use rlx_core::validate_standard_device;
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};
use std::path::{Path, PathBuf};

/// Builder for [`BioClip2Runner`].
#[derive(Debug, Clone, Default)]
pub struct BioClip2RunnerBuilder {
    model_dir: Option<PathBuf>,
    weights: Option<PathBuf>,
    config: Option<BioClip2Config>,
    device: Option<Device>,
    patch_features: bool,
    batch: Option<usize>,
}

impl BioClip2RunnerBuilder {
    /// Model directory containing `open_clip_model.safetensors`,
    /// `open_clip_config.json`, and `tokenizer.json`.
    pub fn model_dir<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.model_dir = Some(p.into());
        self
    }
    /// Override the weights file path (defaults to
    /// `<model_dir>/open_clip_model.safetensors`).
    pub fn weights<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn config(mut self, cfg: BioClip2Config) -> Self {
        self.config = Some(cfg);
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    /// Build the vision tower to output per-patch features instead of
    /// the CLS-projected image embedding. When set, `encode_image_nchw`
    /// returns `[n_patches × width]` (e.g. 256×1024 for ViT-L/14 at
    /// 224 px), the equivalent of DINOv2's dense patch tokens. Disables
    /// the text-tower head as well; zero-shot won't work in this mode.
    pub fn patch_features(mut self, enabled: bool) -> Self {
        self.patch_features = enabled;
        self
    }
    /// Compile both towers for a fixed batch size `n` (default 1). The
    /// batched `encode_images_nchw` / `encode_texts_ids` then run up to
    /// `n` items per graph invocation; single-item methods and zero-shot
    /// chunk transparently. Larger `n` trades memory for throughput.
    pub fn batch(mut self, n: usize) -> Self {
        self.batch = Some(n.max(1));
        self
    }

    pub fn build(self) -> Result<BioClip2Runner> {
        let model_dir = self
            .model_dir
            .clone()
            .ok_or_else(|| anyhow!("model_dir required (call .model_dir(...))"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        validate_standard_device("bioclip2", device)?;

        let cfg = match self.config {
            Some(c) => c,
            None => {
                let cfg_path = model_dir.join("open_clip_config.json");
                if cfg_path.exists() {
                    BioClip2Config::from_open_clip_json(&cfg_path)?
                } else {
                    BioClip2Config::vit_l_14()
                }
            }
        };

        let weights_path = self
            .weights
            .unwrap_or_else(|| model_dir.join("open_clip_model.safetensors"));
        let mut wm = WeightMap::from_file(
            weights_path
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 weights path"))?,
        )?;

        // Host-side embed weights (vision conv1 stem; text token+positional).
        let vision_pre = extract_vision_embed_weights(&mut wm, &cfg)?;
        let text_pre = extract_text_embed_weights(&mut wm, &cfg)?;

        // Build + compile both towers at the configured batch size.
        let batch = self.batch.unwrap_or(1).max(1);
        let vision_built = if self.patch_features {
            crate::flow::build_vision_features_flow(&cfg, &mut wm, batch)?
        } else {
            crate::flow::build_vision_flow(&cfg, &mut wm, batch)?
        };
        let text_built = crate::flow::build_text_flow(&cfg, &mut wm, batch)?;

        // Host-side text projection + logit scale (not part of the graphs).
        let (text_projection, tp_shape) = wm.take("text_projection")?;
        anyhow::ensure!(
            tp_shape == vec![cfg.text.width, cfg.embed_dim],
            "text_projection shape {tp_shape:?} != [{}, {}]",
            cfg.text.width,
            cfg.embed_dim
        );
        let (logit_scale_raw, _) = wm.take("logit_scale")?;
        let logit_scale = logit_scale_raw
            .first()
            .copied()
            .ok_or_else(|| anyhow!("empty logit_scale"))?;

        let vision = rlx_core::flow_util::compile_built(vision_built, device)?;
        let text = rlx_core::flow_util::compile_built(text_built, device)?;

        let tokenizer = ClipTokenizer::from_path(&model_dir, cfg.text.context_length)?;

        Ok(BioClip2Runner {
            cfg,
            device,
            batch,
            vision,
            text,
            vision_pre,
            text_pre,
            tokenizer,
            text_projection,
            logit_scale,
        })
    }
}

/// Resolved BioCLIP-2 runner.
pub struct BioClip2Runner {
    cfg: BioClip2Config,
    device: Device,
    batch: usize,
    vision: CompiledGraph,
    text: CompiledGraph,
    vision_pre: VisionEmbedWeights,
    text_pre: TextEmbedWeights,
    tokenizer: ClipTokenizer,
    text_projection: Vec<f32>,
    logit_scale: f32,
}

impl BioClip2Runner {
    pub fn builder() -> BioClip2RunnerBuilder {
        BioClip2RunnerBuilder::default()
    }
    pub fn config(&self) -> &BioClip2Config {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn logit_scale(&self) -> f32 {
        self.logit_scale
    }
    /// Compiled batch size (graphs run up to this many items per call).
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Encode a CLIP-normalized NCHW pixel tensor `[3·img·img]` into a raw
    /// (unnormalized) image embedding `[embed_dim]` (or `[n_patches·width]`
    /// in `patch_features` mode).
    pub fn encode_image_nchw(&mut self, nchw: &[f32]) -> Result<Vec<f32>> {
        Ok(self
            .encode_images_nchw(&[nchw])?
            .into_iter()
            .next()
            .expect("one input → one output"))
    }

    /// Batched variant of [`Self::encode_image_nchw`]. Items are processed
    /// `batch` at a time (the final chunk is padded internally and the
    /// padding discarded). Returns one embedding per input, in order.
    pub fn encode_images_nchw(&mut self, items: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        let b = self.batch;
        let mut out = Vec::with_capacity(items.len());
        for chunk in items.chunks(b) {
            out.extend(self.run_vision_chunk(chunk)?);
        }
        Ok(out)
    }

    /// Encode an image into a raw image embedding (applies CLIP preprocessing).
    pub fn encode_image(&mut self, img: &DynamicImage) -> Result<Vec<f32>> {
        let nchw = self.preprocess_image(img);
        self.encode_image_nchw(&nchw)
    }

    /// Run one ≤`batch` chunk of vision inputs (pads to `batch`).
    fn run_vision_chunk(&mut self, items: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        debug_assert!(!items.is_empty() && items.len() <= self.batch);
        let b = self.batch;
        let img = self.cfg.vision.image_size;
        let ps = self.cfg.vision.patch_size;
        let seq = self.cfg.vision.seq_len();
        let width = self.cfg.vision.width;
        let mut hidden = Vec::with_capacity(b * seq * width);
        for i in 0..b {
            let nchw = items[i.min(items.len() - 1)];
            hidden.extend_from_slice(&assemble_vision_hidden(&self.vision_pre, nchw, 1, ps, img)?);
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

    /// CLIP-preprocess a decoded image to a normalized NCHW pixel tensor.
    fn preprocess_image(&self, img: &DynamicImage) -> Vec<f32> {
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        clip_normalize_nchw(rgb.as_raw(), h, w, self.cfg.vision.image_size)
    }

    /// Encode raw token ids `[ctx]` into a raw text embedding `[embed_dim]`.
    pub fn encode_text_ids(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        Ok(self
            .encode_texts_ids(std::slice::from_ref(&ids))?
            .into_iter()
            .next()
            .expect("one input → one output"))
    }

    /// Batched text encoding. Each `ids` row must be `ctx`-length; rows are
    /// processed `batch` at a time. Returns one embedding per row.
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

    /// Run one ≤`batch` chunk of token sequences (pads to `batch`), pooling
    /// each at its EOT and projecting through `text_projection` on host.
    fn run_text_chunk(&mut self, ids: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        debug_assert!(!ids.is_empty() && ids.len() <= self.batch);
        let b = self.batch;
        let ctx = self.cfg.text.context_length;
        let width = self.cfg.text.width;
        let embed = self.cfg.embed_dim;
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
        for (i, row_ids) in ids.iter().enumerate() {
            let eot = eot_index(row_ids);
            let seq_out = &flat[i * per..(i + 1) * per];
            let row = &seq_out[eot * width..(eot + 1) * width];
            // pooled @ text_projection ([width, embed], row-major, no transpose).
            let mut feat = vec![0f32; embed];
            for (k, &x) in row.iter().enumerate() {
                if x == 0.0 {
                    continue;
                }
                let base = k * embed;
                for j in 0..embed {
                    feat[j] += x * self.text_projection[base + j];
                }
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

    /// Zero-shot classification logits per image: `exp(logit_scale) *
    /// normalize(image) · normalize(text)ᵀ`. Returns `[n_images][n_labels]`.
    /// Text and image towers run batched at the compiled `batch` size.
    pub fn zeroshot(
        &mut self,
        images: &[DynamicImage],
        labels: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        let scale = self.logit_scale.exp();

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

        let nchw: Vec<Vec<f32>> = images.iter().map(|im| self.preprocess_image(im)).collect();
        let nchw_refs: Vec<&[f32]> = nchw.iter().map(|v| v.as_slice()).collect();
        let imgs: Vec<Vec<f32>> = self
            .encode_images_nchw(&nchw_refs)?
            .into_iter()
            .map(l2_normalize)
            .collect();

        Ok(imgs
            .iter()
            .map(|img| txt.iter().map(|t| scale * dot(img, t)).collect())
            .collect())
    }
}

/// Download the BioCLIP-2 checkpoint files into `dest` via `hf` if missing.
/// (Helper for the CLI; returns `dest` on success.)
pub fn ensure_model_dir(dest: &Path) -> Result<PathBuf> {
    let needed = dest.join("open_clip_model.safetensors");
    if needed.exists() {
        return Ok(dest.to_path_buf());
    }
    anyhow::bail!(
        "model files not found in {dest:?}. Download with:\n  \
         hf download imageomics/bioclip-2 --local-dir {dest:?}"
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
