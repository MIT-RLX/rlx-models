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

//! Vision-tower hyper-parameters, read from a `clip`-architecture mmproj GGUF.

use anyhow::{Context, Result, bail};
use rlx_gguf::{GgufFile, MetaValue};
use std::path::Path;

/// Vision-tower hyper-parameters, from a `clip`-architecture mmproj GGUF.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    pub patch_size: usize,
    /// ViT width (`clip.vision.embedding_length`).
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    /// `clip.vision.image_size`. Informational: Qwen3-VL smart-resizes to the
    /// pixel bounds instead of to a fixed square, so nothing here constrains
    /// the input size.
    pub image_size: usize,
    pub n_ff: usize,
    /// Spatial merge factor; `merge²` patches become one LM token.
    pub n_merge: usize,
    pub eps: f32,
    pub projector_type: String,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// The LM width the projector maps to (`clip.vision.projection_dim`).
    pub llm_hidden_size: usize,
    /// Pixel bounds for [`smart_resize`](super::smart_resize).
    pub image_min_pixels: usize,
    pub image_max_pixels: usize,
    /// Axial-RoPE base. HF's `Qwen3VLVisionConfig` defaults it to 10 000 and
    /// the mmproj carries no key for it.
    pub rope_theta: f32,
    /// Layers with a deepstack side path. Empty in the published checkpoint;
    /// [`build_qwen4exp_vision_flow`](super::build_qwen4exp_vision_flow) refuses a non-empty one rather than
    /// silently dropping the extra features.
    pub deepstack_layers: Vec<usize>,
}

/// Default vision-token floor when the file pins no `image_min_pixels`.
///
/// In post-merge **tokens**, not pixels: fewer tokens is a faster ViT *and* a
/// shorter multimodal LM prefill, at some loss of fine detail.
const DEFAULT_MIN_VISION_TOKENS: usize = 576;
/// Default ceiling, matching Qwen3-VL's processor (`16777216` longest-edge
/// pixels is the preprocessor's bound; this is the token-equivalent cap).
const DEFAULT_MAX_VISION_TOKENS: usize = 4096;

impl VisionConfig {
    /// Per-head width.
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Patch grid must align to this: one merge block is `n_merge` patches a
    /// side.
    pub fn align_size(&self) -> usize {
        self.patch_size * self.n_merge
    }

    /// `merge²` — patches per LM token.
    pub fn merge_sq(&self) -> usize {
        self.n_merge * self.n_merge
    }

    /// Patch grid `(wp, hp)` for an image already sized by [`smart_resize`](super::smart_resize).
    pub fn grid(&self, img_w: usize, img_h: usize) -> (usize, usize) {
        (img_w / self.patch_size, img_h / self.patch_size)
    }

    /// Patch tokens before the merge.
    pub fn n_patch_tokens(&self, img_w: usize, img_h: usize) -> usize {
        let (wp, hp) = self.grid(img_w, img_h);
        wp * hp
    }

    /// LM tokens the image occupies after the merge.
    pub fn n_out_tokens(&self, img_w: usize, img_h: usize) -> usize {
        self.n_patch_tokens(img_w, img_h) / self.merge_sq()
    }

    /// The projector's input width, `n_embd · merge²`.
    pub fn merger_in(&self) -> usize {
        self.n_embd * self.merge_sq()
    }

    pub fn validate(&self) -> Result<()> {
        if self.n_embd == 0 || self.n_head == 0 || !self.n_embd.is_multiple_of(self.n_head) {
            bail!(
                "qwen4exp vision: embedding_length {} is not a multiple of head_count {}",
                self.n_embd,
                self.n_head
            );
        }
        // The axial RoPE splits each head into row and column halves and then
        // duplicates, so the head has to be a multiple of 4.
        if !self.head_dim().is_multiple_of(4) {
            bail!(
                "qwen4exp vision: head_dim {} must be a multiple of 4 for the \
                 axial RoPE (row/column halves, then duplicated)",
                self.head_dim()
            );
        }
        if self.n_merge == 0 || self.patch_size == 0 {
            bail!("qwen4exp vision: patch_size and spatial_merge_size must be > 0");
        }
        if !self.deepstack_layers.is_empty() {
            bail!(
                "qwen4exp vision: this mmproj declares {} deepstack layers; the \
                 published checkpoint has none and the tower does not emit them",
                self.deepstack_layers.len()
            );
        }
        Ok(())
    }

    /// Read from a `clip` mmproj GGUF's metadata.
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let u32k = |k: &str| -> Result<u32> {
            raw.metadata
                .get(k)
                .and_then(MetaValue::as_u32)
                .with_context(|| format!("qwen4exp vision: missing clip key `{k}`"))
        };
        let u32opt = |k: &str| raw.metadata.get(k).and_then(MetaValue::as_u32);
        let f32opt = |k: &str| {
            raw.metadata.get(k).and_then(|v| match v {
                MetaValue::F32(x) => Some(*x),
                MetaValue::F64(x) => Some(*x as f32),
                _ => None,
            })
        };
        let arr3 = |k: &str| -> Result<[f32; 3]> {
            let v: Vec<f32> = match raw.metadata.get(k) {
                Some(MetaValue::Array(a)) => a
                    .iter()
                    .filter_map(|x| match x {
                        MetaValue::F32(f) => Some(*f),
                        MetaValue::F64(f) => Some(*f as f32),
                        _ => None,
                    })
                    .collect(),
                _ => bail!("qwen4exp vision: missing clip key `{k}`"),
            };
            if v.len() != 3 {
                bail!("qwen4exp vision: {k} has {} floats, expected 3", v.len());
            }
            Ok([v[0], v[1], v[2]])
        };

        let n_merge = u32opt("clip.vision.spatial_merge_size")
            .or_else(|| u32opt("clip.vision.projector.scale_factor"))
            .unwrap_or(2) as usize;
        let patch_size = u32k("clip.vision.patch_size")? as usize;
        let per_token = patch_size * patch_size * n_merge * n_merge;

        // `is_deepstack_layers` is a bool-per-layer array; a true entry marks a
        // side path.
        let deepstack_layers = match raw.metadata.get("clip.vision.is_deepstack_layers") {
            Some(MetaValue::Array(a)) => a
                .iter()
                .enumerate()
                .filter_map(|(i, v)| match v {
                    MetaValue::Bool(true) => Some(i),
                    MetaValue::U32(u) if *u != 0 => Some(i),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };

        let cfg = Self {
            patch_size,
            n_embd: u32k("clip.vision.embedding_length")? as usize,
            n_head: u32k("clip.vision.attention.head_count")? as usize,
            n_layer: u32k("clip.vision.block_count")? as usize,
            image_size: u32k("clip.vision.image_size")? as usize,
            n_ff: u32k("clip.vision.feed_forward_length")? as usize,
            n_merge,
            eps: f32opt("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6),
            projector_type: raw
                .metadata
                .get("clip.projector_type")
                .and_then(MetaValue::as_str)
                .unwrap_or_default()
                .to_string(),
            image_mean: arr3("clip.vision.image_mean")?,
            image_std: arr3("clip.vision.image_std")?,
            llm_hidden_size: u32k("clip.vision.projection_dim")? as usize,
            // The file states pixel bounds only rarely; derive them from a
            // token budget, which is the quantity that actually matters.
            image_min_pixels: u32opt("clip.vision.image_min_pixels")
                .map(|v| v as usize)
                .unwrap_or(DEFAULT_MIN_VISION_TOKENS * per_token),
            image_max_pixels: u32opt("clip.vision.image_max_pixels")
                .map(|v| v as usize)
                .unwrap_or(DEFAULT_MAX_VISION_TOKENS * per_token),
            rope_theta: 10_000.0,
            deepstack_layers,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Read from a mmproj GGUF at `path` (metadata only).
    pub fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = GgufFile::from_path(path)
            .with_context(|| format!("qwen4exp vision: reading mmproj metadata from {path:?}"))?;
        Self::from_gguf(&raw)
    }
}
