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

//! SigLIP 2 configuration — mirrors HuggingFace `Siglip2Config`
//! (`SiglipVisionConfig` / `SiglipTextConfig` defaults for the
//! fixed-resolution family, `Siglip2VisionConfig` for NaFlex).
//!
//! The published fixed-resolution `config.json` files are minimal — they
//! carry only `model_type` and the text `vocab_size`, relying entirely on
//! the HF class defaults (base-16 dimensions). [`Siglip2Config::from_hf_config_json`]
//! reproduces those defaults so a bare config parses to the correct base model.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// SigLIP preprocessing mean (RGB, pixels scaled to `[0,1]`). SigLIP uses a
/// symmetric `[-1, 1]` normalization (mean = std = 0.5 on every channel).
pub const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
/// SigLIP preprocessing std (RGB).
pub const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// LayerNorm epsilon used throughout SigLIP (`layer_norm_eps`). Note this is
/// `1e-6`, *not* the `1e-5` of stock `nn.LayerNorm` / OpenCLIP.
pub const LN_EPS: f32 = 1e-6;

/// Which SigLIP 2 architecture family a checkpoint belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// `model_type = "siglip"` — fixed resolution, Conv2d patch stem, no
    /// attention mask.
    Fixed,
    /// `model_type = "siglip2"` — NaFlex: variable resolution, Linear patch
    /// stem on pre-unfolded patches, per-image position-embedding resize,
    /// and a padding attention mask.
    NaFlex,
}

/// Vision tower dimensions.
#[derive(Debug, Clone, Copy)]
pub struct VisionCfg {
    /// Fixed-resolution square input side (ignored for NaFlex).
    pub image_size: usize,
    pub patch_size: usize,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    /// FFN inner dimension (`intermediate_size`; not always `4·width`).
    pub intermediate: usize,
    /// Size of the learned position-embedding table. Fixed: `(image/patch)²`.
    /// NaFlex: `num_patches` (a `√·×√·` grid, default 256 → 16×16).
    pub num_positions: usize,
}

impl VisionCfg {
    /// Per-head attention dimension (`width / heads`).
    pub fn head_dim(&self) -> usize {
        self.width / self.heads
    }
    /// Patches for the fixed-resolution grid.
    pub fn num_patches(&self) -> usize {
        let n = self.image_size / self.patch_size;
        n * n
    }
    /// Fixed-resolution sequence length (== patches; SigLIP has no CLS token).
    pub fn seq_len(&self) -> usize {
        self.num_patches()
    }
    /// Flattened patch length (`3 · patch² `), the patch-embedding input dim.
    pub fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }
    /// Side length of the (square) NaFlex position-embedding grid.
    pub fn pos_grid_side(&self) -> usize {
        (self.num_positions as f64).sqrt().round() as usize
    }
}

/// Text tower dimensions.
#[derive(Debug, Clone, Copy)]
pub struct TextCfg {
    pub context_length: usize,
    pub vocab_size: usize,
    pub width: usize,
    pub heads: usize,
    pub layers: usize,
    pub intermediate: usize,
    /// Output projection of the text head (`projection_size`, defaults to `width`).
    pub projection: usize,
}

impl TextCfg {
    /// Per-head attention dimension (`width / heads`).
    pub fn head_dim(&self) -> usize {
        self.width / self.heads
    }
}

/// Full SigLIP 2 configuration.
#[derive(Debug, Clone, Copy)]
pub struct Siglip2Config {
    pub variant: Variant,
    /// Shared image/text embedding dimension (== text `projection`).
    pub embed_dim: usize,
    pub vision: VisionCfg,
    pub text: TextCfg,
}

impl Siglip2Config {
    /// Canonical `siglip2-base-patch16-224` (fixed resolution). All HF class
    /// defaults for `SiglipVisionConfig` / `SiglipTextConfig`, `patch16`,
    /// `image_size = 224`, multilingual `vocab_size = 256000`.
    pub fn base_patch16_224() -> Self {
        Self::base_fixed(224, 16)
    }

    /// A fixed-resolution base-16 model at an arbitrary square resolution.
    pub fn base_fixed(image_size: usize, patch_size: usize) -> Self {
        let n = image_size / patch_size;
        Self {
            variant: Variant::Fixed,
            embed_dim: 768,
            vision: VisionCfg {
                image_size,
                patch_size,
                width: 768,
                layers: 12,
                heads: 12,
                intermediate: 3072,
                num_positions: n * n,
            },
            text: base_text_cfg(),
        }
    }

    /// Canonical `siglip2-base-patch16-naflex` (variable resolution).
    /// `num_patches = 256` (16×16 position grid).
    pub fn base_naflex() -> Self {
        let mut c = Self::base_fixed(0, 16);
        c.variant = Variant::NaFlex;
        c.vision.num_positions = 256;
        c
    }

    /// Parse a HuggingFace SigLIP `config.json`. Missing dimensions fall back
    /// to the base-16 defaults (the published fixed-res configs omit almost
    /// everything). `model_type` selects the [`Variant`].
    pub fn from_hf_config_json(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("reading siglip config {path:?}"))?;
        let raw: RawConfig = serde_json::from_str(&data)
            .with_context(|| format!("parsing siglip config {path:?}"))?;

        let variant = match raw.model_type.as_deref() {
            Some("siglip2") => Variant::NaFlex,
            _ => Variant::Fixed,
        };
        let def = Self::base_patch16_224();
        let v = raw.vision_config.unwrap_or_default();
        let t = raw.text_config.unwrap_or_default();

        let image_size = v.image_size.unwrap_or(def.vision.image_size);
        let patch_size = v.patch_size.unwrap_or(def.vision.patch_size);
        // NaFlex carries `num_patches` (the position grid); fixed-res derives
        // it from image/patch.
        let num_positions = v.num_patches.unwrap_or_else(|| {
            if variant == Variant::NaFlex {
                256
            } else {
                let n = image_size / patch_size;
                n * n
            }
        });
        let width = v.hidden_size.unwrap_or(def.vision.width);
        let text_width = t.hidden_size.unwrap_or(def.text.width);
        let projection = t.projection_size.unwrap_or(text_width);

        Ok(Self {
            variant,
            embed_dim: projection,
            vision: VisionCfg {
                image_size,
                patch_size,
                width,
                layers: v.num_hidden_layers.unwrap_or(def.vision.layers),
                heads: v.num_attention_heads.unwrap_or(def.vision.heads),
                intermediate: v.intermediate_size.unwrap_or(def.vision.intermediate),
                num_positions,
            },
            text: TextCfg {
                context_length: t.max_position_embeddings.unwrap_or(def.text.context_length),
                vocab_size: t.vocab_size.unwrap_or(def.text.vocab_size),
                width: text_width,
                heads: t.num_attention_heads.unwrap_or(def.text.heads),
                layers: t.num_hidden_layers.unwrap_or(def.text.layers),
                intermediate: t.intermediate_size.unwrap_or(def.text.intermediate),
                projection,
            },
        })
    }
}

fn base_text_cfg() -> TextCfg {
    TextCfg {
        context_length: 64,
        vocab_size: 256_000,
        width: 768,
        heads: 12,
        layers: 12,
        intermediate: 3072,
        projection: 768,
    }
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    model_type: Option<String>,
    vision_config: Option<RawVisionCfg>,
    text_config: Option<RawTextCfg>,
}

#[derive(Debug, Default, Deserialize)]
struct RawVisionCfg {
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    image_size: Option<usize>,
    patch_size: Option<usize>,
    num_patches: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct RawTextCfg {
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    max_position_embeddings: Option<usize>,
    vocab_size: Option<usize>,
    projection_size: Option<usize>,
}
