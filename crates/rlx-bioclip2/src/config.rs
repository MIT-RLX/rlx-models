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

//! BioCLIP-2 configuration — mirrors the model's `open_clip_config.json`.
//!
//! BioCLIP-2 is, architecturally, a stock OpenCLIP **ViT-L-14** (LAION-2B
//! lineage): a 24-layer / 1024-wide bidirectional vision transformer with
//! a 14×14 conv patch stem, and a 12-layer / 768-wide causal text
//! transformer, both projecting into a shared 768-dim embedding space.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// CLIP/OpenAI preprocessing mean (RGB, pixels scaled to `[0,1]`).
pub const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
/// CLIP/OpenAI preprocessing std (RGB).
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// LayerNorm epsilon used throughout OpenCLIP (`nn.LayerNorm` default).
pub const LN_EPS: f32 = 1e-5;

/// Vision tower dimensions.
#[derive(Debug, Clone, Copy)]
pub struct VisionCfg {
    pub image_size: usize,
    pub patch_size: usize,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
}

impl VisionCfg {
    pub fn head_dim(&self) -> usize {
        self.width / self.heads
    }
    pub fn num_patches(&self) -> usize {
        let n = self.image_size / self.patch_size;
        n * n
    }
    /// CLS token + patch tokens.
    pub fn seq_len(&self) -> usize {
        1 + self.num_patches()
    }
    pub fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }
    pub fn mlp_dim(&self) -> usize {
        self.width * 4
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
}

impl TextCfg {
    pub fn head_dim(&self) -> usize {
        self.width / self.heads
    }
    pub fn mlp_dim(&self) -> usize {
        self.width * 4
    }
}

/// Full BioCLIP-2 configuration.
#[derive(Debug, Clone, Copy)]
pub struct BioClip2Config {
    pub embed_dim: usize,
    pub vision: VisionCfg,
    pub text: TextCfg,
}

impl BioClip2Config {
    /// Canonical BioCLIP-2 = OpenCLIP ViT-L-14.
    pub fn vit_l_14() -> Self {
        Self {
            embed_dim: 768,
            vision: VisionCfg {
                image_size: 224,
                patch_size: 14,
                width: 1024,
                layers: 24,
                heads: 16,
            },
            text: TextCfg {
                context_length: 77,
                vocab_size: 49408,
                width: 768,
                heads: 12,
                layers: 12,
            },
        }
    }

    /// Parse a HuggingFace `open_clip_config.json`. Falls back to the
    /// ViT-L-14 default for any field the file omits (open_clip relies on
    /// model-name defaults — e.g. vision/text `heads` are often absent).
    pub fn from_open_clip_json(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("reading open_clip config {path:?}"))?;
        let raw: RawConfig = serde_json::from_str(&data)
            .with_context(|| format!("parsing open_clip config {path:?}"))?;
        let def = Self::vit_l_14();
        let m = raw.model_cfg;
        let v = m.vision_cfg;
        let t = m.text_cfg;
        let width = v.width.unwrap_or(def.vision.width);
        let heads = v.heads.unwrap_or_else(|| {
            v.head_width
                .map(|hw| width / hw)
                .unwrap_or(def.vision.heads)
        });
        Ok(Self {
            embed_dim: m.embed_dim.unwrap_or(def.embed_dim),
            vision: VisionCfg {
                image_size: v.image_size.unwrap_or(def.vision.image_size),
                patch_size: v.patch_size.unwrap_or(def.vision.patch_size),
                width,
                layers: v.layers.unwrap_or(def.vision.layers),
                heads,
            },
            text: TextCfg {
                context_length: t.context_length.unwrap_or(def.text.context_length),
                vocab_size: t.vocab_size.unwrap_or(def.text.vocab_size),
                width: t.width.unwrap_or(def.text.width),
                heads: t.heads.unwrap_or(def.text.heads),
                layers: t.layers.unwrap_or(def.text.layers),
            },
        })
    }
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    model_cfg: RawModelCfg,
}

#[derive(Debug, Deserialize)]
struct RawModelCfg {
    embed_dim: Option<usize>,
    vision_cfg: RawVisionCfg,
    text_cfg: RawTextCfg,
}

#[derive(Debug, Default, Deserialize)]
struct RawVisionCfg {
    image_size: Option<usize>,
    patch_size: Option<usize>,
    width: Option<usize>,
    layers: Option<usize>,
    heads: Option<usize>,
    head_width: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct RawTextCfg {
    context_length: Option<usize>,
    vocab_size: Option<usize>,
    width: Option<usize>,
    heads: Option<usize>,
    layers: Option<usize>,
}
