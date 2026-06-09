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

//! DINOv2 configuration. Mirrors Meta's reference configs.

use serde::Deserialize;
use std::path::Path;

/// ImageNet-1k mean/std applied to RGB pixels in `[0, 1]`.
/// Matches `candle-examples::imagenet::load_image*` and the original
/// DINOv2 PyTorch preprocessing.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// DINOv2 model configuration. `vit_giant` (SwiGLU MLP) is not yet
/// supported — vit_small / vit_base / vit_large are.
#[derive(Debug, Clone, Deserialize)]
pub struct DinoV2Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub img_size: usize,
    pub patch_size: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
    #[serde(default = "default_dinov2_ln_eps")]
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub num_register_tokens: usize,
    /// Number of ImageNet classes for the optional classifier head.
    /// Set to 0 to skip the head entirely (encoder-only output).
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
}

fn default_mlp_ratio() -> f64 {
    4.0
}
fn default_dinov2_ln_eps() -> f64 {
    1e-5
}
fn default_num_classes() -> usize {
    1000
}

impl DinoV2Config {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn new(
        img_size: usize,
        depth: usize,
        embed_dim: usize,
        num_heads: usize,
        num_register_tokens: usize,
    ) -> Self {
        Self {
            hidden_size: embed_dim,
            num_hidden_layers: depth,
            num_attention_heads: num_heads,
            img_size,
            patch_size: 14,
            mlp_ratio: 4.0,
            layer_norm_eps: 1e-5,
            num_register_tokens,
            num_classes: 1000,
        }
    }

    pub fn intermediate_size(&self) -> usize {
        (self.hidden_size as f64 * self.mlp_ratio) as usize
    }
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    pub fn num_patches(&self) -> usize {
        let n = self.img_size / self.patch_size;
        n * n
    }
    pub fn seq_len(&self) -> usize {
        1 + self.num_register_tokens + self.num_patches()
    }
    pub fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }

    pub fn vit_small(img_size: usize) -> Self {
        Self::new(img_size, 12, 384, 6, 0)
    }
    pub fn vit_base(img_size: usize) -> Self {
        Self::new(img_size, 12, 768, 12, 0)
    }
    pub fn vit_large(img_size: usize) -> Self {
        Self::new(img_size, 24, 1024, 16, 0)
    }
}
