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

//! V-JEPA2 configuration — mirrors Meta / HuggingFace `config.json`.

use serde::Deserialize;
use std::path::Path;

/// ImageNet-style mean/std (same as DINOv2 / HF VJEPA2VideoProcessor).
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone, Deserialize)]
pub struct Vjepa2Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(alias = "image_size")]
    pub crop_size: usize,
    pub patch_size: usize,
    pub tubelet_size: usize,
    pub frames_per_clip: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
    #[serde(default = "default_ln_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_in_chans")]
    pub in_chans: usize,
    // Predictor
    #[serde(default = "default_pred_hidden")]
    pub pred_hidden_size: usize,
    #[serde(default = "default_pred_heads")]
    pub pred_num_attention_heads: usize,
    #[serde(default = "default_pred_layers")]
    pub pred_num_hidden_layers: usize,
    #[serde(default = "default_pred_mlp_ratio")]
    pub pred_mlp_ratio: f64,
    #[serde(default = "default_pred_mask_tokens")]
    pub pred_num_mask_tokens: usize,
    #[serde(default = "default_true")]
    pub pred_zero_init_mask_tokens: bool,
    // Attentive pooler (finetuned checkpoints)
    #[serde(default = "default_pooler_layers")]
    pub num_pooler_layers: usize,
    #[serde(default)]
    pub num_classes: usize,
}

fn default_mlp_ratio() -> f64 {
    48.0 / 11.0
}
fn default_ln_eps() -> f64 {
    1e-6
}
fn default_in_chans() -> usize {
    3
}
fn default_pred_hidden() -> usize {
    384
}
fn default_pred_heads() -> usize {
    12
}
fn default_pred_layers() -> usize {
    12
}
fn default_pred_mlp_ratio() -> f64 {
    4.0
}
fn default_pred_mask_tokens() -> usize {
    10
}
fn default_true() -> bool {
    true
}
fn default_pooler_layers() -> usize {
    3
}

impl Vjepa2Config {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    /// `facebook/vjepa2-vitg-fpc64-384` — ViT-G, 64 frames, 384².
    pub fn vit_g_384() -> Self {
        Self {
            hidden_size: 1408,
            num_hidden_layers: 40,
            num_attention_heads: 22,
            crop_size: 384,
            patch_size: 16,
            tubelet_size: 2,
            frames_per_clip: 64,
            mlp_ratio: 48.0 / 11.0,
            layer_norm_eps: 1e-6,
            in_chans: 3,
            pred_hidden_size: 384,
            pred_num_attention_heads: 12,
            pred_num_hidden_layers: 12,
            pred_mlp_ratio: 4.0,
            pred_num_mask_tokens: 10,
            pred_zero_init_mask_tokens: true,
            num_pooler_layers: 3,
            num_classes: 0,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn pred_head_dim(&self) -> usize {
        self.pred_hidden_size / self.pred_num_attention_heads
    }

    pub fn intermediate_size(&self) -> usize {
        (self.hidden_size as f64 * self.mlp_ratio) as usize
    }

    pub fn pred_intermediate_size(&self) -> usize {
        (self.pred_hidden_size as f64 * self.pred_mlp_ratio) as usize
    }

    pub fn pooler_intermediate_size(&self) -> usize {
        (self.hidden_size as f64 * self.mlp_ratio) as usize
    }

    pub fn grid_spatial(&self) -> usize {
        self.crop_size / self.patch_size
    }

    pub fn grid_temporal(&self) -> usize {
        self.frames_per_clip / self.tubelet_size
    }

    pub fn num_patches(&self) -> usize {
        self.grid_temporal() * self.grid_spatial() * self.grid_spatial()
    }

    /// Per-axis RoPE segment sizes (d, h, w). Matches Meta `RoPEAttention`.
    pub fn rope_segment_dims(&self) -> (usize, usize, usize) {
        rope_segment_dims(self.head_dim())
    }

    pub fn pred_rope_segment_dims(&self) -> (usize, usize, usize) {
        rope_segment_dims(self.pred_head_dim())
    }
}

pub fn rope_segment_dims(head_dim: usize) -> (usize, usize, usize) {
    let third = head_dim / 3;
    let seg = 2 * (third / 2);
    (seg, seg, seg)
}
