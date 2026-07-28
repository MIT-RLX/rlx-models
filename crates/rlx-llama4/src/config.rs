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

//! Llama-4 configuration (text tower). Mirrors `Llama4TextConfig`; the layer
//! schedules (`moe_layers`, `no_rope_layers`) are derived exactly as in HF
//! `__post_init__` when not given explicitly.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
struct RopeParams {
    #[serde(default)]
    rope_theta: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Llama4TextConfig {
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_inter")]
    pub intermediate_size: usize,
    #[serde(default = "d_inter_mlp")]
    pub intermediate_size_mlp: usize,
    #[serde(default = "d_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "d_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "d_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "d_local_experts")]
    pub num_local_experts: usize,
    #[serde(default)]
    pub moe_layers: Option<Vec<usize>>,
    #[serde(default = "d_interleave")]
    pub interleave_moe_layer_step: usize,
    #[serde(default = "d_use_qk_norm")]
    pub use_qk_norm: bool,
    #[serde(default)]
    pub no_rope_layers: Option<Vec<usize>>,
    #[serde(default = "d_no_rope_interval")]
    pub no_rope_layer_interval: usize,
    #[serde(default = "d_chunk")]
    pub attention_chunk_size: Option<usize>,
    #[serde(default = "d_attn_temp")]
    pub attn_temperature_tuning: bool,
    #[serde(default = "d_floor")]
    pub floor_scale: usize,
    #[serde(default = "d_attn_scale")]
    pub attn_scale: f32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: Option<RopeParams>,
}

impl Llama4TextConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("reading {}", path.as_ref().display()))?;
        Self::from_json(&text)
    }

    /// Parse either a top-level `Llama4TextConfig` json or a `Llama4Config`
    /// with a nested `text_config`.
    pub fn from_json(text: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct Wrap {
            text_config: Option<Llama4TextConfig>,
        }
        if let Ok(w) = serde_json::from_str::<Wrap>(text) {
            if let Some(t) = w.text_config {
                return Ok(t);
            }
        }
        serde_json::from_str(text).context("parsing llama4 text config")
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(500000.0)
    }

    /// `no_rope_layers[i]`: 1 = RoPE (chunked attention), 0 = NoPE (full attention).
    pub fn no_rope_vec(&self) -> Vec<usize> {
        self.no_rope_layers.clone().unwrap_or_else(|| {
            (0..self.num_hidden_layers)
                .map(|i| usize::from((i + 1) % self.no_rope_layer_interval != 0))
                .collect()
        })
    }
    /// Whether layer `i` applies RoPE (and, if `use_qk_norm`, qk L2-norm).
    pub fn uses_rope(&self, i: usize) -> bool {
        self.no_rope_vec().get(i).copied().unwrap_or(1) == 1
    }

    pub fn moe_layers_vec(&self) -> Vec<usize> {
        self.moe_layers.clone().unwrap_or_else(|| {
            (self.interleave_moe_layer_step.saturating_sub(1)..self.num_hidden_layers)
                .step_by(self.interleave_moe_layer_step.max(1))
                .collect()
        })
    }
    pub fn is_moe_layer(&self, i: usize) -> bool {
        self.moe_layers_vec().contains(&i)
    }
}

fn d_vocab() -> usize {
    202048
}
fn d_hidden() -> usize {
    5120
}
fn d_inter() -> usize {
    8192
}
fn d_inter_mlp() -> usize {
    16384
}
fn d_layers() -> usize {
    48
}
fn d_heads() -> usize {
    40
}
fn d_kv_heads() -> usize {
    8
}
fn d_eps() -> f32 {
    1e-5
}
fn d_max_pos() -> usize {
    4096 * 32
}
fn d_experts_per_tok() -> usize {
    1
}
fn d_local_experts() -> usize {
    16
}
fn d_interleave() -> usize {
    1
}
fn d_use_qk_norm() -> bool {
    true
}
fn d_no_rope_interval() -> usize {
    4
}
fn d_chunk() -> Option<usize> {
    Some(8192)
}
fn d_attn_temp() -> bool {
    true
}
fn d_floor() -> usize {
    8192
}
fn d_attn_scale() -> f32 {
    0.1
}

/// Vision tower configuration (`Llama4VisionConfig`).
#[derive(Debug, Clone, Deserialize)]
pub struct Llama4VisionConfig {
    #[serde(default = "dv_hidden")]
    pub hidden_size: usize,
    #[serde(default = "dv_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "dv_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "dv_inter")]
    pub intermediate_size: usize,
    #[serde(default = "dv_out_dim")]
    pub vision_output_dim: usize,
    #[serde(default = "dv_image")]
    pub image_size: usize,
    #[serde(default = "dv_patch")]
    pub patch_size: usize,
    #[serde(default = "dv_channels")]
    pub num_channels: usize,
    #[serde(default = "dv_eps")]
    pub norm_eps: f32,
    #[serde(default = "dv_ratio")]
    pub pixel_shuffle_ratio: f32,
    #[serde(default = "dv_proj_in")]
    pub projector_input_dim: usize,
    #[serde(default = "dv_proj_out")]
    pub projector_output_dim: usize,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: Option<RopeParams>,
}

impl Llama4VisionConfig {
    pub fn num_patches(&self) -> usize {
        let side = self.image_size / self.patch_size;
        side * side + 1
    }
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(10000.0)
    }
}

fn dv_hidden() -> usize {
    1408
}
fn dv_layers() -> usize {
    34
}
fn dv_heads() -> usize {
    16
}
fn dv_inter() -> usize {
    5632
}
fn dv_out_dim() -> usize {
    4096
}
fn dv_image() -> usize {
    336
}
fn dv_patch() -> usize {
    14
}
fn dv_channels() -> usize {
    3
}
fn dv_eps() -> f32 {
    1e-5
}
fn dv_ratio() -> f32 {
    0.5
}
fn dv_proj_in() -> usize {
    4096
}
fn dv_proj_out() -> usize {
    4096
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scout_defaults_and_schedules() {
        let cfg: Llama4TextConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.num_hidden_layers, 48);
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.num_experts_per_tok, 1);
        // interleave_moe_layer_step=1 → every layer is MoE
        assert_eq!(cfg.moe_layers_vec().len(), 48);
        // NoPE every 4th layer (i=3,7,...): uses_rope false there
        assert!(!cfg.uses_rope(3));
        assert!(cfg.uses_rope(0));
        assert!(cfg.uses_rope(2));
        assert_eq!(cfg.rope_theta(), 500000.0);
    }
}
