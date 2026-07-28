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

//! GLM-4.5 / GLM-4.6 configuration (`Glm4MoeConfig`).

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
struct RopeParams {
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Glm4MoeConfig {
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_inter")]
    pub intermediate_size: usize,
    #[serde(default = "d_moe_inter")]
    pub moe_intermediate_size: usize,
    #[serde(default = "d_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "d_n_shared")]
    pub n_shared_experts: usize,
    #[serde(default = "d_n_routed")]
    pub n_routed_experts: usize,
    #[serde(default = "d_routed_scaling")]
    pub routed_scaling_factor: f32,
    #[serde(default = "d_n_group")]
    pub n_group: usize,
    #[serde(default = "d_topk_group")]
    pub topk_group: usize,
    #[serde(default = "d_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "d_first_k_dense")]
    pub first_k_dense_replace: usize,
    #[serde(default = "d_norm_topk")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub use_qk_norm: bool,
    #[serde(default = "d_partial")]
    pub partial_rotary_factor: f32,
    #[serde(default = "d_attn_bias")]
    pub attention_bias: bool,
    #[serde(default = "d_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "d_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: Option<RopeParams>,
}

impl Glm4MoeConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let t = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("reading {}", path.as_ref().display()))?;
        serde_json::from_str(&t).context("parsing glm4_moe config")
    }
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
    /// Rotary dim = `head_dim * partial_rotary_factor` (rope covers only this prefix).
    pub fn rotary_dim(&self) -> usize {
        let pf = self
            .rope_parameters
            .as_ref()
            .and_then(|r| r.partial_rotary_factor)
            .unwrap_or(self.partial_rotary_factor);
        ((self.head_dim() as f32 * pf) as usize) & !1 // even
    }
    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(10000.0)
    }
    pub fn is_moe_layer(&self, i: usize) -> bool {
        i >= self.first_k_dense_replace
    }
}

fn d_vocab() -> usize {
    151552
}
fn d_hidden() -> usize {
    4096
}
fn d_inter() -> usize {
    10944
}
fn d_moe_inter() -> usize {
    1408
}
fn d_layers() -> usize {
    47
}
fn d_heads() -> usize {
    96
}
fn d_kv_heads() -> usize {
    8
}
fn d_n_shared() -> usize {
    1
}
fn d_n_routed() -> usize {
    128
}
fn d_routed_scaling() -> f32 {
    1.0
}
fn d_n_group() -> usize {
    1
}
fn d_topk_group() -> usize {
    1
}
fn d_experts_per_tok() -> usize {
    8
}
fn d_first_k_dense() -> usize {
    1
}
fn d_norm_topk() -> bool {
    true
}
fn d_partial() -> f32 {
    0.5
}
fn d_attn_bias() -> bool {
    true
}
fn d_rms_eps() -> f32 {
    1e-5
}
fn d_max_pos() -> usize {
    131072
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm_defaults() {
        let cfg: Glm4MoeConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.head_dim(), 4096 / 96);
        assert_eq!(cfg.n_routed_experts, 128);
        assert!(!cfg.is_moe_layer(0) && cfg.is_moe_layer(1));
        assert_eq!(
            cfg.rotary_dim(),
            ((cfg.head_dim() as f32 * 0.5) as usize) & !1
        );
    }
}
