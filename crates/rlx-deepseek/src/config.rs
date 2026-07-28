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

//! DeepSeek-V3 configuration (`DeepseekV3Config`; Kimi-K2 shares this arch).

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
struct RopeParams {
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_type: Option<String>,
    #[serde(default)]
    factor: Option<f32>,
    #[serde(default)]
    mscale_all_dim: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepseekV3Config {
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
    #[serde(default = "d_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "d_n_shared")]
    pub n_shared_experts: usize,
    #[serde(default = "d_n_routed")]
    pub n_routed_experts: usize,
    #[serde(default = "d_routed_scaling")]
    pub routed_scaling_factor: f32,
    #[serde(default = "d_kv_lora")]
    pub kv_lora_rank: usize,
    #[serde(default = "d_q_lora")]
    pub q_lora_rank: Option<usize>,
    #[serde(default = "d_qk_rope")]
    pub qk_rope_head_dim: usize,
    #[serde(default = "d_v_head")]
    pub v_head_dim: usize,
    #[serde(default = "d_qk_nope")]
    pub qk_nope_head_dim: usize,
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
    #[serde(default = "d_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "d_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "d_rope_interleave")]
    pub rope_interleave: bool,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: Option<RopeParams>,
    #[serde(default)]
    rope_scaling: Option<RopeParams>,
}

impl DeepseekV3Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let t = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("reading {}", path.as_ref().display()))?;
        serde_json::from_str(&t).context("parsing deepseek_v3 config")
    }

    /// `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` (192 for V3).
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    pub fn rope_params(&self) -> Option<&RopeParams> {
        self.rope_parameters.as_ref().or(self.rope_scaling.as_ref())
    }
    pub fn rope_theta(&self) -> f32 {
        self.rope_params()
            .and_then(|r| r.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(10000.0)
    }
    pub fn is_moe_layer(&self, i: usize) -> bool {
        i >= self.first_k_dense_replace
    }
    /// Attention score scale `qk_head_dim^-0.5` × YaRN mscale² (when scaled RoPE).
    pub fn attn_score_scale(&self) -> f32 {
        let base = (self.qk_head_dim() as f32).powf(-0.5);
        if let Some(r) = self.rope_params() {
            if r.rope_type
                .as_deref()
                .map(|t| t != "default")
                .unwrap_or(false)
            {
                if let (Some(factor), Some(mscale_all)) = (r.factor, r.mscale_all_dim) {
                    if mscale_all != 0.0 && factor > 1.0 {
                        let mscale = 0.1 * mscale_all * factor.ln() + 1.0;
                        return base * mscale * mscale;
                    }
                }
            }
        }
        base
    }
}

fn d_vocab() -> usize {
    129280
}
fn d_hidden() -> usize {
    7168
}
fn d_inter() -> usize {
    18432
}
fn d_moe_inter() -> usize {
    2048
}
fn d_layers() -> usize {
    61
}
fn d_heads() -> usize {
    128
}
fn d_n_shared() -> usize {
    1
}
fn d_n_routed() -> usize {
    256
}
fn d_routed_scaling() -> f32 {
    2.5
}
fn d_kv_lora() -> usize {
    512
}
fn d_q_lora() -> Option<usize> {
    Some(1536)
}
fn d_qk_rope() -> usize {
    64
}
fn d_v_head() -> usize {
    128
}
fn d_qk_nope() -> usize {
    128
}
fn d_n_group() -> usize {
    8
}
fn d_topk_group() -> usize {
    4
}
fn d_experts_per_tok() -> usize {
    8
}
fn d_first_k_dense() -> usize {
    3
}
fn d_norm_topk() -> bool {
    true
}
fn d_rms_eps() -> f32 {
    1e-6
}
fn d_max_pos() -> usize {
    4096
}
fn d_rope_interleave() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v3_defaults() {
        let cfg: DeepseekV3Config = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.qk_head_dim(), 192);
        assert_eq!(cfg.v_head_dim, 128);
        assert_eq!(cfg.q_lora_rank, Some(1536));
        assert_eq!(cfg.kv_lora_rank, 512);
        assert!(!cfg.is_moe_layer(2) && cfg.is_moe_layer(3));
        assert!((cfg.attn_score_scale() - (192f32).powf(-0.5)).abs() < 1e-6);
    }
}
