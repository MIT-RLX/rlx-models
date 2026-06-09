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

// RLX — LLaDA2 MoE config (`/Users/Shared/TIDE/model/config.json`).

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct LLaDA2MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub num_shared_experts: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default = "default_n_group")]
    pub n_group: usize,
    #[serde(default = "default_topk_group")]
    pub topk_group: usize,
    #[serde(default = "default_routed_scaling")]
    pub routed_scaling_factor: f32,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub use_qk_norm: bool,
    #[serde(default)]
    pub use_qkv_bias: bool,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default)]
    pub embedding_dropout: f64,
    #[serde(default)]
    pub output_dropout: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub moe_router_enable_expert_bias: bool,
    #[serde(default)]
    pub pad_token_id: u32,
    #[serde(default = "default_mask_id")]
    pub mask_token_id: u32,
    #[serde(default = "default_eos_id")]
    pub eos_token_id: u32,
}

fn default_n_group() -> usize {
    8
}
fn default_topk_group() -> usize {
    4
}
fn default_routed_scaling() -> f32 {
    2.5
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    600_000.0
}
fn default_partial_rotary() -> f32 {
    0.5
}
fn default_mask_id() -> u32 {
    156_895
}
fn default_eos_id() -> u32 {
    156_892
}
fn default_hidden_act() -> String {
    "silu".into()
}

impl LLaDA2MoeConfig {
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn from_tide_repo() -> anyhow::Result<Self> {
        Self::from_file(Path::new("/Users/Shared/TIDE/model/config.json"))
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size.unwrap_or(self.hidden_size * 4)
    }

    pub fn expert_ffn_dim(&self) -> usize {
        self.moe_intermediate_size.unwrap_or(512)
    }

    pub fn num_kv_heads(&self) -> usize {
        if self.num_key_value_heads == 0 {
            self.num_attention_heads
        } else {
            self.num_key_value_heads
        }
    }

    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads()
    }

    pub fn rope_dim(&self) -> usize {
        ((self.head_dim() as f32) * self.partial_rotary_factor) as usize
    }

    pub fn is_moe_layer(&self, layer: usize) -> bool {
        self.num_experts > 0 && layer >= self.first_k_dense_replace
    }

    pub fn num_sparse_moe_layers(&self) -> usize {
        self.num_hidden_layers
            .saturating_sub(self.first_k_dense_replace)
    }

    pub fn expert_param_bytes_f32(&self) -> usize {
        let h = self.hidden_size;
        let ff = self.expert_ffn_dim();
        3 * h * ff * std::mem::size_of::<f32>()
    }
}
