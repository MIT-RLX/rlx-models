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

//! Qwen3 configuration. Matches HuggingFace `Qwen3ForCausalLM` config.json.
//!
//! Qwen3 introduces three things over Qwen2 that this struct must capture:
//!   - **GQA** with an explicit `head_dim` (not derived from
//!     `hidden_size / num_attention_heads`), so KV projection width is
//!     `num_key_value_heads * head_dim` rather than `hidden_size`.
//!   - **QK-norm**: per-head RMSNorm on Q and K before RoPE. Weight
//!     shape `[head_dim]`, no bias.
//!   - **Sliding-window attention** (optional, per-layer): `sliding_window`
//!     window size, `max_window_layers` controls how many leading layers
//!     use full attention.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub attention_bias: bool,

    /// Whether the model uses per-head RMS-norm on Q/K *before* RoPE
    /// (a.k.a. "QK-norm"). Qwen 3 has it; Qwen 2 does NOT. Defaults to
    /// `true` to match the historical Qwen 3 build path.
    #[serde(default = "default_qk_norm")]
    pub qk_norm: bool,

    /// Sliding-window size; `None` (or absent) means full causal.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Number of leading layers that use full causal attention; layers
    /// `[max_window_layers, num_hidden_layers)` use sliding window when
    /// `use_sliding_window` is true. HF default: all layers full.
    #[serde(default = "default_max_window_layers")]
    pub max_window_layers: usize,
    #[serde(default)]
    pub use_sliding_window: bool,

    // ─── MoE fields (PLAN.md M1 — for `qwen3-30b-a3b-instruct`) ───
    /// Total number of routed experts per MoE layer (0 = dense model).
    /// HF key: `num_experts` / `n_routed_experts`. GGUF key:
    /// `qwen3.expert_count`.
    #[serde(default, alias = "n_routed_experts")]
    pub num_experts: usize,
    /// Number of experts activated per token (top-k routing).
    /// HF key: `num_experts_per_tok`. GGUF key: `qwen3.expert_used_count`.
    #[serde(default, alias = "num_experts_per_tok")]
    pub num_experts_used: usize,
    /// FFN inner width for each routed expert. When 0 falls back to
    /// `intermediate_size / num_experts_used` to match upstream defaults.
    /// GGUF key: `qwen3.expert_feed_forward_length`.
    #[serde(default)]
    pub expert_ffn_size: usize,
    /// FFN inner width for the always-on shared expert (0 = no shared
    /// expert). GGUF key: `qwen3.expert_shared_feed_forward_length`.
    #[serde(default)]
    pub shared_expert_ffn_size: usize,
    /// Multiplier applied to routed-expert logits before softmax
    /// (default 1.0). GGUF key: `qwen3.expert_weights_scale`.
    #[serde(default = "default_expert_weights_scale")]
    pub expert_weights_scale: f32,
}

fn default_expert_weights_scale() -> f32 {
    1.0
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_hidden_act() -> String {
    "silu".into()
}
fn default_max_window_layers() -> usize {
    usize::MAX
}
fn default_qk_norm() -> bool {
    true
}

impl Qwen3Config {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    /// Repetition factor for GQA: how many Q heads share each KV head.
    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Q projection output width (`num_attention_heads * head_dim`).
    pub fn q_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// K/V projection output width (`num_key_value_heads * head_dim`).
    pub fn kv_proj_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// True when the config carries MoE routing (`num_experts > 0`).
    /// `qwen3-30b-a3b-instruct` and `qwen3-coder-next` MoE variants
    /// will return true; dense Qwen3 returns false.
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    /// Routed-expert SwiGLU inner width. Falls back to
    /// `intermediate_size / num_experts_used` when the explicit
    /// `expert_ffn_size` is absent (matches upstream defaults).
    pub fn expert_ffn_dim(&self) -> usize {
        if self.expert_ffn_size > 0 {
            self.expert_ffn_size
        } else if self.num_experts_used > 0 {
            self.intermediate_size
                .checked_div(self.num_experts_used)
                .unwrap_or(self.intermediate_size)
        } else {
            self.intermediate_size
        }
    }

    /// Shared-expert SwiGLU inner width (0 when there's no shared
    /// expert).
    pub fn shared_expert_ffn_dim(&self) -> usize {
        if self.shared_expert_ffn_size > 0 {
            self.shared_expert_ffn_size
        } else {
            0
        }
    }

    /// Does layer `idx` use sliding-window attention?
    pub fn layer_uses_swa(&self, idx: usize) -> bool {
        self.use_sliding_window && self.sliding_window.is_some() && idx >= self.max_window_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_qwen3_0_6b_like() {
        let json = r#"{
            "vocab_size": 151936,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true
        }"#;
        let cfg: Qwen3Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.kv_group_size(), 2);
        assert_eq!(cfg.q_proj_dim(), 2048);
        assert_eq!(cfg.kv_proj_dim(), 1024);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.rms_norm_eps, 1e-6);
    }

    #[test]
    fn sliding_window_off_by_default() {
        let json = r#"{
            "vocab_size": 100,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 512
        }"#;
        let cfg: Qwen3Config = serde_json::from_str(json).unwrap();
        assert!(!cfg.layer_uses_swa(0));
        assert!(!cfg.layer_uses_swa(1));
    }
}
