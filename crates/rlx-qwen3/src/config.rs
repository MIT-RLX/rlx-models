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
    /// Explicit per-head width. Qwen3 states it (it is *not* always
    /// `hidden_size / num_attention_heads`), but the Qwen2/2.5 family — and
    /// every VL checkpoint built on it — omits the field entirely, where the
    /// implied value is the quotient. Defaulted to 0 here and filled in by
    /// [`Qwen3Config::from_file`] so those configs load instead of failing with
    /// `missing field head_dim`.
    #[serde(default)]
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
    /// GGUF key: `qwen3.expert_feed_forward_length`; HF key:
    /// `moe_intermediate_size`.
    #[serde(default, alias = "moe_intermediate_size")]
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
        Self::from_json_str(&data)
    }

    /// Parse a `config.json`, flat or nested.
    ///
    /// A vision-language checkpoint keeps the language model's fields under
    /// `text_config`, beside a `vision_config` for the tower. Parsing the top
    /// level of one of those fails on `vocab_size` — a long way from saying
    /// "this is a VL config", which is what the error means.
    pub fn from_json_str(data: &str) -> anyhow::Result<Self> {
        let mut value: serde_json::Value = serde_json::from_str(data)?;
        // transformers 5 keeps `rope_theta` / `rope_scaling` in `rope_parameters`.
        rlx_core::config::legacy_rope_keys(&mut value);
        let model_type = value
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("qwen3")
            .to_string();
        let body = match value.get("text_config") {
            Some(nested) if nested.is_object() => {
                // `vocab_size` and `tie_word_embeddings` live at the top level
                // in some exports and in `text_config` in others, so the outer
                // object fills whatever the inner one left out.
                let mut merged = nested.clone();
                if let (Some(m), Some(outer)) = (merged.as_object_mut(), value.as_object()) {
                    for key in ["vocab_size", "tie_word_embeddings", "torch_dtype", "dtype"] {
                        if !m.contains_key(key)
                            && let Some(v) = outer.get(key)
                        {
                            m.insert(key.to_string(), v.clone());
                        }
                    }
                }
                merged
            }
            _ => value,
        };
        let explicit_qk_norm = body.get("qk_norm").is_some();
        let explicit_bias = body.get("attention_bias").is_some();
        let mut cfg: Self = serde_json::from_value(body)?;
        // The rest of the Llama family has no QK-norm, and says nothing of it.
        if !explicit_qk_norm && matches!(model_type.as_str(), "llama" | "mistral" | "qwen2") {
            cfg.qk_norm = false;
        }
        // A Qwen2 attention block adds its Q/K/V bias unconditionally; its
        // config has no flag for it.
        if !explicit_bias && model_type == "qwen2" {
            cfg.attention_bias = true;
        }
        cfg.fill_derived_defaults()?;
        Ok(cfg)
    }

    /// Fill fields the Qwen2/2.5 family omits, which `#[serde(default)]` would
    /// otherwise leave at zero.
    ///
    /// Must be called by *every* path that deserializes a config, not just
    /// [`Self::from_file`] — a VL config arrives through
    /// `Qwen25VlHfConfig::into_runtime` via `serde_json::from_value`, and a
    /// `head_dim` of 0 there does not fail loudly: it reaches the Metal RoPE
    /// lowering as `x_shape[2] / head_dim` and panics with `attempt to divide
    /// by zero`, a long way from the cause.
    pub fn fill_derived_defaults(&mut self) -> anyhow::Result<()> {
        let cfg = self;
        if cfg.head_dim == 0 {
            anyhow::ensure!(
                cfg.num_attention_heads > 0
                    && cfg.hidden_size.is_multiple_of(cfg.num_attention_heads),
                "config omits `head_dim` and hidden_size {} is not divisible by \
                 num_attention_heads {}",
                cfg.hidden_size,
                cfg.num_attention_heads
            );
            cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
        }
        Ok(())
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

    /// Layer `i`'s experts, when the model is MoE: every layer, routing
    /// weights renormalized — the layout of the Qwen3 MoE releases
    /// (`decoder_sparse_step: 1`, no `mlp_only_layers`, `norm_topk_prob`;
    /// an exporter refuses a config that says otherwise).
    pub fn layer_moe(&self, _i: usize) -> Option<rlx_flow::blocks::Qwen3Moe> {
        self.is_moe().then(|| rlx_flow::blocks::Qwen3Moe {
            num_experts: self.num_experts,
            top_k: self.num_experts_used.max(1),
            intermediate: self.expert_ffn_dim(),
            norm_topk_prob: true,
        })
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
    /// A vision-language checkpoint keeps the language model under
    /// `text_config`. Parsing the top level of one fails on `vocab_size`,
    /// which says nothing about the actual problem.
    #[test]
    fn a_nested_vl_config_is_read_as_the_language_model() {
        let json = r#"{
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "vision_config": { "hidden_size": 1152, "patch_size": 16 },
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 6144,
                "vocab_size": 151936,
                "head_dim": 128,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 262144
            }
        }"#;
        let cfg = super::Qwen3Config::from_json_str(json).expect("a VL config parses");
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 28);
    }

    /// Some exports put these at the top level and leave them out of the
    /// nested block, so the outer object fills what the inner one omits.
    #[test]
    fn the_outer_object_supplies_what_the_nested_one_omits() {
        let json = r#"{
            "model_type": "qwen3_vl",
            "vocab_size": 151936,
            "tie_word_embeddings": true,
            "text_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "intermediate_size": 2048,
                "head_dim": 128,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 4096
            }
        }"#;
        let cfg = super::Qwen3Config::from_json_str(json).expect("parses");
        assert_eq!(cfg.vocab_size, 151936);
        assert!(cfg.tie_word_embeddings);
    }

    /// And a plain text config still parses exactly as it did.
    #[test]
    fn a_flat_config_is_unaffected() {
        let json = r#"{
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "intermediate_size": 2048,
            "vocab_size": 100,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096
        }"#;
        let cfg = super::Qwen3Config::from_json_str(json).expect("parses");
        assert_eq!(cfg.vocab_size, 100);
    }

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

    #[test]
    fn rope_theta_is_read_from_rope_parameters() {
        let json = r#"{"text_config": {
            "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16,
            "max_position_embeddings": 512,
            "rope_parameters": {"rope_theta": 5000000, "rope_type": "default"}
        }}"#;
        let cfg = Qwen3Config::from_json_str(json).unwrap();
        assert_eq!(cfg.rope_theta, 5_000_000.0);
    }
}
