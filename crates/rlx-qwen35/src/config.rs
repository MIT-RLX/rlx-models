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

//! Qwen3.5 configuration — parsed from the GGUF metadata keys
//! under the `qwen35.*` prefix. Captures every field needed to
//! eventually wire the hybrid Mamba+Attention forward.
//!
//! Keys (observed on `unsloth/Qwen3.5-0.8B-MTP-GGUF`):
//!   `qwen35.block_count`, `qwen35.nextn_predict_layers`,
//!   `qwen35.embedding_length`, `qwen35.feed_forward_length`,
//!   `qwen35.attention.head_count`, `qwen35.attention.head_count_kv`,
//!   `qwen35.attention.key_length`, `qwen35.attention.value_length`,
//!   `qwen35.attention.layer_norm_rms_epsilon`,
//!   `qwen35.context_length`,
//!   `qwen35.full_attention_interval`,
//!   `qwen35.rope.dimension_count`, `qwen35.rope.freq_base`,
//!   `qwen35.rope.dimension_sections`,
//!   `qwen35.ssm.conv_kernel`, `qwen35.ssm.group_count`,
//!   `qwen35.ssm.inner_size`, `qwen35.ssm.state_size`,
//!   `qwen35.ssm.time_step_rank`.

use anyhow::{Result, anyhow};
use rlx_gguf::{GgufFile, MetaValue};
use std::path::Path;

/// Qwen3.5 model config — fields covering both the per-layer Mamba+
/// Attention block and the MTP head.
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    /// Total layer count (= main layers + `nextn_predict_layers` MTP heads).
    pub num_hidden_layers: usize,
    /// Layers at index `< num_hidden_layers - nextn_predict_layers`
    /// use the hybrid Mamba+Attention block. The remaining
    /// `nextn_predict_layers` layers use standard attention for MTP.
    pub nextn_predict_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Per-head Q dim. The MTP attention head uses this.
    pub key_length: usize,
    /// Per-head V dim.
    pub value_length: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_dim_count: usize,
    pub rope_dim_sections: Vec<usize>,
    /// HF `rope_parameters.mrope_interleaved` — Qwen3.5 / Qwen3-VL default.
    /// When true, rotary pairs cycle THWTHW… instead of contiguous TTT…HHH…WWW.
    pub mrope_interleaved: bool,
    /// HF Qwen3.5 RMSNorm is `(1 + weight) * rms(x)` (zero-init weight).
    /// Bake `+1` into gamma at graph build when true. GGUF converters
    /// (llama.cpp) already add 1 except `linear_attn.norm` — leave false.
    pub rms_norm_offset: bool,
    /// Some Qwen3.5 layers do full attention every N blocks
    /// (interspersed with the Mamba-style blocks). Read but not yet
    /// acted on.
    pub full_attention_interval: usize,
    pub ssm_conv_kernel: usize,
    pub ssm_group_count: usize,
    pub ssm_inner_size: usize,
    pub ssm_state_size: usize,
    pub ssm_time_step_rank: usize,
    pub tie_word_embeddings: bool,
    /// MoE (`qwen35moe`): routed expert count. Zero for dense models.
    pub num_experts: usize,
    /// Top-k experts activated per token.
    pub num_experts_used: usize,
    /// Per-expert FFN inner dim (`qwen35.expert_feed_forward_length`).
    pub expert_ffn_size: usize,
    /// Shared-expert FFN inner dim (`qwen35.expert_shared_feed_forward_length`).
    pub shared_expert_ffn_size: usize,
    /// Router weight multiplier applied after softmax (llama.cpp default 1.0).
    pub expert_weights_scale: f32,
}

/// FastMTP draft vocabulary size used by llama.cpp (PR #20700).
pub const FAST_MTP_VOCAB: usize = 32_000;

/// MTP LM head output width: full vocab, or trimmed for FastMTP draft speed.
pub fn mtp_draft_vocab_size(full_vocab: usize, fast_mtp: bool) -> usize {
    if fast_mtp {
        FAST_MTP_VOCAB.min(full_vocab)
    } else {
        full_vocab
    }
}

impl Qwen35Config {
    /// Read from a GGUF file with `general.architecture` in
    /// `{qwen35, qwen35moe, qwen36, qwen36moe}`. Qwen3.6 reuses the
    /// Qwen3.5 trunk; only the metadata-key prefix differs (`qwen36.*`
    /// vs `qwen35.*`). Returns an error when any required key is missing.
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = raw
            .metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| anyhow!("missing general.architecture"))?;
        let prefix = match arch {
            "qwen35" | "qwen35moe" => "qwen35",
            "qwen36" | "qwen36moe" => "qwen36",
            other => {
                return Err(anyhow!(
                    "expected arch in {{qwen35, qwen35moe, qwen36, qwen36moe}}, got {other}"
                ));
            }
        };
        let is_moe = arch.ends_with("moe");
        // Try the arch-native prefix first, then fall back to the
        // `qwen35.*` prefix. Some early Qwen3.6 GGUF converters reused
        // the Qwen3.5 keys verbatim; this keeps both layouts working
        // without forcing the caller to know which one their file uses.
        let key = |k: &str| -> String { format!("{prefix}.{k}") };
        let alt_key = |k: &str| -> String { format!("qwen35.{k}") };
        let lookup = |k: &str| -> Option<&MetaValue> {
            raw.metadata
                .get(&key(k))
                .or_else(|| raw.metadata.get(&alt_key(k)))
        };
        let u32k = |k: &str| -> Result<u32> {
            lookup(k)
                .and_then(MetaValue::as_u32)
                .ok_or_else(|| anyhow!("missing {prefix} metadata key: {k}"))
        };
        let u32k_opt = |k: &str| -> Option<u32> { lookup(k).and_then(MetaValue::as_u32) };
        let f32k = |k: &str| -> Option<f32> {
            lookup(k).and_then(|v| match v {
                MetaValue::F32(x) => Some(*x),
                _ => None,
            })
        };
        let boolk = |k: &str| -> Option<bool> {
            lookup(k).and_then(|v| match v {
                MetaValue::Bool(b) => Some(*b),
                _ => None,
            })
        };
        let arr_u32k = |k: &str| -> Vec<usize> {
            lookup(k)
                .and_then(|v| match v {
                    MetaValue::Array(a) => Some(
                        a.iter()
                            .filter_map(|x| match x {
                                MetaValue::U32(u) => Some(*u as usize),
                                MetaValue::U64(u) => Some(*u as usize),
                                MetaValue::I32(i) => Some(*i as usize),
                                _ => None,
                            })
                            .collect(),
                    ),
                    _ => None,
                })
                .unwrap_or_default()
        };
        Ok(Self {
            vocab_size: u32k_opt("vocab_size").unwrap_or(151_936) as usize,
            hidden_size: u32k("embedding_length")? as usize,
            intermediate_size: u32k("feed_forward_length")? as usize,
            num_hidden_layers: u32k("block_count")? as usize,
            nextn_predict_layers: u32k_opt("nextn_predict_layers").unwrap_or(0) as usize,
            num_attention_heads: u32k("attention.head_count")? as usize,
            num_key_value_heads: u32k("attention.head_count_kv")? as usize,
            key_length: u32k_opt("attention.key_length").unwrap_or(128) as usize,
            value_length: u32k_opt("attention.value_length").unwrap_or(128) as usize,
            max_position_embeddings: u32k_opt("context_length").unwrap_or(40_960) as usize,
            rms_norm_eps: f32k("attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f64,
            rope_theta: f32k("rope.freq_base").unwrap_or(10_000_000.0) as f64,
            rope_dim_count: u32k_opt("rope.dimension_count").unwrap_or(64) as usize,
            rope_dim_sections: arr_u32k("rope.dimension_sections"),
            mrope_interleaved: false,
            rms_norm_offset: false,
            full_attention_interval: u32k_opt("full_attention_interval").unwrap_or(0) as usize,
            ssm_conv_kernel: u32k_opt("ssm.conv_kernel").unwrap_or(4) as usize,
            ssm_group_count: u32k_opt("ssm.group_count").unwrap_or(0) as usize,
            ssm_inner_size: u32k_opt("ssm.inner_size").unwrap_or(0) as usize,
            ssm_state_size: u32k_opt("ssm.state_size").unwrap_or(0) as usize,
            ssm_time_step_rank: u32k_opt("ssm.time_step_rank").unwrap_or(0) as usize,
            tie_word_embeddings: boolk("tie_word_embeddings").unwrap_or(true),
            num_experts: if is_moe {
                u32k("expert_count")? as usize
            } else {
                0
            },
            num_experts_used: if is_moe {
                u32k("expert_used_count")? as usize
            } else {
                0
            },
            expert_ffn_size: u32k_opt("expert_feed_forward_length").unwrap_or(0) as usize,
            shared_expert_ffn_size: u32k_opt("expert_shared_feed_forward_length").unwrap_or(0)
                as usize,
            expert_weights_scale: f32k("expert_weights_scale").unwrap_or(1.0),
        })
    }

    /// True when the GGUF arch is `qwen35moe`.
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    /// Routed-expert SwiGLU inner width.
    #[allow(clippy::manual_checked_ops)]
    pub fn expert_ffn_dim(&self) -> usize {
        if self.expert_ffn_size > 0 {
            self.expert_ffn_size
        } else if self.num_experts_used > 0 {
            self.intermediate_size / self.num_experts_used
        } else {
            self.intermediate_size
        }
    }

    /// Shared-expert SwiGLU inner width.
    pub fn shared_expert_ffn_dim(&self) -> usize {
        if self.shared_expert_ffn_size > 0 {
            self.shared_expert_ffn_size
        } else {
            self.intermediate_size
        }
    }

    /// Index of the first MTP layer (= `num_hidden_layers -
    /// nextn_predict_layers`). Returns `None` when the file has no
    /// MTP heads.
    pub fn mtp_layer_start(&self) -> Option<usize> {
        if self.nextn_predict_layers == 0 {
            None
        } else {
            Some(self.num_hidden_layers - self.nextn_predict_layers)
        }
    }

    /// Read from a HuggingFace `config.json` (the safetensors
    /// distribution counterpart to GGUF metadata). Maps the standard
    /// HF Qwen2.x / Qwen3.x field names; unknown fields fall back to
    /// the same defaults used by [`Self::from_gguf`]. PLAN.md M1
    /// safetensors load path (HauhauCS catalog rows).
    pub fn from_hf_config_json(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).map_err(|e| anyhow!("qwen35: read {path:?}: {e}"))?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| anyhow!("qwen35: parse {path:?}: {e}"))?;
        Self::from_hf_config_value(&v, path)
    }

    /// Parse HF config JSON (flat or nested `text_config` as in Fara / Qwen3.5-VL).
    pub fn from_hf_config_value(v: &serde_json::Value, path: &Path) -> Result<Self> {
        let top = v
            .as_object()
            .ok_or_else(|| anyhow!("qwen35: {path:?} is not a JSON object"))?;
        // Multimodal checkpoints nest LM hyperparams under `text_config`.
        let text = top
            .get("text_config")
            .and_then(|t| t.as_object())
            .unwrap_or(top);

        let u = |k: &str| -> Option<usize> {
            text.get(k)
                .or_else(|| top.get(k))
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
        };
        let f = |k: &str| -> Option<f64> {
            text.get(k)
                .or_else(|| top.get(k))
                .and_then(|v| v.as_f64())
        };
        let b = |k: &str| -> Option<bool> {
            text.get(k)
                .or_else(|| top.get(k))
                .and_then(|v| v.as_bool())
        };
        let tie_top = top.get("tie_word_embeddings").and_then(|v| v.as_bool());

        let hidden_size =
            u("hidden_size").ok_or_else(|| anyhow!("qwen35: missing hidden_size in {path:?}"))?;
        let intermediate_size = u("intermediate_size")
            .ok_or_else(|| anyhow!("qwen35: missing intermediate_size in {path:?}"))?;
        let num_hidden_layers = u("num_hidden_layers")
            .ok_or_else(|| anyhow!("qwen35: missing num_hidden_layers in {path:?}"))?;
        // Multimodal trunks often ship MTP blocks that must not run as decoder layers.
        let nextn_predict_layers = u("mtp_num_hidden_layers")
            .or_else(|| u("nextn_predict_layers"))
            .unwrap_or(0);
        // Prefer trunk-only depth when present (Fara / Qwen3.5-VL).
        let num_hidden_layers = u("num_hidden_layers")
            .map(|n| n.saturating_sub(nextn_predict_layers))
            .unwrap_or(num_hidden_layers);
        let num_attention_heads = u("num_attention_heads")
            .ok_or_else(|| anyhow!("qwen35: missing num_attention_heads in {path:?}"))?;
        let num_key_value_heads = u("num_key_value_heads").unwrap_or(num_attention_heads);
        let head_dim = u("head_dim").unwrap_or(hidden_size / num_attention_heads.max(1));

        let rope_obj = text
            .get("rope_parameters")
            .or_else(|| text.get("rope_scaling"))
            .or_else(|| top.get("rope_parameters"))
            .or_else(|| top.get("rope_scaling"));
        let rope_dim_sections = rope_obj
            .and_then(|s| s.get("mrope_section"))
            .and_then(|a| a.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mrope_interleaved = rope_obj
            .and_then(|s| s.get("mrope_interleaved"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let partial_rotary = rope_obj
            .and_then(|s| s.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .or_else(|| f("partial_rotary_factor"))
            .unwrap_or(1.0);
        let rope_dim_count = u("rope_dim_count").unwrap_or_else(|| {
            if partial_rotary > 0.0 && partial_rotary < 1.0 {
                ((head_dim as f64) * partial_rotary).round() as usize
            } else {
                head_dim
            }
        });
        let rope_theta = rope_obj
            .and_then(|s| s.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .or_else(|| f("rope_theta"))
            .unwrap_or(10_000_000.0);

        let linear_key_heads = u("linear_num_key_heads");
        let linear_key_dim = u("linear_key_head_dim");
        let linear_value_heads = u("linear_num_value_heads");
        let linear_value_dim = u("linear_value_head_dim");
        let ssm_conv_kernel = u("linear_conv_kernel_dim").or_else(|| u("ssm_conv_kernel")).unwrap_or(4);
        let ssm_group_count = linear_key_heads.unwrap_or(0);
        let ssm_state_size = linear_key_dim.unwrap_or(0);
        let ssm_inner_size = u("ssm_inner_size").unwrap_or_else(|| {
            match (linear_value_heads, linear_value_dim) {
                (Some(nh), Some(dh)) => nh.saturating_mul(dh),
                _ => 0,
            }
        });
        // Must be value heads (not key heads) — matches GDN A_log / dt_bias length.
        let ssm_time_step_rank = u("ssm_time_step_rank")
            .or(linear_value_heads)
            .unwrap_or(0);

        let num_experts = u("num_experts").or_else(|| u("expert_count")).unwrap_or(0);
        let num_experts_used = u("num_experts_per_tok")
            .or_else(|| u("expert_used_count"))
            .unwrap_or(0);
        let expert_ffn_size = u("moe_intermediate_size")
            .or_else(|| u("expert_feed_forward_length"))
            .unwrap_or(0);
        let shared_expert_ffn_size = u("shared_expert_intermediate_size")
            .or_else(|| u("expert_shared_feed_forward_length"))
            .unwrap_or(0);

        Ok(Self {
            vocab_size: u("vocab_size").unwrap_or(248_320),
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            nextn_predict_layers: 0, // trunk-only for multimodal HF dirs
            num_attention_heads,
            num_key_value_heads,
            key_length: head_dim,
            value_length: head_dim,
            max_position_embeddings: u("max_position_embeddings").unwrap_or(262_144),
            rms_norm_eps: f("rms_norm_eps").unwrap_or(1e-6),
            rope_theta,
            rope_dim_count,
            rope_dim_sections,
            mrope_interleaved,
            rms_norm_offset: true,
            full_attention_interval: u("full_attention_interval").unwrap_or(0),
            ssm_conv_kernel,
            ssm_group_count,
            ssm_inner_size,
            ssm_state_size,
            ssm_time_step_rank,
            tie_word_embeddings: tie_top.or_else(|| b("tie_word_embeddings")).unwrap_or(true),
            num_experts,
            num_experts_used,
            expert_ffn_size,
            shared_expert_ffn_size,
            expert_weights_scale: f("router_aux_loss_coef").map(|x| x as f32).unwrap_or(1.0),
        })
    }
}
