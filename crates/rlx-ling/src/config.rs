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

//! `BailingMoeV3Config` (`model_type = "bailing_hybrid"`) — the Ling 3.0 family.
//!
//! Reference: `configuration_bailing_moe_v3.py` on
//! [inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny).
//!
//! Several keys present in the published `config.json` are **dead** in the
//! reference `modeling_bailing_moe_v3.py` and are deliberately not modelled here:
//! `use_qk_norm` (V3 MLA has no q/k norm), `group_norm_size` (`GroupRMSNorm` is
//! defined but never instantiated), `linear_silu`, `up_proj_norm`, `value_norm`,
//! `scale_router_input`, `partial_rotary_factor`/`rotary_dim` (the rotary module
//! overrides both: it rotates the **full** `qk_rope_head_dim` slice).

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

/// Which attention mechanism a decoder layer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    /// Full multi-head latent attention (DeepSeek-style, with a decoupled RoPE head).
    Mla,
    /// Kimi Delta Attention — gated delta-net linear attention.
    Kda,
}

/// Granularity of the sigmoid output gate on the MLA branch
/// (`gated_attention_proj_granularity_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnGate {
    /// No `g_proj`.
    None,
    /// `g_proj: hidden → num_heads`, broadcast across `v_head_dim`.
    HeadWise,
    /// `g_proj: hidden → num_heads * v_head_dim`.
    ElementWise,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LingConfig {
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_inter")]
    pub intermediate_size: usize,
    #[serde(default = "d_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_head_dim")]
    pub head_dim: usize,
    #[serde(default = "d_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "d_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "d_max_pos")]
    pub max_position_embeddings: usize,

    // ── MoE ──
    #[serde(default = "d_num_experts")]
    pub num_experts: usize,
    #[serde(default = "d_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "d_num_shared")]
    pub num_shared_experts: usize,
    #[serde(default = "d_moe_inter")]
    pub moe_intermediate_size: usize,
    #[serde(default = "d_moe_inter")]
    pub moe_shared_expert_intermediate_size: usize,
    #[serde(default = "d_n_group")]
    pub n_group: usize,
    #[serde(default = "d_topk_group")]
    pub topk_group: usize,
    #[serde(default = "d_routed_scaling")]
    pub routed_scaling_factor: f32,
    #[serde(default = "d_first_k_dense")]
    pub first_k_dense_replace: usize,
    #[serde(default = "d_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "d_true")]
    pub moe_router_enable_expert_bias: bool,

    // ── MLA ──
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    #[serde(default = "d_kv_lora")]
    pub kv_lora_rank: usize,
    #[serde(default = "d_qk_nope")]
    pub qk_nope_head_dim: usize,
    #[serde(default = "d_qk_rope")]
    pub qk_rope_head_dim: usize,
    #[serde(default = "d_v_head")]
    pub v_head_dim: usize,
    #[serde(default = "d_true")]
    pub rope_interleave: bool,
    #[serde(default)]
    pub gated_attention_proj_granularity_type: Option<String>,

    // ── KDA (linear attention) ──
    #[serde(default = "d_layer_group")]
    pub layer_group_size: usize,
    #[serde(default = "d_conv_kernel")]
    pub short_conv_kernel_size: usize,
    #[serde(default)]
    pub no_kda_lora: bool,
    #[serde(default)]
    pub kda_safe_gate: bool,
    #[serde(default)]
    pub kda_lower_bound: Option<f32>,

    // ── misc ──
    #[serde(default)]
    pub use_qkv_bias: bool,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub num_nextn_predict_layers: usize,
    #[serde(default = "d_eos")]
    pub eos_token_id: u32,
    #[serde(default = "d_pad")]
    pub pad_token_id: u32,
}

fn d_vocab() -> usize {
    157_184
}
fn d_hidden() -> usize {
    2048
}
fn d_inter() -> usize {
    5120
}
fn d_layers() -> usize {
    20
}
fn d_heads() -> usize {
    16
}
fn d_head_dim() -> usize {
    128
}
fn d_rms_eps() -> f32 {
    1e-6
}
fn d_rope_theta() -> f32 {
    600_000.0
}
fn d_max_pos() -> usize {
    32_768
}
fn d_num_experts() -> usize {
    256
}
fn d_experts_per_tok() -> usize {
    8
}
fn d_num_shared() -> usize {
    1
}
fn d_moe_inter() -> usize {
    512
}
fn d_n_group() -> usize {
    8
}
fn d_topk_group() -> usize {
    4
}
fn d_routed_scaling() -> f32 {
    1.0
}
fn d_first_k_dense() -> usize {
    1
}
fn d_true() -> bool {
    true
}
fn d_kv_lora() -> usize {
    512
}
fn d_qk_nope() -> usize {
    128
}
fn d_qk_rope() -> usize {
    64
}
fn d_v_head() -> usize {
    128
}
fn d_layer_group() -> usize {
    5
}
fn d_conv_kernel() -> usize {
    4
}
fn d_eos() -> u32 {
    156_892
}
fn d_pad() -> u32 {
    156_892
}

impl LingConfig {
    pub fn from_json_str(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_json_str(&std::fs::read_to_string(path)?)
    }

    /// `qk_nope_head_dim + qk_rope_head_dim` — the width Q/K enter attention at.
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// `BailingMoeV3MultiLatentAttention.scaling` — note this is `qk_head_dim`,
    /// **not** `v_head_dim`. Without `rope_scaling` there is no YaRN mscale.
    pub fn attn_score_scale(&self) -> f32 {
        (self.qk_head_dim() as f32).powf(-0.5)
    }

    /// Width of the KDA q/k/v projections (`num_heads * head_dim`).
    pub fn kda_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// `BailingMoeV3DecoderLayer.attention_layer_type`: every `layer_group_size`-th
    /// layer is full attention, as is any layer in the ragged tail past the last
    /// whole group.
    pub fn attn_kind(&self, layer: usize) -> AttnKind {
        let gs = self.layer_group_size.max(1);
        let whole = self.num_hidden_layers / gs * gs;
        if (layer + 1).is_multiple_of(gs) || layer >= whole {
            AttnKind::Mla
        } else {
            AttnKind::Kda
        }
    }

    /// The first `first_k_dense_replace` layers keep a dense SwiGLU MLP.
    pub fn is_moe_layer(&self, layer: usize) -> bool {
        self.num_experts > 0 && layer >= self.first_k_dense_replace
    }

    /// Combined width of the always-on shared expert.
    pub fn shared_intermediate_size(&self) -> usize {
        self.moe_shared_expert_intermediate_size * self.num_shared_experts
    }

    pub fn attn_gate(&self) -> AttnGate {
        match self.gated_attention_proj_granularity_type.as_deref() {
            Some("head_wise") => AttnGate::HeadWise,
            Some("element_wise") => AttnGate::ElementWise,
            _ => AttnGate::None,
        }
    }

    /// `(cos, sin)` tables of shape `[seq, qk_rope_head_dim / 2]` for positions
    /// `0..seq`, laid out for [`rlx_ir::RopeStyle::GptJ`].
    ///
    /// The reference rotary module clones the config with
    /// `head_dim = qk_rope_head_dim` and `partial_rotary_factor = 1.0`, so the
    /// rotary width is the full rope slice regardless of the `partial_rotary_factor`
    /// / `rotary_dim` keys in `config.json`.
    pub fn rope_tables(&self, seq: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.qk_rope_head_dim / 2;
        let mut cos = Vec::with_capacity(seq * half);
        let mut sin = Vec::with_capacity(seq * half);
        for pos in 0..seq {
            for j in 0..half {
                let inv = (self.rope_theta as f64)
                    .powf(-2.0 * (j as f64) / (self.qk_rope_head_dim as f64));
                let ang = pos as f64 * inv;
                cos.push(ang.cos() as f32);
                sin.push(ang.sin() as f32);
            }
        }
        (cos, sin)
    }

    /// Reject configurations the builder cannot honour, with an actionable message.
    pub fn validate(&self) -> Result<()> {
        if !self.rope_interleave {
            anyhow::bail!(
                "rope_interleave=false is unreachable in modeling_bailing_moe_v3.py \
                 (the non-interleaved branch divides by zero); refusing to guess a layout"
            );
        }
        if !self.num_experts.is_multiple_of(self.n_group) {
            anyhow::bail!(
                "num_experts ({}) must be divisible by n_group ({})",
                self.num_experts,
                self.n_group
            );
        }
        if self.kda_safe_gate && self.kda_lower_bound.is_none() {
            anyhow::bail!("kda_safe_gate=true requires kda_lower_bound to be set");
        }
        if self.tie_word_embeddings {
            anyhow::bail!(
                "tie_word_embeddings=true is not supported: the shared lm_head block \
                 reads the tied table from `model.embed_tokens.weight`, but Bailing \
                 names it `model.word_embeddings.weight`. Ling 3.0 ships untied."
            );
        }
        if self.num_nextn_predict_layers > 0 {
            anyhow::bail!(
                "num_nextn_predict_layers={} — MTP heads are not built (inference \
                 ignores them; strip them from the checkpoint or set it to 0)",
                self.num_nextn_predict_layers
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published Ling-3.0-tiny `config.json` (verbatim keys).
    const TINY: &str = include_str!("../fixtures/ling-3.0-tiny-config.json");

    #[test]
    fn parses_published_tiny_config() {
        let cfg = LingConfig::from_json_str(TINY).expect("parse");
        cfg.validate().expect("valid");
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_experts, 128);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.qk_head_dim(), 192);
        assert_eq!(cfg.kda_proj_dim(), 2048);
        assert_eq!(cfg.q_lora_rank, Some(256));
        assert_eq!(cfg.kda_lower_bound, Some(-5.0));
        assert!(cfg.kda_safe_gate && cfg.no_kda_lora);
        assert_eq!(cfg.attn_gate(), AttnGate::HeadWise);
        assert_eq!(cfg.shared_intermediate_size(), 512);
    }

    /// Layer split must match the checkpoint: MLA at 3/7/11/15/19/23, KDA elsewhere,
    /// and only layer 0 keeps a dense MLP.
    #[test]
    fn layer_kinds_match_checkpoint() {
        let cfg = LingConfig::from_json_str(TINY).unwrap();
        let mla: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|&i| cfg.attn_kind(i) == AttnKind::Mla)
            .collect();
        assert_eq!(mla, vec![3, 7, 11, 15, 19, 23]);
        let dense: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|&i| !cfg.is_moe_layer(i))
            .collect();
        assert_eq!(dense, vec![0]);
    }

    /// The ragged tail past the last whole group is forced to full attention.
    #[test]
    fn ragged_tail_is_full_attention() {
        let cfg = LingConfig::from_json_str(
            r#"{"num_hidden_layers":10,"layer_group_size":4,"num_experts":0}"#,
        )
        .unwrap();
        // whole = 10/4*4 = 8 → layers 8,9 are tail; 3,7 are group ends.
        let kinds: Vec<AttnKind> = (0..10).map(|i| cfg.attn_kind(i)).collect();
        use AttnKind::*;
        assert_eq!(
            kinds,
            vec![Kda, Kda, Kda, Mla, Kda, Kda, Kda, Mla, Mla, Mla]
        );
    }

    #[test]
    fn rope_tables_start_at_identity() {
        let cfg = LingConfig::from_json_str(TINY).unwrap();
        let (cos, sin) = cfg.rope_tables(4);
        let half = cfg.qk_rope_head_dim / 2;
        assert_eq!(cos.len(), 4 * half);
        assert!(cos[..half].iter().all(|c| (c - 1.0).abs() < 1e-6));
        assert!(sin[..half].iter().all(|s| s.abs() < 1e-6));
        // Position 1, channel 0 → angle 1 rad.
        assert!((cos[half] - 1.0f32.cos()).abs() < 1e-6);
    }
}
