// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Kimi-K3 configuration (`KimiK3Config` = multimodal wrapper over the
//! `KimiLinearConfig` text model + `KimiVisionConfig` ViT tower).
//!
//! The text model interleaves **KDA** (Kimi Delta Attention, a gated delta-net
//! linear attention) and **MLA** (NoPE multi-head latent attention) layers, with
//! a **LatentMoE** FFN (896 experts / 16 active / 2 shared, sigmoid noaux_tc
//! grouped-topk). `hidden_act = "situ"` — a custom gated activation.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// KDA / full-attention layer schedule + KDA hyper-params (`linear_attn_config`).
#[derive(Debug, Clone, Deserialize)]
pub struct LinearAttnConfig {
    /// 1-indexed layer numbers that use full MLA attention.
    #[serde(default)]
    pub full_attn_layers: Vec<usize>,
    /// 1-indexed layer numbers that use KDA linear attention.
    #[serde(default)]
    pub kda_layers: Vec<usize>,
    #[serde(default = "d_kda_head_dim")]
    pub head_dim: usize,
    #[serde(default = "d_kda_heads")]
    pub num_heads: usize,
    #[serde(default = "d_conv")]
    pub short_conv_kernel_size: usize,
    #[serde(default)]
    pub gate_lower_bound: Option<f32>,
    #[serde(default)]
    pub use_full_rank_gate: bool,
}

fn d_kda_head_dim() -> usize {
    128
}
fn d_kda_heads() -> usize {
    96
}
fn d_conv() -> usize {
    4
}

/// The KimiLinear text-decoder config (`config.json["text_config"]`).
#[derive(Debug, Clone, Deserialize)]
pub struct KimiLinearConfig {
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
    #[serde(default = "d_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "d_situ")]
    pub hidden_act: String,

    // ── MLA (NoPE) ──────────────────────────────────────────
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    #[serde(default = "d_kv_lora")]
    pub kv_lora_rank: usize,
    #[serde(default = "d_nope")]
    pub qk_nope_head_dim: usize,
    #[serde(default = "d_rope")]
    pub qk_rope_head_dim: usize,
    #[serde(default = "d_vdim")]
    pub v_head_dim: usize,
    #[serde(default)]
    pub mla_use_nope: bool,
    #[serde(default)]
    pub mla_use_output_gate: bool,

    // ── LatentMoE ───────────────────────────────────────────
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default = "d_topk")]
    pub num_experts_per_token: usize,
    #[serde(default)]
    pub num_shared_experts: usize,
    #[serde(default = "d_scaling")]
    pub routed_scaling_factor: f32,
    #[serde(default = "d_true")]
    pub moe_renormalize: bool,
    #[serde(default = "d_sigmoid")]
    pub moe_router_activation_func: String,
    #[serde(default = "d_moe_inter")]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub routed_expert_hidden_size: Option<usize>,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    #[serde(default = "d_one")]
    pub moe_layer_freq: usize,
    #[serde(default = "d_true")]
    pub use_grouped_topk: bool,
    #[serde(default = "d_one")]
    pub num_expert_group: usize,
    #[serde(default = "d_one")]
    pub topk_group: usize,
    #[serde(default = "d_noaux")]
    pub topk_method: String,
    #[serde(default)]
    pub latent_moe_use_norm: bool,

    // ── situ activation + attention residuals ───────────────
    #[serde(default)]
    pub activation_situ_beta: Option<f32>,
    #[serde(default)]
    pub activation_situ_linear_beta: Option<f32>,
    #[serde(default)]
    pub attn_res_block_size: Option<usize>,

    #[serde(default = "d_maxpos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// KDA/full-attn schedule. Absent ⇒ every layer is full MLA.
    #[serde(default)]
    pub linear_attn_config: Option<LinearAttnConfig>,
}

fn d_vocab() -> usize {
    163840
}
fn d_hidden() -> usize {
    7168
}
fn d_inter() -> usize {
    33792
}
fn d_layers() -> usize {
    93
}
fn d_heads() -> usize {
    96
}
fn d_eps() -> f32 {
    1e-5
}
fn d_situ() -> String {
    "situ".into()
}
fn d_kv_lora() -> usize {
    512
}
fn d_nope() -> usize {
    128
}
fn d_rope() -> usize {
    64
}
fn d_vdim() -> usize {
    128
}
fn d_topk() -> usize {
    16
}
fn d_scaling() -> f32 {
    1.0
}
fn d_true() -> bool {
    true
}
fn d_sigmoid() -> String {
    "sigmoid".into()
}
fn d_moe_inter() -> usize {
    3072
}
fn d_one() -> usize {
    1
}
fn d_noaux() -> String {
    "noaux_tc".into()
}
fn d_maxpos() -> usize {
    1_048_576
}

impl KimiLinearConfig {
    /// MLA per-head query dim = nope + rope carried dims.
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// `true` if layer `i` (0-indexed) uses KDA linear attention. The config
    /// lists layers 1-indexed, so we test `i + 1`. No `linear_attn_config`
    /// ⇒ every layer is full MLA.
    pub fn is_kda_layer(&self, i: usize) -> bool {
        self.linear_attn_config
            .as_ref()
            .is_some_and(|c| c.kda_layers.contains(&(i + 1)))
    }

    /// `true` if layer `i` uses the MoE FFN (dense for the first
    /// `first_k_dense_replace` layers).
    pub fn is_moe_layer(&self, i: usize) -> bool {
        self.num_experts.is_some() && i >= self.first_k_dense_replace
    }

    pub fn situ_beta(&self) -> f32 {
        self.activation_situ_beta.unwrap_or(1.0)
    }
    pub fn situ_linear_beta(&self) -> Option<f32> {
        self.activation_situ_linear_beta
    }
}

/// The vision ViT tower config (`config.json["vision_config"]`).
#[derive(Debug, Clone, Deserialize)]
pub struct KimiVisionConfig {
    #[serde(default = "d_vis_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_vis_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_vis_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_vis_inter")]
    pub intermediate_size: usize,
    #[serde(default = "d_patch")]
    pub patch_size: usize,
    #[serde(default = "d_mm_hidden")]
    pub mm_hidden_size: usize,
    #[serde(default = "d_merge")]
    pub merge_kernel_size: Vec<usize>,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "d_pos")]
    pub init_pos_emb_height: usize,
    #[serde(default = "d_pos")]
    pub init_pos_emb_width: usize,
    #[serde(default = "d_pos_t")]
    pub init_pos_emb_time: usize,
}

fn d_vis_hidden() -> usize {
    1024
}
fn d_vis_layers() -> usize {
    24
}
fn d_vis_heads() -> usize {
    16
}
fn d_vis_inter() -> usize {
    4096
}
fn d_patch() -> usize {
    14
}
fn d_mm_hidden() -> usize {
    1024
}
fn d_merge() -> Vec<usize> {
    vec![2, 2]
}
fn d_pos() -> usize {
    64
}
fn d_pos_t() -> usize {
    4
}

/// Top-level multimodal config.
#[derive(Debug, Clone, Deserialize)]
pub struct KimiK3Config {
    pub text_config: KimiLinearConfig,
    #[serde(default)]
    pub vision_config: Option<KimiVisionConfig>,
    #[serde(default)]
    pub media_placeholder_token_id: Option<i64>,
    #[serde(default)]
    pub bos_token_id: Option<i64>,
    #[serde(default)]
    pub eos_token_id: Option<i64>,
}

impl KimiK3Config {
    /// Load `config.json` from a model directory (or a direct file path).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let file = if p.is_dir() {
            p.join("config.json")
        } else {
            p.to_path_buf()
        };
        let text = std::fs::read_to_string(&file)
            .with_context(|| format!("reading Kimi-K3 config {file:?}"))?;
        serde_json::from_str(&text).with_context(|| format!("parsing Kimi-K3 config {file:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_layer_schedule() {
        let cfg: KimiLinearConfig = serde_json::from_str(
            r#"{"num_hidden_layers":6,"first_k_dense_replace":1,"num_experts":8,
                "linear_attn_config":{"kda_layers":[1,2,3,5],"full_attn_layers":[4,6],
                "head_dim":32,"num_heads":4,"short_conv_kernel_size":4,"gate_lower_bound":-5.0,
                "use_full_rank_gate":true}}"#,
        )
        .unwrap();
        // 1-indexed schedule: layers 0,1,2,4 are KDA; 3,5 are MLA.
        assert!(cfg.is_kda_layer(0) && cfg.is_kda_layer(1) && cfg.is_kda_layer(2));
        assert!(!cfg.is_kda_layer(3) && cfg.is_kda_layer(4) && !cfg.is_kda_layer(5));
        // layer 0 dense, 1..5 MoE.
        assert!(!cfg.is_moe_layer(0) && cfg.is_moe_layer(1));
    }
}
