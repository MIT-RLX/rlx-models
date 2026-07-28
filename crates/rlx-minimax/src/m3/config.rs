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

//! MiniMax-M3 configuration — the `MiniMaxM3SparseForCausalLM` text backbone
//! (`model_type = "minimax_m3_vl"`, `text_config`).
//!
//! M3 is a mixed dense/sparse MoE decoder: layers whose `moe_layer_freq[i]` is
//! set use the 128-expert MoE, the rest a dense SwiGLU-OAI MLP; layers whose
//! `sparse_attention_freq[i]` is set use **MSA** (block-sparse attention via the
//! lightning indexer), the rest plain causal GQA. Every layer is GQA with
//! per-head Gemma QK-norm and partial (NeoX) RoPE.

use anyhow::{Result, anyhow};
use rlx_gguf::{GgufFile, MetaValue};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// MSA (MiniMax Sparse Attention) indexer parameters.
///
/// Field names mirror the HF `sparse_attention_config` block; the flattened
/// `index_*` names used by the transformers config class are noted in comments.
#[derive(Debug, Clone, Deserialize)]
pub struct SparseAttnConfig {
    /// `index_n_heads` — indexer heads (== `num_key_value_heads`, one per GQA group).
    #[serde(rename = "sparse_num_index_heads", default = "d_index_heads")]
    pub index_n_heads: usize,
    /// `index_head_dim` — indexer projection dim per head.
    #[serde(rename = "sparse_index_dim", default = "d_index_dim")]
    pub index_head_dim: usize,
    /// `index_block_size` — keys per block for max-pool selection.
    #[serde(rename = "sparse_block_size", default = "d_block")]
    pub block_size: usize,
    /// `index_topk_blocks` — top-k blocks kept per query.
    #[serde(rename = "sparse_topk_blocks", default = "d_topk")]
    pub topk_blocks: usize,
    /// `index_local_blocks` — blocks ending at the query's own block that are
    /// always visible (force-included).
    #[serde(rename = "sparse_local_block", default = "d_local")]
    pub local_blocks: usize,
    /// Per-layer flag: `1` = this layer runs sparse (MSA), `0` = full attention.
    #[serde(rename = "sparse_attention_freq", default)]
    pub attention_freq: Vec<u8>,
}

impl Default for SparseAttnConfig {
    fn default() -> Self {
        Self {
            index_n_heads: d_index_heads(),
            index_head_dim: d_index_dim(),
            block_size: d_block(),
            topk_blocks: d_topk(),
            local_blocks: d_local(),
            attention_freq: Vec::new(),
        }
    }
}

/// MiniMax-M3 text decoder config (parsed from `text_config`).
#[derive(Debug, Clone, Deserialize)]
pub struct MiniMaxM3Config {
    /// Token-embedding / LM-head vocabulary size.
    pub vocab_size: usize,
    /// Residual-stream (model) width.
    pub hidden_size: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA; `< num_attention_heads`).
    pub num_key_value_heads: usize,
    /// Per-head dim; when absent, derived as `hidden_size / num_attention_heads`.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Rotary dims actually rotated (partial RoPE; `partial_rotary_factor·head_dim`).
    #[serde(rename = "rotary_dim", default = "d_rotary")]
    pub rotary_dim: usize,
    /// RoPE base frequency (`theta`).
    #[serde(default = "d_rope_theta")]
    pub rope_theta: f64,
    /// RMSNorm epsilon.
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f32,
    /// Inner width of the dense MLP layers.
    #[serde(default = "d_dense_inter")]
    pub dense_intermediate_size: usize,
    /// Inner width of each routed expert.
    #[serde(rename = "intermediate_size", default = "d_moe_inter")]
    pub moe_intermediate_size: usize,
    /// Inner width of the shared expert (defaults to `moe_intermediate_size`).
    #[serde(default)]
    pub shared_intermediate_size: Option<usize>,
    /// Number of routed experts per MoE layer.
    #[serde(default = "d_experts")]
    pub num_local_experts: usize,
    /// Experts selected (top-k) per token.
    #[serde(default = "d_top")]
    pub num_experts_per_tok: usize,
    /// Number of always-on shared experts.
    #[serde(default = "d_shared")]
    pub n_shared_experts: usize,
    /// Scale applied to the routed-expert sum before adding the shared expert.
    #[serde(default = "d_scaling")]
    pub routed_scaling_factor: f32,
    /// SwiGLU-OAI sigmoid gain.
    #[serde(default = "d_alpha")]
    pub swiglu_alpha: f32,
    /// SwiGLU-OAI clamp bound.
    #[serde(default = "d_limit")]
    pub swiglu_limit: f32,
    /// Whether the LM head reuses the token-embedding weights.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Per-layer flag: `1` = MoE MLP, `0` = dense MLP.
    #[serde(default)]
    pub moe_layer_freq: Vec<u8>,
    #[serde(rename = "sparse_attention_config", default)]
    pub sparse: SparseAttnConfig,
    /// Placeholder token id that image features replace (from the top-level config).
    #[serde(default)]
    pub image_token_index: Option<usize>,
}

fn d_index_heads() -> usize {
    4
}
fn d_index_dim() -> usize {
    128
}
fn d_block() -> usize {
    128
}
fn d_topk() -> usize {
    16
}
fn d_local() -> usize {
    1
}
fn d_rotary() -> usize {
    64
}
fn d_rope_theta() -> f64 {
    5_000_000.0
}
fn d_eps() -> f32 {
    1e-6
}
fn d_dense_inter() -> usize {
    12288
}
fn d_moe_inter() -> usize {
    3072
}
fn d_experts() -> usize {
    128
}
fn d_top() -> usize {
    4
}
fn d_shared() -> usize {
    1
}
fn d_scaling() -> f32 {
    2.0
}
fn d_alpha() -> f32 {
    1.702
}
fn d_limit() -> f32 {
    7.0
}

/// MiniMax-M3 vision tower config (parsed from `vision_config`) — a CLIP-style
/// ViT with a Conv patch embed, 3D RoPE, and a spatial-merge projector.
#[derive(Debug, Clone, Deserialize)]
pub struct M3VisionConfig {
    /// ViT hidden width.
    #[serde(default = "dv_hidden")]
    pub hidden_size: usize,
    /// Number of attention heads.
    #[serde(default = "dv_heads")]
    pub num_attention_heads: usize,
    /// Number of encoder layers.
    #[serde(default = "dv_layers")]
    pub num_hidden_layers: usize,
    /// FFN inner width.
    #[serde(default = "dv_inter")]
    pub intermediate_size: usize,
    /// Spatial patch size (`patch × patch`).
    #[serde(default = "dv_patch")]
    pub patch_size: usize,
    /// Temporal patch depth (frames folded into each patch).
    #[serde(default = "dv_temporal")]
    pub temporal_patch_size: usize,
    /// Input image channels.
    #[serde(default = "dv_channels")]
    pub num_channels: usize,
    /// LayerNorm epsilon.
    #[serde(default = "dv_ln_eps")]
    pub layer_norm_eps: f32,
    /// 3D-RoPE base frequency.
    #[serde(default = "dv_rope_theta")]
    pub rope_theta: f64,
    /// Neighbourhood side grouped into channels by the projector (`s × s`).
    #[serde(default = "dv_merge")]
    pub spatial_merge_size: usize,
    /// Text hidden the projector maps into.
    #[serde(default = "dv_proj_dim")]
    pub projection_dim: usize,
    /// Inner width of the two projector MLPs (`projector_hidden_size`).
    #[serde(default = "dv_proj_hidden")]
    pub projector_hidden_size: usize,
}

fn dv_hidden() -> usize {
    1280
}
fn dv_heads() -> usize {
    16
}
fn dv_layers() -> usize {
    32
}
fn dv_inter() -> usize {
    5120
}
fn dv_patch() -> usize {
    14
}
fn dv_temporal() -> usize {
    2
}
fn dv_channels() -> usize {
    3
}
fn dv_ln_eps() -> f32 {
    1e-5
}
fn dv_rope_theta() -> f64 {
    10000.0
}
fn dv_merge() -> usize {
    2
}
fn dv_proj_dim() -> usize {
    6144
}
fn dv_proj_hidden() -> usize {
    6144
}

impl Default for M3VisionConfig {
    fn default() -> Self {
        serde_json::from_str("{}").expect("M3VisionConfig defaults")
    }
}

impl M3VisionConfig {
    /// Per-head dim.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    /// Flattened patch input dim (`channels · temporal · patch²`).
    pub fn patch_dim(&self) -> usize {
        self.num_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }
    /// Rotary dims per axis (`2·((2·(head_dim/2)/3)/2)`), matching HF.
    pub fn axis_dim(&self) -> usize {
        let rope_dims = 2 * (self.head_dim() / 2);
        2 * ((rope_dims / 3) / 2)
    }
    /// Total rotated dims (`3 · axis_dim`); the tail passes through.
    pub fn rot_dim(&self) -> usize {
        3 * self.axis_dim()
    }
    /// Parse `vision_config` from a full HF `config.json`.
    pub fn from_hf_config_json(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).map_err(|e| anyhow!("minimax-m3: read {path:?}: {e}"))?;
        let root: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| anyhow!("minimax-m3: parse {path:?}: {e}"))?;
        let vc = root
            .get("vision_config")
            .ok_or_else(|| anyhow!("minimax-m3: no vision_config in {path:?}"))?;
        serde_json::from_value(vc.clone())
            .map_err(|e| anyhow!("minimax-m3: parse vision_config: {e}"))
    }
}

impl MiniMaxM3Config {
    /// Per-head dim (`head_dim` if present, else `hidden/heads`).
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
    /// Number of leading per-head dims that get RoPE.
    pub fn n_rot(&self) -> usize {
        self.rotary_dim
    }
    /// GQA group size (`heads / kv_heads`).
    pub fn kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
    /// Shared-expert inner width.
    pub fn shared_inter(&self) -> usize {
        self.shared_intermediate_size
            .unwrap_or(self.moe_intermediate_size)
    }
    /// Is layer `i` a MoE layer? (default: yes, when no per-layer table is given.)
    pub fn is_moe_layer(&self, i: usize) -> bool {
        self.moe_layer_freq.get(i).map(|&f| f != 0).unwrap_or(true)
    }
    /// Is layer `i` a sparse (MSA) layer? (default: no.)
    pub fn is_sparse_layer(&self, i: usize) -> bool {
        self.sparse
            .attention_freq
            .get(i)
            .map(|&f| f != 0)
            .unwrap_or(false)
    }

    /// Parse a `text_config`-shaped JSON object.
    pub fn from_text_config_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| anyhow!("minimax-m3: parse text_config: {e}"))
    }

    /// Parse from GGUF metadata (best-effort per llama.cpp ggml-org/llama.cpp#24908;
    /// not yet validated against a real M3 GGUF — HF safetensors is the primary path).
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        Self::from_gguf_meta(&raw.metadata)
    }

    /// Core GGUF parse over a metadata map (factored out so it is unit-testable
    /// without a real `GgufFile`).
    pub fn from_gguf_meta(meta: &HashMap<String, MetaValue>) -> Result<Self> {
        let arch = meta
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| anyhow!("missing general.architecture"))?
            .to_string();
        if !matches!(arch.as_str(), "minimax-m3" | "minimax_m3") {
            return Err(anyhow!("from_gguf: arch `{arch}` is not minimax-m3"));
        }
        // MSA keys were published under the `minimax.*` namespace; standard LM
        // keys use the arch name. Try both prefixes for each key.
        let u = |k: &str| -> Option<u32> {
            meta.get(&format!("{arch}.{k}"))
                .or_else(|| meta.get(&format!("minimax.{k}")))
                .and_then(MetaValue::as_u32)
        };
        let ff = |k: &str| -> Option<f32> {
            meta.get(&format!("{arch}.{k}"))
                .or_else(|| meta.get(&format!("minimax.{k}")))
                .and_then(|v| match v {
                    MetaValue::F32(x) => Some(*x),
                    MetaValue::F64(x) => Some(*x as f32),
                    _ => None,
                })
        };
        let req = |k: &str| u(k).ok_or_else(|| anyhow!("missing gguf key `{k}`"));

        let hidden_size = req("embedding_length")? as usize;
        let num_attention_heads = req("attention.head_count")? as usize;
        let num_key_value_heads =
            u("attention.head_count_kv").unwrap_or(num_attention_heads as u32) as usize;
        let num_hidden_layers = req("block_count")? as usize;
        let head_dim = u("attention.key_length").map(|v| v as usize);
        let hd = head_dim.unwrap_or(hidden_size / num_attention_heads);
        let partial = ff("partial_rotary_factor").unwrap_or(0.5);
        let rotary_dim = u("rope.dimension_count")
            .map(|v| v as usize)
            .unwrap_or((hd as f32 * partial) as usize);
        let leading_dense = u("leading_dense_block_count").unwrap_or(3) as usize;
        let mk_freq = |n: usize, dense: usize| -> Vec<u8> {
            (0..n).map(|i| if i < dense { 0 } else { 1 }).collect()
        };

        let sparse = SparseAttnConfig {
            index_n_heads: u("indexer_head_count").unwrap_or(num_key_value_heads as u32) as usize,
            index_head_dim: u("indexer_head_dim").unwrap_or(128) as usize,
            block_size: u("block_size").unwrap_or(128) as usize,
            topk_blocks: u("top_k_blocks").unwrap_or(16) as usize,
            local_blocks: u("local_blocks").unwrap_or(1) as usize,
            attention_freq: mk_freq(num_hidden_layers, leading_dense),
        };

        Ok(Self {
            vocab_size: u("vocab_size").unwrap_or(200_064) as usize,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rotary_dim,
            rope_theta: ff("rope.freq_base").unwrap_or(5_000_000.0) as f64,
            rms_norm_eps: ff("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            dense_intermediate_size: u("feed_forward_length").unwrap_or(12288) as usize,
            moe_intermediate_size: u("expert_feed_forward_length").unwrap_or(3072) as usize,
            shared_intermediate_size: u("expert_shared_feed_forward_length").map(|v| v as usize),
            num_local_experts: u("expert_count").unwrap_or(128) as usize,
            num_experts_per_tok: u("expert_used_count").unwrap_or(4) as usize,
            n_shared_experts: u("expert_shared_count").unwrap_or(1) as usize,
            routed_scaling_factor: ff("expert_weights_scale").unwrap_or(2.0),
            swiglu_alpha: 1.702,
            swiglu_limit: 7.0,
            tie_word_embeddings: false,
            moe_layer_freq: mk_freq(num_hidden_layers, leading_dense),
            sparse,
            image_token_index: None,
        })
    }

    /// Parse a full HF `config.json`, extracting `text_config` (or the root when
    /// already flat) and lifting `image_token_index` from the top level.
    pub fn from_hf_config_json(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).map_err(|e| anyhow!("minimax-m3: read {path:?}: {e}"))?;
        let root: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| anyhow!("minimax-m3: parse {path:?}: {e}"))?;
        let text = root.get("text_config").unwrap_or(&root);
        let mut cfg: Self = serde_json::from_value(text.clone())
            .map_err(|e| anyhow!("minimax-m3: parse text_config in {path:?}: {e}"))?;
        if cfg.image_token_index.is_none() {
            cfg.image_token_index = root
                .get("image_token_index")
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize);
        }
        Ok(cfg)
    }
}
