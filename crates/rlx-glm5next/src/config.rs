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

//! `glm5next` configuration — read from GGUF metadata or a HF `config.json`.
//!
//! The GGUF converter flattens the HF `text_config` into `glm5next.*` keys. The
//! mapping is not one-to-one, so both readers land on the same struct:
//!
//! ```text
//!   HF text_config                     GGUF metadata
//!   ─────────────────────────────────  ───────────────────────────────────────
//!   num_hidden_layers (45)             block_count (46) − nextn_predict_layers
//!   hidden_size                        embedding_length
//!   intermediate_size                  feed_forward_length
//!   num_attention_heads                attention.head_count
//!   layer_types[i]                     attention.head_count_kv[i]  (0 = KDA)
//!   rms_norm_eps                       attention.layer_norm_rms_epsilon
//!   hc_eps                             attention.layer_norm_epsilon
//!   hc_mult / hc_sinkhorn_iters        hyper_connection.count / .sinkhorn_iterations
//!   linear_attn_config.head_dim        kda.head_dim
//!   linear_attn_config.gate_lower_bound kda.gate_lower_bound
//!   qk_nope_head_dim / v_head_dim      attention.key_length_mla / .value_length_mla
//!   n_routed_experts                   expert_count
//!   moe_intermediate_size              expert_feed_forward_length
//!   routed_scaling_factor              expert_weights_scale
//!   first_k_dense_replace              leading_dense_block_count
//!   swiglu_limit                       swiglu_clamp_exp[i]
//! ```
//!
//! `block_count` counts the MTP block, `num_hidden_layers` does not — the MTP
//! block is layer 45 of 46 in the checkpoint and is skipped by the text flow
//! unless [`Glm5NextConfig::with_mtp`] is set.

use anyhow::{Context, Result, bail};
use rlx_gguf::{GgufFile, MetaValue};
use serde::Deserialize;
use std::path::Path;

/// Which attention mechanism a layer runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    /// Kimi Delta Attention — gated delta-net linear attention. Carries the
    /// model's only positional information (the text model has no RoPE).
    Kda,
    /// NoPE multi-head latent attention behind a DeepSeek sparse-attention
    /// (DSA) indexer.
    MlaDsa,
}

/// Whether a layer runs its own DSA indexer or reuses the previous one's
/// selection (`indexer_types`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexerKind {
    /// This layer computes its own top-k selection.
    Full,
    /// Reuse the top-k selection propagated from the previous full layer.
    Shared,
}

/// GGUF `general.architecture` values this crate claims.
pub const ACCEPTED_ARCHES: &[&str] = &["glm5next"];

/// HF `model_type` for the multimodal wrapper and its text tower.
pub const HF_MODEL_TYPE: &str = "glm5_next";
/// HF `model_type` of `config.json["text_config"]`.
pub const HF_TEXT_MODEL_TYPE: &str = "glm5_next_text";

/// `expert_gating_func` in GGUF: 1 = softmax, 2 = sigmoid.
const GATING_SIGMOID: u32 = 2;

/// GLM-5.3-Flash text-decoder configuration.
#[derive(Debug, Clone)]
pub struct Glm5NextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Dense-MLP width (the first `first_k_dense_replace` layers).
    pub intermediate_size: usize,
    /// Decoder layers **excluding** the MTP block.
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,

    // ── per-layer schedule ──────────────────────────────────
    /// One entry per `num_hidden_layers`.
    pub layer_types: Vec<AttnKind>,
    /// One entry per `num_hidden_layers`.
    pub indexer_types: Vec<IndexerKind>,
    /// Layers `< first_k_dense_replace` use a dense MLP, the rest MoE.
    pub first_k_dense_replace: usize,

    // ── MLA (NoPE) ──────────────────────────────────────────
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    /// Always 0 for GLM-5.3-Flash — the DSA layers are pure NoPE.
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,

    // ── DSA lightning indexer ───────────────────────────────
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    /// Tokens per compressed candidate pool.
    pub index_kpool: usize,
    pub index_kpool_always_select_tail: bool,

    // ── KDA linear attention ────────────────────────────────
    pub linear_num_heads: usize,
    pub linear_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    /// `Some` selects the safe-gate form `lb·σ(exp(A_log)·g)`; `None` the
    /// plain `−exp(A_log)·softplus(g)`.
    pub linear_lower_bound: Option<f32>,

    // ── mHC ─────────────────────────────────────────────────
    /// Number of parallel residual streams (`hc_mult`).
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,

    // ── MoE ─────────────────────────────────────────────────
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub moe_intermediate_size: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    /// SwiGLU clamp: `gate.clamp(max=L)`, `up.clamp(-L, L)`.
    pub swiglu_limit: f32,

    // ── MTP ─────────────────────────────────────────────────
    /// Number of trailing multi-token-prediction blocks in the checkpoint.
    pub num_nextn_predict_layers: usize,
    /// Build the MTP block into the flow (off by default).
    pub with_mtp: bool,
}

impl Glm5NextConfig {
    /// `num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim)`.
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// KDA q/k/v projection width — `linear_num_heads * linear_head_dim`.
    pub fn kda_proj(&self) -> usize {
        self.linear_num_heads * self.linear_head_dim
    }

    /// Total blocks in the checkpoint, MTP included.
    pub fn block_count(&self) -> usize {
        self.num_hidden_layers + self.num_nextn_predict_layers
    }

    pub fn attn_kind(&self, layer: usize) -> AttnKind {
        self.layer_types[layer]
    }

    pub fn indexer_kind(&self, layer: usize) -> IndexerKind {
        self.indexer_types[layer]
    }

    pub fn is_moe_layer(&self, layer: usize) -> bool {
        layer >= self.first_k_dense_replace
    }

    /// Shared-expert MLP width — `n_shared_experts * moe_intermediate_size`.
    pub fn shared_intermediate_size(&self) -> usize {
        self.n_shared_experts * self.moe_intermediate_size
    }

    /// Width of the flattened mHC gate output: `(2 + hc_mult) * hc_mult`.
    pub fn hc_mix(&self) -> usize {
        (2 + self.hc_mult) * self.hc_mult
    }

    /// Every tensor a checkpoint must carry for block `block`, as GGUF names
    /// without the `blk.{i}.` prefix.
    ///
    /// This is the checkpoint contract stated once, in config terms, so it can
    /// be checked against a real file without loading one — see
    /// `tests/tensor_manifest.rs`, which asserts it reproduces the published
    /// GLM-5.3-Flash tensor list exactly for all 46 blocks, and separately that
    /// the emitters request exactly these names and no others.
    ///
    /// `block` may be `num_hidden_layers` (the MTP block), which swaps the two
    /// mHC sites for the `nextn.*` head. Everything else follows from
    /// [`Self::attn_kind`] and [`Self::is_moe_layer`].
    pub fn block_tensor_names(&self, block: usize) -> Vec<String> {
        let mut v: Vec<String> = vec!["attn_norm.weight".into(), "ffn_norm.weight".into()];
        let is_mtp = block >= self.num_hidden_layers;

        // The MTP block carries the next-token head instead of hyper-connections;
        // it is not part of the residual-stream stack.
        if is_mtp {
            v.extend(
                [
                    "nextn.eh_proj.weight",
                    "nextn.enorm.weight",
                    "nextn.hnorm.weight",
                    "nextn.shared_head_norm.weight",
                ]
                .map(String::from),
            );
        } else {
            for site in ["hc_attn", "hc_ffn"] {
                v.push(format!("{site}_fn.weight"));
                v.push(format!("{site}_base.weight"));
                v.push(format!("{site}_scale.weight"));
            }
        }

        // MTP reuses the sparse-attention block.
        let kind = if is_mtp {
            AttnKind::MlaDsa
        } else {
            self.attn_kind(block)
        };
        match kind {
            AttnKind::Kda => {
                v.extend(
                    [
                        "attn_q.weight",
                        "attn_k.weight",
                        "attn_v.weight",
                        "attn_output.weight",
                        "ssm_conv1d_q.weight",
                        "ssm_conv1d_k.weight",
                        "ssm_conv1d_v.weight",
                        "ssm_a",
                        "ssm_dt.bias",
                        "ssm_beta.weight",
                        "ssm_f_a.weight",
                        "ssm_f_b.weight",
                        "ssm_g_a.weight",
                        "ssm_g_b.weight",
                        "ssm_norm.weight",
                    ]
                    .map(String::from),
                );
            }
            AttnKind::MlaDsa => {
                v.extend(
                    [
                        "attn_q_a.weight",
                        "attn_q_a_norm.weight",
                        "attn_q_b.weight",
                        "attn_kv_a_mqa.weight",
                        "attn_kv_a_norm.weight",
                        "attn_k_b.weight",
                        "attn_v_b.weight",
                        "attn_output.weight",
                    ]
                    .map(String::from),
                );
                // A `shared` indexer layer reuses the previous layer's top-k and
                // so carries no indexer weights of its own.
                let own_indexer = is_mtp || self.indexer_kind(block) == IndexerKind::Full;
                if own_indexer {
                    v.extend(
                        [
                            "indexer.attn_q_b.weight",
                            "indexer.attn_k.weight",
                            "indexer.k_norm.weight",
                            "indexer.k_norm.bias",
                            "indexer.proj.weight",
                            "indexer_compressor_gate.weight",
                            "indexer_compressor_ape.weight",
                        ]
                        .map(String::from),
                    );
                }
            }
        }

        // MTP's FFN is always the MoE one.
        if is_mtp || self.is_moe_layer(block) {
            v.extend(
                [
                    "ffn_gate_inp.weight",
                    "exp_probs_b.bias",
                    "ffn_gate_exps.weight",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                    "ffn_gate_shexp.weight",
                    "ffn_up_shexp.weight",
                    "ffn_down_shexp.weight",
                ]
                .map(String::from),
            );
        } else {
            v.extend(["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"].map(String::from));
        }
        v.sort();
        v
    }

    /// The whole checkpoint's tensor names, `blk.{i}.`-prefixed, plus the three
    /// global tensors. See [`Self::block_tensor_names`].
    pub fn tensor_manifest(&self) -> Vec<String> {
        let mut v = vec![
            "token_embd.weight".to_string(),
            "output_norm.weight".to_string(),
        ];
        if !self.tie_word_embeddings {
            v.push("output.weight".to_string());
        }
        for i in 0..self.block_count() {
            v.extend(
                self.block_tensor_names(i)
                    .into_iter()
                    .map(|s| format!("blk.{i}.{s}")),
            );
        }
        v.sort();
        v
    }

    /// Whether a DSA layer at this sequence length selects *every* visible
    /// token, making the sparse mask exactly the causal mask.
    ///
    /// The indexer picks `index_topk / index_kpool` pools of `index_kpool`
    /// tokens each, out of `ceil(seq / index_kpool)` pools. When
    /// `seq <= index_topk` there are never more pools than the budget, so the
    /// selection is the identity and DSA reduces to dense causal attention.
    /// This is an exact algebraic property, not an approximation.
    pub fn dsa_is_dense(&self, seq: usize) -> bool {
        seq <= self.index_topk
    }

    pub fn validate(&self) -> Result<()> {
        if self.layer_types.len() != self.num_hidden_layers {
            bail!(
                "layer_types has {} entries, expected num_hidden_layers = {}",
                self.layer_types.len(),
                self.num_hidden_layers
            );
        }
        if self.indexer_types.len() != self.num_hidden_layers {
            bail!(
                "indexer_types has {} entries, expected num_hidden_layers = {}",
                self.indexer_types.len(),
                self.num_hidden_layers
            );
        }
        if self.qk_rope_head_dim != 0 {
            bail!(
                "glm5next expects NoPE on the DSA layers, got qk_rope_head_dim = {}",
                self.qk_rope_head_dim
            );
        }
        if self.hc_mult == 0 {
            bail!("hc_mult must be >= 1");
        }
        if self.index_kpool == 0 {
            bail!("index_kpool must be >= 1");
        }
        if self.num_experts_per_tok > self.n_routed_experts {
            bail!(
                "num_experts_per_tok ({}) exceeds n_routed_experts ({})",
                self.num_experts_per_tok,
                self.n_routed_experts
            );
        }
        if !self.n_routed_experts.is_multiple_of(self.n_group.max(1)) {
            bail!(
                "n_routed_experts ({}) is not divisible by n_group ({})",
                self.n_routed_experts,
                self.n_group
            );
        }
        // A layer that reuses a previous selection needs one to exist.
        if self.indexer_types.first() == Some(&IndexerKind::Shared) {
            bail!("layer 0 cannot have indexer_types = shared");
        }
        Ok(())
    }
}

// ─────────────────────────── GGUF ───────────────────────────

fn meta<'a>(raw: &'a GgufFile, key: &str) -> Option<&'a MetaValue> {
    raw.metadata.get(key)
}

fn meta_u32(raw: &GgufFile, key: &str) -> Result<u32> {
    meta(raw, key)
        .and_then(MetaValue::as_u32)
        .with_context(|| format!("glm5next: missing or non-integer GGUF key `{key}`"))
}

fn meta_u32_or(raw: &GgufFile, key: &str, default: u32) -> u32 {
    meta(raw, key)
        .and_then(MetaValue::as_u32)
        .unwrap_or(default)
}

fn as_f32(v: &MetaValue) -> Option<f32> {
    match v {
        MetaValue::F32(x) => Some(*x),
        MetaValue::F64(x) => Some(*x as f32),
        _ => None,
    }
}

fn meta_f32_or(raw: &GgufFile, key: &str, default: f32) -> f32 {
    meta(raw, key).and_then(as_f32).unwrap_or(default)
}

fn meta_bool_or(raw: &GgufFile, key: &str, default: bool) -> bool {
    match meta(raw, key) {
        Some(MetaValue::Bool(b)) => *b,
        Some(other) => other.as_u32().map(|v| v != 0).unwrap_or(default),
        None => default,
    }
}

fn meta_array<'a>(raw: &'a GgufFile, key: &str) -> Option<&'a [MetaValue]> {
    match meta(raw, key) {
        Some(MetaValue::Array(v)) => Some(v.as_slice()),
        _ => None,
    }
}

impl Glm5NextConfig {
    /// Parse a `glm5next` GGUF's metadata.
    ///
    /// Only the metadata is read, so this works on the metadata-only first
    /// shard that `llama-gguf-split` emits (unsloth's `…-00001-of-000NN.gguf`
    /// carries all 72 KV pairs and zero tensors).
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = raw
            .metadata
            .get("general.architecture")
            .and_then(|v| match v {
                MetaValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or_default();
        if !ACCEPTED_ARCHES.contains(&arch) {
            bail!(
                "Glm5NextConfig::from_gguf: expected general.architecture ∈ {ACCEPTED_ARCHES:?}, \
                 got {arch:?}"
            );
        }

        let block_count = meta_u32(raw, "glm5next.block_count")? as usize;
        let num_nextn_predict_layers =
            meta_u32_or(raw, "glm5next.nextn_predict_layers", 0) as usize;
        if num_nextn_predict_layers >= block_count {
            bail!(
                "glm5next: nextn_predict_layers ({num_nextn_predict_layers}) >= \
                 block_count ({block_count})"
            );
        }
        let num_hidden_layers = block_count - num_nextn_predict_layers;

        let hidden_size = meta_u32(raw, "glm5next.embedding_length")? as usize;
        let num_attention_heads = meta_u32(raw, "glm5next.attention.head_count")? as usize;

        // `head_count_kv` is per-block: 0 marks a KDA layer, non-zero an MLA
        // layer. It is sized `block_count`, so the MTP tail is trimmed here.
        let kv_counts = meta_array(raw, "glm5next.attention.head_count_kv").with_context(|| {
            "glm5next: attention.head_count_kv must be a per-block array (it is what \
             distinguishes KDA layers from MLA layers)"
        })?;
        if kv_counts.len() < num_hidden_layers {
            bail!(
                "glm5next: attention.head_count_kv has {} entries, need at least {}",
                kv_counts.len(),
                num_hidden_layers
            );
        }
        let layer_types: Vec<AttnKind> = kv_counts[..num_hidden_layers]
            .iter()
            .map(|v| match v.as_u32().unwrap_or(0) {
                0 => AttnKind::Kda,
                _ => AttnKind::MlaDsa,
            })
            .collect();

        // No GGUF key mirrors HF `indexer_types` yet; every MLA layer owning
        // its indexer is the shipped GLM-5.3-Flash configuration.
        let indexer_types = vec![IndexerKind::Full; num_hidden_layers];

        // The KDA head count has no dedicated GGUF key — it equals the
        // attention head count in every published glm5next checkpoint, and the
        // q/k/v projection width (`kda.head_dim * heads`) cross-checks it.
        let linear_head_dim = meta_u32_or(raw, "glm5next.kda.head_dim", 128) as usize;
        let linear_num_heads =
            meta_u32_or(raw, "glm5next.kda.head_count", num_attention_heads as u32) as usize;

        let swiglu_limit = meta_array(raw, "glm5next.swiglu_clamp_exp")
            .and_then(|a| a.first())
            .and_then(as_f32)
            .or_else(|| meta(raw, "glm5next.swiglu_clamp_exp").and_then(as_f32))
            .unwrap_or(f32::INFINITY);

        let gating = meta_u32_or(raw, "glm5next.expert_gating_func", GATING_SIGMOID);
        if gating != GATING_SIGMOID {
            bail!(
                "glm5next: expert_gating_func = {gating}, only sigmoid ({GATING_SIGMOID}) \
                 routing is implemented"
            );
        }

        let cfg = Self {
            vocab_size: meta_u32(raw, "glm5next.vocab_size")? as usize,
            hidden_size,
            intermediate_size: meta_u32(raw, "glm5next.feed_forward_length")? as usize,
            num_hidden_layers,
            num_attention_heads,
            rms_norm_eps: meta_f32_or(raw, "glm5next.attention.layer_norm_rms_epsilon", 1e-5),
            max_position_embeddings: meta_u32_or(raw, "glm5next.context_length", 1 << 20) as usize,
            // glm5next ships an untied `output.weight`.
            tie_word_embeddings: false,

            layer_types,
            indexer_types,
            first_k_dense_replace: meta_u32_or(raw, "glm5next.leading_dense_block_count", 0)
                as usize,

            q_lora_rank: meta_u32(raw, "glm5next.attention.q_lora_rank")? as usize,
            kv_lora_rank: meta_u32(raw, "glm5next.attention.kv_lora_rank")? as usize,
            qk_nope_head_dim: meta_u32(raw, "glm5next.attention.key_length_mla")? as usize,
            qk_rope_head_dim: meta_u32_or(raw, "glm5next.rope.dimension_count", 0) as usize,
            v_head_dim: meta_u32(raw, "glm5next.attention.value_length_mla")? as usize,

            index_n_heads: meta_u32(raw, "glm5next.attention.indexer.head_count")? as usize,
            index_head_dim: meta_u32(raw, "glm5next.attention.indexer.key_length")? as usize,
            index_topk: meta_u32(raw, "glm5next.attention.indexer.top_k")? as usize,
            index_kpool: meta_u32_or(raw, "glm5next.attention.indexer.kpool", 1) as usize,
            index_kpool_always_select_tail: meta_bool_or(
                raw,
                "glm5next.attention.indexer.kpool_always_select_tail",
                true,
            ),

            linear_num_heads,
            linear_head_dim,
            linear_conv_kernel_dim: meta_u32_or(raw, "glm5next.ssm.conv_kernel", 4) as usize,
            linear_lower_bound: meta(raw, "glm5next.kda.gate_lower_bound").and_then(as_f32),

            hc_mult: meta_u32_or(raw, "glm5next.hyper_connection.count", 1) as usize,
            hc_sinkhorn_iters: meta_u32_or(raw, "glm5next.hyper_connection.sinkhorn_iterations", 20)
                as usize,
            hc_eps: meta_f32_or(raw, "glm5next.hyper_connection.epsilon", 1e-6),

            n_routed_experts: meta_u32(raw, "glm5next.expert_count")? as usize,
            num_experts_per_tok: meta_u32(raw, "glm5next.expert_used_count")? as usize,
            n_shared_experts: meta_u32_or(raw, "glm5next.expert_shared_count", 0) as usize,
            moe_intermediate_size: meta_u32(raw, "glm5next.expert_feed_forward_length")? as usize,
            n_group: meta_u32_or(raw, "glm5next.expert_group_count", 1).max(1) as usize,
            topk_group: meta_u32_or(raw, "glm5next.expert_group_used_count", 1).max(1) as usize,
            routed_scaling_factor: meta_f32_or(raw, "glm5next.expert_weights_scale", 1.0),
            norm_topk_prob: meta_bool_or(raw, "glm5next.expert_weights_norm", true),
            swiglu_limit,

            num_nextn_predict_layers,
            with_mtp: false,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse a `glm5next` GGUF at `path` (metadata only — no tensor data is
    /// mapped).
    pub fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = GgufFile::from_path(path)
            .with_context(|| format!("glm5next: reading GGUF metadata from {path:?}"))?;
        Self::from_gguf(&raw)
    }

    /// Parse a split GGUF from all of its shard paths.
    ///
    /// Unsloth's `GLM-5.3-Flash-*-000NN-of-000MM.gguf` sets put every metadata
    /// key in shard 1 (which holds no tensors at all), so
    /// [`Self::from_gguf_path`] on that shard alone is enough for the config —
    /// this exists for callers that already hold the whole set.
    pub fn from_gguf_paths(paths: &[impl AsRef<Path>]) -> Result<Self> {
        let raw =
            GgufFile::from_split_paths(paths).context("glm5next: reading split GGUF metadata")?;
        Self::from_gguf(&raw)
    }
}

// ─────────────────────────── HF config.json ───────────────────────────

#[derive(Debug, Deserialize)]
struct HfLinearAttnConfig {
    #[serde(default)]
    num_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    short_conv_kernel_size: Option<usize>,
    #[serde(default)]
    gate_lower_bound: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct HfTextConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default = "d_eps")]
    rms_norm_eps: f32,
    #[serde(default = "d_ctx")]
    max_position_embeddings: usize,
    #[serde(default)]
    tie_word_embeddings: bool,

    layer_types: Vec<String>,
    #[serde(default)]
    indexer_types: Option<Vec<String>>,
    #[serde(default)]
    first_k_dense_replace: usize,

    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    #[serde(default)]
    qk_rope_head_dim: usize,
    v_head_dim: usize,

    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    #[serde(default = "d_one")]
    index_kpool: usize,
    #[serde(default = "d_true")]
    index_kpool_always_select_tail: bool,

    #[serde(default)]
    linear_attn_config: Option<HfLinearAttnConfig>,

    #[serde(default = "d_one")]
    hc_mult: usize,
    #[serde(default = "d_sinkhorn")]
    hc_sinkhorn_iters: usize,
    #[serde(default = "d_hc_eps")]
    hc_eps: f32,

    n_routed_experts: usize,
    num_experts_per_tok: usize,
    #[serde(default)]
    n_shared_experts: usize,
    moe_intermediate_size: usize,
    #[serde(default = "d_one")]
    n_group: usize,
    #[serde(default = "d_one")]
    topk_group: usize,
    #[serde(default = "d_scale")]
    routed_scaling_factor: f32,
    #[serde(default = "d_true")]
    norm_topk_prob: bool,
    #[serde(default = "d_inf")]
    swiglu_limit: f32,

    #[serde(default)]
    num_nextn_predict_layers: usize,
}

fn d_eps() -> f32 {
    1e-5
}
fn d_hc_eps() -> f32 {
    1e-6
}
fn d_ctx() -> usize {
    1 << 20
}
fn d_one() -> usize {
    1
}
fn d_sinkhorn() -> usize {
    20
}
fn d_true() -> bool {
    true
}
fn d_scale() -> f32 {
    1.0
}
fn d_inf() -> f32 {
    f32::INFINITY
}

#[derive(Debug, Deserialize)]
struct HfConfig {
    #[serde(default)]
    model_type: String,
    text_config: HfTextConfig,
}

impl Glm5NextConfig {
    /// Parse the upstream `zai-org/GLM-5.3-Flash` `config.json`.
    ///
    /// `num_hidden_layers` in HF **excludes** the MTP block (45), while the
    /// GGUF `block_count` includes it (46); both readers land on 45 here.
    pub fn from_hf_json(json: &str) -> Result<Self> {
        let hf: HfConfig =
            serde_json::from_str(json).context("glm5next: parsing HF config.json")?;
        if !hf.model_type.is_empty() && hf.model_type != HF_MODEL_TYPE {
            bail!(
                "glm5next: expected model_type = {HF_MODEL_TYPE:?}, got {:?}",
                hf.model_type
            );
        }
        let t = hf.text_config;

        let layer_types = t
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "linear_attention" => Ok(AttnKind::Kda),
                "deepseek_sparse_attention" | "full_attention" => Ok(AttnKind::MlaDsa),
                other => bail!("glm5next: unknown layer_types entry {other:?}"),
            })
            .collect::<Result<Vec<_>>>()?;

        let indexer_types = match t.indexer_types {
            Some(v) => v
                .iter()
                .map(|s| match s.as_str() {
                    "full" => Ok(IndexerKind::Full),
                    "shared" => Ok(IndexerKind::Shared),
                    other => bail!("glm5next: unknown indexer_types entry {other:?}"),
                })
                .collect::<Result<Vec<_>>>()?,
            None => vec![IndexerKind::Full; t.num_hidden_layers],
        };

        let la = t.linear_attn_config;
        let linear_num_heads = la
            .as_ref()
            .and_then(|c| c.num_heads)
            .unwrap_or(t.num_attention_heads);
        let linear_head_dim = la.as_ref().and_then(|c| c.head_dim).unwrap_or(128);
        let linear_conv_kernel_dim = la
            .as_ref()
            .and_then(|c| c.short_conv_kernel_size)
            .unwrap_or(4);
        let linear_lower_bound = la.as_ref().and_then(|c| c.gate_lower_bound);

        let cfg = Self {
            vocab_size: t.vocab_size,
            hidden_size: t.hidden_size,
            intermediate_size: t.intermediate_size,
            num_hidden_layers: t.num_hidden_layers,
            num_attention_heads: t.num_attention_heads,
            rms_norm_eps: t.rms_norm_eps,
            max_position_embeddings: t.max_position_embeddings,
            tie_word_embeddings: t.tie_word_embeddings,

            layer_types,
            indexer_types,
            first_k_dense_replace: t.first_k_dense_replace,

            q_lora_rank: t.q_lora_rank,
            kv_lora_rank: t.kv_lora_rank,
            qk_nope_head_dim: t.qk_nope_head_dim,
            qk_rope_head_dim: t.qk_rope_head_dim,
            v_head_dim: t.v_head_dim,

            index_n_heads: t.index_n_heads,
            index_head_dim: t.index_head_dim,
            index_topk: t.index_topk,
            index_kpool: t.index_kpool,
            index_kpool_always_select_tail: t.index_kpool_always_select_tail,

            linear_num_heads,
            linear_head_dim,
            linear_conv_kernel_dim,
            linear_lower_bound,

            hc_mult: t.hc_mult,
            hc_sinkhorn_iters: t.hc_sinkhorn_iters,
            hc_eps: t.hc_eps,

            n_routed_experts: t.n_routed_experts,
            num_experts_per_tok: t.num_experts_per_tok,
            n_shared_experts: t.n_shared_experts,
            moe_intermediate_size: t.moe_intermediate_size,
            n_group: t.n_group.max(1),
            topk_group: t.topk_group.max(1),
            routed_scaling_factor: t.routed_scaling_factor,
            norm_topk_prob: t.norm_topk_prob,
            swiglu_limit: t.swiglu_limit,

            num_nextn_predict_layers: t.num_nextn_predict_layers,
            with_mtp: false,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Read `config.json` from disk.
    pub fn from_hf_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let s =
            std::fs::read_to_string(path).with_context(|| format!("glm5next: reading {path:?}"))?;
        Self::from_hf_json(&s)
    }
}
