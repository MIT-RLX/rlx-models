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

//! The GLM-5.3-Flash text decoder flow.
//!
//! ```text
//!   token_embd
//!     → broadcast into hc_mult residual streams          [1, s, H, D]
//!     → 45 × ( mHC(attn) → attn_norm → KDA | MLA+DSA → mHC-expand
//!              mHC(ffn)  → ffn_norm  → MLP | MoE        → mHC-expand )
//!     → mean over streams → output_norm → output
//! ```
//!
//! There is no RoPE input: the DSA layers are NoPE and the KDA layers carry
//! position implicitly, so `input_ids` is the graph's only input.
//!
//! The MTP block (`blk.45`, `nextn.*`) is present in the checkpoint but is not
//! part of the main forward pass; [`Glm5NextConfig::with_mtp`] is reserved for
//! it and this builder currently rejects it rather than emitting a head it has
//! not validated.

use anyhow::{Result, anyhow, bail};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::blocks::LmHeadStage;
use rlx_flow::{BuiltModel, CompileProfile, Emit, FlowStage, ModelFlow, WeightSource};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use std::ops::Range;

use crate::common::rms_norm;
use crate::config::{AttnKind, Glm5NextConfig};
use crate::indexer::IndexerDims;
use crate::kda::{KdaDims, emit_kda_attention};
use crate::mhc::{MhcDims, emit_mhc_expand, emit_mhc_gates, emit_mhc_head, emit_mhc_split};
use crate::mla::{MlaDims, emit_mla_attention};
use crate::moe::{MoeDims, emit_dense_mlp, emit_glm5next_moe};

/// GGUF's token-embedding tensor name.
pub const EMBED_KEY: &str = "token_embd.weight";

/// Which slice of the stack one graph builds, and which ends it owns.
///
/// A whole model is `layers = 0..n, embed_input, produce_logits`; a pipeline
/// rank gets its own range and only the ends that fall to it.
///
/// **What crosses a cut is the mHC stream state `[1, seq, hc_mult, hidden]`,
/// not a hidden state.** The four residual streams run the entire depth of the
/// model and are only averaged by [`emit_mhc_head`] at the very end, so a stage
/// boundary has to carry all of them: collapsing to `[1, seq, hidden]` and
/// re-splitting would broadcast one vector into four identical streams and
/// throw away everything the mHC mixing had accumulated. The practical cost is
/// that a `glm5next` pipeline moves `hc_mult`x (4x) the bytes per token that a
/// plain transformer of the same width would.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSpec {
    /// Global layer indices this graph runs, `[start, end)`.
    pub layers: Range<usize>,
    /// Take `input_ids`, embed, and split into streams. Otherwise adopt the
    /// declared `stream_states` input.
    pub embed_input: bool,
    /// Collapse the streams, final-norm, and run the LM head.
    pub produce_logits: bool,
}

impl BlockSpec {
    /// The whole model in one graph.
    pub fn whole(cfg: &Glm5NextConfig, with_lm_head: bool) -> Self {
        Self {
            layers: 0..cfg.num_hidden_layers,
            embed_input: true,
            produce_logits: with_lm_head,
        }
    }

    /// Name of the tensor this block consumes.
    pub fn input_name(&self) -> &'static str {
        if self.embed_input {
            "input_ids"
        } else {
            STREAM_INPUT
        }
    }
}

/// Graph-input name for a mid-stack block's incoming stream state.
pub const STREAM_INPUT: &str = "stream_states";

/// Does the weight `name` belong to a block with `spec`?
///
/// GGUF spelling: per-layer tensors are `blk.{i}.*`; `token_embd.weight`
/// belongs to whichever block embeds, and also to a logits block when the head
/// is tied to it.
pub fn block_weight_filter(name: &str, cfg: &Glm5NextConfig, spec: &BlockSpec) -> bool {
    if let Some(rest) = name.strip_prefix("blk.") {
        return match rest.split('.').next().unwrap_or("").parse::<usize>() {
            Ok(i) => spec.layers.contains(&i),
            Err(_) => false,
        };
    }
    match name {
        EMBED_KEY => spec.embed_input || (spec.produce_logits && cfg.tie_word_embeddings),
        "output_norm.weight" => spec.produce_logits,
        "output.weight" => spec.produce_logits && !cfg.tie_word_embeddings,
        _ => false,
    }
}

/// Build the `glm5next` text prefill graph for a fixed `seq`.
///
/// Input: `input_ids [1, seq]`. Output: `logits [1, seq, vocab]` when
/// `with_lm_head`, else the final hidden state.
pub fn build_glm5next_text_flow(
    cfg: &Glm5NextConfig,
    weights: &mut WeightMap,
    seq: usize,
    with_lm_head: bool,
) -> Result<BuiltModel> {
    build_glm5next_text_flow_with_source(cfg, &mut WeightMapSource(weights), seq, with_lm_head)
}

/// [`build_glm5next_text_flow`] over any [`WeightSource`] — the entry point for
/// running **packed**.
///
/// Pass `rlx_core::flow_bridge::PackedWeightLoaderSource` wrapping a GGUF
/// loader and every 2-D projection becomes a fused `Op::DequantMatMul` over the
/// quant blob, with no f32 weight ever materialized. At `UD-IQ1_S` that is most
/// of the model's bytes: the projections are `Q5_K` / `Q6_K` / `Q8_0`.
///
/// What stays f32 either way, because it is not a 2-D projection:
/// norms, `ssm_a`, `dt_bias`, `exp_probs_b`, the depthwise `ssm_conv1d_*`
/// kernels, MLA's per-head `attn_k_b` / `attn_v_b`, and the routed expert banks
/// (`GroupedMatMul`, not `MatMul`). The expert banks are the big one — packing
/// them is what a whole-model run still needs.
pub fn build_glm5next_text_flow_with_source(
    cfg: &Glm5NextConfig,
    weights: &mut dyn WeightSource,
    seq: usize,
    with_lm_head: bool,
) -> Result<BuiltModel> {
    build_glm5next_block_with_source(cfg, weights, seq, &BlockSpec::whole(cfg, with_lm_head))
}

/// Build ONE pipeline block: layers `spec.layers`, plus whichever ends of the
/// model `spec` claims.
///
/// A `spec` covering every layer with both ends is exactly
/// [`build_glm5next_text_flow_with_source`], which is how the two stay honest:
/// a single-rank pipeline runs the identical graph as the monolithic model.
pub fn build_glm5next_block_with_source(
    cfg: &Glm5NextConfig,
    weights: &mut dyn WeightSource,
    seq: usize,
    spec: &BlockSpec,
) -> Result<BuiltModel> {
    cfg.validate()?;
    if spec.layers.end > cfg.num_hidden_layers {
        bail!(
            "glm5next: block layers {:?} exceed the model's {} layers",
            spec.layers,
            cfg.num_hidden_layers
        );
    }
    if cfg.with_mtp {
        bail!(
            "glm5next: the MTP block (blk.{}) is not emitted by this flow yet",
            cfg.num_hidden_layers
        );
    }
    if seq == 0 {
        bail!("glm5next: seq must be >= 1");
    }
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;

    let mhc = MhcDims {
        hidden,
        streams: cfg.hc_mult,
        sinkhorn_iters: cfg.hc_sinkhorn_iters,
        eps: cfg.hc_eps,
        norm_eps: eps,
        seq,
    };
    let kda = KdaDims {
        hidden,
        num_heads: cfg.linear_num_heads,
        head_dim: cfg.linear_head_dim,
        conv_kernel: cfg.linear_conv_kernel_dim,
        lower_bound: cfg.linear_lower_bound,
        eps,
        seq,
    };
    let mla = MlaDims {
        hidden,
        num_heads: cfg.num_attention_heads,
        q_lora_rank: cfg.q_lora_rank,
        kv_lora_rank: cfg.kv_lora_rank,
        qk_nope_head_dim: cfg.qk_nope_head_dim,
        v_head_dim: cfg.v_head_dim,
        eps,
        seq,
    };
    let indexer = IndexerDims {
        hidden,
        q_lora_rank: cfg.q_lora_rank,
        n_heads: cfg.index_n_heads,
        head_dim: cfg.index_head_dim,
        topk: cfg.index_topk,
        kpool: cfg.index_kpool,
        always_select_tail: cfg.index_kpool_always_select_tail,
        seq,
        force_emit: false,
    };
    let moe = MoeDims {
        paged: false,
        hidden,
        moe_inter: cfg.moe_intermediate_size,
        n_routed: cfg.n_routed_experts,
        top_k: cfg.num_experts_per_tok,
        n_group: cfg.n_group,
        topk_group: cfg.topk_group,
        routed_scaling: cfg.routed_scaling_factor,
        swiglu_limit: Some(cfg.swiglu_limit),
        seq,
    };
    let swiglu = Some(cfg.swiglu_limit);

    let stream_shape = Shape::new(&[1, seq, cfg.hc_mult, hidden], f);

    let mut flow = ModelFlow::new("glm5next").with_profile(CompileProfile::llama32_prefill());

    flow = if spec.embed_input {
        flow.input("input_ids", Shape::new(&[1, seq], f))
    } else {
        // Mid-stack: the four residual streams arrive whole. See `BlockSpec`.
        flow.input(STREAM_INPUT, stream_shape.clone())
    };
    flow = flow.zero_beta_named("glm5next.zero_beta.hidden", hidden);

    if spec.embed_input {
        flow = flow.embed(EMBED_KEY);
        // Broadcast the embedding into `hc_mult` identical residual streams.
        let s = stream_shape.clone();
        flow = flow.plugin_named("hc_split", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("hc_split needs the embedding"))?
                .hir_id();
            let out = emit_mhc_split(emit, x, mhc);
            Ok(Some(emit.wrap(out, s.clone())))
        });
    } else {
        // Adopt the declared input as the active stream state, unchanged.
        flow = flow.plugin_named("adopt_stream_states", move |emit, _prev| {
            Ok(Some(emit.flow_input(STREAM_INPUT)?))
        });
    }

    for i in spec.layers.clone() {
        let prefix = format!("blk.{i}");
        let kind = cfg.attn_kind(i);
        let is_moe = cfg.is_moe_layer(i);
        let s = stream_shape.clone();
        flow = flow.plugin_named(format!("layer{i}"), move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("layer{i} needs a stream state"))?
                .hir_id();

            // ── attention site ──
            let gates = emit_mhc_gates(emit, &format!("{prefix}.hc_attn"), x, mhc)?;
            let normed = rms_norm(
                emit,
                &format!("{prefix}.attn_norm"),
                gates.collapsed,
                hidden,
                eps,
            )?;
            let branch = match kind {
                AttnKind::Kda => emit_kda_attention(emit, &prefix, normed, kda)?,
                AttnKind::MlaDsa => emit_mla_attention(emit, &prefix, normed, mla, indexer)?,
            };
            let x = emit_mhc_expand(emit, branch, x, gates, mhc);

            // ── FFN site ──
            let gates = emit_mhc_gates(emit, &format!("{prefix}.hc_ffn"), x, mhc)?;
            let normed = rms_norm(
                emit,
                &format!("{prefix}.ffn_norm"),
                gates.collapsed,
                hidden,
                eps,
            )?;
            let branch = if is_moe {
                emit_glm5next_moe(emit, &prefix, normed, moe)?
            } else {
                emit_dense_mlp(emit, &prefix, normed, seq, hidden, swiglu)?
            };
            let out = emit_mhc_expand(emit, branch, x, gates, mhc);

            Ok(Some(emit.wrap(out, s.clone())))
        });
    }

    if !spec.produce_logits {
        // Hand the streams on untouched — the mean is the LAST thing the model
        // does, not something each stage does to its own slice.
        return flow.output("hidden").build_with(weights, None);
    }

    // Collapse the streams (an unweighted mean) before the final norm.
    {
        let hs = Shape::new(&[1, seq, hidden], f);
        flow = flow.plugin_named("hc_head", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("hc_head needs a stream state"))?
                .hir_id();
            let out = emit_mhc_head(emit, x, mhc);
            Ok(Some(emit.wrap(out, hs.clone())))
        });
    }

    // GGUF names these `output_norm.weight` / `output.weight`; the DSL's
    // `final_norm` / `lm_head` helpers default to the HF spelling.
    flow = flow.rms_norm("output_norm.weight", eps);
    let head = if cfg.tie_word_embeddings {
        LmHeadStage::tied(cfg.vocab_size, hidden)
    } else {
        LmHeadStage::separate("output.weight", cfg.vocab_size, hidden)
    };
    flow.raw_stage(FlowStage::LmHead(head))
        .output("logits")
        .build_with(weights, None)
}

/// Reshape helper kept for callers that need the stream state flattened.
pub fn flatten_streams(emit: &mut Emit<'_>, x: HirNodeId, seq: usize, width: usize) -> HirNodeId {
    let mut gb = HirMut::new(emit.hir());
    gb.reshape_(x, vec![seq as i64, width as i64])
}
