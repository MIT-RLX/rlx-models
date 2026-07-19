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

//! Qwen3.5 / Qwen3.6 forward graph builder.
//!
//! End-to-end prefill graph composing the gated-DeltaNet "linear
//! attention" trunk layers, the every-`full_attention_interval`
//! standard attention layers, and (optionally) the MTP head. Mirror
//! of `llama.cpp / src/models/qwen35.cpp` translated into RLX IR.
//!
//! **Status (this slice):**
//! - Trunk linear-attn block: full forward (norm → joint qkv +
//!   gate split → α/β/dt → softplus gate → depthwise conv (k=4)
//!   manually unrolled → SiLU → q/k/v split → L2-norm → GQA
//!   repeat → [`Op::GatedDeltaNet`] → SiLU(z)-gated norm →
//!   `ssm_out` → residual → post-norm → SwiGLU FFN → residual).
//! - Trunk full-attn block: joint Q+gate projection (Qwen3-Next
//!   style) → Q/K norm → standard RoPE → causal attention →
//!   sigmoid-gate multiply → `attn_output` → SwiGLU FFN.
//! - MTP head: enorm(token_embd) ++ hnorm(h_pre) → eh_proj →
//!   one full-attn block → shared head LM.
//!
//! **Status:** trunk + full-attn + MTP head wired; prefill-cache and
//! bucketed decode carry GDN conv/SSM state and KV across steps.
//! Packed K-quants via `Op::DequantMatMul` (CPU + Metal). MRoPE text
//! modality `[p,p,p,0]` implemented in `rope.rs`.
//!
//! **Deviations from llama.cpp** (verify via `qwen35_llama_parity` test):
//! - `ssm_conv1d` uses manual k=4 unroll (not `Op::Conv`)
//! - GQA via narrow+concat (not `ggml_repeat`)
//!
//! Memory: F32 path dequantizes all weights at load; use packed mode
//! for large models (e.g. Qwen3.6-27B Q4_K_M: ~16 GB packed vs ~65 GB F32).

use crate::config::{Qwen35Config, mtp_draft_vocab_size};
use crate::rope;
use crate::weights::{
    MatWeight, Qwen35FullAttnLayer, Qwen35LayerFfn, Qwen35LinearLayer, Qwen35MoeFfn,
    Qwen35MtpLayer, Qwen35TrunkLayer, Qwen35Weights,
};
use anyhow::{Result, anyhow};
use rlx_ir::dynamic::sym;
use rlx_ir::hir::{FusionPolicy, HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::{Activation, MaskKind};
use rlx_ir::quant::QuantScheme;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, Op, Shape};
use std::collections::HashMap;

/// Node id within the HIR builder (lowers to MIR [`rlx_ir::NodeId`]).
type NodeId = HirNodeId;

/// Side channel for packed K-quant weights. `build_qwen35_graph_sized`
/// populates this when a `MatWeight::Packed` source is encountered:
/// `param_name → (loader_key, scheme, [out, in])`. The runner uses
/// `loader_key` to fetch bytes from the still-alive `GgufLoader` via
/// `tensor_bytes_borrowed`, then `compiled.set_param_typed(param_name,
/// bytes, DType::U8)`.
pub type PackedParams = HashMap<String, (String, QuantScheme, Vec<usize>)>;

/// Batch/seq layout — static sizes or symbolic `sym::SEQ` (batch=1 only).
#[derive(Clone, Copy)]
pub struct Qwen35BsLayout {
    batch: usize,
    seq: usize,
    dynamic: bool,
}

impl Qwen35BsLayout {
    pub fn new(batch: usize, seq: usize, dynamic: bool) -> Self {
        if dynamic {
            assert_eq!(batch, 1, "qwen35 dynamic seq requires batch=1");
        }
        Self {
            batch,
            seq,
            dynamic,
        }
    }
}

type BsLayout = Qwen35BsLayout;

impl BsLayout {
    fn bs2(&self, dtype: DType) -> Shape {
        if self.dynamic {
            Shape::from_dims(&[Dim::Static(self.batch), Dim::Dynamic(sym::SEQ)], dtype)
        } else {
            Shape::new(&[self.batch, self.seq], dtype)
        }
    }

    fn bs3(&self, h: usize, dtype: DType) -> Shape {
        if self.dynamic {
            Shape::from_dims(
                &[
                    Dim::Static(self.batch),
                    Dim::Dynamic(sym::SEQ),
                    Dim::Static(h),
                ],
                dtype,
            )
        } else {
            Shape::new(&[self.batch, self.seq, h], dtype)
        }
    }

    fn rows(&self) -> usize {
        self.batch * self.seq
    }

    fn flat2_shape(&self, inner: usize, dtype: DType) -> Shape {
        if self.dynamic {
            Shape::from_dims(&[Dim::Dynamic(sym::SEQ), Dim::Static(inner)], dtype)
        } else {
            Shape::new(&[self.rows(), inner], dtype)
        }
    }

    fn reshape_flat(&self, g: &mut HirMut, x: NodeId, inner: usize) -> NodeId {
        if self.dynamic {
            g.reshape_(x, vec![-1, inner as i64])
        } else {
            g.reshape_(x, vec![self.rows() as i64, inner as i64])
        }
    }

    fn reshape_bsh(&self, g: &mut HirMut, x: NodeId, h: usize) -> NodeId {
        if self.dynamic {
            g.reshape_(x, vec![self.batch as i64, -1, h as i64])
        } else {
            g.reshape_(x, vec![self.batch as i64, self.seq as i64, h as i64])
        }
    }

    fn bsh(&self, h: usize) -> [i64; 3] {
        if self.dynamic {
            [self.batch as i64, -1, h as i64]
        } else {
            [self.batch as i64, self.seq as i64, h as i64]
        }
    }

    fn bsh4(&self, h: usize, d: usize) -> Vec<i64> {
        if self.dynamic {
            vec![self.batch as i64, -1, h as i64, d as i64]
        } else {
            vec![self.batch as i64, self.seq as i64, h as i64, d as i64]
        }
    }

    fn bs4_shape(&self, h: usize, d: usize, dtype: DType) -> Shape {
        if self.dynamic {
            Shape::from_dims(
                &[
                    Dim::Static(self.batch),
                    Dim::Dynamic(sym::SEQ),
                    Dim::Static(h),
                    Dim::Static(d),
                ],
                dtype,
            )
        } else {
            Shape::new(&[self.batch, self.seq, h, d], dtype)
        }
    }
}

/// In/out conv + SSM buffers for decode-mode linear layers.
pub(crate) struct LinearRecurrentIo {
    pub conv_state: NodeId,
    pub ssm_state: NodeId,
}

/// KV cache wiring for full-attention layers.
pub(crate) enum AttnCacheMode<'a> {
    /// Append post-RoPE K and pre-GQA V to graph outputs (prefill seed).
    Export {
        k_out: &'a mut NodeId,
        v_out: &'a mut NodeId,
    },
    /// Single-token decode with cached prefix.
    Decode {
        past_k: NodeId,
        past_v: NodeId,
        past_seq: usize,
        k_out: &'a mut NodeId,
        v_out: &'a mut NodeId,
        /// Optional per-row mask for bucketed / variable-length decode.
        mask: Option<NodeId>,
    },
}

/// Build the Qwen3.5 forward IR.
///
/// When `export_recurrent_state` is true, the graph also expects zero-
/// initialized recurrent inputs per linear layer (`conv_state_l*`,
/// `ssm_state_l*`) and emits them back as outputs together with
/// post-RoPE K / pre-GQA V tensors for each full-attention layer.
/// Output order: `[logits, (optional mtp), per-layer recurrent...]`.
pub fn build_qwen35_graph_sized(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>, PackedParams)> {
    build_qwen35_graph_sized_ext(
        cfg,
        weights,
        batch,
        seq,
        with_lm_head,
        last_logits_only,
        enable_mtp_head,
        false,
        None,
        false,
        false,
        false,
        false,
    )
}

/// Forward graph with optional runtime MRoPE inputs (`rope_cos`/`rope_sin`).
pub fn build_qwen35_graph_sized_ext(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
    export_recurrent_state: bool,
    decode_past_seq: Option<usize>,
    runtime_mrope: bool,
    fast_mtp: bool,
    export_normed_hidden: bool,
    export_trunk_layer_hiddens: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>, PackedParams)> {
    if decode_past_seq.is_none() && !export_recurrent_state && !export_trunk_layer_hiddens {
        let (built, packed) = crate::flow::build_qwen35_prefill_flow_built(
            cfg,
            &weights,
            batch,
            seq,
            with_lm_head,
            last_logits_only,
            enable_mtp_head,
            runtime_mrope,
            fast_mtp,
            export_normed_hidden,
        )?;
        let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
        return Ok((graph, params, packed));
    }

    if export_recurrent_state && decode_past_seq.is_none() && !export_trunk_layer_hiddens {
        let opts = crate::flow::Qwen35PrefillCacheOpts {
            batch,
            seq,
            with_lm_head,
            runtime_mrope,
            dynamic_seq: false,
            prefill_from_hidden: false,
            enable_mtp_head,
            fast_mtp,
            fast_greedy_lm_head: export_normed_hidden,
            profile: None,
        };
        let (built, packed) = crate::flow::build_qwen35_prefill_cache_built(cfg, weights, &opts)?;
        let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
        return Ok((graph, params, packed));
    }

    if export_trunk_layer_hiddens
        && decode_past_seq.is_none()
        && !export_recurrent_state
        && !runtime_mrope
    {
        let opts = crate::flow::Qwen35TrunkExportOpts {
            batch,
            seq,
            with_lm_head,
            last_logits_only,
            enable_mtp_head,
            fast_mtp,
            export_normed_hidden,
            profile: None,
        };
        let (built, packed) = crate::flow::build_qwen35_trunk_export_built(cfg, weights, &opts)?;
        let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
        return Ok((graph, params, packed));
    }

    if let Some(past_seq) = decode_past_seq {
        if !export_recurrent_state
            && !export_trunk_layer_hiddens
            && !runtime_mrope
            && !export_normed_hidden
        {
            let opts = crate::flow::Qwen35DecodeOpts {
                batch,
                past_seq,
                dynamic_past: false,
                use_custom_mask: false,
                enable_mtp_head,
                fast_mtp,
                fast_greedy_lm_head: !with_lm_head,
                profile: None,
            };
            let (built, packed) = crate::flow::build_qwen35_decode_built(cfg, weights, &opts)?;
            let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
            return Ok((graph, params, packed));
        }
    }

    graph_from_qwen35_hir(build_qwen35_hir_sized_ext(
        cfg,
        weights,
        batch,
        seq,
        with_lm_head,
        last_logits_only,
        enable_mtp_head,
        export_recurrent_state,
        decode_past_seq,
        runtime_mrope,
        fast_mtp,
        export_normed_hidden,
        export_trunk_layer_hiddens,
    )?)
}

fn graph_from_qwen35_hir(
    triple: (HirModule, HashMap<String, Vec<f32>>, PackedParams),
) -> Result<(Graph, HashMap<String, Vec<f32>>, PackedParams)> {
    let (hir, params, packed) = triple;
    let (graph, params) = rlx_core::flow_util::graph_from_hir(hir, params)?;
    Ok((graph, params, packed))
}

/// Trunk layer-export graph (per-layer hidden probes, optional LM head).
pub fn build_qwen35_trunk_export_graph(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
    fast_mtp: bool,
    export_normed_hidden: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>, PackedParams)> {
    let opts = crate::flow::Qwen35TrunkExportOpts {
        batch,
        seq,
        with_lm_head,
        last_logits_only,
        enable_mtp_head,
        fast_mtp,
        export_normed_hidden,
        profile: None,
    };
    let (built, packed) = crate::flow::build_qwen35_trunk_export_built(cfg, weights, &opts)?;
    let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
    Ok((graph, params, packed))
}

/// HIR-stage builder — same semantics as [`build_qwen35_graph_sized_ext`].
pub fn build_qwen35_hir_sized_ext(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
    export_recurrent_state: bool,
    decode_past_seq: Option<usize>,
    runtime_mrope: bool,
    fast_mtp: bool,
    export_normed_hidden: bool,
    export_trunk_layer_hiddens: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, PackedParams)> {
    build_qwen35_hir_sized_impl(
        cfg,
        weights,
        batch,
        seq,
        with_lm_head,
        last_logits_only,
        enable_mtp_head,
        export_recurrent_state,
        decode_past_seq,
        runtime_mrope,
        false,
        fast_mtp,
        export_normed_hidden,
        export_trunk_layer_hiddens,
        false,
        false,
        false,
    )
}

/// Prefill-cache HIR fed by runtime hidden states (VLM: vision rows spliced on host).
pub fn build_qwen35_prefill_hidden_cache_hir_dynamic_ext(
    cfg: &Qwen35Config,
    weights: impl Into<std::sync::Arc<Qwen35Weights>>,
    batch: usize,
    max_seq: usize,
    runtime_mrope: bool,
    enable_mtp_head: bool,
    fast_mtp: bool,
    fast_greedy_lm_head: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, PackedParams)> {
    build_qwen35_prefill_cache_hir_assembled(
        cfg,
        weights,
        batch,
        max_seq,
        !fast_greedy_lm_head,
        true,
        enable_mtp_head,
        runtime_mrope,
        true,
        true,
        fast_mtp,
        fast_greedy_lm_head,
    )
}

/// Prefill graph that seeds [`super::cache::Qwen35DecodeCache`].
pub fn build_qwen35_prefill_cache_graph(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    batch: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>, PackedParams)> {
    build_qwen35_prefill_cache_graph_ext(cfg, weights, batch, seq, false, false, false, false)
}

/// Prefill-cache HIR with optional runtime MRoPE inputs (multimodal).
pub fn build_qwen35_prefill_cache_hir_ext(
    cfg: &Qwen35Config,
    weights: impl Into<std::sync::Arc<Qwen35Weights>>,
    batch: usize,
    seq: usize,
    runtime_mrope: bool,
    enable_mtp_head: bool,
    fast_mtp: bool,
    fast_greedy_lm_head: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, PackedParams)> {
    build_qwen35_prefill_cache_hir_assembled(
        cfg,
        weights,
        batch,
        seq,
        !fast_greedy_lm_head,
        true,
        enable_mtp_head,
        runtime_mrope,
        false,
        false,
        fast_mtp,
        fast_greedy_lm_head,
    )
}

/// Prefill-cache HIR with symbolic seq dim (`sym::SEQ`) for dynamic compile cache.
pub fn build_qwen35_prefill_cache_hir_dynamic_ext(
    cfg: &Qwen35Config,
    weights: impl Into<std::sync::Arc<Qwen35Weights>>,
    batch: usize,
    max_seq: usize,
    runtime_mrope: bool,
    enable_mtp_head: bool,
    fast_mtp: bool,
    fast_greedy_lm_head: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, PackedParams)> {
    build_qwen35_prefill_cache_hir_assembled(
        cfg,
        weights,
        batch,
        max_seq,
        !fast_greedy_lm_head,
        true,
        enable_mtp_head,
        runtime_mrope,
        true,
        false,
        fast_mtp,
        fast_greedy_lm_head,
    )
}

/// Prefill-cache graph with optional runtime MRoPE inputs (multimodal).
pub fn build_qwen35_prefill_cache_graph_ext(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    batch: usize,
    seq: usize,
    runtime_mrope: bool,
    enable_mtp_head: bool,
    fast_mtp: bool,
    fast_greedy_lm_head: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>, PackedParams)> {
    let opts = crate::flow::Qwen35PrefillCacheOpts {
        batch,
        seq,
        with_lm_head: !fast_greedy_lm_head,
        runtime_mrope,
        dynamic_seq: false,
        prefill_from_hidden: false,
        enable_mtp_head,
        fast_mtp,
        fast_greedy_lm_head,
        profile: None,
    };
    let (built, packed) = crate::flow::build_qwen35_prefill_cache_built(cfg, weights, &opts)?;
    let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
    Ok((graph, params, packed))
}

/// Single-token decode graph at prefix length `past_seq`.
pub fn build_qwen35_decode_graph(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    batch: usize,
    past_seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>, PackedParams)> {
    build_qwen35_decode_graph_ext(cfg, weights, batch, past_seq, false, false, false, false)
}

/// Decode HIR with symbolic past length (`sym::PAST_SEQ`) for dynamic compile cache.
pub fn build_qwen35_decode_hir_dynamic_ext(
    cfg: &Qwen35Config,
    weights: impl Into<std::sync::Arc<Qwen35Weights>>,
    batch: usize,
    max_past_seq: usize,
    enable_mtp_head: bool,
    fast_mtp: bool,
    fast_greedy_lm_head: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, PackedParams)> {
    build_qwen35_decode_hir_assembled(
        cfg,
        weights,
        batch,
        !fast_greedy_lm_head,
        true,
        enable_mtp_head,
        Some(max_past_seq),
        true,
        false,
        fast_mtp,
    )
}

/// Decode HIR graph with optional custom attention mask (bucketed cache).
pub fn build_qwen35_decode_hir_ext(
    cfg: &Qwen35Config,
    weights: impl Into<std::sync::Arc<Qwen35Weights>>,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
    enable_mtp_head: bool,
    fast_mtp: bool,
    fast_greedy_lm_head: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, PackedParams)> {
    build_qwen35_decode_hir_assembled(
        cfg,
        weights,
        batch,
        !fast_greedy_lm_head,
        true,
        enable_mtp_head,
        Some(past_seq),
        false,
        use_custom_mask,
        fast_mtp,
    )
}

/// Decode graph with optional custom attention mask (bucketed cache).
pub fn build_qwen35_decode_graph_ext(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
    enable_mtp_head: bool,
    fast_mtp: bool,
    fast_greedy_lm_head: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>, PackedParams)> {
    let opts = crate::flow::Qwen35DecodeOpts {
        batch,
        past_seq,
        dynamic_past: false,
        use_custom_mask,
        enable_mtp_head,
        fast_mtp,
        fast_greedy_lm_head,
        profile: None,
    };
    let (built, packed) = crate::flow::build_qwen35_decode_built(cfg, weights, &opts)?;
    let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
    Ok((graph, params, packed))
}

/// Decode-step HIR assembly (delegates to native [`crate::flow::build_qwen35_decode_model_flow`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_qwen35_decode_hir_assembled(
    cfg: &Qwen35Config,
    weights: impl Into<std::sync::Arc<Qwen35Weights>>,
    batch: usize,
    with_lm_head: bool,
    _last_logits_only: bool,
    enable_mtp_head: bool,
    decode_past_seq: Option<usize>,
    dynamic_past_seq: bool,
    use_custom_mask: bool,
    fast_mtp: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, PackedParams)> {
    let past_seq = decode_past_seq
        .ok_or_else(|| anyhow!("qwen35 decode assembly requires decode_past_seq"))?;
    let opts = crate::flow::Qwen35DecodeOpts {
        batch,
        past_seq,
        dynamic_past: dynamic_past_seq,
        use_custom_mask,
        enable_mtp_head,
        fast_mtp,
        fast_greedy_lm_head: !with_lm_head,
        profile: None,
    };
    crate::flow::build_qwen35_decode_model_flow(cfg, weights.into(), &opts)
}

/// Prefill-cache HIR (delegates to native [`crate::flow::build_qwen35_prefill_cache_model_flow`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_qwen35_prefill_cache_hir_assembled(
    cfg: &Qwen35Config,
    weights: impl Into<std::sync::Arc<Qwen35Weights>>,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    _last_logits_only: bool,
    enable_mtp_head: bool,
    runtime_mrope: bool,
    dynamic_seq: bool,
    prefill_from_hidden: bool,
    fast_mtp: bool,
    export_normed_hidden: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, PackedParams)> {
    let opts = crate::flow::Qwen35PrefillCacheOpts {
        batch,
        seq,
        runtime_mrope,
        dynamic_seq,
        prefill_from_hidden,
        enable_mtp_head,
        fast_mtp,
        with_lm_head,
        fast_greedy_lm_head: export_normed_hidden,
        profile: None,
    };
    crate::flow::build_qwen35_prefill_cache_model_flow(cfg, weights.into(), &opts)
}

/// Trunk layer-export HIR (delegates to [`crate::flow::build_qwen35_trunk_export_model_flow`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_qwen35_trunk_export_hir_assembled(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
    fast_mtp: bool,
    export_normed_hidden: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, PackedParams)> {
    anyhow::ensure!(
        last_logits_only,
        "export_trunk_layer_hiddens requires last_logits_only=true"
    );
    let opts = crate::flow::Qwen35TrunkExportOpts {
        batch,
        seq,
        with_lm_head,
        last_logits_only,
        enable_mtp_head,
        fast_mtp,
        export_normed_hidden,
        profile: None,
    };
    crate::flow::build_qwen35_trunk_export_model_flow(cfg, weights, &opts)
}

/// Runtime-MRoPE prefill HIR (delegates to [`Qwen35Flow::runtime_mrope`]).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn build_qwen35_runtime_mrope_prefill_hir_assembled(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
    fast_mtp: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, PackedParams)> {
    crate::flow::build_qwen35_runtime_mrope_prefill_flow(
        cfg,
        &weights,
        batch,
        seq,
        with_lm_head,
        last_logits_only,
        enable_mtp_head,
        fast_mtp,
    )
}

fn build_qwen35_hir_sized_impl(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
    export_recurrent_state: bool,
    decode_past_seq: Option<usize>,
    runtime_mrope: bool,
    use_custom_mask: bool,
    fast_mtp: bool,
    export_normed_hidden: bool,
    export_trunk_layer_hiddens: bool,
    dynamic_seq: bool,
    dynamic_past_seq: bool,
    prefill_from_hidden: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, PackedParams)> {
    if decode_past_seq.is_none()
        && !export_recurrent_state
        && !export_trunk_layer_hiddens
        && !use_custom_mask
        && !dynamic_seq
        && !dynamic_past_seq
        && !prefill_from_hidden
    {
        let (hir, params, packed) = crate::flow::build_qwen35_prefill_flow_ext(
            cfg,
            &weights,
            batch,
            seq,
            with_lm_head,
            last_logits_only,
            enable_mtp_head,
            runtime_mrope,
            fast_mtp,
            export_normed_hidden,
        )?;
        return Ok((hir, params, packed));
    }

    if (decode_past_seq.is_some() || dynamic_past_seq)
        && !export_recurrent_state
        && !runtime_mrope
        && !export_normed_hidden
        && !export_trunk_layer_hiddens
        && !dynamic_seq
        && !prefill_from_hidden
    {
        return build_qwen35_decode_hir_assembled(
            cfg,
            weights,
            batch,
            with_lm_head,
            last_logits_only,
            enable_mtp_head,
            decode_past_seq,
            dynamic_past_seq,
            use_custom_mask,
            fast_mtp,
        );
    }

    if export_recurrent_state
        && decode_past_seq.is_none()
        && !dynamic_past_seq
        && !export_trunk_layer_hiddens
    {
        return build_qwen35_prefill_cache_hir_assembled(
            cfg,
            weights,
            batch,
            seq,
            with_lm_head,
            last_logits_only,
            enable_mtp_head,
            runtime_mrope,
            dynamic_seq,
            prefill_from_hidden,
            fast_mtp,
            export_normed_hidden,
        );
    }

    if export_trunk_layer_hiddens
        && decode_past_seq.is_none()
        && !export_recurrent_state
        && !dynamic_past_seq
        && !dynamic_seq
        && !prefill_from_hidden
        && !runtime_mrope
    {
        return build_qwen35_trunk_export_hir_assembled(
            cfg,
            weights,
            batch,
            seq,
            with_lm_head,
            last_logits_only,
            enable_mtp_head,
            fast_mtp,
            export_normed_hidden,
        );
    }

    build_qwen35_hir_fallback_assembled(
        cfg,
        weights,
        batch,
        seq,
        with_lm_head,
        last_logits_only,
        enable_mtp_head,
        export_recurrent_state,
        decode_past_seq,
        runtime_mrope,
        use_custom_mask,
        fast_mtp,
        export_normed_hidden,
        export_trunk_layer_hiddens,
        dynamic_seq,
        dynamic_past_seq,
        prefill_from_hidden,
    )
}

/// Uncommon flag combinations not covered by native ModelFlow paths:
/// decode mixed with export_recurrent / runtime_mrope / trunk_export / VLM hidden;
/// `dynamic_seq` on non-cache prefill; decode + `export_normed_hidden`; etc.
#[allow(clippy::too_many_arguments)]
fn build_qwen35_hir_fallback_assembled(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
    export_recurrent_state: bool,
    decode_past_seq: Option<usize>,
    runtime_mrope: bool,
    use_custom_mask: bool,
    fast_mtp: bool,
    export_normed_hidden: bool,
    export_trunk_layer_hiddens: bool,
    dynamic_seq: bool,
    dynamic_past_seq: bool,
    prefill_from_hidden: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, PackedParams)> {
    let n_embd = cfg.hidden_size;
    let n_vocab = if weights.token_embd.is_empty() {
        cfg.vocab_size
    } else {
        weights.token_embd.len() / n_embd
    };
    if n_vocab == 0 {
        return Err(anyhow!("qwen35: vocab_size could not be inferred"));
    }

    let bs = BsLayout::new(
        batch,
        seq,
        dynamic_seq && decode_past_seq.is_none() && export_recurrent_state,
    );

    let mut hir = HirModule::new("qwen35").with_fusion_policy(FusionPolicy::Direct);
    let mut g = HirMut::new(&mut hir);
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let mut packed: PackedParams = HashMap::new();

    // ── MRoPE cos/sin cache (shared by all full-attn + MTP layers) ─
    let head_dim = cfg.key_length;
    let n_rot = cfg.rope_dim_count;
    let head_half = head_dim / 2;
    let (cos_id, sin_id) = if decode_past_seq.is_some() {
        let cos_id = g.input("rope_cos", Shape::new(&[1, head_half], DType::F32));
        let sin_id = g.input("rope_sin", Shape::new(&[1, head_half], DType::F32));
        (cos_id, sin_id)
    } else if runtime_mrope {
        let rope_shape = if bs.dynamic {
            Shape::from_dims(
                &[Dim::Dynamic(sym::SEQ), Dim::Static(head_half)],
                DType::F32,
            )
        } else {
            Shape::new(&[seq, head_half], DType::F32)
        };
        let cos_id = g.input("rope_cos", rope_shape.clone());
        let sin_id = g.input("rope_sin", rope_shape);
        (cos_id, sin_id)
    } else {
        let (cos_data, sin_data) =
            rope::build_mrope_tables(cfg, cfg.max_position_embeddings, head_half);
        let cos_id = g.param(
            "qwen35.rope.cos",
            Shape::new(&[cfg.max_position_embeddings, head_half], DType::F32),
        );
        params.insert("qwen35.rope.cos".into(), cos_data);
        let sin_id = g.param(
            "qwen35.rope.sin",
            Shape::new(&[cfg.max_position_embeddings, head_half], DType::F32),
        );
        params.insert("qwen35.rope.sin".into(), sin_data);
        (cos_id, sin_id)
    };

    // ── Input: token ids or pre-spliced hidden states (VLM) ─────
    // Token IDs are I32 — backends auto-convert from f32 host buffers
    // (Metal arena.write_from_f32, MLX mc::astype). Declaring F32 here
    // caused gather/take to reinterpret float bits as ints (garbled tokens).
    let input_ids = if prefill_from_hidden {
        if enable_mtp_head {
            Some(g.input("input_ids", bs.bs2(DType::I32)))
        } else {
            None
        }
    } else {
        Some(g.input("input_ids", bs.bs2(DType::I32)))
    };

    let mut h = if prefill_from_hidden {
        g.input("prefill_hidden", bs.bs3(n_embd, DType::F32))
    } else {
        // Embedding table param.
        let embed_w = register_param(
            &mut g,
            &mut params,
            "token_embd.weight",
            weights.token_embd.to_vec(),
            Shape::new(&[n_vocab, n_embd], DType::F32),
        );

        // Hidden states `[batch, seq, n_embd]`.
        g.gather_(embed_w, input_ids.expect("input_ids"), 0)
    };

    if prefill_from_hidden {
        if weights.token_embd.is_empty() {
            return Err(anyhow!("qwen35: prefill_from_hidden requires token_embd"));
        }
        register_param(
            &mut g,
            &mut params,
            "token_embd.weight",
            weights.token_embd.to_vec(),
            Shape::new(&[n_vocab, n_embd], DType::F32),
        );
    }

    let need_last_token =
        last_logits_only && (with_lm_head || export_normed_hidden || export_trunk_layer_hiddens);
    let last_token_idx = if need_last_token {
        Some(g.input("last_token_idx", Shape::new(&[batch], DType::I32)))
    } else {
        None
    };

    let mut trunk_layer_hiddens: Vec<NodeId> = Vec::new();
    if export_trunk_layer_hiddens {
        anyhow::ensure!(
            last_logits_only,
            "export_trunk_layer_hiddens requires last_logits_only=true"
        );
        let idx = last_token_idx.expect("last_token_idx for trunk layer export");
        trunk_layer_hiddens.push(gather_last_token(&mut g, h, batch, idx));
    }

    let decode_mask_id = if use_custom_mask {
        let past_seq = decode_past_seq
            .ok_or_else(|| anyhow!("qwen35: use_custom_mask requires decode_past_seq"))?;
        Some(g.input("mask", Shape::new(&[batch, past_seq + seq], DType::F32)))
    } else {
        None
    };

    // ── Trunk layers ───────────────────────────────────────────
    let _n_main = cfg.num_hidden_layers - cfg.nextn_predict_layers;
    let interval = cfg.full_attention_interval.max(1);
    let mut recurrent_outputs: Vec<NodeId> = Vec::new();

    for (il, layer) in weights.trunk_layers.iter().enumerate() {
        let _is_full_attn = ((il + 1) % interval) == 0;
        match layer {
            Qwen35TrunkLayer::Linear(lin) => {
                let recurrent = if export_recurrent_state || decode_past_seq.is_some() {
                    let conv_state = g.input(
                        format!("conv_state_l{il}"),
                        Shape::new(
                            &[batch, cfg.ssm_conv_kernel - 1, linear_conv_channels(cfg)],
                            DType::F32,
                        ),
                    );
                    let n_v_heads = cfg.ssm_time_step_rank;
                    let n_state = cfg.ssm_state_size;
                    let ssm_state = g.input(
                        format!("ssm_state_l{il}"),
                        Shape::new(&[batch, n_v_heads, n_state, n_state], DType::F32),
                    );
                    Some(LinearRecurrentIo {
                        conv_state,
                        ssm_state,
                    })
                } else {
                    None
                };
                let mut layer_recur_out = Vec::new();
                h = build_linear_layer(
                    &mut g,
                    &mut params,
                    &mut packed,
                    cfg,
                    il,
                    lin,
                    bs,
                    h,
                    recurrent.as_ref(),
                    &mut layer_recur_out,
                )?;
                recurrent_outputs.extend(layer_recur_out);
            }
            Qwen35TrunkLayer::FullAttn(fa) => {
                let mut k_export = HirNodeId(0);
                let mut v_export = HirNodeId(0);
                let attn_cache = if export_recurrent_state {
                    Some(AttnCacheMode::Export {
                        k_out: &mut k_export,
                        v_out: &mut v_export,
                    })
                } else if decode_past_seq.is_some() {
                    let past_len = decode_past_seq.unwrap();
                    let kv_cols = cfg.num_key_value_heads * head_dim;
                    let past_kv_shape = if dynamic_past_seq {
                        Shape::from_dims(
                            &[
                                Dim::Static(batch),
                                Dim::Dynamic(sym::PAST_SEQ),
                                Dim::Static(kv_cols),
                            ],
                            DType::F32,
                        )
                    } else {
                        Shape::new(&[batch, past_len, kv_cols], DType::F32)
                    };
                    let past_k = g.input(format!("past_k_l{il}"), past_kv_shape.clone());
                    let past_v = g.input(format!("past_v_l{il}"), past_kv_shape);
                    Some(AttnCacheMode::Decode {
                        past_k,
                        past_v,
                        past_seq: past_len,
                        k_out: &mut k_export,
                        v_out: &mut v_export,
                        mask: decode_mask_id,
                    })
                } else {
                    None
                };
                h = build_full_attn_layer(
                    &mut g,
                    &mut params,
                    &mut packed,
                    cfg,
                    il,
                    fa,
                    bs,
                    h,
                    cos_id,
                    sin_id,
                    head_dim,
                    n_rot,
                    attn_cache,
                    None,
                )?;
                if export_recurrent_state || decode_past_seq.is_some() {
                    recurrent_outputs.push(k_export);
                    recurrent_outputs.push(v_export);
                }
            }
        }
        if export_trunk_layer_hiddens {
            let idx = last_token_idx.expect("last_token_idx for trunk layer export");
            trunk_layer_hiddens.push(gather_last_token(&mut g, h, batch, idx));
        }
    }

    // Snapshot pre-norm hidden state (MTP input) before the final
    // RMS norm. Kept at full `[batch, seq, n_embd]` so the MTP head
    // sees every token's hidden state — mirrors llama.cpp #23198,
    // which moved the output-row gather to *after* `h_pre_norm` so
    // the MTP draft path keeps a dense pre-norm even when the LM
    // head only emits last-token logits.
    let h_pre_norm = h;

    // Final RMS norm — applied to the (possibly narrowed) hidden
    // state. Narrowing here saves the [seq-1, n_embd] norm and the
    // [seq-1, vocab] matmul that would otherwise run for outputs we
    // discard. The MTP head is unaffected because it consumes
    // `h_pre_norm` (still full seq) above.
    let h_for_norm = if last_logits_only && (with_lm_head || export_normed_hidden) {
        if decode_past_seq.is_some() {
            // Single-token decode graph: last position is always index 0.
            g.narrow_(h, 1, seq - 1, 1)
        } else {
            // Prefill / predict: select the last *real* prompt token per row
            // (input is padded to the compiled `seq` bucket).
            let idx = last_token_idx.expect("last_token_idx for narrowed norm");
            gather_last_token(&mut g, h, batch, idx)
        }
    } else {
        h
    };
    let out_norm = register_param(
        &mut g,
        &mut params,
        "output_norm.weight",
        weights.output_norm.clone(),
        Shape::new(&[n_embd], DType::F32),
    );
    let out_norm_beta = synth_zero(&mut g, &mut params, "output_norm.beta", n_embd);
    let h_norm = g.rms_norm(h_for_norm, out_norm, out_norm_beta, cfg.rms_norm_eps as f32);
    let h_logits_in = h_norm;

    let mut outputs = Vec::new();

    if export_trunk_layer_hiddens {
        outputs.extend(trunk_layer_hiddens);
    }

    if export_normed_hidden {
        outputs.push(h_norm);
    }

    if with_lm_head {
        // LM head: tied to token_embd if no separate `output` weight.
        // Packed-aware when `weights.output` carries K-quant bytes.
        let logit_rows = if last_logits_only { 1 } else { seq };
        let logit_shape = Shape::new(&[batch, logit_rows, n_vocab], DType::F32);
        let logits = match &weights.output {
            Some(w) => {
                let head = proj_mat(
                    &mut g,
                    &mut params,
                    &mut packed,
                    "output.weight",
                    w,
                    n_embd,
                    n_vocab,
                );
                emit_proj(&mut g, h_logits_in, head, w, logit_shape)
            }
            None => {
                if let Some(w) = &weights.token_embd_lm {
                    let head = proj_mat(
                        &mut g,
                        &mut params,
                        &mut packed,
                        "lm_head.tied_t",
                        w,
                        n_embd,
                        n_vocab,
                    );
                    emit_proj(&mut g, h_logits_in, head, w, logit_shape)
                } else {
                    let embed_t = transpose_2d(&weights.token_embd, n_vocab, n_embd);
                    let tied = register_param(
                        &mut g,
                        &mut params,
                        "lm_head.tied_t",
                        embed_t,
                        Shape::new(&[n_embd, n_vocab], DType::F32),
                    );
                    g.mm(h_logits_in, tied)
                }
            }
        };
        outputs.push(logits);
    }

    // ── MTP head (optional) ────────────────────────────────────
    if enable_mtp_head {
        let mtp_layer = weights
            .mtp_layers
            .first()
            .ok_or_else(|| anyhow!("qwen35: MTP requested but no MTP layers loaded"))?;
        let mtp_il = cfg.num_hidden_layers - cfg.nextn_predict_layers;
        let mtp_logits = build_mtp_head(
            &mut g,
            &mut params,
            &mut packed,
            cfg,
            mtp_il,
            mtp_layer,
            batch,
            seq,
            input_ids.ok_or_else(|| anyhow!("qwen35: MTP head requires input_ids"))?,
            h_pre_norm,
            &weights.token_embd,
            n_vocab,
            cos_id,
            sin_id,
            head_dim,
            n_rot,
            fast_mtp,
            last_token_idx,
        )?;
        outputs.push(mtp_logits);
    }

    if outputs.is_empty() {
        if with_lm_head || export_normed_hidden {
            outputs.push(h_norm);
        } else {
            outputs.push(h);
        }
    }
    outputs.extend(recurrent_outputs);
    g.set_outputs(outputs);
    Ok((hir, params, packed))
}

/// Layer-probe HIR (delegates to [`crate::flow::build_qwen35_layer_probe_model_flow`]).
#[allow(dead_code)]
pub(crate) fn build_qwen35_layer_probe_hir_assembled(
    cfg: &Qwen35Config,
    weights: &Qwen35Weights,
    il: usize,
    batch: usize,
    seq: usize,
    export_post_attn: bool,
) -> Result<(HirModule, HashMap<String, Vec<f32>>, PackedParams)> {
    crate::flow::build_qwen35_layer_probe_model_flow(
        cfg,
        weights.clone(),
        il,
        batch,
        seq,
        export_post_attn,
    )
}

/// Single trunk-layer probe: external `trunk_h` → one layer → full-seq hidden out.
/// When `export_post_attn` is true and the layer is full-attention, also exports
/// the post-attention residual (pre-FFN) as a second output.
pub fn build_qwen35_layer_probe_graph(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    il: usize,
    batch: usize,
    seq: usize,
    export_post_attn: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>, PackedParams)> {
    let mut flow = crate::flow::Qwen35LayerProbeFlow::new(cfg, weights, il, batch, seq);
    if export_post_attn {
        flow = flow.export_post_attn();
    }
    let (built, packed) = flow.build()?;
    let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
    Ok((graph, params, packed))
}

/// Prefix trunk graph: first `n_layers` blocks, full-seq hidden out (no LM head).
pub fn build_qwen35_prefix_graph(
    cfg: &Qwen35Config,
    mut weights: Qwen35Weights,
    n_layers: usize,
    batch: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>, PackedParams)> {
    anyhow::ensure!(
        n_layers <= weights.trunk_layers.len(),
        "prefix graph: n_layers={n_layers} > trunk depth {}",
        weights.trunk_layers.len()
    );
    weights.trunk_layers.truncate(n_layers);
    build_qwen35_graph_sized_ext(
        cfg, weights, batch, seq, false, false, false, false, None, false, false, false, false,
    )
}

/// Last-token row gather for trunk-export / prefill-cache taps.
pub(crate) fn emit_qwen35_gather_last_token(
    g: &mut HirMut,
    h: HirNodeId,
    batch: usize,
    last_token_idx: HirNodeId,
) -> HirNodeId {
    gather_last_token(g, h, batch, last_token_idx)
}

fn gather_last_token(g: &mut HirMut, h: NodeId, batch: usize, last_token_idx: NodeId) -> NodeId {
    let idx_2d = g.reshape_(last_token_idx, vec![batch as i64, 1]);
    g.gather_(h, idx_2d, 1)
}

fn linear_conv_channels(cfg: &Qwen35Config) -> usize {
    let n_state = cfg.ssm_state_size;
    let n_k_heads = cfg.ssm_group_count;
    let n_v_heads = cfg.ssm_time_step_rank;
    let key_dim = n_state * n_k_heads;
    let value_dim = n_state * n_v_heads;
    key_dim * 2 + value_dim
}

// ── Trunk linear-attention (gated DeltaNet) layer ──────────────
pub(crate) fn build_linear_layer(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    cfg: &Qwen35Config,
    il: usize,
    lin: &Qwen35LinearLayer,
    bs: BsLayout,
    h_in: NodeId,
    recurrent: Option<&LinearRecurrentIo>,
    recur_out: &mut Vec<NodeId>,
) -> Result<NodeId> {
    let batch = bs.batch;
    let seq = bs.seq;
    let n_embd = cfg.hidden_size;
    let _n_ff = cfg.intermediate_size;
    let n_state = cfg.ssm_state_size;
    let n_k_heads = cfg.ssm_group_count;
    let n_v_heads = cfg.ssm_time_step_rank;
    let key_dim = n_state * n_k_heads;
    let value_dim = n_state * n_v_heads;
    let conv_channels = key_dim * 2 + value_dim;
    let k_conv = cfg.ssm_conv_kernel;

    // Declare bias / 1-D-shape params up front. The
    // `FuseMatMulBiasAct` pass walks nodes in declaration order and
    // tries to fuse `MatMul → Add(rank1_bias) → Activation`; if the
    // bias is declared *after* the matmul it's not yet in the
    // rewriter's id_map and the pass panics. Declaring rank-1
    // params first is the contract every other builder in this
    // crate follows.
    let dt_bias = param(
        g,
        params,
        &name(il, "ssm_dt.bias"),
        &lin.ssm_dt_bias,
        &[n_v_heads],
    );
    let ssm_a_p = param(g, params, &name(il, "ssm_a"), &lin.ssm_a, &[n_v_heads]);

    // attn_norm (pre-norm)
    let attn_norm_w = param(
        g,
        params,
        &name(il, "attn_norm.weight"),
        &lin.attn_norm,
        &[n_embd],
    );
    let attn_norm_b = synth_zero(g, params, &name(il, "attn_norm.beta"), n_embd);
    let x = g.rms_norm(h_in, attn_norm_w, attn_norm_b, cfg.rms_norm_eps as f32);

    // Reshape to 2D for matmul: [batch*seq, n_embd].
    let x_2d = bs.reshape_flat(g, x, n_embd);

    let _rows = bs.rows();
    // Fused qkv projection (key_dim*2 + value_dim channels).
    let qkv_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "attn_qkv.weight"),
        &lin.attn_qkv,
        n_embd,
        conv_channels,
    );
    let qkv = emit_proj(
        g,
        x_2d,
        qkv_w,
        &lin.attn_qkv,
        bs.flat2_shape(conv_channels, DType::F32),
    );
    // → [batch*seq, conv_channels]

    // Gate projection z.
    let gate_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "attn_gate.weight"),
        &lin.attn_gate,
        n_embd,
        value_dim,
    );
    let z = emit_proj(
        g,
        x_2d,
        gate_w,
        &lin.attn_gate,
        bs.flat2_shape(value_dim, DType::F32),
    );
    // → [batch*seq, value_dim]

    // alpha = ssm_alpha @ x ; shape [batch*seq, n_v_heads]
    let alpha_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "ssm_alpha.weight"),
        &lin.ssm_alpha,
        n_embd,
        n_v_heads,
    );
    let alpha = emit_proj(
        g,
        x_2d,
        alpha_w,
        &lin.ssm_alpha,
        bs.flat2_shape(n_v_heads, DType::F32),
    );

    // beta = sigmoid(ssm_beta @ x)
    let beta_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "ssm_beta.weight"),
        &lin.ssm_beta,
        n_embd,
        n_v_heads,
    );
    let beta_pre = emit_proj(
        g,
        x_2d,
        beta_w,
        &lin.ssm_beta,
        bs.flat2_shape(n_v_heads, DType::F32),
    );
    let beta = activation(g, Activation::Sigmoid, beta_pre);

    // gate_g = softplus(alpha + ssm_dt_bias) * ssm_a
    //   ssm_dt_bias: [n_v_heads], broadcast over [batch*seq, n_v_heads].
    let alpha_biased = g.add(alpha, dt_bias);
    let alpha_softplus = softplus(g, alpha_biased);
    let gate_g = g.mul(alpha_softplus, ssm_a_p);

    // Reshape gate/beta to [batch, seq, n_v_heads] for the
    // GatedDeltaNet kernel signature.
    let gate_g_3d = bs.reshape_bsh(g, gate_g, n_v_heads);
    let beta_3d = bs.reshape_bsh(g, beta, n_v_heads);

    // Depthwise causal 1-D conv via `Op::Conv` (NCHW `[B,C,1,W]`).
    let qkv_3d = bs.reshape_bsh(g, qkv, conv_channels);
    let (conv_out, conv_padded) = if let Some(rec) = recurrent {
        let padded = g.concat_(vec![rec.conv_state, qkv_3d], 1);
        let out = if bs.dynamic {
            depthwise_conv1d_op_dynamic(
                g,
                params,
                &name(il, "ssm_conv1d.weight"),
                &lin.ssm_conv1d,
                padded,
                batch,
                conv_channels,
                k_conv,
            )?
        } else {
            let width = (k_conv - 1) + seq;
            depthwise_conv1d_op(
                g,
                params,
                &name(il, "ssm_conv1d.weight"),
                &lin.ssm_conv1d,
                padded,
                batch,
                width,
                seq,
                conv_channels,
                k_conv,
            )?
        };
        // Template bakes max_seq; `sync_narrow_ops` reclamps after dim bind.
        let new_conv = g.narrow_(padded, 1, seq, k_conv - 1);
        recur_out.push(new_conv);
        // SSM state is exported after GatedDeltaNet below (must run after the
        // in-place carry update — see materialize there).
        (out, padded)
    } else if bs.dynamic {
        (
            depthwise_conv1d_causal_dynamic(
                g,
                params,
                &name(il, "ssm_conv1d.weight"),
                &lin.ssm_conv1d,
                qkv_3d,
                batch,
                conv_channels,
                k_conv,
            )?,
            qkv_3d,
        )
    } else {
        (
            depthwise_conv1d_causal(
                g,
                params,
                &name(il, "ssm_conv1d.weight"),
                &lin.ssm_conv1d,
                qkv_3d,
                batch,
                seq,
                conv_channels,
                k_conv,
            )?,
            qkv_3d,
        )
    };
    let _ = conv_padded;
    let conv_silu = g.silu(conv_out);
    // → [batch, seq, conv_channels]

    // Split convolved channels into q_conv, k_conv, v_conv.
    let q_part = g.narrow_(conv_silu, 2, 0, key_dim);
    let k_part = g.narrow_(conv_silu, 2, key_dim, key_dim);
    let v_part = g.narrow_(conv_silu, 2, key_dim * 2, value_dim);

    // Reshape into per-head: [batch, seq, n_*_heads, n_state].
    let q_heads = g.reshape_(q_part, bs.bsh4(n_k_heads, n_state));
    let k_heads = g.reshape_(k_part, bs.bsh4(n_k_heads, n_state));
    let v_heads = g.reshape_(v_part, bs.bsh4(n_v_heads, n_state));

    // L2 normalize Q and K along the last (state) dim.
    let q_l2 = l2_norm(g, q_heads, cfg.rms_norm_eps as f32);
    let k_l2 = l2_norm(g, k_heads, cfg.rms_norm_eps as f32);

    // GQA-repeat k from n_k_heads to n_v_heads if needed.
    let (q_rep, k_rep) = if n_k_heads == n_v_heads {
        (q_l2, k_l2)
    } else {
        let factor = n_v_heads / n_k_heads;
        if factor * n_k_heads != n_v_heads {
            return Err(anyhow!(
                "qwen35 layer {il}: n_v_heads={n_v_heads} must be a multiple \
                 of n_k_heads={n_k_heads} (gqa)"
            ));
        }
        (
            repeat_heads(g, q_l2, bs, n_k_heads, n_state, factor),
            repeat_heads(g, k_l2, bs, n_k_heads, n_state, factor),
        )
    };

    // GatedDeltaNet scan: returns [batch, seq, n_v_heads, n_state].
    let scan_out_shape = bs.bs4_shape(n_v_heads, n_state, DType::F32);
    let scan_out = if let Some(rec) = recurrent {
        let y = g.gated_delta_net_carry(
            q_rep,
            k_rep,
            v_heads,
            gate_g_3d,
            beta_3d,
            rec.ssm_state,
            n_state,
            scan_out_shape,
        );
        // Materialize post-update SSM state into a new buffer ordered after the
        // scan. Exporting the raw input node is unreliable on discrete GPUs:
        // D2H of an Input-as-output can observe the pre-upload zeros even when
        // GDN mutated the arena in place (CPU/Metal unified memory hides this).
        // `ssm * (1 + 0*sum(y))` is a broadcast mul that depends on `y`.
        let scan_sum = g.sum(y, vec![0, 1, 2, 3], false);
        let one = scalar_const(g, 1.0);
        let zero = scalar_const(g, 0.0);
        let zero_dep = g.mul(scan_sum, zero);
        let scale = g.add(one, zero_dep);
        let ssm_out = g.mul(rec.ssm_state, scale);
        recur_out.push(ssm_out);
        y
    } else {
        g.gated_delta_net(
            q_rep,
            k_rep,
            v_heads,
            gate_g_3d,
            beta_3d,
            n_state,
            scan_out_shape,
        )
    };

    // Gated norm: ssm_norm over the last (state) dim, multiplied
    // by silu(z) per element. z was [batch*seq, value_dim] = per
    // (head, state). Reshape both to [batch, seq, n_v_heads,
    // n_state] and apply.
    let z_4d = g.reshape_(z, bs.bsh4(n_v_heads, n_state));
    let z_silu = g.silu(z_4d);

    let ssm_norm_w = param(
        g,
        params,
        &name(il, "ssm_norm.weight"),
        &lin.ssm_norm,
        &[n_state],
    );
    let ssm_norm_b = synth_zero(g, params, &name(il, "ssm_norm.beta"), n_state);
    let scan_normed = g.rms_norm(scan_out, ssm_norm_w, ssm_norm_b, cfg.rms_norm_eps as f32);
    let scan_gated = g.mul(scan_normed, z_silu);

    // Reshape back to [batch*seq, value_dim] for ssm_out.
    let scan_flat = bs.reshape_flat(g, scan_gated, value_dim);

    // ssm_out: [value_dim → n_embd]. Packed-aware.
    let ssm_out_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "ssm_out.weight"),
        &lin.ssm_out,
        value_dim,
        n_embd,
    );
    let attn_out_2d = emit_proj(
        g,
        scan_flat,
        ssm_out_w,
        &lin.ssm_out,
        bs.flat2_shape(n_embd, DType::F32),
    );
    let attn_out = bs.reshape_bsh(g, attn_out_2d, n_embd);

    // Residual.
    let h_post_attn = g.add(h_in, attn_out);

    // Post-attention norm + FFN + residual.
    let h_ffn = build_layer_ffn(
        g,
        params,
        cfg,
        il,
        h_post_attn,
        bs,
        &lin.attn_post_norm,
        &lin.ffn,
        packed,
    )?;

    Ok(h_ffn)
}

// ── Trunk full-attention (every full_attention_interval) layer ─
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_full_attn_layer(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    cfg: &Qwen35Config,
    il: usize,
    fa: &Qwen35FullAttnLayer,
    bs: BsLayout,
    h_in: NodeId,
    cos_id: NodeId,
    sin_id: NodeId,
    head_dim: usize,
    n_rot: usize,
    attn_cache: Option<AttnCacheMode<'_>>,
    export_post_attn: Option<&mut NodeId>,
) -> Result<NodeId> {
    let batch = bs.batch;
    let seq = bs.seq;
    let n_embd = cfg.hidden_size;
    let _n_ff = cfg.intermediate_size;
    let n_head = cfg.num_attention_heads;
    let n_kv_head = cfg.num_key_value_heads;
    let q_gate_cols = n_head * head_dim * 2;
    let kv_cols = n_kv_head * head_dim;
    let kv_dim = n_head * head_dim;

    // pre-norm
    let attn_norm_w = param(
        g,
        params,
        &name(il, "attn_norm.weight"),
        &fa.attn_norm,
        &[n_embd],
    );
    let attn_norm_b = synth_zero(g, params, &name(il, "attn_norm.beta"), n_embd);
    let x = g.rms_norm(h_in, attn_norm_w, attn_norm_b, cfg.rms_norm_eps as f32);
    let x_2d = bs.reshape_flat(g, x, n_embd);

    let _rows = bs.rows();
    // Joint Q + gate projection (Qwen3-Next).
    let q_gate_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "attn_q.weight"),
        &fa.attn_q_gate,
        n_embd,
        q_gate_cols,
    );
    let q_gate = emit_proj(
        g,
        x_2d,
        q_gate_w,
        &fa.attn_q_gate,
        bs.flat2_shape(q_gate_cols, DType::F32),
    );
    // Layout per qwen35.cpp ggml_view_3d: the n_head*2 axis is
    // (gate, q) interleaved per head, but ggml's strides imply
    // Q is at offset 0 and gate at offset n_embd_head_k. Equivalent
    // to splitting [batch*seq, n_head, head_dim*2] into [...,
    // head_dim] (q) and [..., head_dim] (gate).
    let q_gate_4d = g.reshape_(q_gate, bs.bsh4(n_head, head_dim * 2));
    let q_heads = g.narrow_(q_gate_4d, 3, 0, head_dim);
    let gate_heads = g.narrow_(q_gate_4d, 3, head_dim, head_dim);
    let q_packed = bs.reshape_bsh(g, q_heads, kv_dim);
    let gate_packed = bs.reshape_bsh(g, gate_heads, kv_dim);

    // K, V projections → [B, S, n_kv * head_dim].
    let k_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "attn_k.weight"),
        &fa.attn_k,
        n_embd,
        kv_cols,
    );
    let k_proj = emit_proj(
        g,
        x_2d,
        k_w,
        &fa.attn_k,
        bs.flat2_shape(kv_cols, DType::F32),
    );
    let v_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "attn_v.weight"),
        &fa.attn_v,
        n_embd,
        kv_cols,
    );
    let v_proj = emit_proj(
        g,
        x_2d,
        v_w,
        &fa.attn_v,
        bs.flat2_shape(kv_cols, DType::F32),
    );
    let k_packed = bs.reshape_bsh(g, k_proj, n_kv_head * head_dim);
    let v_packed = bs.reshape_bsh(g, v_proj, n_kv_head * head_dim);

    // Per-head Q/K norm → [B, S, heads * head_dim] (qwen3 layout).
    let q_normed = per_head_rms(
        g,
        params,
        &name(il, "attn_q_norm"),
        &fa.attn_q_norm,
        q_packed,
        bs,
        n_head,
        head_dim,
        cfg.rms_norm_eps as f32,
    );
    let k_normed = per_head_rms(
        g,
        params,
        &name(il, "attn_k_norm"),
        &fa.attn_k_norm,
        k_packed,
        bs,
        n_kv_head,
        head_dim,
        cfg.rms_norm_eps as f32,
    );

    // MRoPE (text modality) on Q/K.
    let q_rot = g.rope_n(q_normed, cos_id, sin_id, head_dim, n_rot);
    let k_rot = g.rope_n(k_normed, cos_id, sin_id, head_dim, n_rot);

    let (k_cat, v_cat, attn_seq, attn_mask) = match attn_cache {
        Some(AttnCacheMode::Export { k_out, v_out }) => {
            *k_out = k_rot;
            *v_out = v_packed;
            (k_rot, v_packed, seq, None)
        }
        Some(AttnCacheMode::Decode {
            past_k,
            past_v,
            past_seq,
            k_out,
            v_out,
            mask,
        }) => {
            let new_k = g.concat_(vec![past_k, k_rot], 1);
            let new_v = g.concat_(vec![past_v, v_packed], 1);
            *k_out = new_k;
            *v_out = new_v;
            (new_k, new_v, past_seq + seq, mask)
        }
        None => (k_rot, v_packed, seq, None),
    };

    // sigmoid gate → [B, S, n_head * head_dim]
    let gate_sig = activation(g, Activation::Sigmoid, gate_packed);

    // GQA repeat: widen K/V from n_kv_head to n_head along head dim.
    let group = n_head / n_kv_head;
    let k_full = if group == 1 {
        k_cat
    } else {
        repeat_heads_packed(g, k_cat, batch, attn_seq, n_kv_head, head_dim, group)
    };
    let v_full = if group == 1 {
        v_cat
    } else {
        repeat_heads_packed(g, v_cat, batch, attn_seq, n_kv_head, head_dim, group)
    };

    let attn_shape = bs.bs3(kv_dim, DType::F32);
    let attn_out = if let Some(mask) = attn_mask {
        g.attention(q_rot, k_full, v_full, mask, n_head, head_dim, attn_shape)
    } else {
        g.add_node(
            Op::Attention {
                num_heads: n_head,
                head_dim,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q_rot, k_full, v_full],
            attn_shape,
        )
    };

    let attn_gated = g.mul(attn_out, gate_sig);

    // Output projection.
    let attn_gated_2d = bs.reshape_flat(g, attn_gated, kv_dim);
    let out_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "attn_output.weight"),
        &fa.attn_output,
        kv_dim,
        n_embd,
    );
    let attn_out_proj = emit_proj(
        g,
        attn_gated_2d,
        out_w,
        &fa.attn_output,
        bs.flat2_shape(n_embd, DType::F32),
    );
    let attn_out_3d = bs.reshape_bsh(g, attn_out_proj, n_embd);

    // Residual.
    let h_post_attn = g.add(h_in, attn_out_3d);
    if let Some(out) = export_post_attn {
        *out = h_post_attn;
    }

    // FFN.
    let h_ffn = build_layer_ffn(
        g,
        params,
        cfg,
        il,
        h_post_attn,
        bs,
        &fa.attn_post_norm,
        &fa.ffn,
        packed,
    )?;

    Ok(h_ffn)
}

// ── MTP head (NextN) ───────────────────────────────────────────
#[allow(clippy::too_many_arguments)]
fn build_mtp_head(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    cfg: &Qwen35Config,
    il: usize,
    mtp: &Qwen35MtpLayer,
    batch: usize,
    seq: usize,
    input_ids: NodeId,
    h_pre_norm: NodeId,
    trunk_token_embd: &[f32],
    n_vocab: usize,
    cos_id: NodeId,
    sin_id: NodeId,
    head_dim: usize,
    n_rot: usize,
    fast_mtp: bool,
    last_token_idx: Option<NodeId>,
) -> Result<NodeId> {
    let n_embd = cfg.hidden_size;
    let n_ff = cfg.intermediate_size;
    let n_head = cfg.num_attention_heads;
    let n_kv_head = cfg.num_key_value_heads;
    let mtp_vocab = mtp_draft_vocab_size(n_vocab, fast_mtp);
    let fa = &mtp.base;
    let eps = cfg.rms_norm_eps as f32;

    let hnorm_w = param(
        g,
        params,
        &name(il, "nextn.hnorm.weight"),
        &mtp.hnorm,
        &[n_embd],
    );
    let hnorm_b = synth_zero(g, params, &name(il, "nextn.hnorm.beta"), n_embd);

    let embed_bytes: Vec<f32> = match &mtp.embed_tokens {
        Some(MatWeight::F32(v)) => {
            if mtp_vocab < n_vocab {
                v[..mtp_vocab * n_embd].to_vec()
            } else {
                v.clone()
            }
        }
        Some(MatWeight::Packed { .. }) => {
            trunk_token_embd[..mtp_vocab * n_embd.min(trunk_token_embd.len())].to_vec()
        }
        None => {
            if mtp_vocab < n_vocab {
                trunk_token_embd[..mtp_vocab * n_embd].to_vec()
            } else {
                trunk_token_embd.to_vec()
            }
        }
    };
    let embed_w = register_param(
        g,
        params,
        &name(il, "nextn.embed_tokens.weight"),
        embed_bytes,
        Shape::new(&[mtp_vocab, n_embd], DType::F32),
    );

    let enorm_w = param(
        g,
        params,
        &name(il, "nextn.enorm.weight"),
        &mtp.enorm,
        &[n_embd],
    );
    let enorm_b = synth_zero(g, params, &name(il, "nextn.enorm.beta"), n_embd);

    let eh_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "nextn.eh_proj.weight"),
        &mtp.eh_proj,
        2 * n_embd,
        n_embd,
    );
    let fa_attn_norm_w = param(
        g,
        params,
        &name(il, "attn_norm.weight"),
        &fa.attn_norm,
        &[n_embd],
    );
    let fa_attn_norm_b = synth_zero(g, params, &name(il, "attn_norm.beta"), n_embd);
    let q_gate_cols = n_head * head_dim * 2;
    let kv_cols = n_kv_head * head_dim;
    let fa_q_gate_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "attn_q.weight"),
        &fa.attn_q_gate,
        n_embd,
        q_gate_cols,
    );
    let fa_k_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "attn_k.weight"),
        &fa.attn_k,
        n_embd,
        kv_cols,
    );
    let fa_v_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "attn_v.weight"),
        &fa.attn_v,
        n_embd,
        kv_cols,
    );
    let fa_q_norm_w = param(
        g,
        params,
        &name(il, "attn_q_norm.weight"),
        &fa.attn_q_norm,
        &[head_dim],
    );
    let fa_q_norm_b = synth_zero(g, params, &name(il, "attn_q_norm.beta"), head_dim);
    let fa_k_norm_w = param(
        g,
        params,
        &name(il, "attn_k_norm.weight"),
        &fa.attn_k_norm,
        &[head_dim],
    );
    let fa_k_norm_b = synth_zero(g, params, &name(il, "attn_k_norm.beta"), head_dim);
    let kv_dim = n_head * head_dim;
    let fa_o_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "attn_output.weight"),
        &fa.attn_output,
        kv_dim,
        n_embd,
    );
    let fa_post_norm_w = param(
        g,
        params,
        &name(il, "attn_post_norm.weight"),
        &fa.attn_post_norm,
        &[n_embd],
    );
    let fa_post_norm_b = synth_zero(g, params, &name(il, "attn_post_norm.beta"), n_embd);

    if cfg.is_moe() {
        return build_mtp_head_moe(
            g,
            params,
            packed,
            cfg,
            il,
            mtp,
            batch,
            seq,
            input_ids,
            h_pre_norm,
            trunk_token_embd,
            n_vocab,
            cos_id,
            sin_id,
            head_dim,
            n_rot,
            fast_mtp,
            last_token_idx,
        );
    }

    let (fa_gate_src, fa_up_src, fa_down_src) = match &fa.ffn {
        Qwen35LayerFfn::Dense { gate, up, down } => (gate, up, down),
        Qwen35LayerFfn::Moe(_) => unreachable!("MoE handled above"),
    };
    let fa_gate_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "ffn_gate.weight"),
        fa_gate_src,
        n_embd,
        n_ff,
    );
    let fa_up_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "ffn_up.weight"),
        fa_up_src,
        n_embd,
        n_ff,
    );
    let fa_down_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "ffn_down.weight"),
        fa_down_src,
        n_ff,
        n_embd,
    );
    let head_norm_w = if let Some(w) = &mtp.shared_head_norm {
        param(
            g,
            params,
            &name(il, "nextn.shared_head_norm.weight"),
            w,
            &[n_embd],
        )
    } else {
        match params.get("output_norm.weight").cloned() {
            Some(w) => register_param(
                g,
                params,
                &name(il, "nextn.shared_head_norm.weight_fallback"),
                w,
                Shape::new(&[n_embd], DType::F32),
            ),
            None => synth_zero(
                g,
                params,
                &name(il, "nextn.shared_head_norm.placeholder"),
                n_embd,
            ),
        }
    };
    let head_norm_b = synth_zero(g, params, &name(il, "nextn.shared_head_norm.beta"), n_embd);

    let (lm_head_w, _) = if let Some(w) = &mtp.shared_head_head {
        let out_vocab = if mtp_vocab < n_vocab {
            mtp_vocab
        } else {
            n_vocab
        };
        // Packed K-quant LM head weights now lower through
        // `DequantMatMul` directly (per-weight scheme dispatch in
        // `lower_qwen35_mtp_head`), so we no longer need the F32
        // fallback path that used to tie to the trunk token embedding.
        if matches!(w, MatWeight::Packed { .. }) {
            let scheme = scheme_of(w);
            let id = proj_mat(
                g,
                params,
                packed,
                &name(il, "nextn.shared_head_head.weight"),
                w,
                n_embd,
                out_vocab,
            );
            (id, scheme)
        } else {
            let mat = match w {
                MatWeight::F32(data) if mtp_vocab < n_vocab => {
                    MatWeight::F32(data[..mtp_vocab * n_embd].to_vec())
                }
                _ => w.clone(),
            };
            let id = proj_mat(
                g,
                params,
                packed,
                &name(il, "nextn.shared_head_head.weight"),
                &mat,
                n_embd,
                out_vocab,
            );
            (id, scheme_of(&mat))
        }
    } else {
        let bytes = transpose_2d(
            if mtp_vocab < n_vocab {
                &trunk_token_embd[..mtp_vocab * n_embd]
            } else {
                trunk_token_embd
            },
            mtp_vocab,
            n_embd,
        );
        let id = register_param(
            g,
            params,
            &name(il, "nextn.shared_head_head.tied_t"),
            bytes,
            Shape::new(&[n_embd, mtp_vocab], DType::F32),
        );
        (id, None)
    };

    let last_idx = if let Some(id) = last_token_idx {
        id
    } else {
        let data: Vec<f32> = vec![(seq - 1) as f32; batch];
        register_param(
            g,
            params,
            &name(il, "nextn.last_token_idx"),
            data,
            Shape::new(&[batch], DType::F32),
        )
    };

    let logit_vocab = if mtp_vocab < n_vocab {
        mtp_vocab
    } else {
        n_vocab
    };
    let logits = g.0.qwen35_mtp_head(
        h_pre_norm,
        input_ids,
        cos_id,
        sin_id,
        last_idx,
        embed_w,
        hnorm_w,
        hnorm_b,
        enorm_w,
        enorm_b,
        eh_w,
        fa_attn_norm_w,
        fa_attn_norm_b,
        fa_q_gate_w,
        fa_k_w,
        fa_v_w,
        fa_q_norm_w,
        fa_q_norm_b,
        fa_k_norm_w,
        fa_k_norm_b,
        fa_o_w,
        fa_post_norm_w,
        fa_post_norm_b,
        fa_gate_w,
        fa_up_w,
        fa_down_w,
        head_norm_w,
        head_norm_b,
        lm_head_w,
        n_head,
        n_kv_head,
        head_dim,
        n_rot,
        n_embd,
        n_ff,
        mtp_vocab,
        eps,
        Shape::new(&[batch, 1, logit_vocab], DType::F32),
    );
    Ok(logits)
}

// ── Helpers ────────────────────────────────────────────────────

fn build_ffn(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &Qwen35Config,
    il: usize,
    h_in: NodeId,
    bs: BsLayout,
    attn_post_norm: &[f32],
    ffn_gate: &MatWeight,
    ffn_up: &MatWeight,
    ffn_down: &MatWeight,
    n_ff: usize,
    packed: &mut PackedParams,
) -> Result<NodeId> {
    let n_embd = cfg.hidden_size;
    let post_norm_w = param(
        g,
        params,
        &name(il, "post_attention_norm.weight"),
        attn_post_norm,
        &[n_embd],
    );
    let post_norm_b = synth_zero(g, params, &name(il, "post_attention_norm.beta"), n_embd);
    let h_normed = g.rms_norm(h_in, post_norm_w, post_norm_b, cfg.rms_norm_eps as f32);
    let h_2d = bs.reshape_flat(g, h_normed, n_embd);

    let gate_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "ffn_gate.weight"),
        ffn_gate,
        n_embd,
        n_ff,
    );
    let up_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "ffn_up.weight"),
        ffn_up,
        n_embd,
        n_ff,
    );
    let down_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "ffn_down.weight"),
        ffn_down,
        n_ff,
        n_embd,
    );

    let _rows = bs.rows();
    let gate = emit_proj(g, h_2d, gate_w, ffn_gate, bs.flat2_shape(n_ff, DType::F32));
    let up = emit_proj(g, h_2d, up_w, ffn_up, bs.flat2_shape(n_ff, DType::F32));
    let gate_silu = g.silu(gate);
    let swiglu = g.mul(gate_silu, up);
    let down = emit_proj(
        g,
        swiglu,
        down_w,
        ffn_down,
        bs.flat2_shape(n_embd, DType::F32),
    );
    let ffn_out = bs.reshape_bsh(g, down, n_embd);
    Ok(g.add(h_in, ffn_out))
}

fn build_layer_ffn(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &Qwen35Config,
    il: usize,
    h_in: NodeId,
    bs: BsLayout,
    attn_post_norm: &[f32],
    ffn: &Qwen35LayerFfn,
    packed: &mut PackedParams,
) -> Result<NodeId> {
    match ffn {
        Qwen35LayerFfn::Dense { gate, up, down } => build_ffn(
            g,
            params,
            cfg,
            il,
            h_in,
            bs,
            attn_post_norm,
            gate,
            up,
            down,
            cfg.intermediate_size,
            packed,
        ),
        Qwen35LayerFfn::Moe(moe) => {
            build_moe_ffn(g, params, cfg, il, h_in, bs, attn_post_norm, moe, packed)
        }
    }
}

fn build_moe_ffn(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &Qwen35Config,
    il: usize,
    h_in: NodeId,
    bs: BsLayout,
    attn_post_norm: &[f32],
    moe: &Qwen35MoeFfn,
    packed: &mut PackedParams,
) -> Result<NodeId> {
    let n_embd = cfg.hidden_size;
    let n_ff = cfg.expert_ffn_dim();
    let n_ff_shared = cfg.shared_expert_ffn_dim();
    let n_expert = cfg.num_experts;
    let top_k = cfg.num_experts_used.max(1);
    let scale = cfg.expert_weights_scale;

    let post_norm_w = param(
        g,
        params,
        &name(il, "post_attention_norm.weight"),
        attn_post_norm,
        &[n_embd],
    );
    let post_norm_b = synth_zero(g, params, &name(il, "post_attention_norm.beta"), n_embd);
    let h_normed = g.rms_norm(h_in, post_norm_w, post_norm_b, cfg.rms_norm_eps as f32);
    let h_2d = bs.reshape_flat(g, h_normed, n_embd);
    let rows = bs.rows();

    let router_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "ffn_gate_inp.weight"),
        &moe.router,
        n_embd,
        n_expert,
    );
    let logits = emit_proj(
        g,
        h_2d,
        router_w,
        &moe.router,
        bs.flat2_shape(n_expert, DType::F32),
    );
    let mut probs = g.sm(logits, -1);
    if (scale - 1.0).abs() > f32::EPSILON {
        let scale_n = scalar_const(g, scale);
        probs = g.mul(probs, scale_n);
    }

    let top_idx_2d = g.add_node(
        Op::TopK { k: top_k },
        vec![probs],
        Shape::new(&[rows, top_k], DType::F32),
    );
    let top_probs_2d = g.gather_(probs, top_idx_2d, 1);

    let (gate_w, gate_src) = expert_mat_param(
        g,
        params,
        packed,
        &name(il, "ffn_gate_exps.weight"),
        &moe.gate_exps,
        n_expert,
        n_embd,
        n_ff,
    )?;
    let (up_w, up_src) = expert_mat_param(
        g,
        params,
        packed,
        &name(il, "ffn_up_exps.weight"),
        &moe.up_exps,
        n_expert,
        n_embd,
        n_ff,
    )?;
    let (down_w, down_src) = expert_mat_param(
        g,
        params,
        packed,
        &name(il, "ffn_down_exps.weight"),
        &moe.down_exps,
        n_expert,
        n_ff,
        n_embd,
    )?;

    let mut moe_acc: Option<NodeId> = None;
    for ki in 0..top_k {
        let expert_col = g.narrow_(top_idx_2d, 1, ki, 1);
        let expert_idx = g.reshape_(expert_col, vec![rows as i64]);
        let prob_col = g.narrow_(top_probs_2d, 1, ki, 1);
        let prob_2d = g.reshape_(prob_col, vec![rows as i64, 1]);

        let gate = emit_grouped_proj(
            g,
            h_2d,
            gate_w,
            &gate_src,
            expert_idx,
            bs.flat2_shape(n_ff, DType::F32),
        );
        let up = emit_grouped_proj(
            g,
            h_2d,
            up_w,
            &up_src,
            expert_idx,
            bs.flat2_shape(n_ff, DType::F32),
        );
        let gate_silu = g.silu(gate);
        let swiglu = g.mul(gate_silu, up);
        let down = emit_grouped_proj(
            g,
            swiglu,
            down_w,
            &down_src,
            expert_idx,
            bs.flat2_shape(n_embd, DType::F32),
        );
        let weighted = g.mul(down, prob_2d);
        moe_acc = Some(match moe_acc {
            None => weighted,
            Some(acc) => g.add(acc, weighted),
        });
    }
    let moe_flat = moe_acc.expect("top_k >= 1");

    // Gated shared expert (llama.cpp qwen35moe::build_layer_ffn).
    let shared_router_w = register_param(
        g,
        params,
        &name(il, "ffn_gate_inp_shexp.weight"),
        moe.shared_router.clone(),
        Shape::new(&[n_embd, 1], DType::F32),
    );
    let shared_logits = g.mm(h_2d, shared_router_w);
    let shared_gate = g.activation(
        Activation::Sigmoid,
        shared_logits,
        Shape::new(&[rows, 1], DType::F32),
    );
    let s_gate_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "ffn_gate_shexp.weight"),
        &moe.shared_gate,
        n_embd,
        n_ff_shared,
    );
    let s_up_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "ffn_up_shexp.weight"),
        &moe.shared_up,
        n_embd,
        n_ff_shared,
    );
    let s_down_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "ffn_down_shexp.weight"),
        &moe.shared_down,
        n_ff_shared,
        n_embd,
    );
    let s_gate = emit_proj(
        g,
        h_2d,
        s_gate_w,
        &moe.shared_gate,
        bs.flat2_shape(n_ff_shared, DType::F32),
    );
    let s_up = emit_proj(
        g,
        h_2d,
        s_up_w,
        &moe.shared_up,
        bs.flat2_shape(n_ff_shared, DType::F32),
    );
    let s_gate_silu = g.silu(s_gate);
    let s_swiglu = g.mul(s_gate_silu, s_up);
    let s_down = emit_proj(
        g,
        s_swiglu,
        s_down_w,
        &moe.shared_down,
        bs.flat2_shape(n_embd, DType::F32),
    );
    let shared_out = g.mul(s_down, shared_gate);
    let combined = g.add(moe_flat, shared_out);
    let ffn_out = bs.reshape_bsh(g, combined, n_embd);

    Ok(g.add(h_in, ffn_out))
}

fn emit_grouped_proj(
    g: &mut HirMut,
    input: NodeId,
    weight: NodeId,
    weight_src: &MatWeight,
    expert_idx: NodeId,
    out_shape: Shape,
) -> NodeId {
    match weight_src {
        MatWeight::F32(_) => g.add_node(
            Op::GroupedMatMul,
            vec![input, weight, expert_idx],
            out_shape,
        ),
        MatWeight::Packed { scheme, .. } => g.add_node(
            Op::DequantGroupedMatMul { scheme: *scheme },
            vec![input, weight, expert_idx],
            out_shape,
        ),
    }
}

fn expert_mat_param(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    name: &str,
    weight: &MatWeight,
    num_experts: usize,
    k: usize,
    n: usize,
) -> Result<(NodeId, MatWeight)> {
    match weight {
        MatWeight::F32(data) => {
            if data.len() != num_experts * k * n {
                return Err(anyhow!(
                    "MoE expert weight {name}: len {} != {num_experts}*{k}*{n}",
                    data.len()
                ));
            }
            Ok((
                register_param(
                    g,
                    params,
                    name,
                    data.clone(),
                    Shape::new(&[num_experts, k, n], DType::F32),
                ),
                weight.clone(),
            ))
        }
        MatWeight::Packed { key, scheme, shape } => {
            if *shape != [num_experts, k, n] {
                return Err(anyhow!(
                    "MoE packed expert {name}: shape {shape:?} != [{num_experts}, {k}, {n}]"
                ));
            }
            let block_elems = scheme.gguf_block_size() as usize;
            let block_bytes = scheme.gguf_block_bytes() as usize;
            let bytes_per_expert = (k * n) / block_elems * block_bytes;
            let total_bytes = bytes_per_expert * num_experts;
            let id = g.param(name, Shape::new(&[total_bytes], DType::U8));
            packed.insert(
                name.to_string(),
                (key.clone(), *scheme, vec![num_experts, k, n]),
            );
            Ok((id, weight.clone()))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_mtp_head_moe(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    cfg: &Qwen35Config,
    il: usize,
    mtp: &Qwen35MtpLayer,
    batch: usize,
    seq: usize,
    input_ids: NodeId,
    h_pre_norm: NodeId,
    trunk_token_embd: &[f32],
    n_vocab: usize,
    cos_id: NodeId,
    sin_id: NodeId,
    head_dim: usize,
    n_rot: usize,
    fast_mtp: bool,
    last_token_idx: Option<NodeId>,
) -> Result<NodeId> {
    let n_embd = cfg.hidden_size;
    let _n_head = cfg.num_attention_heads;
    let _n_kv_head = cfg.num_key_value_heads;
    let mtp_vocab = mtp_draft_vocab_size(n_vocab, fast_mtp);
    let fa = &mtp.base;
    let eps = cfg.rms_norm_eps as f32;
    let bs = BsLayout::new(batch, seq, false);

    let hnorm_w = param(
        g,
        params,
        &name(il, "nextn.hnorm.weight"),
        &mtp.hnorm,
        &[n_embd],
    );
    let hnorm_b = synth_zero(g, params, &name(il, "nextn.hnorm.beta"), n_embd);
    let enorm_w = param(
        g,
        params,
        &name(il, "nextn.enorm.weight"),
        &mtp.enorm,
        &[n_embd],
    );
    let enorm_b = synth_zero(g, params, &name(il, "nextn.enorm.beta"), n_embd);
    let eh_w = proj_mat(
        g,
        params,
        packed,
        &name(il, "nextn.eh_proj.weight"),
        &mtp.eh_proj,
        2 * n_embd,
        n_embd,
    );

    let embed_bytes: Vec<f32> = match &mtp.embed_tokens {
        Some(MatWeight::F32(v)) => {
            if mtp_vocab < n_vocab {
                v[..mtp_vocab * n_embd].to_vec()
            } else {
                v.clone()
            }
        }
        Some(MatWeight::Packed { .. }) => {
            trunk_token_embd[..mtp_vocab * n_embd.min(trunk_token_embd.len())].to_vec()
        }
        None => {
            if mtp_vocab < n_vocab {
                trunk_token_embd[..mtp_vocab * n_embd].to_vec()
            } else {
                trunk_token_embd.to_vec()
            }
        }
    };
    let embed_w = register_param(
        g,
        params,
        &name(il, "nextn.embed_tokens.weight"),
        embed_bytes,
        Shape::new(&[mtp_vocab, n_embd], DType::F32),
    );

    let h_normed = g.rms_norm(h_pre_norm, hnorm_w, hnorm_b, eps);
    let tok_embd = g.gather_(embed_w, input_ids, 0);
    let e_normed = g.rms_norm(tok_embd, enorm_w, enorm_b, eps);
    let concat = g.concat_(vec![e_normed, h_normed], 2);
    let concat_2d = bs.reshape_flat(g, concat, 2 * n_embd);
    let cur_2d = g.mm(concat_2d, eh_w);
    let cur = bs.reshape_bsh(g, cur_2d, n_embd);

    let h_post_attn = build_full_attn_layer(
        g, params, packed, cfg, il, fa, bs, cur, cos_id, sin_id, head_dim, n_rot, None, None,
    )?;
    let h_ffn = build_layer_ffn(
        g,
        params,
        cfg,
        il,
        h_post_attn,
        bs,
        &fa.attn_post_norm,
        &fa.ffn,
        packed,
    )?;

    let last_idx = if let Some(id) = last_token_idx {
        id
    } else {
        let data: Vec<f32> = vec![(seq - 1) as f32; batch];
        register_param(
            g,
            params,
            &name(il, "nextn.last_token_idx"),
            data,
            Shape::new(&[batch], DType::F32),
        )
    };
    let idx_2d = g.reshape_(last_idx, vec![batch as i64, 1]);
    let last = g.gather_(h_ffn, idx_2d, 1);

    let head_norm_w = if let Some(w) = &mtp.shared_head_norm {
        param(
            g,
            params,
            &name(il, "nextn.shared_head_norm.weight"),
            w,
            &[n_embd],
        )
    } else {
        match params.get("output_norm.weight").cloned() {
            Some(w) => register_param(
                g,
                params,
                &name(il, "nextn.shared_head_norm.weight_fallback"),
                w,
                Shape::new(&[n_embd], DType::F32),
            ),
            None => synth_zero(
                g,
                params,
                &name(il, "nextn.shared_head_norm.placeholder"),
                n_embd,
            ),
        }
    };
    let head_norm_b = synth_zero(g, params, &name(il, "nextn.shared_head_norm.beta"), n_embd);
    let last_norm = g.rms_norm(last, head_norm_w, head_norm_b, eps);

    let logit_vocab = if mtp_vocab < n_vocab {
        mtp_vocab
    } else {
        n_vocab
    };
    let lm_head_w = if let Some(w) = &mtp.shared_head_head {
        let mat = match w {
            MatWeight::F32(data) if mtp_vocab < n_vocab => {
                MatWeight::F32(data[..mtp_vocab * n_embd].to_vec())
            }
            _ => w.clone(),
        };
        proj_mat(
            g,
            params,
            packed,
            &name(il, "nextn.shared_head_head.weight"),
            &mat,
            n_embd,
            logit_vocab,
        )
    } else {
        let bytes = transpose_2d(
            if mtp_vocab < n_vocab {
                &trunk_token_embd[..mtp_vocab * n_embd]
            } else {
                trunk_token_embd
            },
            mtp_vocab,
            n_embd,
        );
        register_param(
            g,
            params,
            &name(il, "nextn.shared_head_head.tied_t"),
            bytes,
            Shape::new(&[n_embd, logit_vocab], DType::F32),
        )
    };
    let logits = g.mm(last_norm, lm_head_w);
    Ok(logits)
}

fn depthwise_conv1d_op(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    weight: &[f32],
    padded_bsc: NodeId,
    batch: usize,
    width: usize,
    out_seq: usize,
    channels: usize,
    k: usize,
) -> Result<NodeId> {
    debug_assert_eq!(width, out_seq + k - 1);
    let bcw = g.transpose_(padded_bsc, vec![0, 2, 1]);
    let nchw = g.reshape_(bcw, vec![batch as i64, channels as i64, 1, width as i64]);
    let w_data = pack_depthwise_conv_weight(weight, k, channels);
    let w = register_param(
        g,
        params,
        name,
        w_data,
        Shape::new(&[channels, 1, 1, k], DType::F32),
    );
    let conv = g.add_node(
        Op::Conv {
            kernel_size: vec![1, k],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: channels,
        },
        vec![nchw, w],
        Shape::new(&[batch, channels, 1, out_seq], DType::F32),
    );
    let bcs = g.reshape_(conv, vec![batch as i64, channels as i64, out_seq as i64]);
    Ok(g.transpose_(bcs, vec![0, 2, 1]))
}

fn pack_depthwise_conv_weight(weight: &[f32], k: usize, channels: usize) -> Vec<f32> {
    // GGUF `ssm_conv1d` is stored innermost-first as `[k, channels]`
    // (tap fastest). After shape reverse the logical label is
    // `[channels, k]` but byte order is unchanged: index `tap + c*k`.
    let mut out = vec![0f32; channels * k];
    for c in 0..channels {
        for ki in 0..k {
            out[c * k + ki] = weight[ki + c * k];
        }
    }
    out
}

fn depthwise_conv1d_causal(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    weight: &[f32],
    input: NodeId,
    batch: usize,
    seq: usize,
    channels: usize,
    k: usize,
) -> Result<NodeId> {
    let pad_shape = Shape::new(&[batch, k - 1, channels], DType::F32);
    let pad_name = format!("{name}.causal_pad");
    let pad_data = vec![0f32; batch * (k - 1) * channels];
    let pad = register_param(g, params, &pad_name, pad_data, pad_shape);
    let padded = g.concat_(vec![pad, input], 1);
    depthwise_conv1d_op(
        g,
        params,
        name,
        weight,
        padded,
        batch,
        (k - 1) + seq,
        seq,
        channels,
        k,
    )
}

/// Depthwise causal conv with symbolic `sym::SEQ` (dynamic prefill specialization).
fn depthwise_conv1d_op_dynamic(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    weight: &[f32],
    padded_bsc: NodeId,
    batch: usize,
    channels: usize,
    k: usize,
) -> Result<NodeId> {
    let bcw = g.transpose_(padded_bsc, vec![0, 2, 1]);
    let nchw = g.reshape_(bcw, vec![batch as i64, channels as i64, 1, -1]);
    let w_data = pack_depthwise_conv_weight(weight, k, channels);
    let w = register_param(
        g,
        params,
        name,
        w_data,
        Shape::new(&[channels, 1, 1, k], DType::F32),
    );
    let conv = g.add_node(
        Op::Conv {
            kernel_size: vec![1, k],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: channels,
        },
        vec![nchw, w],
        Shape::from_dims(
            &[
                Dim::Static(batch),
                Dim::Static(channels),
                Dim::Static(1),
                Dim::Dynamic(sym::SEQ),
            ],
            DType::F32,
        ),
    );
    let bcs = g.reshape_(conv, vec![batch as i64, channels as i64, -1]);
    Ok(g.transpose_(bcs, vec![0, 2, 1]))
}

fn depthwise_conv1d_causal_dynamic(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    weight: &[f32],
    input: NodeId,
    batch: usize,
    channels: usize,
    k: usize,
) -> Result<NodeId> {
    let pad_shape = Shape::new(&[batch, k - 1, channels], DType::F32);
    let pad_name = format!("{name}.causal_pad");
    let pad_data = vec![0f32; batch * (k - 1) * channels];
    let pad = register_param(g, params, &pad_name, pad_data, pad_shape);
    let padded = g.concat_(vec![pad, input], 1);
    depthwise_conv1d_op_dynamic(g, params, name, weight, padded, batch, channels, k)
}

/// L2 normalize along the last dim (ggml `L2_NORM`):
/// `out = x / max(sqrt(sum(x²)), eps)`.
fn l2_norm(g: &mut HirMut, x: NodeId, eps: f32) -> NodeId {
    let rank = g.shape(x).rank();
    let last = rank - 1;
    let sq = g.mul(x, x);
    let sumsq = g.sum(sq, vec![last], true);
    let rms = g.sqrt(sumsq);
    let eps_p = scalar_const(g, eps);
    // max(rms, eps) = eps + relu(rms - eps) — fewer ops than the abs form.
    let diff = g.sub(rms, eps_p);
    let relu = activation(g, Activation::Relu, diff);
    let denom = g.add(eps_p, relu);
    g.div(x, denom)
}

/// `softplus(x) = log(1 + exp(x))`.
fn softplus(g: &mut HirMut, x: NodeId) -> NodeId {
    let ex = activation(g, Activation::Exp, x);
    let one = scalar_const(g, 1.0);
    let sum = g.add(ex, one);
    activation(g, Activation::Log, sum)
}

/// Repeat each KV head `factor` times on the packed last axis of a
/// `[B, S, n_kv * head_dim]` tensor → `[B, S, n_kv * factor * head_dim]`.
fn repeat_heads_packed(
    g: &mut HirMut,
    x: NodeId,
    _batch: usize,
    _seq: usize,
    in_heads: usize,
    head_dim: usize,
    factor: usize,
) -> NodeId {
    let last_ax = g.shape(x).rank() - 1;
    let mut pieces = Vec::with_capacity(in_heads * factor);
    for h in 0..in_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..factor {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}

/// Per-head RMSNorm on `[B, S, heads * head_dim]`.
#[allow(clippy::too_many_arguments)]
fn per_head_rms(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    weight_name: &str,
    weight: &[f32],
    x: NodeId,
    bs: BsLayout,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> NodeId {
    let r = g.reshape_(x, bs.bsh4(heads, head_dim));
    let gamma = param(
        g,
        params,
        &format!("{weight_name}.weight"),
        weight,
        &[head_dim],
    );
    let beta = synth_zero(g, params, &format!("{weight_name}.beta"), head_dim);
    let n = g.rms_norm(r, gamma, beta, eps);
    g.reshape_(n, bs.bsh(heads * head_dim).to_vec())
}

/// Repeat each head `factor` times along the head axis (axis = 2,
/// for a [b, s, h, d] tensor). Concatenates `factor` narrows for
/// each source head.
fn repeat_heads(
    g: &mut HirMut,
    x: NodeId,
    bs: BsLayout,
    in_heads: usize,
    head_dim: usize,
    factor: usize,
) -> NodeId {
    let _ = (bs, head_dim);
    let mut pieces = Vec::with_capacity(in_heads * factor);
    for h in 0..in_heads {
        let slice = g.narrow_(x, 2, h, 1);
        for _ in 0..factor {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, 2)
}

fn activation(g: &mut HirMut, kind: Activation, x: NodeId) -> NodeId {
    let s = g.shape(x).clone();
    g.activation(kind, x, s)
}

/// Register a `MatWeight` as a graph param. F32 takes the same path
/// as `param()` (with a [in, out] transpose so the matmul convention
/// matches). Packed registers as a U8 byte tensor + records the
/// scheme/shape in `packed`. Returns `(node, scheme_or_none, in, out)`.
fn scheme_of(weight: &MatWeight) -> Option<QuantScheme> {
    match weight {
        MatWeight::F32(_) => None,
        MatWeight::Packed { scheme, .. } => Some(*scheme),
    }
}

fn proj_mat(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    name: &str,
    weight: &MatWeight,
    expected_in: usize,
    expected_out: usize,
) -> NodeId {
    match weight {
        MatWeight::F32(data) => {
            assert_eq!(
                data.len(),
                expected_in * expected_out,
                "proj_mat F32 {name}: len {} != in {expected_in} * out {expected_out}",
                data.len()
            );
            param(
                g,
                params,
                name,
                &transpose_2d(data, expected_out, expected_in),
                &[expected_in, expected_out],
            )
        }
        MatWeight::Packed { key, scheme, shape } => {
            // Total byte count = elements × bytes-per-block /
            // block-size. We can compute it from scheme + shape.
            let n_elements: usize = shape.iter().product();
            let bytes_per_block = scheme.gguf_block_bytes() as usize;
            let block_size = scheme.gguf_block_size() as usize;
            assert!(
                n_elements.is_multiple_of(block_size),
                "proj_mat packed {name}: {n_elements} elems not aligned to \
                 block {block_size} for {scheme:?}"
            );
            let n_blocks = n_elements / block_size;
            let total_bytes = n_blocks * bytes_per_block;
            let id = g.param(name, Shape::new(&[total_bytes], DType::U8));
            packed.insert(
                name.to_string(),
                (key.clone(), *scheme, vec![expected_out, expected_in]),
            );
            id
        }
    }
}

/// Emit either `MatMul` (F32 weights) or `DequantMatMul` (packed),
/// based on whether `proj_mat` saw a packed source for `weight_node`.
/// The caller passes the *expected* out_shape (post-matmul); the
/// packed path needs it because DequantMatMul shape can't be
/// inferred from the U8 bytes alone.
fn emit_proj(
    g: &mut HirMut,
    input: NodeId,
    weight_node: NodeId,
    weight_src: &MatWeight,
    out_shape: Shape,
) -> NodeId {
    match weight_src {
        MatWeight::F32(_) => g.mm(input, weight_node),
        MatWeight::Packed { scheme, .. } => g.add_node(
            Op::DequantMatMul { scheme: *scheme },
            vec![input, weight_node],
            out_shape,
        ),
    }
}

fn scalar_const(g: &mut HirMut, value: f32) -> NodeId {
    // Encode the scalar f32 as a 4-byte Constant payload.
    let bytes = value.to_le_bytes().to_vec();
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[1], DType::F32),
    )
}

fn param(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: &[f32],
    shape: &[usize],
) -> NodeId {
    register_param(
        g,
        params,
        name,
        data.to_vec(),
        Shape::new(shape, DType::F32),
    )
}

pub(crate) fn register_param(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: Vec<f32>,
    shape: Shape,
) -> NodeId {
    let id = g.param(name, shape);
    params.insert(name.to_string(), data);
    id
}

fn synth_zero(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    len: usize,
) -> NodeId {
    let id = g.param(name, Shape::new(&[len], DType::F32));
    params.insert(name.to_string(), vec![0f32; len]);
    id
}

fn name(il: usize, suffix: &str) -> String {
    format!("blk.{il}.{suffix}")
}

/// Tier-0 flow entry — emit one GDN trunk layer (prefill, no recurrent carry).
pub fn emit_qwen35_gdn_prefill_layer(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    cfg: &Qwen35Config,
    layer_idx: usize,
    lin: &Qwen35LinearLayer,
    bs: Qwen35BsLayout,
    h_in: HirNodeId,
) -> Result<HirNodeId> {
    build_linear_layer(
        g,
        params,
        packed,
        cfg,
        layer_idx,
        lin,
        bs,
        h_in,
        None,
        &mut Vec::new(),
    )
}

/// Tier-0 flow entry — one decode trunk layer (GDN recurrent or full-attn KV).
#[allow(clippy::too_many_arguments)]
pub fn emit_qwen35_decode_trunk_layer(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    cfg: &Qwen35Config,
    il: usize,
    layer: &Qwen35TrunkLayer,
    bs: Qwen35BsLayout,
    h_in: HirNodeId,
    cos_id: HirNodeId,
    sin_id: HirNodeId,
    past_len: usize,
    dynamic_past_seq: bool,
    decode_mask: Option<HirNodeId>,
    recur_out: &mut Vec<NodeId>,
) -> Result<HirNodeId> {
    let batch = bs.batch;
    let head_dim = cfg.key_length;
    let n_rot = cfg.rope_dim_count;
    match layer {
        Qwen35TrunkLayer::Linear(lin) => {
            let conv_state = g.input(
                format!("conv_state_l{il}"),
                Shape::new(
                    &[batch, cfg.ssm_conv_kernel - 1, linear_conv_channels(cfg)],
                    DType::F32,
                ),
            );
            let n_v_heads = cfg.ssm_time_step_rank;
            let n_state = cfg.ssm_state_size;
            let ssm_state = g.input(
                format!("ssm_state_l{il}"),
                Shape::new(&[batch, n_v_heads, n_state, n_state], DType::F32),
            );
            let recurrent = LinearRecurrentIo {
                conv_state,
                ssm_state,
            };
            build_linear_layer(
                g,
                params,
                packed,
                cfg,
                il,
                lin,
                bs,
                h_in,
                Some(&recurrent),
                recur_out,
            )
        }
        Qwen35TrunkLayer::FullAttn(fa) => {
            let mut k_export = HirNodeId(0);
            let mut v_export = HirNodeId(0);
            let kv_cols = cfg.num_key_value_heads * head_dim;
            let past_kv_shape = if dynamic_past_seq {
                Shape::from_dims(
                    &[
                        Dim::Static(batch),
                        Dim::Dynamic(sym::PAST_SEQ),
                        Dim::Static(kv_cols),
                    ],
                    DType::F32,
                )
            } else {
                Shape::new(&[batch, past_len, kv_cols], DType::F32)
            };
            let past_k = g.input(format!("past_k_l{il}"), past_kv_shape.clone());
            let past_v = g.input(format!("past_v_l{il}"), past_kv_shape);
            let attn_cache = AttnCacheMode::Decode {
                past_k,
                past_v,
                past_seq: past_len,
                k_out: &mut k_export,
                v_out: &mut v_export,
                mask: decode_mask,
            };
            let h = build_full_attn_layer(
                g,
                params,
                packed,
                cfg,
                il,
                fa,
                bs,
                h_in,
                cos_id,
                sin_id,
                head_dim,
                n_rot,
                Some(attn_cache),
                None,
            )?;
            recur_out.push(k_export);
            recur_out.push(v_export);
            Ok(h)
        }
    }
}

/// Tier-0 flow entry — one prefill-cache trunk layer (GDN export + full-attn K/V export).
#[allow(clippy::too_many_arguments)]
pub fn emit_qwen35_prefill_cache_trunk_layer(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    cfg: &Qwen35Config,
    il: usize,
    layer: &Qwen35TrunkLayer,
    bs: Qwen35BsLayout,
    h_in: HirNodeId,
    cos_id: HirNodeId,
    sin_id: HirNodeId,
    recur_out: &mut Vec<NodeId>,
) -> Result<HirNodeId> {
    let batch = bs.batch;
    let head_dim = cfg.key_length;
    let n_rot = cfg.rope_dim_count;
    match layer {
        Qwen35TrunkLayer::Linear(lin) => {
            let conv_state = g.input(
                format!("conv_state_l{il}"),
                Shape::new(
                    &[batch, cfg.ssm_conv_kernel - 1, linear_conv_channels(cfg)],
                    DType::F32,
                ),
            );
            let n_v_heads = cfg.ssm_time_step_rank;
            let n_state = cfg.ssm_state_size;
            let ssm_state = g.input(
                format!("ssm_state_l{il}"),
                Shape::new(&[batch, n_v_heads, n_state, n_state], DType::F32),
            );
            let recurrent = LinearRecurrentIo {
                conv_state,
                ssm_state,
            };
            build_linear_layer(
                g,
                params,
                packed,
                cfg,
                il,
                lin,
                bs,
                h_in,
                Some(&recurrent),
                recur_out,
            )
        }
        Qwen35TrunkLayer::FullAttn(fa) => {
            let mut k_export = HirNodeId(0);
            let mut v_export = HirNodeId(0);
            let attn_cache = AttnCacheMode::Export {
                k_out: &mut k_export,
                v_out: &mut v_export,
            };
            let h = build_full_attn_layer(
                g,
                params,
                packed,
                cfg,
                il,
                fa,
                bs,
                h_in,
                cos_id,
                sin_id,
                head_dim,
                n_rot,
                Some(attn_cache),
                None,
            )?;
            recur_out.push(k_export);
            recur_out.push(v_export);
            Ok(h)
        }
    }
}

/// Tier-0 flow entry — emit one full-attention trunk layer (prefill, no KV cache).
pub fn emit_qwen35_full_attn_prefill_layer(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    cfg: &Qwen35Config,
    layer_idx: usize,
    fa: &Qwen35FullAttnLayer,
    bs: Qwen35BsLayout,
    h_in: HirNodeId,
    cos_id: HirNodeId,
    sin_id: HirNodeId,
) -> Result<HirNodeId> {
    emit_qwen35_full_attn_prefill_layer_ext(
        g, params, packed, cfg, layer_idx, fa, bs, h_in, cos_id, sin_id, None,
    )
}

/// Full-attention prefill layer; optional post-attention residual export (layer probe).
pub fn emit_qwen35_full_attn_prefill_layer_ext(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    cfg: &Qwen35Config,
    layer_idx: usize,
    fa: &Qwen35FullAttnLayer,
    bs: Qwen35BsLayout,
    h_in: HirNodeId,
    cos_id: HirNodeId,
    sin_id: HirNodeId,
    export_post_attn: Option<&mut HirNodeId>,
) -> Result<HirNodeId> {
    let head_dim = cfg.key_length;
    let n_rot = cfg.rope_dim_count;
    build_full_attn_layer(
        g,
        params,
        packed,
        cfg,
        layer_idx,
        fa,
        bs,
        h_in,
        cos_id,
        sin_id,
        head_dim,
        n_rot,
        None,
        export_post_attn,
    )
}

/// Single trunk-layer probe (external hidden in → one block out).
#[allow(clippy::too_many_arguments)]
pub fn emit_qwen35_layer_probe_layer(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    cfg: &Qwen35Config,
    il: usize,
    layer: &Qwen35TrunkLayer,
    bs: Qwen35BsLayout,
    h_in: HirNodeId,
    cos_id: HirNodeId,
    sin_id: HirNodeId,
    export_post_attn: Option<&mut HirNodeId>,
) -> Result<HirNodeId> {
    match layer {
        Qwen35TrunkLayer::Linear(lin) => {
            emit_qwen35_gdn_prefill_layer(g, params, packed, cfg, il, lin, bs, h_in)
        }
        Qwen35TrunkLayer::FullAttn(fa) => emit_qwen35_full_attn_prefill_layer_ext(
            g,
            params,
            packed,
            cfg,
            il,
            fa,
            bs,
            h_in,
            cos_id,
            sin_id,
            export_post_attn,
        ),
    }
}

/// Final norm, optional LM head, and optional MTP head (prefill).
#[allow(clippy::too_many_arguments)]
pub fn emit_qwen35_prefill_tail(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    cfg: &Qwen35Config,
    weights: &Qwen35Weights,
    batch: usize,
    seq: usize,
    h_for_lm: HirNodeId,
    h_pre_norm: HirNodeId,
    input_ids: HirNodeId,
    cos_id: HirNodeId,
    sin_id: HirNodeId,
    with_lm_head: bool,
    enable_mtp_head: bool,
    export_normed_hidden: bool,
    fast_mtp: bool,
    last_token_idx: Option<HirNodeId>,
) -> Result<(Option<HirNodeId>, Option<HirNodeId>, Option<HirNodeId>)> {
    let n_embd = cfg.hidden_size;
    let n_vocab = if weights.token_embd.is_empty() {
        cfg.vocab_size
    } else {
        weights.token_embd.len() / n_embd
    };
    let head_dim = cfg.key_length;
    let n_rot = cfg.rope_dim_count;

    let out_norm = register_param(
        g,
        params,
        "output_norm.weight",
        weights.output_norm.clone(),
        Shape::new(&[n_embd], DType::F32),
    );
    let out_norm_beta = synth_zero(g, params, "output_norm.beta", n_embd);
    let h_norm = g.rms_norm(h_for_lm, out_norm, out_norm_beta, cfg.rms_norm_eps as f32);

    let mut logits = None;
    if with_lm_head {
        let logit_rows = if last_token_idx.is_some() { 1 } else { seq };
        let logit_shape = Shape::new(&[batch, logit_rows, n_vocab], DType::F32);
        let lm = match &weights.output {
            Some(w) => {
                let head = proj_mat(g, params, packed, "output.weight", w, n_embd, n_vocab);
                emit_proj(g, h_norm, head, w, logit_shape)
            }
            None => {
                if let Some(w) = &weights.token_embd_lm {
                    let head = proj_mat(g, params, packed, "lm_head.tied_t", w, n_embd, n_vocab);
                    emit_proj(g, h_norm, head, w, logit_shape)
                } else {
                    let embed_t = transpose_2d(&weights.token_embd, n_vocab, n_embd);
                    let tied = register_param(
                        g,
                        params,
                        "lm_head.tied_t",
                        embed_t,
                        Shape::new(&[n_embd, n_vocab], DType::F32),
                    );
                    g.mm(h_norm, tied)
                }
            }
        };
        logits = Some(if let Some(idx) = last_token_idx {
            gather_last_token(g, lm, batch, idx)
        } else {
            lm
        });
    }

    let mut mtp_logits = None;
    if enable_mtp_head {
        let mtp_layer = weights
            .mtp_layers
            .first()
            .ok_or_else(|| anyhow!("qwen35: MTP requested but no MTP layers loaded"))?;
        let mtp_il = cfg.num_hidden_layers - cfg.nextn_predict_layers;
        mtp_logits = Some(build_mtp_head(
            g,
            params,
            packed,
            cfg,
            mtp_il,
            mtp_layer,
            batch,
            seq,
            input_ids,
            h_pre_norm,
            &weights.token_embd,
            n_vocab,
            cos_id,
            sin_id,
            head_dim,
            n_rot,
            fast_mtp,
            last_token_idx,
        )?);
    }

    let normed = export_normed_hidden.then_some(h_norm);
    Ok((logits, mtp_logits, normed))
}

fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        data.len(),
        rows * cols,
        "transpose_2d: len {} != rows {rows} * cols {cols}",
        data.len()
    );
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

#[cfg(test)]
mod conv_weight_tests {
    use super::pack_depthwise_conv_weight;

    #[test]
    fn pack_depthwise_conv_respects_gguf_tap_major_layout() {
        // GGUF bytes for shape [k=2, channels=3] (tap fastest).
        let weight = vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        let packed = pack_depthwise_conv_weight(&weight, 2, 3);
        assert_eq!(packed, vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
    }

    #[test]
    fn l2_norm_matches_ggml_formula() {
        // ggml: y = x / max(sqrt(sum(x²)), eps)
        let x = [3.0f32, 4.0];
        let eps = 1e-6f32;
        let sumsq: f32 = x.iter().map(|v| v * v).sum();
        let scale = 1.0 / sumsq.sqrt().max(eps);
        let want = [x[0] * scale, x[1] * scale];
        let got = [
            x[0] / (sumsq.sqrt().max(eps)),
            x[1] / (sumsq.sqrt().max(eps)),
        ];
        assert!((got[0] - want[0]).abs() < 1e-6);
        assert!((got[1] - want[1]).abs() < 1e-6);
    }
}
