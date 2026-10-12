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

//! The `qwen4exp` text decoder flow.
//!
//! ```text
//!   token_embd
//!     → broadcast into hc_count residual streams              [1, s, hc, D]
//!     → 48 × ( [PLE on layer 1]
//!              hc_mix(attn) → GDN | gated-attention+QSA → hc_combine
//!              hc_mix(ffn)  → 512-expert MoE               → hc_combine )
//!     → hc_mix(output)                                       [1, s, D]
//!     → output
//! ```
//!
//! **There is no final norm stage.** `output_hc_*` is a third
//! hyper-connection module, and its grouped RMSNorm *is* the output norm; the
//! checkpoint carries no `output_norm.weight` at all. Adding one, or norming
//! again after the collapse, double-normalizes the logits.
//!
//! Graph inputs:
//!
//! | name | shape | when |
//! |---|---|---|
//! | `input_ids` | `[1, seq]` | always |
//! | `rope_cos` / `rope_sin` | `[seq, head_dim/2]` | [`FlowOptions::runtime_mrope`] |
//! | `ple_embeddings.{i}` | `[seq, ple_embed_dim]` | one per PLE module |
//!
//! `ple_embeddings.{i}` is an input rather than a gather because the table it
//! comes from is 51 billion weights — see [`crate::ple`], and
//! [`crate::ple::prepare_ple_embeddings`] for filling it.

use anyhow::{Result, anyhow, bail};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::blocks::LmHeadStage;
use rlx_flow::{BuiltModel, CompileProfile, FlowStage, ModelFlow, WeightSource};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};

use crate::attention::{AttnDims, emit_attention};
use crate::config::{LayerKind, Qwen4ExpConfig};
use crate::gdn::{GdnDims, emit_gdn};
use crate::hc::{HcDims, emit_hc_combine, emit_hc_mix, emit_hc_split};
use crate::indexer::QsaDims;
use crate::moe::{MoeDims, emit_moe};
use crate::ple::{PleDims, emit_ple, ple_embed_input};
use crate::rope::{mrope_tables, table_width};

/// GGUF's token-embedding tensor name.
pub const EMBED_KEY: &str = "token_embd.weight";
/// Graph-input name for the M-RoPE cosine table.
pub const ROPE_COS_INPUT: &str = "rope_cos";
/// Graph-input name for the M-RoPE sine table.
pub const ROPE_SIN_INPUT: &str = "rope_sin";

/// What the flow's head produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputKind {
    /// `[1, seq, vocab]` — collapse the streams through the final mixer, then
    /// the LM head.
    Logits,
    /// `[1, seq, hidden]` — the collapsed, normed hidden state. The final mixer
    /// *is* the output norm, so this is already normalized.
    Hidden,
    /// `[1, seq, hc_count · hidden]` — the hyper-connection residual, **before**
    /// the final mixer.
    ///
    /// This is what the MTP block reads (`res->t_h_nextn` in llama.cpp): the
    /// draft head is fed the four residual streams, not the collapsed hidden
    /// state. Collapsing and re-splitting would broadcast one vector into four
    /// identical streams and throw away everything the depth-wise mixing
    /// accumulated. See [`crate::mtp`].
    HcResidual,
}

/// Knobs that change the graph's shape rather than its weights.
#[derive(Debug, Clone, Copy)]
pub struct FlowOptions {
    /// Prompt length this graph is built for.
    pub seq: usize,
    /// What the head produces.
    pub output: OutputKind,
    /// Declare `rope_cos` / `rope_sin` as graph inputs instead of baking the
    /// text positions `0..seq` in as parameters.
    ///
    /// Needed for anything whose positions are not `0..seq`: a continued
    /// prefill, or a multimodal batch, whose 3-D M-RoPE positions the caller
    /// computes.
    pub runtime_mrope: bool,
    /// Run the QSA selection even where it is provably the identity.
    ///
    /// Only useful for testing: see [`QsaDims::force_emit`].
    pub force_qsa: bool,
    /// Take an `inputs_embeds` input instead of gathering the embedding table
    /// in the graph.
    ///
    /// This is the multimodal seam: vision rows are spliced into the embedding
    /// stream on the host (see `crate::vision`), so the graph never needs to
    /// know an image was there. It also keeps a 1.2 GB `token_embd` off the
    /// device.
    pub host_embed: bool,
}

impl FlowOptions {
    pub fn prefill(seq: usize) -> Self {
        Self {
            seq,
            output: OutputKind::Logits,
            runtime_mrope: false,
            force_qsa: false,
            host_embed: false,
        }
    }
}

/// Does the weight `name` belong to the text flow at all?
///
/// Useful for narrowing a loader to the tensors a text-only build reads — in
/// particular it excludes `per_layer_token_embd.weight`, which is most of the
/// checkpoint and never becomes a graph parameter.
pub fn is_text_flow_weight(name: &str, cfg: &Qwen4ExpConfig) -> bool {
    if name == "per_layer_token_embd.weight" {
        return false;
    }
    if let Some(rest) = name.strip_prefix("blk.") {
        return match rest.split('.').next().unwrap_or("").parse::<usize>() {
            Ok(i) => i < cfg.num_hidden_layers,
            Err(_) => false,
        };
    }
    matches!(
        name,
        EMBED_KEY
            | "output.weight"
            | "output_hc_norm.weight"
            | "output_hc_down.weight"
            | "output_hc_up.weight"
    )
}

/// Build the `qwen4exp` text prefill graph for a fixed `seq`.
///
/// Input: `input_ids [1, seq]` (plus [`PLE_EMBED_INPUT`](crate::ple::PLE_EMBED_INPUT) when the model has a
/// PLE layer). Output: `logits [1, seq, vocab]` when `with_lm_head`, else the
/// final hidden state.
pub fn build_qwen4exp_text_flow(
    cfg: &Qwen4ExpConfig,
    weights: &mut WeightMap,
    seq: usize,
    with_lm_head: bool,
) -> Result<BuiltModel> {
    build_qwen4exp_text_flow_with_source(cfg, &mut WeightMapSource(weights), seq, with_lm_head)
}

/// [`build_qwen4exp_text_flow`] over any [`WeightSource`] — the entry point for
/// running **packed**.
///
/// Pass `rlx_core::flow_bridge::PackedWeightLoaderSource` wrapping a GGUF
/// loader and every 2-D projection becomes a fused `Op::DequantMatMul`, and
/// every routed expert bank an `Op::DequantGroupedMatMul`, with no f32 weight
/// materialized. For this model that is nearly everything that matters: the
/// 512-expert banks are 48 × 3 × `[512, 640, 2560]`.
///
/// What stays f32 either way, because it is not a 2-D projection: the norm
/// gains, `ssm_a`, `ssm_dt.bias`, the depthwise `ssm_conv1d` / `ple_conv1d`
/// kernels, the router (`ffn_gate_inp`) and the shared expert's scalar gate.
pub fn build_qwen4exp_text_flow_with_source(
    cfg: &Qwen4ExpConfig,
    weights: &mut dyn WeightSource,
    seq: usize,
    with_lm_head: bool,
) -> Result<BuiltModel> {
    let mut opts = FlowOptions::prefill(seq);
    opts.output = if with_lm_head {
        OutputKind::Logits
    } else {
        OutputKind::Hidden
    };
    build_qwen4exp_text_flow_with_options(cfg, weights, opts)
}

/// [`build_qwen4exp_text_flow_with_source`] with the full [`FlowOptions`] —
/// runtime M-RoPE tables and the QSA `force_emit` escape.
pub fn build_qwen4exp_text_flow_with_options(
    cfg: &Qwen4ExpConfig,
    weights: &mut dyn WeightSource,
    opts: FlowOptions,
) -> Result<BuiltModel> {
    cfg.validate()?;
    let seq = opts.seq;
    if seq == 0 {
        bail!("qwen4exp: seq must be >= 1");
    }
    if cfg.with_mtp {
        bail!(
            "qwen4exp: the MTP block (blk.{}) is not emitted by this flow yet",
            cfg.num_hidden_layers
        );
    }

    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let rope_width = table_width(cfg);

    let hc = HcDims::from_config(cfg, seq);
    let gdn = GdnDims::from_config(cfg, seq);
    let attn = AttnDims::from_config(cfg, seq);
    let qsa = QsaDims {
        force_emit: opts.force_qsa,
        ..QsaDims::from_config(cfg, seq)
    };
    let moe = MoeDims::from_config(cfg, seq);
    let ple = PleDims::from_config(cfg, seq);

    let stream_shape = Shape::new(&[1, seq, cfg.hc_count, hidden], f);
    let cfg_hc_dim = cfg.hc_dim();

    let mut flow = ModelFlow::new("qwen4exp").with_profile(CompileProfile::llama32_prefill());
    flow = flow.input("input_ids", Shape::new(&[1, seq], f));
    if opts.runtime_mrope {
        let s = Shape::new(&[seq, rope_width], f);
        flow = flow
            .input(ROPE_COS_INPUT, s.clone())
            .input(ROPE_SIN_INPUT, s);
    }
    for idx in 0..cfg.ple_layers.len() {
        flow = flow.input(
            ple_embed_input(idx),
            Shape::new(&[seq, cfg.ple_embed_dim()], f),
        );
    }

    // Publish the M-RoPE tables under stable names so every attention layer
    // and every QSA indexer reads the same ones.
    {
        let runtime = opts.runtime_mrope;
        let (cos_data, sin_data) = if runtime {
            (Vec::new(), Vec::new())
        } else {
            mrope_tables(cfg, seq, None)
        };
        flow = flow.plugin_named("rope_tables", move |emit, prev| {
            let (cos, sin) = if runtime {
                (
                    emit.flow_input(ROPE_COS_INPUT)?.hir_id(),
                    emit.flow_input(ROPE_SIN_INPUT)?.hir_id(),
                )
            } else {
                let s = Shape::new(&[seq, rope_width], DType::F32);
                (
                    emit.synth_param("qwen4exp.rope.cos", cos_data.clone(), s.clone()),
                    emit.synth_param("qwen4exp.rope.sin", sin_data.clone(), s),
                )
            };
            emit.set_named(ROPE_COS_INPUT, cos);
            emit.set_named(ROPE_SIN_INPUT, sin);
            Ok(prev)
        });
    }

    flow = if opts.host_embed {
        flow.embed_host(EMBED_KEY, hidden)
    } else {
        flow.embed(EMBED_KEY)
    };

    // Broadcast the embedding into `hc_count` identical residual streams.
    {
        let s = stream_shape.clone();
        flow = flow.plugin_named("hc_split", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("hc_split needs the embedding"))?
                .hir_id();
            let out = emit_hc_split(emit, x, hc);
            Ok(Some(emit.wrap(out, s.clone())))
        });
    }

    for i in 0..cfg.num_hidden_layers {
        let prefix = format!("blk.{i}");
        let kind = cfg.layer_kind(i);
        let ple_idx = cfg.ple_layer_index(i);
        let s = stream_shape.clone();
        flow = flow.plugin_named(format!("layer{i}"), move |emit, prev| {
            let mut x = prev
                .ok_or_else(|| anyhow!("layer{i} needs a stream state"))?
                .hir_id();

            // PLE adds to the residual directly — it is not a sublayer and
            // does not go through a hyper-connection.
            if let Some(idx) = ple_idx {
                let emb = emit.flow_input(&ple_embed_input(idx))?.hir_id();
                x = emit_ple(emit, &prefix, x, emb, ple)?;
            }

            let cos = emit.named(ROPE_COS_INPUT)?;
            let sin = emit.named(ROPE_SIN_INPUT)?;

            // ── token mixer ──
            let gates = emit_hc_mix(emit, &format!("{prefix}.hc_attn"), x, hc, true)?;
            let branch = match kind {
                LayerKind::Linear => emit_gdn(emit, &prefix, gates.mixed, gdn)?,
                LayerKind::Indexed => {
                    emit_attention(emit, &prefix, gates.mixed, cos, sin, attn, Some(qsa))?
                }
            };
            let x = emit_hc_combine(emit, branch, x, gates, hc);

            // ── MoE ──
            let gates = emit_hc_mix(emit, &format!("{prefix}.hc_ffn"), x, hc, true)?;
            let branch = emit_moe(emit, &prefix, gates.mixed, moe)?;
            let out = emit_hc_combine(emit, branch, x, gates, hc);

            Ok(Some(emit.wrap(out, s.clone())))
        });
    }

    // The MTP block reads the residual before the collapse, so that output
    // skips the final mixer entirely rather than collapsing and re-splitting.
    if opts.output == OutputKind::HcResidual {
        let rs = Shape::new(&[1, seq, cfg.hc_dim()], f);
        flow = flow.plugin_named("hc_residual", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("hc_residual needs a stream state"))?
                .hir_id();
            let mut gb = HirMut::new(emit.hir());
            let out = gb.reshape_(x, vec![1, seq as i64, cfg_hc_dim as i64]);
            Ok(Some(emit.wrap(out, rs.clone())))
        });
        return flow.output("hc_residual").build_with(weights, None);
    }

    // The final mixer collapses the streams AND is the output norm.
    {
        let hs = Shape::new(&[1, seq, hidden], f);
        flow = flow.plugin_named("hc_head", move |emit, prev| {
            let x = prev
                .ok_or_else(|| anyhow!("hc_head needs a stream state"))?
                .hir_id();
            let gates = emit_hc_mix(emit, "output_hc", x, hc, false)?;
            Ok(Some(emit.wrap(gates.mixed, hs.clone())))
        });
    }

    if opts.output == OutputKind::Hidden {
        return flow.output("hidden").build_with(weights, None);
    }

    let head = if cfg.tie_word_embeddings {
        LmHeadStage::tied(cfg.vocab_size, hidden)
    } else {
        LmHeadStage::separate("output.weight", cfg.vocab_size, hidden)
    };
    flow.raw_stage(FlowStage::LmHead(head))
        .output("logits")
        .build_with(weights, None)
}
