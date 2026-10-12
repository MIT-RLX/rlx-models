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

//! The multi-token-prediction (MTP) draft block — `blk.48`.
//!
//! One extra gated-attention + MoE layer, fed by the **trunk's
//! hyper-connection residual** and the next token's embedding:
//!
//! ```text
//!   e      = RMSNorm(token_embd[t+1], nextn.enorm)        [s, D]
//!   h_s    = GroupedRMSNorm(trunk residual, nextn.hnorm)  [s, hc, D]
//!   res_s  = nextn.eh_proj · [ e ; h_s ]                  per stream
//!   ── one normal qwen4exp layer: hc_attn → QSA attention → hc_ffn → MoE ──
//!   next_h = the residual, which the NEXT draft step reads as its `h`
//!   logits = output · hc_mix(res, nextn.hc_head_*)
//! ```
//!
//! Three things make this more than "another layer":
//!
//! * **It reads the residual, not the hidden state.** `h` is
//!   `[seq, hc_count · hidden]` — all four streams, taken *before* the trunk's
//!   final mixer. Collapsing and re-splitting would broadcast one vector into
//!   four identical streams and discard everything the depth-wise mixing
//!   accumulated. Build the trunk with [`crate::OutputKind::HcResidual`] to get
//!   it, and [`build_qwen4exp_head_flow`] to turn the same residual into the
//!   trunk's own logits without running the trunk twice.
//! * **`eh_proj` is applied per stream, and its input is a concatenation on the
//!   feature axis.** The converter builds it as
//!   `cat([fc_embedding, fc_hidden], dim=1)`, so `eh_proj([e ; h_s])` *is*
//!   `fc_embedding(e) + fc_hidden(h_s)` — the embedding half is shared across
//!   streams, the hidden half is not. Concatenating on the stream axis instead
//!   type-checks and is a different function.
//! * **It has its own output mixer** (`nextn.hc_head_*`), separate from the
//!   trunk's `output_hc_*`, but shares the trunk's `output.weight`.
//!
//! The MTP ships as its own `mtp-*.gguf` — `block_count = 49`,
//! `nextn_predict_layers = 1`, and no `output_hc_*` — which is why
//! [`crate::Qwen4ExpConfig::block_count`] gates the extra block on `with_mtp`.

use anyhow::{Result, anyhow, bail};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{BuiltModel, CompileProfile, Emit, ModelFlow, WeightSource};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::attention::{AttnCache, AttnCacheOut, AttnDims, emit_attention_with_cache};
use crate::common::{grouped_rms_norm, linear, rms_norm};
use crate::config::Qwen4ExpConfig;
use crate::flow::{EMBED_KEY, OutputKind};
use crate::hc::{HcDims, emit_hc_combine, emit_hc_mix};
use crate::indexer::QsaDims;
use crate::moe::{MoeDims, emit_moe};
use crate::rope::{mrope_tables, table_width};

/// Graph-input name for the trunk's hyper-connection residual.
pub const HC_RESIDUAL_INPUT: &str = "hc_residual";

/// Everything one MTP block needs that is not a weight.
#[derive(Debug, Clone, Copy)]
pub struct MtpDims {
    pub hc: HcDims,
    pub attn: AttnDims,
    pub moe: MoeDims,
    /// `None` only if the checkpoint has no indexer on the MTP layer; the
    /// published one does (`attention.compress_ratios[48] = 4`).
    pub qsa: Option<QsaDims>,
}

/// What one MTP block produces.
#[derive(Debug, Clone, Copy)]
pub struct MtpOut {
    /// `[1, seq, hc_count · hidden]` — the residual the *next* draft step reads
    /// as its `h`.
    pub next_h: HirNodeId,
    /// `[1, seq, hidden]` — collapsed through `nextn.hc_head_*`, ready for the
    /// LM head. Already normalized: the mixer is the norm.
    pub collapsed: HirNodeId,
    /// Present when a key/value cache was supplied.
    pub cache: Option<AttnCacheOut>,
}

/// Emit the MTP block for GGUF block `prefix` (`blk.{num_hidden_layers}`).
///
/// * `tok_embd` — `[1, seq, hidden]`, the embedding of the token being
///   predicted *from* (the reference feeds `input_ids` shifted by one).
/// * `h` — `[1, seq, hc_count · hidden]`, the trunk's residual.
#[allow(clippy::too_many_arguments)]
pub fn emit_mtp_block(
    emit: &mut Emit<'_>,
    prefix: &str,
    tok_embd: HirNodeId,
    h: HirNodeId,
    cos: HirNodeId,
    sin: HirNodeId,
    d: MtpDims,
    cache: Option<AttnCache>,
) -> Result<MtpOut> {
    let hidden = d.hc.hidden;
    let hc = d.hc.streams;
    let seq = d.hc.seq;
    let (si, hci, di) = (seq as i64, hc as i64, hidden as i64);
    let width = hc * hidden;

    let h_flat = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(h, vec![si, width as i64])
    };
    // Grouped over one stream, with a `[hidden, hc]` gain — the same norm shape
    // as every other hyper-connection site.
    let h_norm = grouped_rms_norm(
        emit,
        &format!("{prefix}.nextn.hnorm"),
        h_flat,
        hc,
        hidden,
        d.hc.norm_eps,
        d.hc.norm_offset,
    )?;
    let e2d = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(tok_embd, vec![si, di])
    };
    let e_norm = rms_norm(
        emit,
        &format!("{prefix}.nextn.enorm"),
        e2d,
        hidden,
        d.hc.norm_eps,
        d.hc.norm_offset,
    )?;

    // `[e ; h_s]` on the FEATURE axis, per stream: the embedding half is shared
    // across streams, the hidden half is each stream's own.
    let joined = {
        let mut gb = HirMut::new(emit.hir());
        let h3 = gb.reshape_(h_norm, vec![si, hci, di]);
        let e3 = gb.reshape_(e_norm, vec![si, 1, di]);
        let e3 = gb.expand_(e3, vec![si, hci, di]);
        let cat = gb.concat_(vec![e3, h3], 2); // [seq, hc, 2·hidden]
        gb.reshape_(cat, vec![(seq * hc) as i64, 2 * di])
    };
    let projected = linear(emit, &format!("{prefix}.nextn.eh_proj.weight"), joined)?;
    let mut x = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(projected, vec![1, si, hci, di])
    };

    // ── one ordinary qwen4exp layer ──
    let gates = emit_hc_mix(emit, &format!("{prefix}.hc_attn"), x, d.hc, true)?;
    let (branch, cache_out) =
        emit_attention_with_cache(emit, prefix, gates.mixed, cos, sin, d.attn, d.qsa, cache)?;
    x = emit_hc_combine(emit, branch, x, gates, d.hc);

    let gates = emit_hc_mix(emit, &format!("{prefix}.hc_ffn"), x, d.hc, true)?;
    let branch = emit_moe(emit, prefix, gates.mixed, d.moe)?;
    x = emit_hc_combine(emit, branch, x, gates, d.hc);

    // The draft head's own mixer — not the trunk's `output_hc_*`.
    let head = emit_hc_mix(emit, &format!("{prefix}.nextn.hc_head"), x, d.hc, false)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(MtpOut {
        next_h: gb.reshape_(x, vec![1, si, width as i64]),
        collapsed: gb.reshape_(head.mixed, vec![1, si, di]),
        cache: cache_out,
    })
}

/// Dims for the MTP block, derived from the config at prompt length `seq`.
pub fn mtp_dims(cfg: &Qwen4ExpConfig, seq: usize, force_qsa: bool) -> MtpDims {
    MtpDims {
        hc: HcDims::from_config(cfg, seq),
        attn: AttnDims::from_config(cfg, seq),
        moe: MoeDims::from_config(cfg, seq),
        qsa: Some(QsaDims {
            force_emit: force_qsa,
            ..QsaDims::from_config(cfg, seq)
        }),
    }
}

/// Build the MTP draft graph for a fixed `seq`.
///
/// Inputs: `input_ids [1, seq]` (the *draft* tokens), `hc_residual
/// [1, seq, hc_count · hidden]` (the trunk's), and `rope_cos` / `rope_sin`
/// `[seq, head_dim/2]`. Output per `output`: [`OutputKind::Logits`] for the
/// draft's predictions, or [`OutputKind::HcResidual`] for the residual a second
/// draft step would consume.
pub fn build_qwen4exp_mtp_flow(
    cfg: &Qwen4ExpConfig,
    weights: &mut WeightMap,
    seq: usize,
    output: OutputKind,
) -> Result<BuiltModel> {
    build_qwen4exp_mtp_flow_with(cfg, &mut WeightMapSource(weights), seq, output)
}

/// [`build_qwen4exp_mtp_flow`] over any [`WeightSource`].
pub fn build_qwen4exp_mtp_flow_with(
    cfg: &Qwen4ExpConfig,
    weights: &mut dyn WeightSource,
    seq: usize,
    output: OutputKind,
) -> Result<BuiltModel> {
    cfg.validate()?;
    if seq == 0 {
        bail!("qwen4exp: seq must be >= 1");
    }
    if cfg.num_nextn_predict_layers == 0 {
        bail!(
            "qwen4exp: this config declares no MTP block \
             (nextn_predict_layers = 0). The MTP ships as its own \
             `mtp-*.gguf`; parse that file's metadata, which sets \
             block_count = {} and nextn_predict_layers = 1.",
            cfg.num_hidden_layers + 1
        );
    }
    if output == OutputKind::Hidden {
        bail!(
            "qwen4exp: the MTP block's `Hidden` output is ambiguous — use \
             `Logits` for the draft's predictions or `HcResidual` for the \
             residual a second draft step reads"
        );
    }
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let hc_dim = cfg.hc_dim();
    let rope_width = table_width(cfg);
    let prefix = format!("blk.{}", cfg.num_hidden_layers);
    let d = mtp_dims(cfg, seq, false);
    let (cos_data, sin_data) = mrope_tables(cfg, seq, None);
    let vocab = cfg.vocab_size;

    let out_shape = match output {
        OutputKind::Logits => Shape::new(&[1, seq, vocab], f),
        OutputKind::HcResidual => Shape::new(&[1, seq, hc_dim], f),
        OutputKind::Hidden => unreachable!("rejected above"),
    };

    let flow = ModelFlow::new("qwen4exp_mtp")
        .with_profile(CompileProfile::llama32_prefill())
        .input("input_ids", Shape::new(&[1, seq], f))
        .input(HC_RESIDUAL_INPUT, Shape::new(&[1, seq, hc_dim], f))
        .embed(EMBED_KEY)
        .plugin_named("mtp", move |emit, prev| {
            let tok_embd = prev
                .ok_or_else(|| anyhow!("the MTP flow needs the embedding output"))?
                .hir_id();
            let h = emit.flow_input(HC_RESIDUAL_INPUT)?.hir_id();
            let s = Shape::new(&[seq, rope_width], DType::F32);
            let cos = emit.synth_param("qwen4exp.mtp.rope.cos", cos_data.clone(), s.clone());
            let sin = emit.synth_param("qwen4exp.mtp.rope.sin", sin_data.clone(), s);
            let out = emit_mtp_block(emit, &prefix, tok_embd, h, cos, sin, d, None)?;
            let id = match output {
                OutputKind::HcResidual => out.next_h,
                OutputKind::Logits => {
                    let x2d = {
                        let mut gb = HirMut::new(emit.hir());
                        gb.reshape_(out.collapsed, vec![seq as i64, hidden as i64])
                    };
                    let y = crate::common::linear(emit, "output.weight", x2d)?;
                    let mut gb = HirMut::new(emit.hir());
                    gb.reshape_(y, vec![1, seq as i64, vocab as i64])
                }
                OutputKind::Hidden => unreachable!("rejected above"),
            };
            Ok(Some(emit.wrap(id, out_shape.clone())))
        });

    flow.output("mtp").build_with(weights, None)
}

/// Build just the trunk's head: `hc_residual` → final mixer → LM head.
///
/// A speculative loop needs the trunk's residual (for the draft block) *and*
/// the trunk's logits. Rather than running the trunk twice, build it once with
/// [`OutputKind::HcResidual`] and feed that residual here. The two together are
/// exactly [`OutputKind::Logits`] — `tests/mtp.rs` checks that.
pub fn build_qwen4exp_head_flow(
    cfg: &Qwen4ExpConfig,
    weights: &mut WeightMap,
    seq: usize,
) -> Result<BuiltModel> {
    build_qwen4exp_head_flow_with(cfg, &mut WeightMapSource(weights), seq)
}

/// [`build_qwen4exp_head_flow`] over any [`WeightSource`].
pub fn build_qwen4exp_head_flow_with(
    cfg: &Qwen4ExpConfig,
    weights: &mut dyn WeightSource,
    seq: usize,
) -> Result<BuiltModel> {
    cfg.validate()?;
    if seq == 0 {
        bail!("qwen4exp: seq must be >= 1");
    }
    let f = DType::F32;
    let hidden = cfg.hidden_size;
    let hc_dim = cfg.hc_dim();
    let vocab = cfg.vocab_size;
    let hc = HcDims {
        hidden,
        streams: cfg.hc_count,
        lowrank: cfg.hc_lowrank,
        norm_eps: cfg.rms_norm_eps,
        norm_offset: cfg.rms_norm_offset,
        seq,
    };
    let out_shape = Shape::new(&[1, seq, vocab], f);

    let flow = ModelFlow::new("qwen4exp_head")
        .with_profile(CompileProfile::llama32_prefill())
        .input(HC_RESIDUAL_INPUT, Shape::new(&[1, seq, hc_dim], f))
        .plugin_named("head", move |emit, _prev| {
            let h = emit.flow_input(HC_RESIDUAL_INPUT)?.hir_id();
            let x4 = {
                let mut gb = HirMut::new(emit.hir());
                gb.reshape_(h, vec![1, seq as i64, hc.streams as i64, hidden as i64])
            };
            let gates = emit_hc_mix(emit, "output_hc", x4, hc, false)?;
            let x2d = {
                let mut gb = HirMut::new(emit.hir());
                gb.reshape_(gates.mixed, vec![seq as i64, hidden as i64])
            };
            let y = crate::common::linear(emit, "output.weight", x2d)?;
            let mut gb = HirMut::new(emit.hir());
            let id = gb.reshape_(y, vec![1, seq as i64, vocab as i64]);
            Ok(Some(emit.wrap(id, out_shape.clone())))
        });

    flow.output("logits").build_with(weights, None)
}
