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

//! Hyper-connections — `Qwen4ExpTextGatedResidual`.
//!
//! Qwen3.8-Flash-Next carries `hc_count = 4` parallel residual streams instead
//! of one, and **replaces every layer norm with a hyper-connection**: there is
//! no `attn_norm`, no `ffn_norm` and no `output_norm` anywhere in the
//! checkpoint. Each of the three sites (attention, MoE, and the final collapse
//! before the LM head) is one of these modules.
//!
//! ```text
//!   xn     = GroupedRMSNorm(reshape(x, [s, hc·D]))          [s, hc·D]
//!   lo     = silu(xn @ W_downᵀ / hc)                        [s, lowrank]
//!   gate   = σ(lo @ W_upᵀ)                                  [s, hc·D]
//!   mixed  = mean_h (gate · xn)                             [s, D]
//!   ── the sublayer runs on `mixed` ──
//!   inject = 2·σ(xn @ W_injectᵀ / hc)                       [s, hc]
//!   x'     = x + y ⊗ inject                                 [1, s, hc, D]
//! ```
//!
//! Three details are load-bearing, and all three differ from the sibling
//! implementations in `rlx_glm5next::mhc` and `rlx_motif::mhc` — **do not
//! cross-port between them**:
//!
//! * **The scatter is identity, not a learned mixer.** There is no `comb`
//!   matrix: every stream receives the *same* sublayer output, scaled only by
//!   its own `inject` weight. GLM-5.3-Flash multiplies the residual by a
//!   Sinkhorn-projected `[hc, hc]` matrix at the same point; doing that here
//!   would mix streams the reference keeps independent.
//! * **The input norm is grouped and weighted.** It reduces over one stream
//!   (`group_size = hidden_size`) but scales with a full `hc·D` gain. GLM's is
//!   unweighted and reduces over all four streams at once — a different
//!   normalizer *and* a different parameterization.
//! * **Both gates divide by `hc` before the nonlinearity**, and `inject` is
//!   `2·σ(·)` rather than `σ(·)`. The factor 2 centres the scatter weight on 1,
//!   so an untrained `inject` is a plain residual add; dropping it halves every
//!   sublayer contribution.
//!
//! Every tensor here is tiny apart from the two low-rank projections, so the
//! whole block is a few dozen nodes and needs no custom kernel on any backend.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::common::{grouped_rms_norm, linear, scalar, sigmoid};
use crate::config::Qwen4ExpConfig;

/// Static shape/hyper-parameters of one hyper-connection site.
#[derive(Debug, Clone, Copy)]
pub struct HcDims {
    pub hidden: usize,
    /// `hc_count` — parallel residual streams.
    pub streams: usize,
    /// `hc_lowrank` — rank of the learned input mixer.
    pub lowrank: usize,
    pub norm_eps: f32,
    /// Whether the norm gain still needs the reference's `1 +`.
    pub norm_offset: bool,
    pub seq: usize,
}

impl HcDims {
    /// Derive from the model config, for a step of `seq` positions.
    ///
    /// There is exactly one statement of how a config becomes these dimensions, so
    /// prefill, chunked prefill, the decode step, the MTP block and the tests
    /// cannot disagree about what model they are building. They used to hold four
    /// copies of it.
    pub fn from_config(cfg: &Qwen4ExpConfig, seq: usize) -> Self {
        HcDims {
            hidden: cfg.hidden_size,
            streams: cfg.hc_count,
            lowrank: cfg.hc_lowrank,
            norm_eps: cfg.rms_norm_eps,
            norm_offset: cfg.rms_norm_offset,
            seq,
        }
    }
    /// `hc_count · hidden_size` — the residual's flattened width.
    pub fn width(&self) -> usize {
        self.streams * self.hidden
    }
}

/// What one hyper-connection site produces before its sublayer runs.
#[derive(Debug, Clone, Copy)]
pub struct HcGates {
    /// `[1, seq, hidden]` — the streams collapsed to a single sequence, ready
    /// for the sublayer. This is already normalized: the sublayer has no input
    /// norm of its own.
    pub mixed: HirNodeId,
    /// `[seq, hc]` — how the sublayer output is scattered back over streams.
    ///
    /// `None` at the final mixer, which has no `_inject` tensor because nothing
    /// is scattered back. [`emit_hc_combine`] panics on `None` rather than
    /// inventing a weight: a site built without a scatter cannot have one.
    pub inject: Option<HirNodeId>,
}

/// Broadcast the embedding into `hc` identical residual streams — the shape the
/// first decoder layer expects.
pub fn emit_hc_split(emit: &mut Emit<'_>, x: HirNodeId, d: HcDims) -> HirNodeId {
    let (hi, si, di) = (d.streams as i64, d.seq as i64, d.hidden as i64);
    let mut gb = HirMut::new(emit.hir());
    let x4 = gb.reshape_(x, vec![1, si, 1, di]);
    gb.expand_(x4, vec![1, si, hi, di])
}

/// Emit the mixer of one hyper-connection site.
///
/// `prefix` is the GGUF stem — `blk.{i}.hc_attn`, `blk.{i}.hc_ffn` or
/// `output_hc` — and `x` is the `[1, seq, hc, hidden]` stream state.
///
/// `with_inject = false` is the final mixer (`use_combine = False` upstream):
/// it has no `_inject` tensor because nothing is scattered back. That mixer
/// **is** the model's output norm.
pub fn emit_hc_mix(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    d: HcDims,
    with_inject: bool,
) -> Result<HcGates> {
    let f = DType::F32;
    let (h, s) = (d.streams, d.seq);
    let (hi, si, di) = (h as i64, s as i64, d.hidden as i64);
    let width = d.width();

    let flat = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(x, vec![si, width as i64])
    };
    let xn = grouped_rms_norm(
        emit,
        &format!("{prefix}_norm"),
        flat,
        h,
        d.hidden,
        d.norm_eps,
        d.norm_offset,
    )?;

    let inv_hc = scalar(emit, &format!("{prefix}.inv_hc"), 1.0 / h as f32);
    let two = scalar(emit, &format!("{prefix}.two"), 2.0);

    // low-rank input mixer: hc·D → lowrank → hc·D
    let lo = linear(emit, &format!("{prefix}_down.weight"), xn)?;
    let lo = {
        let mut gb = HirMut::new(emit.hir());
        let scaled = gb.mul(lo, inv_hc);
        gb.silu(scaled)
    };
    let gate = linear(emit, &format!("{prefix}_up.weight"), lo)?;

    // The injection weights read the SAME normed input as the mixer, so they
    // must be projected before `xn` is consumed by the collapse below.
    let inject_raw = if with_inject {
        Some(linear(emit, &format!("{prefix}_inject.weight"), xn)?)
    } else {
        None
    };

    let mut gb = HirMut::new(emit.hir());

    // mixed = mean over streams of (σ(gate) · xn)
    let g = sigmoid(&mut gb, gate);
    let gated = gb.mul(xn, g);
    let gated4 = gb.reshape_(gated, vec![1, si, hi, di]);
    let mixed = gb.mean(gated4, vec![2], false);
    let mixed = gb.reshape_(mixed, vec![1, si, di]);

    // `2·σ(·)` centres the scatter weight on 1, so an untrained `inject` is a
    // plain residual add. Dropping the 2 halves every sublayer contribution.
    let inject = inject_raw.map(|raw| {
        let t = gb.mul(raw, inv_hc);
        let t = gb.add_node(
            rlx_ir::op::Op::Activation(rlx_ir::op::Activation::Sigmoid),
            vec![t],
            Shape::new(&[s, h], f),
        );
        gb.mul(t, two)
    });

    Ok(HcGates { mixed, inject })
}

/// Scatter a sublayer output back over the residual streams:
/// `residual + y ⊗ inject`.
///
/// * `y` — `[1, seq, hidden]`, the sublayer's output.
/// * `residual` — `[1, seq, hc, hidden]`, the stream state *before* the site.
///
/// The comb is the identity: each stream adds the same `y`, weighted only by
/// its own `inject` column.
pub fn emit_hc_combine(
    emit: &mut Emit<'_>,
    y: HirNodeId,
    residual: HirNodeId,
    gates: HcGates,
    d: HcDims,
) -> HirNodeId {
    let (hi, si, di) = (d.streams as i64, d.seq as i64, d.hidden as i64);
    let inject = gates
        .inject
        .expect("emit_hc_combine on a site built with `with_inject = false`");
    let mut gb = HirMut::new(emit.hir());
    let w = gb.reshape_(inject, vec![1, si, hi, 1]);
    let y4 = gb.reshape_(y, vec![1, si, 1, di]);
    let placed = gb.mul(y4, w);
    gb.add(residual, placed)
}
