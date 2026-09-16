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

//! mHC — Manifold-Constrained Hyper-Connections.
//!
//! GLM-5.3-Flash does not carry a single residual stream. The hidden state is
//! `[1, seq, H, D]` with `H = hc_mult = 4` parallel streams, and each of the two
//! sublayer sites per decoder layer (attention, FFN) is wrapped in a learned,
//! input-dependent mixing of them:
//!
//! ```text
//!   flat        = UnweightedRMSNorm(reshape(x, [seq, H·D]))
//!   mix         = flat @ fn                                   [seq, (2+H)·H]
//!   pre|post|comb = split(mix, [H, H, H·H])
//!
//!   pre   = σ(pre·scale₀  + base₀) + ε                        [seq, H]
//!   post  = 2·σ(post·scale₁ + base₁)                          [seq, H]
//!   comb  = Sinkhorn(softmax(comb·scale₂ + base₂))            [seq, H, H]
//!
//!   collapsed = Σ_h pre[h]·x[h]                               [1, seq, D]
//!   ── sublayer runs on `collapsed` ──
//!   x'  = post ⊗ y + combᵀ · x                                [1, seq, H, D]
//! ```
//!
//! Three details are load-bearing and differ from the sibling implementation in
//! `rlx_motif::mhc` — do not cross-port between them:
//!
//! * **The input norm is unweighted.** There is no `gamma` in the checkpoint;
//!   the norm is a plain `x·rsqrt(mean(x²) + rms_norm_eps)`. `rlx-motif` loads a
//!   `rms_norm.weight`.
//! * **`comb` is a softmax, not a sigmoid.** `Glm5NextTextHyperConnection`
//!   softmaxes the logits along the last axis, adds `hc_eps`, then Sinkhorns.
//!   `rlx-motif` starts from `exp(clamp(·))`.
//! * **Sinkhorn starts with a *column* normalization.** The reference does one
//!   column pass, then `hc_sinkhorn_iters - 1` rounds of (row, column) — so
//!   there are `iters` column passes but only `iters - 1` row passes, and the
//!   result is column-stochastic, not exactly doubly stochastic. Emitting
//!   `iters` symmetric rounds gives a subtly different matrix.
//!
//! Every gate is `H`-wide with `H = 4`, so the whole block is a few dozen tiny
//! nodes and needs no custom kernel on any backend.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

/// Static shape/hyper-parameters of one mHC site.
#[derive(Debug, Clone, Copy)]
pub struct MhcDims {
    pub hidden: usize,
    /// `hc_mult` — number of parallel residual streams.
    pub streams: usize,
    pub sinkhorn_iters: usize,
    /// `hc_eps` — the additive floor on `pre` and inside Sinkhorn.
    pub eps: f32,
    /// `rms_norm_eps` — the *input norm's* epsilon, not [`Self::eps`].
    pub norm_eps: f32,
    pub seq: usize,
}

impl MhcDims {
    /// `(2 + H) · H` — width of the fused gate projection.
    pub fn mix(&self) -> usize {
        (2 + self.streams) * self.streams
    }
}

/// What one mHC site produces before the sublayer runs.
#[derive(Debug, Clone, Copy)]
pub struct MhcGates {
    /// `[1, seq, hidden]` — the streams collapsed to a single sequence, ready
    /// for the sublayer's input LayerNorm.
    pub collapsed: HirNodeId,
    /// `[seq, H]` — how the sublayer output is broadcast back over streams.
    pub post: HirNodeId,
    /// `[seq, H, H]` — the (near) doubly-stochastic stream mixer.
    pub comb: HirNodeId,
}

/// `x · rsqrt(mean(x²) + eps)` over the last axis, with no learned gain.
fn unweighted_rms_norm(
    emit: &mut Emit<'_>,
    name: &str,
    x: HirNodeId,
    width: usize,
    eps: f32,
) -> HirNodeId {
    // `Op::RmsNorm` carries the gain and bias as operands, so a unit gain and a
    // zero bias reproduce the parameter-free form exactly while still riding the
    // backends' fused RMSNorm kernels.
    let ones = emit.synth_param(
        &format!("{name}.ones"),
        vec![1.0; width],
        Shape::new(&[width], DType::F32),
    );
    let zeros = emit.synth_zeros(&format!("{name}.zb"), width);
    let mut gb = HirMut::new(emit.hir());
    gb.rms_norm(x, ones, zeros, eps)
}

/// Emit the gates for one mHC site.
///
/// `prefix` is the GGUF stem — `blk.{i}.hc_attn` or `blk.{i}.hc_ffn` — and `x`
/// is the `[1, seq, H, hidden]` stream state.
pub fn emit_mhc_gates(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    d: MhcDims,
) -> Result<MhcGates> {
    let f = DType::F32;
    let (h, s) = (d.streams, d.seq);
    let (hi, si) = (h as i64, s as i64);
    let flat_width = h * d.hidden;

    let base = emit.load_param(&format!("{prefix}_base.weight"), false)?;
    let scale = emit.load_param(&format!("{prefix}_scale.weight"), false)?;

    let flat = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(x, vec![si, flat_width as i64])
    };
    let normed = unweighted_rms_norm(
        emit,
        &format!("{prefix}.in_norm"),
        flat,
        flat_width,
        d.norm_eps,
    );

    let eps = emit.synth_param(&format!("{prefix}.eps"), vec![d.eps], Shape::new(&[1], f));
    let two = emit.synth_param(&format!("{prefix}.two"), vec![2.0], Shape::new(&[1], f));

    // GGUF stores `fn` as [mix, H·hidden]. It is a plain 2-D projection, so it
    // goes through `linear` and takes the packed path when the source offers one.
    let mixed = crate::common::linear(emit, &format!("{prefix}_fn.weight"), normed)?; // [seq, mix]

    let mut gb = HirMut::new(emit.hir());

    // `scale` is [3]: one learned multiplier per output group.
    let s_pre = gb.narrow_(scale, 0, 0, 1);
    let s_post = gb.narrow_(scale, 0, 1, 1);
    let s_comb = gb.narrow_(scale, 0, 2, 1);

    let pre_w = gb.narrow_(mixed, 1, 0, h);
    let post_w = gb.narrow_(mixed, 1, h, h);
    let comb_w = gb.narrow_(mixed, 1, 2 * h, h * h);
    let pre_b = gb.narrow_(base, 0, 0, h);
    let post_b = gb.narrow_(base, 0, h, h);
    let comb_b = gb.narrow_(base, 0, 2 * h, h * h);

    // pre = σ(w·scale + b) + ε
    let pre = {
        let t = gb.mul(pre_w, s_pre);
        let t = gb.add(t, pre_b);
        let t = gb.add_node(
            Op::Activation(Activation::Sigmoid),
            vec![t],
            Shape::new(&[s, h], f),
        );
        gb.add(t, eps)
    };

    // post = 2·σ(w·scale + b)   — no epsilon here, and the coefficient is 2.
    let post = {
        let t = gb.mul(post_w, s_post);
        let t = gb.add(t, post_b);
        let t = gb.add_node(
            Op::Activation(Activation::Sigmoid),
            vec![t],
            Shape::new(&[s, h], f),
        );
        gb.mul(t, two)
    };

    // comb = Sinkhorn(softmax(w·scale + b))
    let comb = {
        let logits = gb.mul(comb_w, s_comb);
        let logits = gb.reshape_(logits, vec![si, hi, hi]);
        let b3 = gb.reshape_(comb_b, vec![1, hi, hi]);
        let logits = gb.add(logits, b3);
        let sm = gb.sm(logits, -1);
        let mut c = gb.add(sm, eps);
        // The reference normalizes columns once *before* the loop, then runs
        // `iters - 1` (row, column) rounds.
        c = sinkhorn_axis(&mut gb, c, 1, eps);
        for _ in 1..d.sinkhorn_iters.max(1) {
            c = sinkhorn_axis(&mut gb, c, 2, eps);
            c = sinkhorn_axis(&mut gb, c, 1, eps);
        }
        c
    };

    // collapsed = Σ_h pre[h] · x[h]
    let collapsed = {
        let p = gb.reshape_(pre, vec![1, si, hi, 1]);
        let w = gb.mul(x, p);
        let summed = gb.sum(w, vec![2], false); // [1, seq, hidden]
        gb.reshape_(summed, vec![1, si, d.hidden as i64])
    };

    Ok(MhcGates {
        collapsed,
        post,
        comb,
    })
}

/// One Sinkhorn half-step: `c / (sum(c, axis) + eps)`.
fn sinkhorn_axis(gb: &mut HirMut<'_>, c: HirNodeId, axis: usize, eps: HirNodeId) -> HirNodeId {
    let s = gb.sum(c, vec![axis], true);
    let s = gb.add(s, eps);
    gb.div(c, s)
}

/// Re-expand a sublayer output back over the residual streams:
/// `post ⊗ y + combᵀ · residual`.
///
/// * `y` — `[1, seq, hidden]`, the sublayer's output.
/// * `residual` — `[1, seq, H, hidden]`, the stream state *before* the site.
pub fn emit_mhc_expand(
    emit: &mut Emit<'_>,
    y: HirNodeId,
    residual: HirNodeId,
    gates: MhcGates,
    d: MhcDims,
) -> HirNodeId {
    let (h, s) = (d.streams, d.seq);
    let (hi, si, di) = (h as i64, s as i64, d.hidden as i64);

    let mut gb = HirMut::new(emit.hir());
    // post[.., h, 1] * y[.., 1, d] → [1, seq, H, hidden]
    let p = gb.reshape_(gates.post, vec![1, si, hi, 1]);
    let y4 = gb.reshape_(y, vec![1, si, 1, di]);
    let placed = gb.mul(p, y4);

    // combᵀ @ residual, batched over seq.
    let comb_t = gb.transpose_(gates.comb, vec![0, 2, 1]); // [seq, H, H]
    let res3 = gb.reshape_(residual, vec![si, hi, di]);
    let mixed = gb.mm(comb_t, res3); // [seq, H, hidden]
    let mixed = gb.reshape_(mixed, vec![1, si, hi, di]);

    gb.add(placed, mixed)
}

/// The final stream collapse before the output norm — `Glm5NextTextHyperHead`
/// is an unweighted mean, *not* a learned reduction.
pub fn emit_mhc_head(emit: &mut Emit<'_>, x: HirNodeId, d: MhcDims) -> HirNodeId {
    let mut gb = HirMut::new(emit.hir());
    let m = gb.mean(x, vec![2], false);
    gb.reshape_(m, vec![1, d.seq as i64, d.hidden as i64])
}

/// Broadcast the embedding into `H` identical residual streams — the shape the
/// first decoder layer expects.
pub fn emit_mhc_split(emit: &mut Emit<'_>, x: HirNodeId, d: MhcDims) -> HirNodeId {
    let (hi, si, di) = (d.streams as i64, d.seq as i64, d.hidden as i64);
    let mut gb = HirMut::new(emit.hir());
    let x4 = gb.reshape_(x, vec![1, si, 1, di]);
    gb.expand_(x4, vec![1, si, hi, di])
}
