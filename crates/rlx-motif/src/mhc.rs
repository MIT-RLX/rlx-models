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

//! MHC — Manifold-constrained Hyper-Connections (<https://arxiv.org/abs/2512.24880>).
//!
//! Motif-3 does not carry a single residual stream: the hidden state is
//! `[batch, seq, E, dim]` with `E = mhc_expansion_rate` parallel streams, and
//! every sublayer is wrapped in a learned, input-dependent mixing of them:
//!
//! ```text
//!   x_norm         = RMSNorm_{E·D}(flatten(x))
//!   h_pre  [.,E]   = σ(clamp(α_pre ·W_pre (x_norm) + b_pre , ±10))
//!   h_post [.,E]   = c·σ(clamp(α_post·W_post(x_norm) + b_post, ±10))
//!   h_res  [.,E,E] = Sinkhorn(α_res·W_res(x_norm) + b_res)     ← doubly stochastic
//!
//!   branch_in      = Σ_e h_pre[e]·x[e]                          (E streams → 1)
//!   x'             = h_res · x + h_post ⊗ branch_out            (1 → E streams)
//! ```
//!
//! `Sinkhorn` is 20 alternating row/column normalizations of `exp(clamp(m,±20))`,
//! emitted inline — `E` is 4, so the whole thing is 120 tiny nodes per gate and
//! runs on every backend without a custom kernel.
//!
//! Two MHC blocks live in each decoder layer (`mhc_attn`, `mhc_ffn`), and their
//! RMSNorm epsilon is hardcoded `1e-6` upstream — *not* `config.rms_norm_eps`.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, BinaryOp, Op};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::config::{MHC_GATE_CLAMP, MHC_NORM_EPS, MHC_SINKHORN_CLAMP, MHC_SINKHORN_FLOOR};

#[derive(Debug, Clone, Copy)]
pub struct MhcDims {
    pub hidden: usize,
    /// `mhc_expansion_rate` — number of parallel residual streams.
    pub expansion: usize,
    pub sinkhorn_iters: usize,
    /// `1 + mhc_h_post_alpha_end` (`MotifDecoderLayer` passes this in; the
    /// `MHCLayer` default of 2.0 is never used by Motif-3).
    pub h_post_coeff: f32,
    pub seq: usize,
}

/// The three gates of one MHC block.
#[derive(Debug, Clone, Copy)]
pub struct MhcGates {
    /// `[1, seq, E]` — reduction weights, one per stream.
    pub h_pre: HirNodeId,
    /// `[1, seq, E]` — broadcast weights for the sublayer output.
    pub h_post: HirNodeId,
    /// `[1, seq, E, E]` — doubly stochastic stream mixing matrix.
    pub h_res: HirNodeId,
}

fn clamp_to(gb: &mut HirMut<'_>, x: HirNodeId, lo: HirNodeId, hi: HirNodeId) -> HirNodeId {
    let s = gb.shape(x).clone();
    let up = gb.add_node(Op::Binary(BinaryOp::Max), vec![x, lo], s.clone());
    gb.add_node(Op::Binary(BinaryOp::Min), vec![up, hi], s)
}

/// `σ(clamp(α · W(x_norm) + b, ±10))` → `[1, seq, E]`.
#[allow(clippy::too_many_arguments)]
fn sigmoid_gate(
    gb: &mut HirMut<'_>,
    x_norm: HirNodeId,
    w: HirNodeId,
    alpha: HirNodeId,
    bias: HirNodeId,
    (lo, hi): (HirNodeId, HirNodeId),
    (seq, e): (usize, usize),
) -> HirNodeId {
    let lin = gb.mm(x_norm, w);
    let scaled = gb.mul(lin, alpha);
    let biased = gb.add(scaled, bias);
    let clamped = clamp_to(gb, biased, lo, hi);
    gb.add_node(
        Op::Activation(Activation::Sigmoid),
        vec![clamped],
        Shape::new(&[1, seq, e], DType::F32),
    )
}

/// Emit `MHCLayer.forward` for `prefix` (`…layers.{i}.mhc_attn` / `.mhc_ffn`)
/// over a `[1, seq, E, hidden]` hidden state.
pub fn emit_mhc_gates(
    emit: &mut Emit<'_>,
    prefix: &str,
    x: HirNodeId,
    d: MhcDims,
) -> Result<MhcGates> {
    let f = DType::F32;
    let (e, s) = (d.expansion, d.seq);
    let (ei, si) = (e as i64, s as i64);
    let flat = e * d.hidden;

    let gamma = emit.load_param(&format!("{prefix}.rms_norm.weight"), false)?;
    let zb = emit.synth_param(
        &format!("{prefix}.rms_norm.zb"),
        vec![0.0; flat],
        Shape::new(&[flat], f),
    );
    let w_pre = emit.load_param(&format!("{prefix}.proj_pre.weight"), true)?;
    let w_post = emit.load_param(&format!("{prefix}.proj_post.weight"), true)?;
    let w_res = emit.load_param(&format!("{prefix}.proj_res.weight"), true)?;
    let b_pre = emit.load_param(&format!("{prefix}.bias_pre"), false)?;
    let b_post = emit.load_param(&format!("{prefix}.bias_post"), false)?;
    let b_res = emit.load_param(&format!("{prefix}.bias_res"), false)?;
    let a_pre = emit.load_param(&format!("{prefix}.alpha_pre"), false)?;
    let a_post = emit.load_param(&format!("{prefix}.alpha_post"), false)?;
    let a_res = emit.load_param(&format!("{prefix}.alpha_res"), false)?;

    let gate_lo = emit.synth_param(
        &format!("{prefix}.clamp.lo"),
        vec![-MHC_GATE_CLAMP],
        Shape::new(&[1], f),
    );
    let gate_hi = emit.synth_param(
        &format!("{prefix}.clamp.hi"),
        vec![MHC_GATE_CLAMP],
        Shape::new(&[1], f),
    );
    let sk_lo = emit.synth_param(
        &format!("{prefix}.sinkhorn.lo"),
        vec![-MHC_SINKHORN_CLAMP],
        Shape::new(&[1], f),
    );
    let sk_hi = emit.synth_param(
        &format!("{prefix}.sinkhorn.hi"),
        vec![MHC_SINKHORN_CLAMP],
        Shape::new(&[1], f),
    );
    let floor = emit.synth_param(
        &format!("{prefix}.sinkhorn.floor"),
        vec![MHC_SINKHORN_FLOOR],
        Shape::new(&[1], f),
    );
    let coeff = (d.h_post_coeff != 1.0).then(|| {
        emit.synth_param(
            &format!("{prefix}.h_post.coeff"),
            vec![d.h_post_coeff],
            Shape::new(&[1], f),
        )
    });

    let mut gb = HirMut::new(emit.hir());
    let x_flat = gb.reshape_(x, vec![1, si, flat as i64]);
    let x_norm = gb.rms_norm(x_flat, gamma, zb, MHC_NORM_EPS);

    let bounds = (gate_lo, gate_hi);
    let h_pre = sigmoid_gate(&mut gb, x_norm, w_pre, a_pre, b_pre, bounds, (s, e));
    let h_post = sigmoid_gate(&mut gb, x_norm, w_post, a_post, b_post, bounds, (s, e));
    let h_post = match coeff {
        Some(c) => gb.mul(h_post, c),
        None => h_post,
    };

    // ── Sinkhorn-Knopp → doubly stochastic [1, s, E, E] ──
    let res = gb.mm(x_norm, w_res);
    let res = gb.reshape_(res, vec![1, si, ei, ei]);
    let res = gb.mul(res, a_res);
    let res = gb.add(res, b_res);
    let res = clamp_to(&mut gb, res, sk_lo, sk_hi);
    let mut m = gb.exp(res);
    for _ in 0..d.sinkhorn_iters {
        let rows = gb.sum(m, vec![3], true); // [1,s,E,1]
        let rows = gb.add_node(
            Op::Binary(BinaryOp::Max),
            vec![rows, floor],
            Shape::new(&[1, s, e, 1], f),
        );
        m = gb.div(m, rows);
        let cols = gb.sum(m, vec![2], true); // [1,s,1,E]
        let cols = gb.add_node(
            Op::Binary(BinaryOp::Max),
            vec![cols, floor],
            Shape::new(&[1, s, 1, e], f),
        );
        m = gb.div(m, cols);
    }

    Ok(MhcGates {
        h_pre,
        h_post,
        h_res: m,
    })
}

/// `MHCLayer.apply_h_pre` — collapse the `E` streams into the sublayer input:
/// `[1, s, E, D] → [1, s, D]`.
pub fn apply_h_pre(gb: &mut HirMut<'_>, x: HirNodeId, h_pre: HirNodeId, d: MhcDims) -> HirNodeId {
    let w = gb.reshape_(h_pre, vec![1, d.seq as i64, d.expansion as i64, 1]);
    let scaled = gb.mul(x, w);
    gb.sum(scaled, vec![2], false)
}

/// `h_res · x + h_post ⊗ branch` — the MHC residual combine, `[1, s, E, D]`.
///
/// The `h_res` contraction is `einsum("bsij,bsjd->bsid")`, which is exactly a
/// batched matmul over the `[1, s]` leading axes.
pub fn combine(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    branch: HirNodeId,
    gates: MhcGates,
    d: MhcDims,
) -> HirNodeId {
    let (si, ei, di) = (d.seq as i64, d.expansion as i64, d.hidden as i64);
    let mixed = gb.mm(gates.h_res, x);
    let w = gb.reshape_(gates.h_post, vec![1, si, ei, 1]);
    let b = gb.reshape_(branch, vec![1, si, 1, di]);
    let posted = gb.mul(b, w);
    gb.add(mixed, posted)
}

/// Host-side Sinkhorn-Knopp reference for one `E × E` block — the tests compare
/// the emitted graph against this.
pub fn sinkhorn_reference(m: &[f32], e: usize, iters: usize) -> Vec<f32> {
    let mut out: Vec<f32> = m
        .iter()
        .map(|&v| v.clamp(-MHC_SINKHORN_CLAMP, MHC_SINKHORN_CLAMP).exp())
        .collect();
    for _ in 0..iters {
        for r in 0..e {
            let s = out[r * e..(r + 1) * e]
                .iter()
                .sum::<f32>()
                .max(MHC_SINKHORN_FLOOR);
            for c in 0..e {
                out[r * e + c] /= s;
            }
        }
        for c in 0..e {
            let s = (0..e)
                .map(|r| out[r * e + c])
                .sum::<f32>()
                .max(MHC_SINKHORN_FLOOR);
            for r in 0..e {
                out[r * e + c] /= s;
            }
        }
    }
    out
}
