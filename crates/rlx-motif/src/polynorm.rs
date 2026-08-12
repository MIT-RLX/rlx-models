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

//! PolyNorm — Motif's trainable polynomial activation
//! ([PolyCom](https://arxiv.org/html/2411.03884v1)), used in place of SiLU in
//! every Motif FFN:
//!
//! ```text
//!   n(y)      = y / √(mean(y²) + 1e-6)                      (over the last axis)
//!   poly(x)   = w₀·n(x³) + w₁·n(x²) + w₂·n(x) + b
//!   out       = poly(clamp(gate)) · clamp(up) · output_scale
//! ```
//!
//! `w = σ(weight)` and (for the MoE experts only) `b = clamp(bias, ±bias_clamp)`
//! are folded host-side by [`crate::weights::prepare_checkpoint`] into a single
//! `[…, 4]` coefficient row `[w₀, w₁, w₂, b]`, so the graph never sees a sigmoid
//! on a parameter — and the per-expert variant becomes one `Gather` of that
//! table by routed expert id.
//!
//! Two upstream variants differ in their clamps, and the difference is real:
//! `PolyNormTorch` (dense MLP / shared expert) clamps only `gate` and `up`,
//! while `GroupedPolyNorm` (routed experts) *additionally* clamps the bias and
//! the product. [`PolyNormSpec::clamp_result`] selects between them.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{BinaryOp, Op};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

/// Which PolyNorm flavour to emit.
#[derive(Debug, Clone, Copy)]
pub struct PolyNormSpec {
    /// `_norm`'s epsilon ([`crate::config::POLYNORM_EPS`]).
    pub eps: f32,
    /// `config.hidden_clamp` — applied to `gate`, `up` (both variants) and the
    /// product (`clamp_result` only).
    pub hidden_clamp: Option<f32>,
    /// `config.polynorm_output_scale`.
    pub output_scale: f32,
    /// `GroupedPolyNorm` clamps `poly · up` before scaling; `PolyNormTorch`
    /// does not.
    pub clamp_result: bool,
}

/// `min(max(x, lo), hi)`.
fn clamp_to(gb: &mut HirMut<'_>, x: HirNodeId, lo: HirNodeId, hi: HirNodeId) -> HirNodeId {
    let s = gb.shape(x).clone();
    let up = gb.add_node(Op::Binary(BinaryOp::Max), vec![x, lo], s.clone());
    gb.add_node(Op::Binary(BinaryOp::Min), vec![up, hi], s)
}

/// Emit `poly(gate) · up · output_scale`.
///
/// * `name` — unique prefix for the synthesized norm gain/bias and clamp bounds.
/// * `coeff` — `[…, 4]` node holding `[w₀, w₁, w₂, b]`, broadcastable against
///   `gate` once its last axis is sliced to width 1 (`[1, 4]` for a shared
///   activation, `[rows, 4]` for the per-expert gather).
/// * `width` — size of `gate`'s last axis (the norm reduces over it).
pub fn emit_poly_norm_mul(
    emit: &mut Emit<'_>,
    name: &str,
    gate: HirNodeId,
    up: HirNodeId,
    coeff: HirNodeId,
    width: usize,
    spec: PolyNormSpec,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let ones = emit.synth_param(
        &format!("{name}.poly.gain"),
        vec![1.0; width],
        Shape::new(&[width], f),
    );
    let zeros = emit.synth_param(
        &format!("{name}.poly.zb"),
        vec![0.0; width],
        Shape::new(&[width], f),
    );
    let bounds = spec.hidden_clamp.map(|c| {
        (
            emit.synth_param(&format!("{name}.poly.lo"), vec![-c], Shape::new(&[1], f)),
            emit.synth_param(&format!("{name}.poly.hi"), vec![c], Shape::new(&[1], f)),
        )
    });
    let scale = (spec.output_scale != 1.0).then(|| {
        emit.synth_param(
            &format!("{name}.poly.scale"),
            vec![spec.output_scale],
            Shape::new(&[1], f),
        )
    });

    let mut gb = HirMut::new(emit.hir());
    let (gate, up) = match bounds {
        Some((lo, hi)) => (
            clamp_to(&mut gb, gate, lo, hi),
            clamp_to(&mut gb, up, lo, hi),
        ),
        None => (gate, up),
    };

    let axis = gb.shape(coeff).rank() - 1;
    let w0 = gb.narrow_(coeff, axis, 0, 1);
    let w1 = gb.narrow_(coeff, axis, 1, 1);
    let w2 = gb.narrow_(coeff, axis, 2, 1);
    let bias = gb.narrow_(coeff, axis, 3, 1);

    let x2 = gb.mul(gate, gate);
    let x3 = gb.mul(x2, gate);
    let n3 = gb.rms_norm(x3, ones, zeros, spec.eps);
    let n2 = gb.rms_norm(x2, ones, zeros, spec.eps);
    let n1 = gb.rms_norm(gate, ones, zeros, spec.eps);

    let t3 = gb.mul(n3, w0);
    let t2 = gb.mul(n2, w1);
    let t1 = gb.mul(n1, w2);
    let poly = gb.add(t3, t2);
    let poly = gb.add(poly, t1);
    let poly = gb.add(poly, bias);

    let out = gb.mul(poly, up);
    let out = match (spec.clamp_result, bounds) {
        (true, Some((lo, hi))) => clamp_to(&mut gb, out, lo, hi),
        _ => out,
    };
    Ok(match scale {
        Some(s) => gb.mul(out, s),
        None => out,
    })
}

/// Host-side reference for [`emit_poly_norm_mul`] on one row — the shape the
/// tests compare against, and the exact arithmetic of `GroupedPolyNorm`
/// (`clamp_result = true`) / `PolyNormTorch` (`false`).
pub fn poly_norm_row(gate: &[f32], up: &[f32], coeff: [f32; 4], spec: PolyNormSpec) -> Vec<f32> {
    let clamp = |v: f32| match spec.hidden_clamp {
        Some(c) => v.clamp(-c, c),
        None => v,
    };
    let g: Vec<f32> = gate.iter().map(|&v| clamp(v)).collect();
    let u: Vec<f32> = up.iter().map(|&v| clamp(v)).collect();
    let norm = |p: &dyn Fn(f32) -> f32| -> Vec<f32> {
        let vals: Vec<f32> = g.iter().map(|&v| p(v)).collect();
        let ms = vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32;
        let inv = 1.0 / (ms + spec.eps).sqrt();
        vals.into_iter().map(|v| v * inv).collect()
    };
    let n3 = norm(&|v: f32| v * v * v);
    let n2 = norm(&|v: f32| v * v);
    let n1 = norm(&|v: f32| v);
    (0..g.len())
        .map(|i| {
            let poly = coeff[0] * n3[i] + coeff[1] * n2[i] + coeff[2] * n1[i] + coeff[3];
            let mut out = poly * u[i];
            if spec.clamp_result {
                out = clamp(out);
            }
            out * spec.output_scale
        })
        .collect()
}
