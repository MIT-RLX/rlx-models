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

//! Shared in-graph primitives for MiniMax-M3: Gemma `(1+w)` RMSNorm (whole-row
//! and per-head), and the SwiGLU-OAI (gpt-oss clamped) activation. None of these
//! exist as a single rlx op, so they are composed from `rms_norm` / `Clamp` /
//! `Sigmoid` / mul.

use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

/// Gemma-style RMSNorm over the last dim: `x / rms(x) * (1 + weight)`.
///
/// `key` is the weight tensor key (`.weight` appended). `dim` is the normalized
/// width. A zero-beta and a ones vector are synthesized (keyed off `key`).
pub fn gemma_rmsnorm(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    dim: usize,
    eps: f32,
) -> anyhow::Result<HirNodeId> {
    let w = emit.load_param(&format!("{key}.weight"), false)?;
    let ones = emit.synth_param(
        &format!("{key}.m3_ones"),
        vec![1.0f32; dim],
        Shape::new(&[dim], DType::F32),
    );
    let zb = emit.synth_param(
        &format!("{key}.m3_zb"),
        vec![0.0f32; dim],
        Shape::new(&[dim], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    let one_plus_w = gb.add(ones, w);
    Ok(gb.rms_norm(x, one_plus_w, zb, eps))
}

/// Per-head Gemma RMSNorm: reshape `[b,s,heads*head_dim]` → `[b*s*heads, head_dim]`,
/// normalize over `head_dim`, reshape back. Weight is shared across heads.
pub fn per_head_gemma_rmsnorm(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> anyhow::Result<HirNodeId> {
    let w = emit.load_param(&format!("{key}.weight"), false)?;
    let ones = emit.synth_param(
        &format!("{key}.m3_ones"),
        vec![1.0f32; head_dim],
        Shape::new(&[head_dim], DType::F32),
    );
    let zb = emit.synth_param(
        &format!("{key}.m3_zb"),
        vec![0.0f32; head_dim],
        Shape::new(&[head_dim], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    let one_plus_w = gb.add(ones, w);
    let flat = (batch * seq * heads) as i64;
    let r = gb.reshape_(x, vec![flat, head_dim as i64]);
    let n = gb.rms_norm(r, one_plus_w, zb, eps);
    Ok(gb.reshape_(n, vec![batch as i64, seq as i64, (heads * head_dim) as i64]))
}

/// SwiGLU-OAI over separate `gate`/`up` activations, each `[rows, inter]`:
/// `gate=clamp(g, max=limit)`, `up=clamp(u, -limit, limit)`,
/// `glu = gate·sigmoid(alpha·gate)`, `out = (up + 1)·glu`. Returns `[rows, inter]`.
///
/// `alpha_node` is a `[1]` const holding `swiglu_alpha`; `one_node` a `[1]` const
/// holding `1.0` (both synthesized once by the caller and reused).
pub fn swiglu_oai(
    gb: &mut HirMut,
    gate: HirNodeId,
    up: HirNodeId,
    rows: usize,
    inter: usize,
    limit: f32,
    alpha_node: HirNodeId,
    one_node: HirNodeId,
) -> HirNodeId {
    let f = DType::F32;
    let sh = Shape::new(&[rows, inter], f);
    let gate_c = gb.add_node(
        Op::Clamp {
            min: f32::NEG_INFINITY,
            max: limit,
        },
        vec![gate],
        sh.clone(),
    );
    let up_c = gb.add_node(
        Op::Clamp {
            min: -limit,
            max: limit,
        },
        vec![up],
        sh.clone(),
    );
    let ag = gb.mul(gate_c, alpha_node);
    let sig = gb.activation(Activation::Sigmoid, ag, sh.clone());
    let glu = gb.mul(gate_c, sig);
    let up1 = gb.add(up_c, one_node);
    gb.mul(up1, glu)
}
