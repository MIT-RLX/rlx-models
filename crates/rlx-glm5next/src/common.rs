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

//! Small shared emitters.

use anyhow::Result;
use rlx_flow::{Emit, GgufPackedBank, GgufPackedLinear};
use rlx_ir::hir::HirMut;
use rlx_ir::op::{BinaryOp, Op};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

/// `x @ Wᵀ` for a GGUF-stored weight, packed when the source offers it.
///
/// Two paths, chosen by the [`rlx_flow::WeightSource`]:
///
/// * **Packed.** If the source hands out a GGUF quant blob for `key` (i.e. the
///   flow was built through `rlx_core::flow_bridge::PackedWeightLoaderSource`),
///   the U8 blob is registered as a graph param and the projection becomes a
///   fused `Op::DequantMatMul` — the f32 weight is never materialized. For
///   GLM-5.3-Flash that is most of the model: at `UD-IQ1_S` the 2-D projections
///   are `Q5_K` / `Q6_K` / `Q8_0`, ~5.5 bits/weight against f32's 32.
/// * **Dense.** Otherwise the weight is loaded as f32. The GGUF loader reverses
///   GGML's innermost-first dims, so it arrives in torch `[out, in]` order and
///   `transpose = true` turns it into the `[in, out]` operand `mm` contracts
///   against.
///
/// The packed blob is already `[out_dim, in_dim]` and `DequantMatMul`
/// transposes internally, so the two agree without the caller knowing which
/// ran. This mirrors `FlowCtx::linear`, which hand-rolled `HirMut` blocks like
/// this crate's cannot use directly.
pub fn linear(emit: &mut Emit<'_>, key: &str, x: HirNodeId) -> Result<HirNodeId> {
    // A projection already registered by an earlier stage: reuse the param
    // rather than re-`take_packed`-ing a consuming loader.
    if let Some(&(wq, out_dim, scheme)) = emit.state.packed_linears.get(key) {
        let mut gb = HirMut::new(emit.hir());
        return Ok(emit_dequant(&mut gb, x, wq, out_dim, scheme));
    }
    if let Some(packed) = emit.weights.take_packed(key)? {
        let GgufPackedLinear {
            w_q,
            scheme,
            out_dim,
            ..
        } = packed;
        // `\0q` keeps the blob's param name distinct from the f32 key.
        let wq_key = format!("{key}\0q");
        let wq = emit
            .hir()
            .param(&wq_key, Shape::new(&[w_q.len()], DType::U8));
        emit.state.typed_params.push((wq_key, w_q, DType::U8));
        emit.state
            .packed_linears
            .insert(key.to_string(), (wq, out_dim, scheme));
        let mut gb = HirMut::new(emit.hir());
        return Ok(emit_dequant(&mut gb, x, wq, out_dim, scheme));
    }
    let w = emit.load_param(key, true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

/// `DequantMatMul` with the output shape being `x`'s with the last dim replaced
/// by `out_dim` — leading dims (including a dynamic `m`) are preserved, so one
/// packed graph runs at any sequence length.
fn emit_dequant(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    wq: HirNodeId,
    out_dim: usize,
    scheme: QuantScheme,
) -> HirNodeId {
    let mut dims: Vec<usize> = gb
        .shape(x)
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let last = dims.len() - 1;
    dims[last] = out_dim;
    let out = Shape::new(&dims, DType::F32);
    gb.0.dequant_matmul(x, wq, None, None, scheme, out)
}

/// RMSNorm with a GGUF gain and no bias.
pub fn rms_norm(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    width: usize,
    eps: f32,
) -> Result<HirNodeId> {
    let g = emit.load_param(&format!("{key}.weight"), false)?;
    let zb = emit.synth_zeros(&format!("{key}.zb"), width);
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.rms_norm(x, g, zb, eps))
}

/// `min(x, hi)` — the reference's `gate.clamp(min=None, max=swiglu_limit)`.
pub fn clamp_max(gb: &mut HirMut<'_>, x: HirNodeId, hi: HirNodeId) -> HirNodeId {
    let s = gb.shape(x).clone();
    gb.add_node(Op::Binary(BinaryOp::Min), vec![x, hi], s)
}

/// `clamp(x, lo, hi)` — the reference's `up.clamp(-limit, limit)`.
pub fn clamp_both(gb: &mut HirMut<'_>, x: HirNodeId, lo: HirNodeId, hi: HirNodeId) -> HirNodeId {
    let s = gb.shape(x).clone();
    let up = gb.add_node(Op::Binary(BinaryOp::Max), vec![x, lo], s.clone());
    gb.add_node(Op::Binary(BinaryOp::Min), vec![up, hi], s)
}

/// A scalar `[1]` synthetic parameter.
pub fn scalar(emit: &mut Emit<'_>, name: &str, v: f32) -> HirNodeId {
    emit.synth_param(name, vec![v], Shape::new(&[1], DType::F32))
}

/// Clamped SwiGLU: `silu(min(gate, L)) · clamp(up, −L, L)`.
///
/// With `limit = None` this is the plain `silu(gate) · up`.
pub fn clamped_swiglu(
    emit: &mut Emit<'_>,
    name: &str,
    gate: HirNodeId,
    up: HirNodeId,
    limit: Option<f32>,
) -> HirNodeId {
    let bounds = limit.filter(|l| l.is_finite()).map(|l| {
        (
            scalar(emit, &format!("{name}.swiglu_lo"), -l),
            scalar(emit, &format!("{name}.swiglu_hi"), l),
        )
    });
    let mut gb = HirMut::new(emit.hir());
    let (gate, up) = match bounds {
        Some((lo, hi)) => (
            clamp_max(&mut gb, gate, hi),
            clamp_both(&mut gb, up, lo, hi),
        ),
        None => (gate, up),
    };
    let a = gb.silu(gate);
    gb.mul(a, up)
}

/// A packed MoE expert bank registered as a graph param, if the source has one.
///
/// The grouped sibling of [`linear`]: returns the U8 blob's node plus the dims
/// `Op::DequantGroupedMatMul` needs. `None` means the caller should fall back to
/// an F32 `GroupedMatMul` over a dequantized bank.
///
/// GGUF's `[experts, out, in]` order is already the op's slab layout, so unlike
/// the F32 path there is no `[E, N, K] → [E, K, N]` transpose — which also
/// avoids constant-folding a second copy of the bank into the arena.
pub fn packed_bank(
    emit: &mut Emit<'_>,
    key: &str,
) -> Result<Option<(HirNodeId, QuantScheme, usize)>> {
    if let Some(&(wq, out_dim, scheme)) = emit.state.packed_linears.get(key) {
        return Ok(Some((wq, scheme, out_dim)));
    }
    let Some(bank) = emit.weights.take_packed_bank(key)? else {
        return Ok(None);
    };
    let GgufPackedBank {
        w_q,
        scheme,
        out_dim,
        ..
    } = bank;
    let wq_key = format!("{key}\0q");
    let wq = emit
        .hir()
        .param(&wq_key, Shape::new(&[w_q.len()], DType::U8));
    emit.state.typed_params.push((wq_key, w_q, DType::U8));
    // Reuse `packed_linears` as the registry: a bank is keyed the same way and
    // only ever read back as (node, out_dim, scheme).
    emit.state
        .packed_linears
        .insert(key.to_string(), (wq, out_dim, scheme));
    Ok(Some((wq, scheme, out_dim)))
}
