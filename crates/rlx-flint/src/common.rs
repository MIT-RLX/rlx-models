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

//! Small shared emitters over HF-named weights.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

/// `x @ Wᵀ` for an HF `nn.Linear` weight stored `[out, in]`.
///
/// The weight loads transposed to `[in, out]` under the flow's
/// [`rlx_flow::WeightStorage`]: the per-layer graphs use `Auto`, so a bf16
/// checkpoint read through [`crate::CheckpointRef`] stays **bf16** in the arena
/// (its own values — lossless — at half the memory and bandwidth of f32) and
/// the matmul reads it directly. `x` may have any leading dims.
pub fn linear(emit: &mut Emit<'_>, key: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_matrix(key, true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

/// A `[1]` synthetic scalar.
pub fn scalar(emit: &mut Emit<'_>, name: &str, v: f32) -> HirNodeId {
    emit.synth_param(name, vec![v], Shape::new(&[1], DType::F32))
}

/// RMSNorm over the last axis with the checkpoint's gain used verbatim
/// (`x · rsqrt(mean(x²) + eps) · w` — no `1 +`).
pub fn rms_norm(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    width: usize,
    eps: f32,
) -> Result<HirNodeId> {
    let g = emit.load_param(key, false)?;
    let zb = emit.synth_zeros(&format!("{key}.zb"), width);
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.rms_norm(x, g, zb, eps))
}

/// Per-head RMSNorm on a packed `[1, seq, heads · head_dim]` tensor —
/// `apply_qk_norm`, one `[head_dim]` gain shared by every head.
pub fn per_head_rms(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<HirNodeId> {
    let g = emit.load_param(key, false)?;
    let zb = emit.synth_zeros(&format!("{key}.zb"), head_dim);
    let mut gb = HirMut::new(emit.hir());
    let seq = gb.shape(x).dim(1).unwrap_static() as i64;
    let r = gb.reshape_(x, vec![1, seq, heads as i64, head_dim as i64]);
    let n = gb.rms_norm(r, g, zb, eps);
    Ok(gb.reshape_(n, vec![1, seq, (heads * head_dim) as i64]))
}

/// `down(silu(gate(x)) · up(x))` — `Qwen2MoeMLP`, used for the dense layers,
/// the shared expert, and (via the checkpoint's per-expert names) nothing else:
/// routed experts go through [`crate::moe`]'s grouped GEMMs.
pub fn swiglu(emit: &mut Emit<'_>, prefix: &str, x2d: HirNodeId) -> Result<HirNodeId> {
    let g = linear(emit, &format!("{prefix}.gate_proj.weight"), x2d)?;
    let u = linear(emit, &format!("{prefix}.up_proj.weight"), x2d)?;
    let h = {
        let mut gb = HirMut::new(emit.hir());
        let a = gb.silu(g);
        gb.mul(a, u)
    };
    linear(emit, &format!("{prefix}.down_proj.weight"), h)
}
