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

//! Jamba attention: plain causal GQA with **no RoPE** (positional information
//! comes from the interleaved Mamba layers). Bias-free q/k/v/o.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

#[derive(Debug, Clone, Copy)]
pub struct JambaAttnDims {
    pub hidden: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub seq: usize,
}

fn linear(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

fn repeat_kv(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    num_kv: usize,
    head_dim: usize,
    group: usize,
) -> HirNodeId {
    if group == 1 {
        return x;
    }
    let last = gb.shape(x).rank() - 1;
    let mut pieces = Vec::with_capacity(num_kv * group);
    for hh in 0..num_kv {
        let s = gb.narrow_(x, last, hh * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(s);
        }
    }
    gb.concat_(pieces, last)
}

/// Emit the attention sub-block for `model.layers.{i}.self_attn` (`prefix`).
pub fn emit_jamba_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    normed: HirNodeId,
    d: JambaAttnDims,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let q_dim = d.num_heads * d.head_dim;
    let group = d.num_heads / d.num_kv_heads;

    let q = linear(emit, &format!("{prefix}.q_proj"), normed)?;
    let k = linear(emit, &format!("{prefix}.k_proj"), normed)?;
    let v = linear(emit, &format!("{prefix}.v_proj"), normed)?;
    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let k_rep = repeat_kv(&mut gb, k, d.num_kv_heads, d.head_dim, group);
        let v_rep = repeat_kv(&mut gb, v, d.num_kv_heads, d.head_dim, group);
        gb.attention_kind(
            q,
            k_rep,
            v_rep,
            d.num_heads,
            d.head_dim,
            MaskKind::Causal,
            Shape::new(&[1, d.seq, q_dim], f),
        )
    };
    linear(emit, &format!("{prefix}.o_proj"), attn)
}
