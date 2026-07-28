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

//! GLM-4.6 attention: GQA with **partial** NeoX RoPE (rope covers only the
//! `rotary_dim` prefix of each head) and optional per-head qk RMSNorm.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, RopeStyle, Shape};

pub const ROPE_COS: &str = "rope_cos";
pub const ROPE_SIN: &str = "rope_sin";

#[derive(Debug, Clone, Copy)]
pub struct GlmAttnDims {
    pub hidden: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub eps: f32,
    pub seq: usize,
    pub attention_bias: bool,
    pub use_qk_norm: bool,
}

fn proj(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId, bias: bool) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let b = if bias {
        Some(emit.load_param(&format!("{prefix}.bias"), false)?)
    } else {
        None
    };
    let mut gb = HirMut::new(emit.hir());
    let mm = gb.mm(x, w);
    Ok(match b {
        Some(b) => gb.add(mm, b),
        None => mm,
    })
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

/// Per-head qk RMSNorm over `head_dim`.
fn qk_norm(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    seq: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<HirNodeId> {
    let g = emit.load_param(&format!("{key}.weight"), false)?;
    let zb = emit.synth_param(
        &format!("{key}.zb"),
        vec![0.0; head_dim],
        Shape::new(&[head_dim], DType::F32),
    );
    let mut gb = HirMut::new(emit.hir());
    let xr = gb.reshape_(x, vec![1, (seq * heads) as i64, head_dim as i64]);
    let n = gb.rms_norm(xr, g, zb, eps);
    Ok(gb.reshape_(n, vec![1, seq as i64, (heads * head_dim) as i64]))
}

/// Emit the attention sub-block for `model.layers.{i}.self_attn` (`prefix`).
pub fn emit_glm_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    normed: HirNodeId,
    d: GlmAttnDims,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let q_dim = d.num_heads * d.head_dim;
    let group = d.num_heads / d.num_kv_heads;

    let mut q = proj(emit, &format!("{prefix}.q_proj"), normed, d.attention_bias)?;
    let mut k = proj(emit, &format!("{prefix}.k_proj"), normed, d.attention_bias)?;
    let v = proj(emit, &format!("{prefix}.v_proj"), normed, d.attention_bias)?;

    if d.use_qk_norm {
        q = qk_norm(
            emit,
            &format!("{prefix}.q_norm"),
            q,
            d.seq,
            d.num_heads,
            d.head_dim,
            d.eps,
        )?;
        k = qk_norm(
            emit,
            &format!("{prefix}.k_norm"),
            k,
            d.seq,
            d.num_kv_heads,
            d.head_dim,
            d.eps,
        )?;
    }

    let cos = emit.flow_input(ROPE_COS)?.hir_id();
    let sin = emit.flow_input(ROPE_SIN)?.hir_id();
    let attn = {
        let mut gb = HirMut::new(emit.hir());
        let qr = gb.rope_n_styled(q, cos, sin, d.head_dim, d.rotary_dim, RopeStyle::NeoX);
        let kr = gb.rope_n_styled(k, cos, sin, d.head_dim, d.rotary_dim, RopeStyle::NeoX);
        let k_rep = repeat_kv(&mut gb, kr, d.num_kv_heads, d.head_dim, group);
        let v_rep = repeat_kv(&mut gb, v, d.num_kv_heads, d.head_dim, group);
        gb.attention_kind(
            qr,
            k_rep,
            v_rep,
            d.num_heads,
            d.head_dim,
            MaskKind::Causal,
            Shape::new(&[1, d.seq, q_dim], f),
        )
    };
    proj(emit, &format!("{prefix}.o_proj"), attn, false)
}
