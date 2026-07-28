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

//! Llama-4 self-attention (`Llama4TextAttention`).
//!
//! iRoPE: RoPE layers apply complex/interleaved rotary ([`RopeStyle::GptJ`])
//! then L2-normalize q/k per head (when `use_qk_norm`); NoPE layers skip both.
//! Temperature tuning and chunked attention are no-ops for `seq < chunk_size`
//! (v1 target), so all layers use plain causal attention with scale
//! `head_dim^-0.5`. The RoPE cos/sin `[seq, head_dim/2]` are fed as the shared
//! [`ROPE_COS`]/[`ROPE_SIN`] graph inputs.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, RopeStyle, Shape};

pub const ROPE_COS: &str = "rope_cos";
pub const ROPE_SIN: &str = "rope_sin";

#[derive(Debug, Clone, Copy)]
pub struct AttnDims {
    pub hidden: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub seq: usize,
}

fn linear(emit: &mut Emit<'_>, prefix: &str, x: HirNodeId) -> Result<HirNodeId> {
    let w = emit.load_param(&format!("{prefix}.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(x, w))
}

/// GQA repeat-interleave of key/value heads along the feature axis.
fn repeat_kv(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> HirNodeId {
    if group == 1 {
        return x;
    }
    let last = gb.shape(x).rank() - 1;
    let mut pieces = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let s = gb.narrow_(x, last, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(s);
        }
    }
    gb.concat_(pieces, last)
}

/// L2 norm over `head_dim` per head (RMSNorm with unit gain, no learned weight).
fn l2_norm(
    emit: &mut Emit<'_>,
    tag: &str,
    x: HirNodeId,
    seq: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> HirNodeId {
    let f = DType::F32;
    let ones = emit.synth_param(
        &format!("{tag}.l2ones"),
        vec![1.0; head_dim],
        Shape::new(&[head_dim], f),
    );
    let zeros = emit.synth_param(
        &format!("{tag}.l2zeros"),
        vec![0.0; head_dim],
        Shape::new(&[head_dim], f),
    );
    let dim = heads * head_dim;
    let mut gb = HirMut::new(emit.hir());
    let xr = gb.reshape_(x, vec![1, (seq * heads) as i64, head_dim as i64]);
    let n = gb.rms_norm(xr, ones, zeros, eps);
    gb.reshape_(n, vec![1, seq as i64, dim as i64])
}

/// Emit the attention sub-block for `model.layers.{i}.self_attn` (`prefix`),
/// consuming a pre-normed `[1, seq, hidden]` and producing `[1, seq, hidden]`.
pub fn emit_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    normed: HirNodeId,
    d: AttnDims,
    use_rope: bool,
    use_qk_norm: bool,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let q_dim = d.num_heads * d.head_dim;
    let group = d.num_heads / d.num_kv_heads;

    let mut q = linear(emit, &format!("{prefix}.q_proj"), normed)?;
    let mut k = linear(emit, &format!("{prefix}.k_proj"), normed)?;
    let v = linear(emit, &format!("{prefix}.v_proj"), normed)?;

    if use_rope {
        let cos = emit.flow_input(ROPE_COS)?.hir_id();
        let sin = emit.flow_input(ROPE_SIN)?.hir_id();
        let mut gb = HirMut::new(emit.hir());
        q = gb.rope_styled(q, cos, sin, d.head_dim, RopeStyle::GptJ);
        k = gb.rope_styled(k, cos, sin, d.head_dim, RopeStyle::GptJ);
    }
    if use_rope && use_qk_norm {
        q = l2_norm(
            emit,
            &format!("{prefix}.q"),
            q,
            d.seq,
            d.num_heads,
            d.head_dim,
            d.eps,
        );
        k = l2_norm(
            emit,
            &format!("{prefix}.k"),
            k,
            d.seq,
            d.num_kv_heads,
            d.head_dim,
            d.eps,
        );
    }

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
