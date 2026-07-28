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

//! MiniMax-M3 self-attention (`self_attn`).
//!
//! GQA (bias-free q/k/v/o) with **per-head Gemma QK-norm** applied before
//! **partial NeoX RoPE** (`n_rot < head_dim`). `full_attention` layers use a
//! causal mask; `minimax_m3_sparse` layers instead add the MSA block-sparse
//! additive bias from [`super::indexer::emit_msa_bias`]. `score_scale` is the
//! default `head_dim^-0.5`.

use anyhow::Result;
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, HirGraphExt, HirNodeId, RopeStyle, Shape};

use super::indexer::{IndexerDims, emit_msa_bias};
use super::ops::per_head_gemma_rmsnorm;
use super::{ROPE_COS, ROPE_SIN};

/// Dimensions for one MiniMax-M3 attention block.
#[derive(Debug, Clone, Copy)]
pub struct M3AttnDims {
    /// Residual-stream width.
    pub hidden: usize,
    /// Query heads.
    pub num_heads: usize,
    /// Key/value heads (GQA).
    pub num_kv_heads: usize,
    /// Per-head dim.
    pub head_dim: usize,
    /// Leading per-head dims that receive RoPE (partial RoPE).
    pub n_rot: usize,
    /// RMSNorm epsilon (Gemma QK-norm).
    pub eps: f32,
    /// Sequence length.
    pub seq: usize,
    /// `true` = MSA block-sparse attention; `false` = full causal.
    pub sparse: bool,
    /// MSA indexer head dim (sparse layers).
    pub index_head_dim: usize,
    /// MSA block size (keys per block).
    pub block_size: usize,
    /// MSA top-k blocks kept per query.
    pub topk_blocks: usize,
    /// MSA always-visible local blocks.
    pub local_blocks: usize,
}

fn repeat_kv(
    g: &mut HirMut,
    x: HirNodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> HirNodeId {
    if group == 1 {
        return x;
    }
    let last_ax = g.shape(x).rank() - 1;
    let mut pieces = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}

/// Emit `self_attn` for `{prefix}` over a layer-normed `[1, seq, hidden]` input.
pub fn emit_m3_attention(
    emit: &mut Emit<'_>,
    prefix: &str,
    normed: HirNodeId,
    d: M3AttnDims,
) -> Result<HirNodeId> {
    let f = DType::F32;
    let h = d.num_heads;
    let kv = d.num_kv_heads;
    let hd = d.head_dim;
    let seq = d.seq;
    let group = h / kv;

    // q/k/v projections (bias-free).
    let q = {
        let w = emit.load_param(&format!("{prefix}.q_proj.weight"), true)?;
        let mut gb = HirMut::new(emit.hir());
        gb.mm(normed, w)
    };
    let k = {
        let w = emit.load_param(&format!("{prefix}.k_proj.weight"), true)?;
        let mut gb = HirMut::new(emit.hir());
        gb.mm(normed, w)
    };
    let v = {
        let w = emit.load_param(&format!("{prefix}.v_proj.weight"), true)?;
        let mut gb = HirMut::new(emit.hir());
        gb.mm(normed, w)
    };

    // Per-head Gemma QK-norm before RoPE.
    let q = per_head_gemma_rmsnorm(emit, &format!("{prefix}.q_norm"), q, 1, seq, h, hd, d.eps)?;
    let k = per_head_gemma_rmsnorm(emit, &format!("{prefix}.k_norm"), k, 1, seq, kv, hd, d.eps)?;

    let cos = emit.flow_input(ROPE_COS)?.hir_id();
    let sin = emit.flow_input(ROPE_SIN)?.hir_id();

    let (q, k_rep, v_rep) = {
        let mut gb = HirMut::new(emit.hir());
        let q = gb.rope_n_styled(q, cos, sin, hd, d.n_rot, RopeStyle::NeoX);
        let k = gb.rope_n_styled(k, cos, sin, hd, d.n_rot, RopeStyle::NeoX);
        let k_rep = repeat_kv(&mut gb, k, kv, hd, group);
        let v_rep = repeat_kv(&mut gb, v, kv, hd, group);
        (q, k_rep, v_rep)
    };

    let out_shape = Shape::new(&[1, seq, h * hd], f);
    let attn = if d.sparse {
        let bias = emit_msa_bias(
            emit,
            prefix,
            normed,
            IndexerDims {
                hidden: d.hidden,
                num_heads: h,
                index_n_heads: kv,
                index_head_dim: d.index_head_dim,
                n_rot: d.n_rot,
                block_size: d.block_size,
                topk_blocks: d.topk_blocks,
                local_blocks: d.local_blocks,
                eps: d.eps,
                seq,
            },
        )?;
        let mut gb = HirMut::new(emit.hir());
        gb.attention_bias(q, k_rep, v_rep, bias, h, hd, out_shape)
    } else {
        let mut gb = HirMut::new(emit.hir());
        gb.attention_kind(q, k_rep, v_rep, h, hd, MaskKind::Causal, out_shape)
    };

    // Output projection.
    let o_w = emit.load_param(&format!("{prefix}.o_proj.weight"), true)?;
    let mut gb = HirMut::new(emit.hir());
    Ok(gb.mm(attn, o_w))
}
