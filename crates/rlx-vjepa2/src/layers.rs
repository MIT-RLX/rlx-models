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

//! Shared transformer block + attention kernels for V-JEPA2 submodules.

use super::rope::apply_vjepa2_rope;
use super::weights::Vjepa2BlockWeights;
use anyhow::Result;
use rlx_core::host_kernels::{gelu_tanh, layer_norm, linear, matmul, matmul_bt, softmax_rows};

#[allow(clippy::too_many_arguments)]
pub fn block_forward(
    x: &mut [f32],
    block: &Vjepa2BlockWeights,
    batch: usize,
    seq: usize,
    embed: usize,
    num_heads: usize,
    head_dim: usize,
    d_dim: usize,
    h_dim: usize,
    w_dim: usize,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    eps: f32,
    position_ids: Option<&[usize]>,
) -> Result<()> {
    let rows = batch * seq;
    let n1 = layer_norm(x, &block.norm1_w, &block.norm1_b, embed, eps)?;
    let attn = attention_rope(
        &n1,
        batch,
        seq,
        embed,
        num_heads,
        head_dim,
        d_dim,
        h_dim,
        w_dim,
        grid_t,
        grid_h,
        grid_w,
        position_ids,
        &block.q_w_t,
        &block.q_b,
        &block.k_w_t,
        &block.k_b,
        &block.v_w_t,
        &block.v_b,
        &block.proj_w_t,
        &block.proj_b,
    )?;
    for i in 0..x.len() {
        x[i] += attn[i];
    }

    let n2 = layer_norm(x, &block.norm2_w, &block.norm2_b, embed, eps)?;
    let hidden = block.mlp_fc1_b.len();
    let mut mlp = linear(
        &n2,
        rows,
        embed,
        &block.mlp_fc1_w_t,
        hidden,
        &block.mlp_fc1_b,
    )?;
    gelu_tanh(&mut mlp);
    let ffn = linear(
        &mlp,
        rows,
        hidden,
        &block.mlp_fc2_w_t,
        embed,
        &block.mlp_fc2_b,
    )?;
    for i in 0..x.len() {
        x[i] += ffn[i];
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn attention_rope(
    x: &[f32],
    batch: usize,
    l: usize,
    embed: usize,
    num_heads: usize,
    head_dim: usize,
    d_dim: usize,
    h_dim: usize,
    w_dim: usize,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    position_ids: Option<&[usize]>,
    q_w_t: &[f32],
    q_b: &[f32],
    k_w_t: &[f32],
    k_b: &[f32],
    v_w_t: &[f32],
    v_b: &[f32],
    proj_w_t: &[f32],
    proj_b: &[f32],
) -> Result<Vec<f32>> {
    let rows = batch * l;
    let q_proj = linear(x, rows, embed, q_w_t, embed, q_b)?;
    let k_proj = linear(x, rows, embed, k_w_t, embed, k_b)?;
    let v_proj = linear(x, rows, embed, v_w_t, embed, v_b)?;

    let bh = batch * num_heads;
    let mut q = vec![0f32; bh * l * head_dim];
    let mut k = vec![0f32; bh * l * head_dim];
    let mut v = vec![0f32; bh * l * head_dim];
    repack_heads(&q_proj, &mut q, batch, l, num_heads, head_dim);
    repack_heads(&k_proj, &mut k, batch, l, num_heads, head_dim);
    repack_heads(&v_proj, &mut v, batch, l, num_heads, head_dim);

    apply_vjepa2_rope(
        &mut q,
        &mut k,
        bh,
        l,
        head_dim,
        grid_t,
        grid_h,
        grid_w,
        d_dim,
        h_dim,
        w_dim,
        position_ids,
    );

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0f32; bh * l * head_dim];
    let mut scores = vec![0f32; l * l];

    for bhi in 0..bh {
        let q_h = &q[bhi * l * head_dim..(bhi + 1) * l * head_dim];
        let k_h = &k[bhi * l * head_dim..(bhi + 1) * l * head_dim];
        let v_h = &v[bhi * l * head_dim..(bhi + 1) * l * head_dim];
        matmul_bt(q_h, k_h, &mut scores, l, head_dim, l, scale);
        softmax_rows(&mut scores, l, l);
        let out_h = &mut attn_out[bhi * l * head_dim..(bhi + 1) * l * head_dim];
        matmul(&scores, v_h, out_h, l, l, head_dim);
    }

    let mut packed = vec![0f32; rows * embed];
    for bi in 0..batch {
        for li in 0..l {
            for hd in 0..num_heads {
                let src = ((bi * num_heads + hd) * l + li) * head_dim;
                let dst = (bi * l + li) * embed + hd * head_dim;
                packed[dst..dst + head_dim].copy_from_slice(&attn_out[src..src + head_dim]);
            }
        }
    }
    linear(&packed, rows, embed, proj_w_t, embed, proj_b)
}

/// Standard multi-head self-attention (no RoPE) for the attentive pooler.
pub fn attention_plain(
    x: &[f32],
    batch: usize,
    l: usize,
    embed: usize,
    num_heads: usize,
    head_dim: usize,
    q_w_t: &[f32],
    q_b: &[f32],
    k_w_t: &[f32],
    k_b: &[f32],
    v_w_t: &[f32],
    v_b: &[f32],
    proj_w_t: &[f32],
    proj_b: &[f32],
) -> Result<Vec<f32>> {
    attention_rope(
        x, batch, l, embed, num_heads, head_dim, 0, 0, 0, 1, 1, 1, None, q_w_t, q_b, k_w_t, k_b,
        v_w_t, v_b, proj_w_t, proj_b,
    )
}

/// Cross-attention: `queries` attend to `context` (K/V from context, Q from queries).
pub fn cross_attention(
    queries: &[f32],
    context: &[f32],
    batch: usize,
    l_q: usize,
    l_kv: usize,
    embed: usize,
    num_heads: usize,
    head_dim: usize,
    q_w_t: &[f32],
    q_b: &[f32],
    k_w_t: &[f32],
    k_b: &[f32],
    v_w_t: &[f32],
    v_b: &[f32],
) -> Result<Vec<f32>> {
    let q_rows = batch * l_q;
    let kv_rows = batch * l_kv;
    let q_proj = linear(queries, q_rows, embed, q_w_t, embed, q_b)?;
    let k_proj = linear(context, kv_rows, embed, k_w_t, embed, k_b)?;
    let v_proj = linear(context, kv_rows, embed, v_w_t, embed, v_b)?;

    let bh = batch * num_heads;
    let mut q = vec![0f32; bh * l_q * head_dim];
    let mut k = vec![0f32; bh * l_kv * head_dim];
    let mut v = vec![0f32; bh * l_kv * head_dim];
    repack_heads(&q_proj, &mut q, batch, l_q, num_heads, head_dim);
    repack_heads(&k_proj, &mut k, batch, l_kv, num_heads, head_dim);
    repack_heads(&v_proj, &mut v, batch, l_kv, num_heads, head_dim);

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0f32; bh * l_q * head_dim];
    let mut scores = vec![0f32; l_q * l_kv];

    for bhi in 0..bh {
        let q_h = &q[bhi * l_q * head_dim..(bhi + 1) * l_q * head_dim];
        let k_h = &k[bhi * l_kv * head_dim..(bhi + 1) * l_kv * head_dim];
        let v_h = &v[bhi * l_kv * head_dim..(bhi + 1) * l_kv * head_dim];
        matmul_bt(q_h, k_h, &mut scores, l_q, head_dim, l_kv, scale);
        softmax_rows(&mut scores, l_q, l_kv);
        let out_h = &mut attn_out[bhi * l_q * head_dim..(bhi + 1) * l_q * head_dim];
        matmul(&scores, v_h, out_h, l_q, l_kv, head_dim);
    }

    let mut packed = vec![0f32; q_rows * embed];
    for bi in 0..batch {
        for li in 0..l_q {
            for hd in 0..num_heads {
                let src = ((bi * num_heads + hd) * l_q + li) * head_dim;
                let dst = (bi * l_q + li) * embed + hd * head_dim;
                packed[dst..dst + head_dim].copy_from_slice(&attn_out[src..src + head_dim]);
            }
        }
    }
    Ok(packed)
}

pub fn repack_heads(
    src: &[f32],
    dst: &mut [f32],
    batch: usize,
    l: usize,
    num_heads: usize,
    head_dim: usize,
) {
    let e = num_heads * head_dim;
    for bi in 0..batch {
        for li in 0..l {
            for h in 0..num_heads {
                let s = (bi * l + li) * e + h * head_dim;
                let d = ((bi * num_heads + h) * l + li) * head_dim;
                dst[d..d + head_dim].copy_from_slice(&src[s..s + head_dim]);
            }
        }
    }
}

/// Gather rows from `[seq, dim]` flat tokens by index list → `[indices.len(), dim]`.
pub fn gather_rows(tokens: &[f32], indices: &[usize], seq: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; indices.len() * dim];
    for (oi, &idx) in indices.iter().enumerate() {
        debug_assert!(idx < seq);
        let src = idx * dim;
        let dst = oi * dim;
        out[dst..dst + dim].copy_from_slice(&tokens[src..src + dim]);
    }
    out
}
