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

//! Host-side FLUX.2 layer primitives (CPU reference path).

use super::weights::{
    Flux2DualAttnWeights, Flux2FeedForwardWeights, Flux2ParallelAttnWeights, LinearWeights,
    RmsNormWeight,
};
use anyhow::Result;
use rlx_tensor::{layer_norm, linear, matmul, matmul_bt};

pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

pub fn linear_no_bias(x: &[f32], rows: usize, lw: &LinearWeights) -> Result<Vec<f32>> {
    linear(
        x,
        rows,
        lw.in_dim,
        &lw.w_t,
        lw.out_dim,
        &vec![0.0f32; lw.out_dim],
    )
}

pub fn linear_w(x: &[f32], rows: usize, lw: &LinearWeights) -> Result<Vec<f32>> {
    linear(x, rows, lw.in_dim, &lw.w_t, lw.out_dim, &lw.bias)
}

pub fn layer_norm_no_affine(x: &[f32], dim: usize, eps: f32) -> Result<Vec<f32>> {
    let gamma = vec![1.0f32; dim];
    let beta = vec![0.0f32; dim];
    layer_norm(x, &gamma, &beta, dim, eps)
}

pub fn rms_norm_heads(x: &[f32], scale: &RmsNormWeight, dim: usize) -> Result<Vec<f32>> {
    let beta = vec![0.0f32; dim];
    layer_norm(x, &scale.scale, &beta, dim, 1e-6)
}

pub fn swiglu(x: &[f32], dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let rows = x.len() / dim;
    let mut out = vec![0.0f32; rows * half];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        for c in 0..half {
            let a = row[c];
            let b = row[c + half];
            let s = a / (1.0 + (-a).exp());
            out[r * half + c] = s * b;
        }
    }
    out
}

pub fn feed_forward(
    ff: &Flux2FeedForwardWeights,
    x: &[f32],
    rows: usize,
    dim: usize,
) -> Result<Vec<f32>> {
    let h = linear_w(x, rows, &ff.linear_in)?;
    let inner = ff.linear_in.out_dim / 2;
    let activated = swiglu(&h, ff.linear_in.out_dim);
    linear(
        &activated,
        rows,
        inner,
        &ff.linear_out.w_t,
        dim,
        &ff.linear_out.bias,
    )
}

pub fn timestep_embedding(timesteps: &[f32], dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut emb = vec![0.0f32; timesteps.len() * dim];
    for (bi, &t) in timesteps.iter().enumerate() {
        for i in 0..half {
            let freq = 1.0 / 10000f32.powf(i as f32 / half as f32);
            let angle = t * freq;
            emb[bi * dim + i] = angle.cos();
            emb[bi * dim + half + i] = angle.sin();
        }
    }
    emb
}

pub fn time_guidance_embed(
    timestep: &[f32],
    guidance: Option<&[f32]>,
    tg: &super::weights::Flux2TimestepGuidanceWeights,
    _dim: usize,
) -> Result<Vec<f32>> {
    let ch = tg.timestep_linear1.in_dim;
    let t_proj = timestep_embedding(timestep, ch);
    let mut emb = linear_w(&t_proj, timestep.len(), &tg.timestep_linear1)?;
    silu(&mut emb);
    let mut out = linear_w(&emb, timestep.len(), &tg.timestep_linear2)?;
    if let (Some(g), Some(g1), Some(g2)) = (guidance, &tg.guidance_linear1, &tg.guidance_linear2) {
        let g_proj = timestep_embedding(g, ch);
        let mut g_emb = linear_w(&g_proj, g.len(), g1)?;
        silu(&mut g_emb);
        let g_out = linear_w(&g_emb, g.len(), g2)?;
        for i in 0..out.len() {
            out[i] += g_out[i];
        }
    }
    Ok(out)
}

/// Dual-timestep embedding: average of embed(t) and embed(t′) (flow-map / Diamond Maps).
pub fn time_guidance_embed_dual(
    timestep: &[f32],
    timestep_target: &[f32],
    guidance: Option<&[f32]>,
    tg: &super::weights::Flux2TimestepGuidanceWeights,
    tg_target: &super::weights::Flux2TimestepGuidanceWeights,
    dim: usize,
) -> Result<Vec<f32>> {
    ensure_same_batch(timestep, timestep_target)?;
    let e1 = time_guidance_embed(timestep, guidance, tg, dim)?;
    let e2 = time_guidance_embed(timestep_target, guidance, tg_target, dim)?;
    Ok(e1
        .iter()
        .zip(e2.iter())
        .map(|(a, b)| 0.5 * (a + b))
        .collect())
}

fn ensure_same_batch(a: &[f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        anyhow::bail!("timestep batches must match: {} vs {}", a.len(), b.len());
    }
    Ok(())
}

/// Returns two modulation tuples: `((shift_msa, scale_msa, gate_msa), (shift_mlp, ...))`.
#[allow(clippy::type_complexity)]
pub fn double_stream_mod(
    temb: &[f32],
    batch: usize,
    dim: usize,
    lw: &LinearWeights,
) -> Result<(
    (Vec<f32>, Vec<f32>, Vec<f32>),
    (Vec<f32>, Vec<f32>, Vec<f32>),
)> {
    let mut h = temb.to_vec();
    silu(&mut h);
    let mod_out = linear_w(&h, batch, lw)?;
    let d = dim;
    let mut shift_msa = vec![0.0f32; batch * d];
    let mut scale_msa = vec![0.0f32; batch * d];
    let mut gate_msa = vec![0.0f32; batch * d];
    let mut shift_mlp = vec![0.0f32; batch * d];
    let mut scale_mlp = vec![0.0f32; batch * d];
    let mut gate_mlp = vec![0.0f32; batch * d];
    for b in 0..batch {
        let row = &mod_out[b * 6 * d..(b + 1) * 6 * d];
        shift_msa[b * d..(b + 1) * d].copy_from_slice(&row[0..d]);
        scale_msa[b * d..(b + 1) * d].copy_from_slice(&row[d..2 * d]);
        gate_msa[b * d..(b + 1) * d].copy_from_slice(&row[2 * d..3 * d]);
        shift_mlp[b * d..(b + 1) * d].copy_from_slice(&row[3 * d..4 * d]);
        scale_mlp[b * d..(b + 1) * d].copy_from_slice(&row[4 * d..5 * d]);
        gate_mlp[b * d..(b + 1) * d].copy_from_slice(&row[5 * d..6 * d]);
    }
    Ok((
        (shift_msa, scale_msa, gate_msa),
        (shift_mlp, scale_mlp, gate_mlp),
    ))
}

pub fn single_stream_mod(
    temb: &[f32],
    batch: usize,
    dim: usize,
    lw: &LinearWeights,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut h = temb.to_vec();
    silu(&mut h);
    let mod_out = linear_w(&h, batch, lw)?;
    let mut shift = vec![0.0f32; batch * dim];
    let mut scale = vec![0.0f32; batch * dim];
    let mut gate = vec![0.0f32; batch * dim];
    for b in 0..batch {
        let row = &mod_out[b * 3 * dim..(b + 1) * 3 * dim];
        shift[b * dim..(b + 1) * dim].copy_from_slice(&row[0..dim]);
        scale[b * dim..(b + 1) * dim].copy_from_slice(&row[dim..2 * dim]);
        gate[b * dim..(b + 1) * dim].copy_from_slice(&row[2 * dim..3 * dim]);
    }
    Ok((shift, scale, gate))
}

pub fn modulate(
    x: &[f32],
    shift: &[f32],
    scale: &[f32],
    dim: usize,
    batch: usize,
    seq: usize,
) -> Vec<f32> {
    let mut out = x.to_vec();
    for b in 0..batch {
        for t in 0..seq {
            for d in 0..dim {
                let i = (b * seq + t) * dim + d;
                out[i] = (1.0 + scale[b * dim + d]) * out[i] + shift[b * dim + d];
            }
        }
    }
    out
}

pub fn modulate_scale_shift(
    x: &[f32],
    scale: &[f32],
    shift: &[f32],
    dim: usize,
    batch: usize,
    seq: usize,
) -> Vec<f32> {
    let mut out = x.to_vec();
    for b in 0..batch {
        for t in 0..seq {
            for d in 0..dim {
                let i = (b * seq + t) * dim + d;
                out[i] = out[i] * (1.0 + scale[b * dim + d]) + shift[b * dim + d];
            }
        }
    }
    out
}

pub fn gate_mul(x: &[f32], gate: &[f32], dim: usize, batch: usize, seq: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for b in 0..batch {
        for t in 0..seq {
            for d in 0..dim {
                let i = (b * seq + t) * dim + d;
                out[i] = gate[b * dim + d] * x[i];
            }
        }
    }
    out
}

pub fn dual_attention(
    attn: &Flux2DualAttnWeights,
    hidden: &[f32],
    encoder: &[f32],
    batch: usize,
    img_seq: usize,
    txt_seq: usize,
    heads: usize,
    head_dim: usize,
    dim: usize,
    cos: &[f32],
    sin: &[f32],
    rope_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let img_rows = batch * img_seq;
    let txt_rows = batch * txt_seq;
    let mut q = linear_w(hidden, img_rows, &attn.to_q)?;
    let mut k = linear_w(hidden, img_rows, &attn.to_k)?;
    let mut v = linear_w(hidden, img_rows, &attn.to_v)?;
    let mut eq = linear_w(encoder, txt_rows, &attn.add_q)?;
    let mut ek = linear_w(encoder, txt_rows, &attn.add_k)?;
    let mut ev = linear_w(encoder, txt_rows, &attn.add_v)?;

    repack_bh(&mut q, batch, img_seq, heads, head_dim);
    repack_bh(&mut k, batch, img_seq, heads, head_dim);
    repack_bh(&mut v, batch, img_seq, heads, head_dim);
    repack_bh(&mut eq, batch, txt_seq, heads, head_dim);
    repack_bh(&mut ek, batch, txt_seq, heads, head_dim);
    repack_bh(&mut ev, batch, txt_seq, heads, head_dim);

    q = rms_norm_heads(&q, &attn.norm_q, head_dim)?;
    k = rms_norm_heads(&k, &attn.norm_k, head_dim)?;
    eq = rms_norm_heads(&eq, &attn.norm_added_q, head_dim)?;
    ek = rms_norm_heads(&ek, &attn.norm_added_k, head_dim)?;

    let total_seq = txt_seq + img_seq;
    let mut cq = vec![0.0f32; batch * total_seq * heads * head_dim];
    let mut ck = vec![0.0f32; batch * total_seq * heads * head_dim];
    let mut cv = vec![0.0f32; batch * total_seq * heads * head_dim];
    concat_seq(&eq, &q, batch, txt_seq, img_seq, heads, head_dim, &mut cq);
    concat_seq(&ek, &k, batch, txt_seq, img_seq, heads, head_dim, &mut ck);
    concat_seq(&ev, &v, batch, txt_seq, img_seq, heads, head_dim, &mut cv);

    super::rope::apply_flux2_qk_rope(
        &mut cq, &mut ck, cos, sin, batch, total_seq, heads, head_dim, rope_dim,
    );

    let scale = 1.0 / (head_dim as f32).sqrt();
    let attn_out = mha_from_qkv(&cq, &ck, &cv, batch, total_seq, heads, head_dim, scale)?;
    let mut packed = vec![0.0f32; batch * total_seq * dim];
    unpack_seq(
        &attn_out,
        batch,
        total_seq,
        heads,
        head_dim,
        dim,
        &mut packed,
    );

    let txt_packed = &packed[..batch * txt_seq * dim];
    let img_packed = &packed[batch * txt_seq * dim..];
    let enc_out = linear(
        txt_packed,
        batch * txt_seq,
        dim,
        &attn.to_add_out.w_t,
        dim,
        &attn.to_add_out.bias,
    )?;
    let img_out = linear(
        img_packed,
        batch * img_seq,
        dim,
        &attn.to_out.w_t,
        dim,
        &attn.to_out.bias,
    )?;
    Ok((enc_out, img_out))
}

pub fn parallel_attention(
    attn: &Flux2ParallelAttnWeights,
    hidden: &[f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    dim: usize,
    mlp_hidden: usize,
    cos: &[f32],
    sin: &[f32],
    rope_dim: usize,
) -> Result<Vec<f32>> {
    let rows = batch * seq;
    let fused = linear_w(hidden, rows, &attn.to_qkv_mlp)?;
    let qkv_dim = dim * 3;
    let qkv = &fused[..rows * qkv_dim];
    let mlp_in = &fused[rows * qkv_dim..];

    let mut q = qkv[..rows * dim].to_vec();
    let mut k = qkv[rows * dim..rows * 2 * dim].to_vec();
    let mut v = qkv[rows * 2 * dim..rows * 3 * dim].to_vec();

    repack_bh(&mut q, batch, seq, heads, head_dim);
    repack_bh(&mut k, batch, seq, heads, head_dim);
    repack_bh(&mut v, batch, seq, heads, head_dim);
    q = rms_norm_heads(&q, &attn.norm_q, head_dim)?;
    k = rms_norm_heads(&k, &attn.norm_k, head_dim)?;
    super::rope::apply_flux2_qk_rope(
        &mut q, &mut k, cos, sin, batch, seq, heads, head_dim, rope_dim,
    );
    let scale = 1.0 / (head_dim as f32).sqrt();
    let attn_out = mha_from_qkv(&q, &k, &v, batch, seq, heads, head_dim, scale)?;
    let mut packed = vec![0.0f32; rows * dim];
    unpack_seq(&attn_out, batch, seq, heads, head_dim, dim, &mut packed);
    let mlp_act = swiglu(mlp_in, mlp_hidden * 2);
    let mut cat = vec![0.0f32; rows * (dim + mlp_hidden)];
    for r in 0..rows {
        cat[r * (dim + mlp_hidden)..r * (dim + mlp_hidden) + dim]
            .copy_from_slice(&packed[r * dim..(r + 1) * dim]);
        cat[r * (dim + mlp_hidden) + dim..(r + 1) * (dim + mlp_hidden)]
            .copy_from_slice(&mlp_act[r * mlp_hidden..(r + 1) * mlp_hidden]);
    }
    linear_w(&cat, rows, &attn.to_out)
}

fn repack_bh(flat: &mut [f32], batch: usize, seq: usize, heads: usize, head_dim: usize) {
    let dim = heads * head_dim;
    let mut tmp = vec![0.0f32; batch * seq * heads * head_dim];
    for b in 0..batch {
        for t in 0..seq {
            for h in 0..heads {
                let src = (b * seq + t) * dim + h * head_dim;
                let dst = (b * seq + t) * heads * head_dim + h * head_dim;
                tmp[dst..dst + head_dim].copy_from_slice(&flat[src..src + head_dim]);
            }
        }
    }
    flat.copy_from_slice(&tmp);
}

fn unpack_seq(
    bh: &[f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    dim: usize,
    out: &mut [f32],
) {
    for b in 0..batch {
        for t in 0..seq {
            for h in 0..heads {
                let src = (b * seq + t) * heads * head_dim + h * head_dim;
                let dst = (b * seq + t) * dim + h * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&bh[src..src + head_dim]);
            }
        }
    }
}

fn concat_seq(
    a: &[f32],
    b: &[f32],
    batch: usize,
    len_a: usize,
    len_b: usize,
    heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let stride = heads * head_dim;
    for bi in 0..batch {
        for t in 0..len_a {
            let src = (bi * len_a + t) * stride;
            let dst = (bi * (len_a + len_b) + t) * stride;
            out[dst..dst + stride].copy_from_slice(&a[src..src + stride]);
        }
        for t in 0..len_b {
            let src = (bi * len_b + t) * stride;
            let dst = (bi * (len_a + len_b) + len_a + t) * stride;
            out[dst..dst + stride].copy_from_slice(&b[src..src + stride]);
        }
    }
}

fn mha_from_qkv(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    let bh = batch * heads;
    let l = seq;
    let mut out = vec![0.0f32; bh * l * head_dim];
    let mut scores = vec![0.0f32; l * l];
    for bi in 0..batch {
        for h in 0..heads {
            let bhi = bi * heads + h;
            let q_h = &q[bhi * l * head_dim..(bhi + 1) * l * head_dim];
            let k_h = &k[bhi * l * head_dim..(bhi + 1) * l * head_dim];
            let v_h = &v[bhi * l * head_dim..(bhi + 1) * l * head_dim];
            matmul_bt(q_h, k_h, &mut scores, l, head_dim, l, scale);
            softmax_rows(&mut scores, l, l);
            let o_h = &mut out[bhi * l * head_dim..(bhi + 1) * l * head_dim];
            matmul(&scores, v_h, o_h, l, l, head_dim);
        }
    }
    Ok(out)
}

fn softmax_rows(scores: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut scores[r * cols..(r + 1) * cols];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

pub fn ada_layer_norm_continuous(
    x: &[f32],
    temb: &[f32],
    batch: usize,
    seq: usize,
    dim: usize,
    norm_linear: &LinearWeights,
    eps: f32,
) -> Result<Vec<f32>> {
    let mut h = temb.to_vec();
    silu(&mut h);
    let emb = linear_w(&h, batch, norm_linear)?;
    let half = dim;
    let scale = &emb[..batch * half];
    let shift = &emb[batch * half..batch * 2 * half];
    let normed = layer_norm_no_affine(x, dim, eps)?;
    let mut out = normed.clone();
    for b in 0..batch {
        for t in 0..seq {
            for d in 0..dim {
                let i = (b * seq + t) * dim + d;
                out[i] = normed[i] * (1.0 + scale[b * dim + d]) + shift[b * dim + d];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod dual_temb_tests {
    use super::*;
    use crate::config::Flux2Config;
    use crate::weights::extract_flux2_weights;
    use crate::{prepare_weight_map, synthetic_weights};

    #[test]
    fn dual_temb_averages() {
        let cfg = Flux2Config::tiny();
        let w = extract_flux2_weights(prepare_weight_map(synthetic_weights(&cfg)), &cfg).unwrap();
        let t1 = [500.0f32];
        let t2 = [250.0f32];
        let a = time_guidance_embed(&t1, None, &w.time_guidance, cfg.inner_dim()).unwrap();
        let b = time_guidance_embed(&t2, None, &w.time_guidance, cfg.inner_dim()).unwrap();
        let dual = time_guidance_embed_dual(
            &t1,
            &t2,
            None,
            &w.time_guidance,
            &w.time_guidance,
            cfg.inner_dim(),
        )
        .unwrap();
        assert_eq!(dual.len(), a.len());
        for i in 0..dual.len() {
            assert!((dual[i] - 0.5 * (a[i] + b[i])).abs() < 1e-5);
        }
    }
}
