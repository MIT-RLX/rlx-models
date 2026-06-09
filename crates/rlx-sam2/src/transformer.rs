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

//! SAM 2 two-way transformer — host-side.
//!
//! Mirrors `sam2/modeling/sam/transformer.py::TwoWayTransformer` and
//! `TwoWayAttentionBlock`. Structurally identical to SAM v1 (2 layers
//! of self-attn → cross t→i → ReLU MLP → cross i→t, then final
//! token→image attn + LayerNorm) — the only differences are:
//!
//!   - Weight key prefix: `sam_mask_decoder.transformer.*` instead of
//!     `mask_decoder.transformer.*`.
//!   - The cross-attention `downsample_rate` is configurable in the
//!     reference (defaults to 2 for the decoder transformer, matching
//!     v1). We keep the rate as a parameter on
//!     [`extract_two_way_transformer_weights`].
//!
//! Decoder transformer compute is small (q_n ≤ ~10 tokens, k_n = 64²),
//! so staying on the CPU is the right tradeoff vs. growing the IR
//! surface with multi-shape cross-attention.

use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

/// Weights for one `Attention` layer (`embed_dim → internal_dim → embed_dim`).
pub struct Sam2AttentionWeights {
    pub q_w: Vec<f32>, // [internal_dim, embed_dim]
    pub q_b: Vec<f32>,
    pub k_w: Vec<f32>,
    pub k_b: Vec<f32>,
    pub v_w: Vec<f32>,
    pub v_b: Vec<f32>,
    pub out_w: Vec<f32>, // [embed_dim, internal_dim]
    pub out_b: Vec<f32>,
    pub num_heads: usize,
    pub embed_dim: usize,
    pub internal_dim: usize,
}

pub struct Sam2TwoWayAttentionBlockWeights {
    pub self_attn: Sam2AttentionWeights,
    pub norm1_g: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub cross_token_to_image: Sam2AttentionWeights,
    pub norm2_g: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub mlp_lin1_w: Vec<f32>,
    pub mlp_lin1_b: Vec<f32>,
    pub mlp_lin2_w: Vec<f32>,
    pub mlp_lin2_b: Vec<f32>,
    pub norm3_g: Vec<f32>,
    pub norm3_b: Vec<f32>,
    pub cross_image_to_token: Sam2AttentionWeights,
    pub norm4_g: Vec<f32>,
    pub norm4_b: Vec<f32>,
    pub skip_first_layer_pe: bool,
}

pub struct Sam2TwoWayTransformerWeights {
    pub layers: Vec<Sam2TwoWayAttentionBlockWeights>,
    pub final_attn_token_to_image: Sam2AttentionWeights,
    pub norm_final_g: Vec<f32>,
    pub norm_final_b: Vec<f32>,
    pub embed_dim: usize,
}

fn load_attention(
    weights: &mut WeightMap,
    prefix: &str,
    embed_dim: usize,
    num_heads: usize,
    downsample_rate: usize,
) -> Result<Sam2AttentionWeights> {
    let internal_dim = embed_dim / downsample_rate;
    let (q_w, sh) = weights.take(&format!("{prefix}.q_proj.weight"))?;
    ensure!(
        sh == vec![internal_dim, embed_dim],
        "{prefix}.q_proj.weight shape {sh:?} not [{internal_dim}, {embed_dim}]"
    );
    let (q_b, _) = weights.take(&format!("{prefix}.q_proj.bias"))?;
    let (k_w, _) = weights.take(&format!("{prefix}.k_proj.weight"))?;
    let (k_b, _) = weights.take(&format!("{prefix}.k_proj.bias"))?;
    let (v_w, _) = weights.take(&format!("{prefix}.v_proj.weight"))?;
    let (v_b, _) = weights.take(&format!("{prefix}.v_proj.bias"))?;
    let (out_w, sh) = weights.take(&format!("{prefix}.out_proj.weight"))?;
    ensure!(
        sh == vec![embed_dim, internal_dim],
        "{prefix}.out_proj.weight shape {sh:?} not [{embed_dim}, {internal_dim}]"
    );
    let (out_b, _) = weights.take(&format!("{prefix}.out_proj.bias"))?;
    Ok(Sam2AttentionWeights {
        q_w,
        q_b,
        k_w,
        k_b,
        v_w,
        v_b,
        out_w,
        out_b,
        num_heads,
        embed_dim,
        internal_dim,
    })
}

pub(super) fn extract_two_way_transformer_weights(
    weights: &mut WeightMap,
    embed_dim: usize,
    depth: usize,
    num_heads: usize,
    mlp_dim: usize,
) -> Result<Sam2TwoWayTransformerWeights> {
    let mut layers = Vec::with_capacity(depth);
    for i in 0..depth {
        let p = format!("sam_mask_decoder.transformer.layers.{i}");
        let self_attn =
            load_attention(weights, &format!("{p}.self_attn"), embed_dim, num_heads, 1)?;
        let (norm1_g, _) = weights.take(&format!("{p}.norm1.weight"))?;
        let (norm1_b, _) = weights.take(&format!("{p}.norm1.bias"))?;
        let cross_t2i = load_attention(
            weights,
            &format!("{p}.cross_attn_token_to_image"),
            embed_dim,
            num_heads,
            2,
        )?;
        let (norm2_g, _) = weights.take(&format!("{p}.norm2.weight"))?;
        let (norm2_b, _) = weights.take(&format!("{p}.norm2.bias"))?;
        let (mlp_lin1_w, sh) = weights.take(&format!("{p}.mlp.layers.0.weight"))?;
        ensure!(
            sh == vec![mlp_dim, embed_dim],
            "{p}.mlp.layers.0.weight shape {sh:?} not [{mlp_dim}, {embed_dim}]"
        );
        let (mlp_lin1_b, _) = weights.take(&format!("{p}.mlp.layers.0.bias"))?;
        let (mlp_lin2_w, _) = weights.take(&format!("{p}.mlp.layers.1.weight"))?;
        let (mlp_lin2_b, _) = weights.take(&format!("{p}.mlp.layers.1.bias"))?;
        let (norm3_g, _) = weights.take(&format!("{p}.norm3.weight"))?;
        let (norm3_b, _) = weights.take(&format!("{p}.norm3.bias"))?;
        let cross_i2t = load_attention(
            weights,
            &format!("{p}.cross_attn_image_to_token"),
            embed_dim,
            num_heads,
            2,
        )?;
        let (norm4_g, _) = weights.take(&format!("{p}.norm4.weight"))?;
        let (norm4_b, _) = weights.take(&format!("{p}.norm4.bias"))?;
        layers.push(Sam2TwoWayAttentionBlockWeights {
            self_attn,
            norm1_g,
            norm1_b,
            cross_token_to_image: cross_t2i,
            norm2_g,
            norm2_b,
            mlp_lin1_w,
            mlp_lin1_b,
            mlp_lin2_w,
            mlp_lin2_b,
            norm3_g,
            norm3_b,
            cross_image_to_token: cross_i2t,
            norm4_g,
            norm4_b,
            skip_first_layer_pe: i == 0,
        });
    }
    let final_attn = load_attention(
        weights,
        "sam_mask_decoder.transformer.final_attn_token_to_image",
        embed_dim,
        num_heads,
        2,
    )?;
    let (norm_final_g, _) = weights.take("sam_mask_decoder.transformer.norm_final_attn.weight")?;
    let (norm_final_b, _) = weights.take("sam_mask_decoder.transformer.norm_final_attn.bias")?;
    Ok(Sam2TwoWayTransformerWeights {
        layers,
        final_attn_token_to_image: final_attn,
        norm_final_g,
        norm_final_b,
        embed_dim,
    })
}

// ─── Host-side execution ─────────────────────────────────────────

/// Standard scaled-dot-product multi-head attention.
/// All inputs `[B, N_*, embed_dim]`. `b` is the batch dim.
pub fn sam2_attention_forward(
    w: &Sam2AttentionWeights,
    q: &[f32],
    q_n: usize,
    k: &[f32],
    k_n: usize,
    v: &[f32],
    v_n: usize,
    b: usize,
) -> Vec<f32> {
    let e = w.embed_dim;
    let id = w.internal_dim;
    let nh = w.num_heads;
    let dh = id / nh;
    let scale = 1.0 / (dh as f32).sqrt();

    let q_p = linear(q, &w.q_w, &w.q_b, b * q_n, e, id);
    let k_p = linear(k, &w.k_w, &w.k_b, b * k_n, e, id);
    let v_p = linear(v, &w.v_w, &w.v_b, b * v_n, e, id);

    let q_h = separate_heads(&q_p, b, q_n, nh, dh);
    let k_h = separate_heads(&k_p, b, k_n, nh, dh);
    let v_h = separate_heads(&v_p, b, v_n, nh, dh);

    let mut out_h = vec![0f32; b * nh * q_n * dh];
    let mut scores = vec![0f32; q_n * k_n];
    for bi in 0..b {
        for h in 0..nh {
            let q_off = ((bi * nh) + h) * q_n * dh;
            let k_off = ((bi * nh) + h) * k_n * dh;
            let v_off = ((bi * nh) + h) * v_n * dh;
            let out_off = ((bi * nh) + h) * q_n * dh;

            for i in 0..q_n {
                for j in 0..k_n {
                    let mut acc = 0f32;
                    for d in 0..dh {
                        acc += q_h[q_off + i * dh + d] * k_h[k_off + j * dh + d];
                    }
                    scores[i * k_n + j] = acc * scale;
                }
            }
            for i in 0..q_n {
                let row = &mut scores[i * k_n..(i + 1) * k_n];
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0f32;
                for v in row.iter_mut() {
                    *v = (*v - m).exp();
                    s += *v;
                }
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
            for i in 0..q_n {
                for d in 0..dh {
                    let mut acc = 0f32;
                    for j in 0..k_n {
                        acc += scores[i * k_n + j] * v_h[v_off + j * dh + d];
                    }
                    out_h[out_off + i * dh + d] = acc;
                }
            }
        }
    }

    let merged = recombine_heads(&out_h, b, q_n, nh, dh);
    linear(&merged, &w.out_w, &w.out_b, b * q_n, id, e)
}

/// Standard PyTorch Linear: `y = x @ W^T + b` where `W: [out, in]`.
pub fn linear(x: &[f32], w: &[f32], b: &[f32], rows: usize, in_d: usize, out_d: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * out_d];
    for r in 0..rows {
        for o in 0..out_d {
            let mut acc = b[o];
            for k in 0..in_d {
                acc += x[r * in_d + k] * w[o * in_d + k];
            }
            out[r * out_d + o] = acc;
        }
    }
    out
}

fn separate_heads(x: &[f32], b: usize, n: usize, nh: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0f32; b * nh * n * dh];
    for bi in 0..b {
        for i in 0..n {
            for h in 0..nh {
                for d in 0..dh {
                    out[((bi * nh + h) * n + i) * dh + d] =
                        x[(bi * n + i) * (nh * dh) + h * dh + d];
                }
            }
        }
    }
    out
}

fn recombine_heads(x: &[f32], b: usize, n: usize, nh: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0f32; b * n * nh * dh];
    for bi in 0..b {
        for h in 0..nh {
            for i in 0..n {
                for d in 0..dh {
                    out[(bi * n + i) * (nh * dh) + h * dh + d] =
                        x[((bi * nh + h) * n + i) * dh + d];
                }
            }
        }
    }
    out
}

/// LayerNorm over the last axis. `x: [rows, n]` (two-pass variance).
pub fn layer_norm_last(x: &mut [f32], rows: usize, n: usize, g: &[f32], b: &[f32], eps: f32) {
    for r in 0..rows {
        let row = &mut x[r * n..(r + 1) * n];
        let mut mean = 0f32;
        for v in row.iter() {
            mean += *v;
        }
        mean /= n as f32;
        let mut var = 0f32;
        for v in row.iter() {
            let d = *v - mean;
            var += d * d;
        }
        var /= n as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for k in 0..n {
            row[k] = (row[k] - mean) * inv * g[k] + b[k];
        }
    }
}

/// Stack / IR final norm — matches compiled [`Op::LayerNorm`] on CPU.
pub fn layer_norm_last_cpu(x: &mut [f32], rows: usize, n: usize, g: &[f32], b: &[f32], eps: f32) {
    let mut tmp = vec![0f32; n];
    for r in 0..rows {
        let base = r * n;
        rlx_cpu::kernels::layer_norm_row(&x[base..base + n], g, b, &mut tmp, n, eps);
        x[base..base + n].copy_from_slice(&tmp);
    }
}

pub(super) fn add_inplace(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// One `TwoWayAttentionBlock` forward.
pub fn two_way_attention_block_forward(
    w: &Sam2TwoWayAttentionBlockWeights,
    queries: Vec<f32>,
    keys: Vec<f32>,
    query_pe: &[f32],
    key_pe: &[f32],
    b: usize,
    q_n: usize,
    k_n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let e = w.self_attn.embed_dim;

    // ── Self attention block ──
    let mut queries = if w.skip_first_layer_pe {
        sam2_attention_forward(&w.self_attn, &queries, q_n, &queries, q_n, &queries, q_n, b)
    } else {
        let mut q = queries.clone();
        add_inplace(&mut q, query_pe);
        let attn_out = sam2_attention_forward(&w.self_attn, &q, q_n, &q, q_n, &queries, q_n, b);
        let mut out = queries;
        add_inplace(&mut out, &attn_out);
        out
    };
    layer_norm_last(&mut queries, b * q_n, e, &w.norm1_g, &w.norm1_b, 1e-5);

    // ── Cross attention, tokens attending to image ──
    let mut q_pe = queries.clone();
    add_inplace(&mut q_pe, query_pe);
    let mut k_pe = keys.clone();
    add_inplace(&mut k_pe, key_pe);
    let attn_out = sam2_attention_forward(
        &w.cross_token_to_image,
        &q_pe,
        q_n,
        &k_pe,
        k_n,
        &keys,
        k_n,
        b,
    );
    add_inplace(&mut queries, &attn_out);
    layer_norm_last(&mut queries, b * q_n, e, &w.norm2_g, &w.norm2_b, 1e-5);

    // ── MLP (ReLU activation per reference's `MLPBlock`) ──
    let mlp_dim = w.mlp_lin1_b.len();
    let mut mlp_mid = linear(&queries, &w.mlp_lin1_w, &w.mlp_lin1_b, b * q_n, e, mlp_dim);
    relu_inplace(&mut mlp_mid);
    let mlp_out = linear(&mlp_mid, &w.mlp_lin2_w, &w.mlp_lin2_b, b * q_n, mlp_dim, e);
    add_inplace(&mut queries, &mlp_out);
    layer_norm_last(&mut queries, b * q_n, e, &w.norm3_g, &w.norm3_b, 1e-5);

    // ── Cross attention, image attending to tokens ──
    let mut q_pe = queries.clone();
    add_inplace(&mut q_pe, query_pe);
    let mut k_pe = keys.clone();
    add_inplace(&mut k_pe, key_pe);
    let attn_out = sam2_attention_forward(
        &w.cross_image_to_token,
        &k_pe,
        k_n,
        &q_pe,
        q_n,
        &queries,
        q_n,
        b,
    );
    let mut keys = keys;
    add_inplace(&mut keys, &attn_out);
    layer_norm_last(&mut keys, b * k_n, e, &w.norm4_g, &w.norm4_b, 1e-5);

    (queries, keys)
}

/// Top-level two-way transformer forward.
///
/// `image_embedding`: NCHW `[B, C, H, W]` (flat).
/// `image_pe`: same shape.
/// `point_embedding`: `[B, q_n, E]`.
///
/// Returns `(queries, keys)` where queries is `[B, q_n, E]` and keys
/// is `[B, H*W, E]` (after the final LN on queries).
pub fn two_way_transformer_forward(
    w: &Sam2TwoWayTransformerWeights,
    image_embedding: &[f32],
    image_pe: &[f32],
    point_embedding: &[f32],
    b: usize,
    c: usize,
    h: usize,
    ww: usize,
    q_n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let k_n = h * ww;
    let mut image_seq = vec![0f32; b * k_n * c];
    let mut image_pe_seq = vec![0f32; b * k_n * c];
    for bi in 0..b {
        for y in 0..h {
            for x in 0..ww {
                for ch in 0..c {
                    let src = (bi * c + ch) * h * ww + y * ww + x;
                    let dst = (bi * k_n + y * ww + x) * c + ch;
                    image_seq[dst] = image_embedding[src];
                    image_pe_seq[dst] = image_pe[src];
                }
            }
        }
    }

    let mut queries = point_embedding.to_vec();
    let mut keys = image_seq;

    for layer in &w.layers {
        let (q, k) = two_way_attention_block_forward(
            layer,
            queries,
            keys,
            point_embedding,
            &image_pe_seq,
            b,
            q_n,
            k_n,
        );
        queries = q;
        keys = k;
    }

    // Final cross attention token → image
    let mut q_pe = queries.clone();
    add_inplace(&mut q_pe, point_embedding);
    let mut k_pe = keys.clone();
    add_inplace(&mut k_pe, &image_pe_seq);
    let attn_out = sam2_attention_forward(
        &w.final_attn_token_to_image,
        &q_pe,
        q_n,
        &k_pe,
        k_n,
        &keys,
        k_n,
        b,
    );
    add_inplace(&mut queries, &attn_out);
    layer_norm_last(
        &mut queries,
        b * q_n,
        w.embed_dim,
        &w.norm_final_g,
        &w.norm_final_b,
        1e-5,
    );

    (queries, keys)
}
