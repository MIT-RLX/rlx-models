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

//! SAM 2 memory attention — host-side.
//!
//! Mirrors `sam2/modeling/memory_attention.py` and the RoPE-attention
//! helper from `sam2/modeling/sam/transformer.py`. The memory attention
//! is the video-tracking core: each layer self-attends the current
//! frame's image tokens, then cross-attends them to the memory bank
//! (spatial memory tokens from prior frames + object-pointer tokens
//! from prior frame decoder outputs).
//!
//! ## Per-layer structure
//!
//! ```text
//!   tgt = curr_image_tokens   # [B, N_img, d_model]
//!   memory = [spatial_memory_tokens; object_pointer_tokens]  # [B, N_mem, kv_in_dim]
//!
//!   # ── Self-attention (RoPE on Q and K) ──
//!   tgt = tgt + SelfAttn(LN(tgt), LN(tgt), LN(tgt))
//!
//!   # ── Cross-attention to memory (RoPE on Q + spatial part of K) ──
//!   #   `num_k_exclude_rope = number of obj_ptr tokens in K` — those
//!   #   get no rotary encoding (they're positionless).
//!   tgt = tgt + CrossAttn(LN(tgt), memory + memory_pos, memory,
//!                         num_k_exclude_rope=N_obj_ptr)
//!
//!   # ── FFN ──
//!   tgt = tgt + Linear(ReLU(Linear(LN(tgt))))
//! ```
//!
//! ## Axial 2-D RoPE
//!
//! Reference's `compute_axial_cis(dim, end_x, end_y, theta)` builds a
//! per-position complex rotation table where the first `dim/2` head
//! channels rotate by x-coordinate frequencies and the second `dim/2`
//! by y-coordinate frequencies. `apply_rotary_enc` then multiplies
//! each contiguous (real, imag) channel pair by the complex factor.
//! When the memory bank holds `r` frames, `repeat_freqs_k=True`
//! interleave-repeats the per-position freqs `r` times along the
//! sequence axis so K's rotation matches the temporal stacking.

use super::config::Sam2MemoryConfig;
use super::transformer::{layer_norm_last, layer_norm_last_cpu, linear};
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

// ─── Weight structs ─────────────────────────────────────────────────

pub struct Sam2RoPEAttnWeights {
    pub q_w: Vec<f32>, // [internal_dim, embedding_dim]
    pub q_b: Vec<f32>,
    pub k_w: Vec<f32>, // [internal_dim, kv_in_dim]
    pub k_b: Vec<f32>,
    pub v_w: Vec<f32>, // [internal_dim, kv_in_dim]
    pub v_b: Vec<f32>,
    pub out_w: Vec<f32>, // [embedding_dim, internal_dim]
    pub out_b: Vec<f32>,
    pub embedding_dim: usize,
    pub kv_in_dim: usize,
    pub internal_dim: usize,
    pub num_heads: usize,
    pub rope_theta: f32,
    pub rope_feat_size: [usize; 2],
    pub rope_k_repeat: bool,
}

pub struct Sam2MemoryAttentionLayerWeights {
    pub self_attn: Sam2RoPEAttnWeights,
    pub cross_attn: Sam2RoPEAttnWeights,
    pub norm1_g: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub norm2_g: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub norm3_g: Vec<f32>,
    pub norm3_b: Vec<f32>,
    pub linear1_w: Vec<f32>, // [dim_ff, d_model]
    pub linear1_b: Vec<f32>,
    pub linear2_w: Vec<f32>, // [d_model, dim_ff]
    pub linear2_b: Vec<f32>,
    pub pos_enc_at_attn: bool,
    pub pos_enc_at_cross_attn_queries: bool,
    pub pos_enc_at_cross_attn_keys: bool,
    pub d_model: usize,
}

pub struct Sam2MemoryAttentionWeights {
    pub layers: Vec<Sam2MemoryAttentionLayerWeights>,
    pub norm_g: Vec<f32>,
    pub norm_b: Vec<f32>,
    pub d_model: usize,
    pub pos_enc_at_input: bool,
}

// ─── Weight extraction ─────────────────────────────────────────────

fn load_rope_attn(
    weights: &mut WeightMap,
    prefix: &str,
    cfg: &Sam2MemoryConfig,
    is_self: bool,
) -> Result<Sam2RoPEAttnWeights> {
    let d = cfg.d_model;
    let internal_dim = d; // downsample_rate=1 in published configs
    let kv_in_dim = if is_self { d } else { cfg.kv_in_dim };
    let (q_w, sh) = weights.take(&format!("{prefix}.q_proj.weight"))?;
    ensure!(
        sh == vec![internal_dim, d],
        "{prefix}.q_proj.weight shape {sh:?} not [{internal_dim}, {d}]"
    );
    let (q_b, _) = weights.take(&format!("{prefix}.q_proj.bias"))?;
    let (k_w, sh) = weights.take(&format!("{prefix}.k_proj.weight"))?;
    ensure!(
        sh == vec![internal_dim, kv_in_dim],
        "{prefix}.k_proj.weight shape {sh:?} not [{internal_dim}, {kv_in_dim}]"
    );
    let (k_b, _) = weights.take(&format!("{prefix}.k_proj.bias"))?;
    let (v_w, _) = weights.take(&format!("{prefix}.v_proj.weight"))?;
    let (v_b, _) = weights.take(&format!("{prefix}.v_proj.bias"))?;
    let (out_w, sh) = weights.take(&format!("{prefix}.out_proj.weight"))?;
    ensure!(
        sh == vec![d, internal_dim],
        "{prefix}.out_proj.weight shape {sh:?} not [{d}, {internal_dim}]"
    );
    let (out_b, _) = weights.take(&format!("{prefix}.out_proj.bias"))?;
    Ok(Sam2RoPEAttnWeights {
        q_w,
        q_b,
        k_w,
        k_b,
        v_w,
        v_b,
        out_w,
        out_b,
        embedding_dim: d,
        kv_in_dim,
        internal_dim,
        num_heads: cfg.num_heads,
        rope_theta: cfg.rope_theta,
        rope_feat_size: cfg.rope_feat_size,
        rope_k_repeat: cfg.rope_k_repeat,
    })
}

pub fn extract_memory_attention_weights(
    weights: &mut WeightMap,
    cfg: &Sam2MemoryConfig,
) -> Result<Sam2MemoryAttentionWeights> {
    let mut layers = Vec::with_capacity(cfg.num_layers);
    for i in 0..cfg.num_layers {
        let p = format!("memory_attention.layers.{i}");
        let self_attn = load_rope_attn(
            weights,
            &format!("{p}.self_attn"),
            cfg,
            /*is_self=*/ true,
        )?;
        let cross_attn = load_rope_attn(
            weights,
            &format!("{p}.cross_attn_image"),
            cfg,
            /*is_self=*/ false,
        )?;
        let (norm1_g, _) = weights.take(&format!("{p}.norm1.weight"))?;
        let (norm1_b, _) = weights.take(&format!("{p}.norm1.bias"))?;
        let (norm2_g, _) = weights.take(&format!("{p}.norm2.weight"))?;
        let (norm2_b, _) = weights.take(&format!("{p}.norm2.bias"))?;
        let (norm3_g, _) = weights.take(&format!("{p}.norm3.weight"))?;
        let (norm3_b, _) = weights.take(&format!("{p}.norm3.bias"))?;
        let (linear1_w, sh) = weights.take(&format!("{p}.linear1.weight"))?;
        ensure!(
            sh == vec![cfg.dim_feedforward, cfg.d_model],
            "{p}.linear1.weight shape {sh:?} not [{}, {}]",
            cfg.dim_feedforward,
            cfg.d_model
        );
        let (linear1_b, _) = weights.take(&format!("{p}.linear1.bias"))?;
        let (linear2_w, _) = weights.take(&format!("{p}.linear2.weight"))?;
        let (linear2_b, _) = weights.take(&format!("{p}.linear2.bias"))?;
        layers.push(Sam2MemoryAttentionLayerWeights {
            self_attn,
            cross_attn,
            norm1_g,
            norm1_b,
            norm2_g,
            norm2_b,
            norm3_g,
            norm3_b,
            linear1_w,
            linear1_b,
            linear2_w,
            linear2_b,
            pos_enc_at_attn: cfg.pos_enc_at_attn,
            pos_enc_at_cross_attn_queries: cfg.pos_enc_at_cross_attn_queries,
            pos_enc_at_cross_attn_keys: cfg.pos_enc_at_cross_attn_keys,
            d_model: cfg.d_model,
        });
    }
    let (norm_g, _) = weights.take("memory_attention.norm.weight")?;
    let (norm_b, _) = weights.take("memory_attention.norm.bias")?;
    Ok(Sam2MemoryAttentionWeights {
        layers,
        norm_g,
        norm_b,
        d_model: cfg.d_model,
        pos_enc_at_input: cfg.pos_enc_at_input,
    })
}

// ─── Forward ────────────────────────────────────────────────────────

/// Memory attention forward.
///
/// `curr`: current frame's image tokens `[N_img, d_model]` (B=1).
/// `curr_pos`: same-shape positional encoding (sinusoidal 2-D, from
///     the FpnNeck stride-32 level).
/// `memory`: memory bank `[N_mem, kv_in_dim]` — concatenation of
///     `[spatial_tokens; object_pointer_tokens]` in that order.
/// `memory_pos`: same-shape positional encoding for memory. Object-
///     pointer tokens may use zeros for their pos slots — they're
///     excluded from RoPE via `num_obj_ptr_tokens`.
/// `num_obj_ptr_tokens`: count of obj-ptr tokens at the *end* of
///     memory (i.e. `N_mem - num_obj_ptr_tokens` spatial tokens).
pub fn memory_attention_forward(
    w: &Sam2MemoryAttentionWeights,
    curr: &[f32],
    curr_pos: &[f32],
    memory: &[f32],
    memory_pos: &[f32],
    n_img: usize,
    n_mem: usize,
    kv_in_dim: usize,
    num_obj_ptr_tokens: usize,
) -> Result<Vec<f32>> {
    let d = w.d_model;
    ensure!(curr.len() == n_img * d, "curr len mismatch");
    ensure!(curr_pos.len() == n_img * d, "curr_pos len mismatch");
    ensure!(memory.len() == n_mem * kv_in_dim, "memory len mismatch");
    ensure!(
        memory_pos.len() == n_mem * kv_in_dim,
        "memory_pos len mismatch"
    );

    // Apply 0.1·curr_pos at input (reference uses `output = output + 0.1 * curr_pos`).
    let mut output = curr.to_vec();
    if w.pos_enc_at_input {
        for i in 0..output.len() {
            output[i] += 0.1 * curr_pos[i];
        }
    }

    for layer in &w.layers {
        output = memory_attention_layer_forward(
            layer,
            output,
            curr_pos,
            memory,
            memory_pos,
            n_img,
            n_mem,
            kv_in_dim,
            num_obj_ptr_tokens,
        )?;
    }

    layer_norm_last(&mut output, n_img, d, &w.norm_g, &w.norm_b, 1e-5);
    Ok(output)
}

/// Layers + input pos only (no stack final norm).
pub fn memory_attention_forward_layers_only(
    w: &Sam2MemoryAttentionWeights,
    curr: &[f32],
    curr_pos: &[f32],
    memory: &[f32],
    memory_pos: &[f32],
    n_img: usize,
    n_mem: usize,
    kv_in_dim: usize,
    num_obj_ptr_tokens: usize,
) -> Result<Vec<f32>> {
    let _d = w.d_model;
    let mut output = curr.to_vec();
    if w.pos_enc_at_input {
        for i in 0..output.len() {
            output[i] += 0.1 * curr_pos[i];
        }
    }
    for layer in &w.layers {
        output = memory_attention_layer_forward(
            layer,
            output,
            curr_pos,
            memory,
            memory_pos,
            n_img,
            n_mem,
            kv_in_dim,
            num_obj_ptr_tokens,
        )?;
    }
    Ok(output)
}

/// Same as [`memory_attention_forward`] but stack final norm uses the CPU/IR
/// kernel (`layer_norm_last_cpu`). Use when comparing to compiled IR that ends
/// with `Op::LayerNorm`.
pub fn memory_attention_forward_ir_stack(
    w: &Sam2MemoryAttentionWeights,
    curr: &[f32],
    curr_pos: &[f32],
    memory: &[f32],
    memory_pos: &[f32],
    n_img: usize,
    n_mem: usize,
    kv_in_dim: usize,
    num_obj_ptr_tokens: usize,
) -> Result<Vec<f32>> {
    let d = w.d_model;
    let mut output = memory_attention_forward_layers_only(
        w,
        curr,
        curr_pos,
        memory,
        memory_pos,
        n_img,
        n_mem,
        kv_in_dim,
        num_obj_ptr_tokens,
    )?;
    layer_norm_last_cpu(&mut output, n_img, d, &w.norm_g, &w.norm_b, 1e-5);
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn memory_attention_layer_forward(
    w: &Sam2MemoryAttentionLayerWeights,
    mut tgt: Vec<f32>,
    query_pos: &[f32],
    memory: &[f32],
    memory_pos: &[f32],
    n_img: usize,
    n_mem: usize,
    kv_in_dim: usize,
    num_obj_ptr_tokens: usize,
) -> Result<Vec<f32>> {
    let d = w.d_model;

    // ── Self-attention ──
    let mut tgt2 = tgt.clone();
    layer_norm_last(&mut tgt2, n_img, d, &w.norm1_g, &w.norm1_b, 1e-5);
    let q_in = if w.pos_enc_at_attn {
        let mut x = tgt2.clone();
        for i in 0..x.len() {
            x[i] += query_pos[i];
        }
        x
    } else {
        tgt2.clone()
    };
    let k_in = q_in.clone();
    let v_in = tgt2.clone();
    let sa_out = rope_attn_forward(
        &w.self_attn,
        &q_in,
        n_img,
        &k_in,
        n_img,
        &v_in,
        n_img,
        d,
        d,
        /*num_k_exclude_rope=*/ 0,
    );
    for i in 0..tgt.len() {
        tgt[i] += sa_out[i];
    }

    // ── Cross-attention to memory ──
    let mut tgt2 = tgt.clone();
    layer_norm_last(&mut tgt2, n_img, d, &w.norm2_g, &w.norm2_b, 1e-5);
    let q_in = if w.pos_enc_at_cross_attn_queries {
        let mut x = tgt2.clone();
        for i in 0..x.len() {
            x[i] += query_pos[i];
        }
        x
    } else {
        tgt2
    };
    let k_in = if w.pos_enc_at_cross_attn_keys {
        let mut x = memory.to_vec();
        for i in 0..x.len() {
            x[i] += memory_pos[i];
        }
        x
    } else {
        memory.to_vec()
    };
    let ca_out = rope_attn_forward(
        &w.cross_attn,
        &q_in,
        n_img,
        &k_in,
        n_mem,
        memory,
        n_mem,
        d,
        kv_in_dim,
        num_obj_ptr_tokens,
    );
    for i in 0..tgt.len() {
        tgt[i] += ca_out[i];
    }

    // ── FFN ──
    let mut tgt2 = tgt.clone();
    layer_norm_last(&mut tgt2, n_img, d, &w.norm3_g, &w.norm3_b, 1e-5);
    let dim_ff = w.linear1_b.len();
    let mut mid = linear(&tgt2, &w.linear1_w, &w.linear1_b, n_img, d, dim_ff);
    // Reference uses ReLU activation in `memory_attention` (`activation:
    // relu` in the YAML).
    for v in mid.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    let down = linear(&mid, &w.linear2_w, &w.linear2_b, n_img, dim_ff, d);
    for i in 0..tgt.len() {
        tgt[i] += down[i];
    }

    Ok(tgt)
}

#[allow(clippy::too_many_arguments)]
fn rope_attn_forward(
    w: &Sam2RoPEAttnWeights,
    q: &[f32],
    q_n: usize,
    k: &[f32],
    k_n: usize,
    v: &[f32],
    v_n: usize,
    q_in_dim: usize,
    kv_in_dim: usize,
    num_k_exclude_rope: usize,
) -> Vec<f32> {
    let d = w.embedding_dim;
    let id = w.internal_dim;
    let nh = w.num_heads;
    let dh = id / nh;
    let scale = 1.0 / (dh as f32).sqrt();
    let _ = q_in_dim;

    // 1) Projections.
    let q_p = linear(q, &w.q_w, &w.q_b, q_n, d, id);
    let k_p = linear(k, &w.k_w, &w.k_b, k_n, kv_in_dim, id);
    let v_p = linear(v, &w.v_w, &w.v_b, v_n, kv_in_dim, id);

    // 2) Separate heads: [N, id] → [nh, N, dh] (B=1 implicit).
    let q_h = separate_heads_b1(&q_p, q_n, nh, dh);
    let mut k_h = separate_heads_b1(&k_p, k_n, nh, dh);
    let v_h = separate_heads_b1(&v_p, v_n, nh, dh);

    // 3) Apply axial 2-D RoPE to Q and the first `k_n - num_k_exclude_rope`
    //    K positions. Memory bank may be `r` spatial frames stacked → use
    //    `rope_k_repeat=true` to repeat-interleave the freqs `r` times.
    let num_k_rope = k_n.saturating_sub(num_k_exclude_rope);
    let [end_x, end_y] = w.rope_feat_size;
    let spatial = end_x * end_y;
    let q_h = super::axial_rope::apply_axial_rope_2d(
        &q_h,
        nh,
        q_n,
        dh,
        end_x,
        end_y,
        w.rope_theta,
        /*repeat_factor=*/ 1,
    );
    if num_k_rope > 0 {
        let r = if w.rope_k_repeat && num_k_rope >= spatial && num_k_rope.is_multiple_of(spatial) {
            num_k_rope / spatial
        } else {
            1
        };
        let mut k_prefix = vec![0f32; nh * num_k_rope * dh];
        for h in 0..nh {
            let src = &k_h[h * k_n * dh..(h * k_n + num_k_rope) * dh];
            let dst = &mut k_prefix[h * num_k_rope * dh..(h + 1) * num_k_rope * dh];
            dst.copy_from_slice(src);
        }
        let rotated = super::axial_rope::apply_axial_rope_2d(
            &k_prefix,
            nh,
            num_k_rope,
            dh,
            end_x,
            end_y,
            w.rope_theta,
            r,
        );
        for h in 0..nh {
            let src = &rotated[h * num_k_rope * dh..(h + 1) * num_k_rope * dh];
            let dst = &mut k_h[h * k_n * dh..(h * k_n + num_k_rope) * dh];
            dst.copy_from_slice(src);
        }
    }

    // 4) Scaled dot-product attention (no mask).
    let mut out_h = vec![0f32; nh * q_n * dh];
    let mut scores = vec![0f32; q_n * k_n];
    for h in 0..nh {
        for i in 0..q_n {
            for j in 0..k_n {
                let mut acc = 0f32;
                for dd in 0..dh {
                    acc += q_h[(h * q_n + i) * dh + dd] * k_h[(h * k_n + j) * dh + dd];
                }
                scores[i * k_n + j] = acc * scale;
            }
        }
        for i in 0..q_n {
            let row = &mut scores[i * k_n..(i + 1) * k_n];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0f32;
            for vv in row.iter_mut() {
                *vv = (*vv - m).exp();
                s += *vv;
            }
            for vv in row.iter_mut() {
                *vv /= s;
            }
        }
        for i in 0..q_n {
            for dd in 0..dh {
                let mut acc = 0f32;
                for j in 0..k_n {
                    acc += scores[i * k_n + j] * v_h[(h * v_n + j) * dh + dd];
                }
                out_h[(h * q_n + i) * dh + dd] = acc;
            }
        }
    }

    // 5) Recombine heads → [q_n, id]
    let merged = recombine_heads_b1(&out_h, q_n, nh, dh);

    // 6) Output projection.
    linear(&merged, &w.out_w, &w.out_b, q_n, id, d)
}

fn separate_heads_b1(x: &[f32], n: usize, nh: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0f32; nh * n * dh];
    for i in 0..n {
        for h in 0..nh {
            for d in 0..dh {
                out[(h * n + i) * dh + d] = x[i * (nh * dh) + h * dh + d];
            }
        }
    }
    out
}

fn recombine_heads_b1(x: &[f32], n: usize, nh: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * nh * dh];
    for h in 0..nh {
        for i in 0..n {
            for d in 0..dh {
                out[i * (nh * dh) + h * dh + d] = x[(h * n + i) * dh + d];
            }
        }
    }
    out
}
