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

//! Attention blocks, norms, and MLP for HOCT.
//!
//! Gated multi-head attention: `q_proj` / `kv_proj` emit Q‖gate and K‖V, 3D RoPE
//! on Q/K, optional distance bias in the mask, then `out *= σ(gate)`.

use crate::config::HoctConfig;
use crate::rope3d::apply_rope3d;
use crate::weights::{AttnWeights, BlockWeights, MlpWeights};
use ndarray::{Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4, Axis, s};

pub fn rms_norm(x: &ArrayView3<f32>, weight: &[f32], eps: f32) -> Array3<f32> {
    let b = x.len_of(Axis(0));
    let n = x.len_of(Axis(1));
    let c = x.len_of(Axis(2));
    let mut out = Array3::<f32>::zeros((b, n, c));
    for bi in 0..b {
        for ni in 0..n {
            let mut sq = 0.0f32;
            for ci in 0..c {
                sq += x[[bi, ni, ci]] * x[[bi, ni, ci]];
            }
            let inv = 1.0 / (sq / c as f32 + eps).sqrt();
            for ci in 0..c {
                out[[bi, ni, ci]] = x[[bi, ni, ci]] * inv * weight[ci];
            }
        }
    }
    out
}

pub fn layer_norm(x: &ArrayView3<f32>, weight: &[f32], bias: &[f32], eps: f32) -> Array3<f32> {
    let b = x.len_of(Axis(0));
    let n = x.len_of(Axis(1));
    let c = x.len_of(Axis(2));
    let mut out = Array3::<f32>::zeros((b, n, c));
    for bi in 0..b {
        for ni in 0..n {
            let mut mean = 0.0f32;
            for ci in 0..c {
                mean += x[[bi, ni, ci]];
            }
            mean /= c as f32;
            let mut var = 0.0f32;
            for ci in 0..c {
                let d = x[[bi, ni, ci]] - mean;
                var += d * d;
            }
            var /= c as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for ci in 0..c {
                out[[bi, ni, ci]] = (x[[bi, ni, ci]] - mean) * inv * weight[ci] + bias[ci];
            }
        }
    }
    out
}

pub fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_56 * (x + 0.044_715 * x * x * x)).tanh())
}

pub fn linear2d(
    x: &ArrayView2<f32>,
    w: &[f32],
    out_f: usize,
    in_f: usize,
    bias: Option<&[f32]>,
) -> Array2<f32> {
    let n = x.len_of(Axis(0));
    let mut out = Array2::<f32>::zeros((n, out_f));
    for i in 0..n {
        for o in 0..out_f {
            // Match Torch fp32 matmul accumulation order (not f64 — that drifts).
            let mut s = 0.0f32;
            for ii in 0..in_f {
                s += x[[i, ii]] * w[o * in_f + ii];
            }
            if let Some(b) = bias {
                s += b[o];
            }
            out[[i, o]] = s;
        }
    }
    out
}

pub fn linear3d(
    x: &ArrayView3<f32>,
    w: &[f32],
    out_f: usize,
    in_f: usize,
    bias: Option<&[f32]>,
) -> Array3<f32> {
    let b = x.len_of(Axis(0));
    let n = x.len_of(Axis(1));
    let mut out = Array3::<f32>::zeros((b, n, out_f));
    for bi in 0..b {
        let row = x.slice(s![bi, .., ..]);
        let y = linear2d(&row, w, out_f, in_f, bias);
        out.slice_mut(s![bi, .., ..]).assign(&y);
    }
    out
}

pub fn mlp(cfg: &HoctConfig, x: &ArrayView3<f32>, w: &MlpWeights) -> Array3<f32> {
    let h1 = linear3d(
        x,
        &w.fc1_weight,
        cfg.mlp_hidden(),
        cfg.hidden_dim,
        Some(&w.fc1_bias),
    );
    let mut h1a = h1.clone();
    h1a.mapv_inplace(gelu_tanh);
    let h2 = linear3d(
        &h1a.view(),
        &w.fc2_weight,
        cfg.hidden_dim,
        cfg.mlp_hidden(),
        Some(&w.fc2_bias),
    );
    h2.mapv(gelu_tanh)
}

fn scaled_dot_product_attn(
    q: &ArrayView4<f32>,
    k: &ArrayView4<f32>,
    v: &ArrayView4<f32>,
    mask: &ArrayView4<f32>,
    head_dim: usize,
) -> Array4<f32> {
    let b = q.len_of(Axis(0));
    let h = q.len_of(Axis(1));
    let n = q.len_of(Axis(2));
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut out = Array4::<f32>::zeros((b, h, n, head_dim));

    let n_k = k.len_of(Axis(2));
    let mask_heads = mask.len_of(Axis(1));
    for bi in 0..b {
        for hi in 0..h {
            let mh = if mask_heads == 1 { 0 } else { hi };
            for qi in 0..n {
                let mut scores = vec![0.0f64; n_k];
                for kj in 0..n_k {
                    let mut dot = 0.0f64;
                    for d in 0..head_dim {
                        dot += q[[bi, hi, qi, d]] as f64 * k[[bi, hi, kj, d]] as f64;
                    }
                    scores[kj] = dot * scale + mask[[bi, mh, qi, kj]] as f64;
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores
                    .iter()
                    .map(|v| {
                        if *v == f64::NEG_INFINITY || v.is_nan() {
                            0.0
                        } else {
                            (v - max).exp()
                        }
                    })
                    .collect();
                let sum: f64 = exps.iter().sum::<f64>().max(1e-12);
                for d in 0..head_dim {
                    let mut acc = 0.0f64;
                    for kj in 0..n_k {
                        acc += (exps[kj] / sum) * v[[bi, hi, kj, d]] as f64;
                    }
                    out[[bi, hi, qi, d]] = acc as f32;
                }
            }
        }
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Gated multi-head attention (shared by node and edge blocks).
pub fn gated_attention(
    cfg: &HoctConfig,
    q_in: &ArrayView3<f32>,
    kv_in: &ArrayView3<f32>,
    pos: &ArrayView3<f32>,
    attn_mask: &ArrayView4<f32>,
    w: &AttnWeights,
) -> Array3<f32> {
    let b = q_in.len_of(Axis(0));
    let n = q_in.len_of(Axis(1));
    let h = cfg.num_heads;
    let hd = cfg.head_dim;
    let c = cfg.hidden_dim;

    let q_flat = linear3d(q_in, &w.q_proj_weight, cfg.q_proj_out(), c, None);
    let kv_flat = linear3d(kv_in, &w.kv_proj_weight, 2 * cfg.qkv_dim(), c, None);

    let mut q_heads = Array4::<f32>::zeros((b, h, n, hd * 2));
    for bi in 0..b {
        for ni in 0..n {
            for hi in 0..h {
                for d in 0..(hd * 2) {
                    q_heads[[bi, hi, ni, d]] = q_flat[[bi, ni, hi * hd * 2 + d]];
                }
            }
        }
    }

    let (q_part, gate) = {
        let mut q = Array4::<f32>::zeros((b, h, n, hd));
        let mut g = Array4::<f32>::zeros((b, h, n, hd));
        for bi in 0..b {
            for hi in 0..h {
                for ni in 0..n {
                    for d in 0..hd {
                        q[[bi, hi, ni, d]] = q_heads[[bi, hi, ni, d]];
                        g[[bi, hi, ni, d]] = q_heads[[bi, hi, ni, d + hd]];
                    }
                }
            }
        }
        (q, g)
    };

    let mut k = Array4::<f32>::zeros((b, h, n, hd));
    let mut v = Array4::<f32>::zeros((b, h, n, hd));
    let heads_stride = h * hd;
    for bi in 0..b {
        for ni in 0..n {
            for hi in 0..h {
                for d in 0..hd {
                    k[[bi, hi, ni, d]] = kv_flat[[bi, ni, hi * hd + d]];
                    v[[bi, hi, ni, d]] = kv_flat[[bi, ni, heads_stride + hi * hd + d]];
                }
            }
        }
    }

    let pos3 = pos.to_owned();
    let q_rope = apply_rope3d(
        cfg,
        &q_part.view(),
        &pos3,
        &w.log_freq,
        &w.reflect_vec,
        &w.eye,
    );
    let k_rope = apply_rope3d(cfg, &k.view(), &pos3, &w.log_freq, &w.reflect_vec, &w.eye);

    let attn_out =
        scaled_dot_product_attn(&q_rope.view(), &k_rope.view(), &v.view(), attn_mask, hd);

    let mut gated = attn_out;
    for bi in 0..b {
        for hi in 0..h {
            for ni in 0..n {
                for d in 0..hd {
                    gated[[bi, hi, ni, d]] *= sigmoid(gate[[bi, hi, ni, d]]);
                }
            }
        }
    }

    let mut merged = Array3::<f32>::zeros((b, n, c));
    for bi in 0..b {
        for ni in 0..n {
            for hi in 0..h {
                for d in 0..hd {
                    merged[[bi, ni, hi * hd + d]] = gated[[bi, hi, ni, d]];
                }
            }
        }
    }

    linear3d(&merged.view(), &w.proj_weight, c, c, Some(&w.proj_bias))
}

pub fn node_block(
    cfg: &HoctConfig,
    x: &ArrayView3<f32>,
    pos: &ArrayView3<f32>,
    attn_mask: &ArrayView4<f32>,
    w: &BlockWeights,
) -> Array3<f32> {
    let normed = rms_norm(x, &w.norm1_x_weight, cfg.rms_eps);
    let attn_out = gated_attention(cfg, &normed.view(), &normed.view(), pos, attn_mask, &w.attn);
    let mut h = x.to_owned();
    h += &attn_out;
    let mlp_in = rms_norm(&h.view(), &w.norm2_weight, cfg.rms_eps);
    let mlp_out = mlp(cfg, &mlp_in.view(), &w.mlp);
    h + &mlp_out
}

pub fn edge_cross_block(
    cfg: &HoctConfig,
    h_e: &ArrayView3<f32>,
    f_e: &ArrayView3<f32>,
    pos: &ArrayView3<f32>,
    attn_mask: &ArrayView4<f32>,
    w: &BlockWeights,
) -> Array3<f32> {
    let q = rms_norm(h_e, &w.norm1_x_weight, cfg.rms_eps);
    let kv = rms_norm(f_e, &w.norm1_y_weight, cfg.rms_eps);
    let attn_out = gated_attention(cfg, &q.view(), &kv.view(), pos, attn_mask, &w.attn);
    let mut h = h_e.to_owned();
    h += &attn_out;
    let mlp_in = rms_norm(&h.view(), &w.norm2_weight, cfg.rms_eps);
    let mlp_out = mlp(cfg, &mlp_in.view(), &w.mlp);
    h + &mlp_out
}

pub fn edge_self_block(
    cfg: &HoctConfig,
    e: &ArrayView3<f32>,
    pos: &ArrayView3<f32>,
    attn_mask: &ArrayView4<f32>,
    w: &BlockWeights,
) -> Array3<f32> {
    let q = rms_norm(e, &w.norm1_x_weight, cfg.rms_eps);
    let kv = rms_norm(e, &w.norm1_y_weight, cfg.rms_eps);
    let attn_out = gated_attention(cfg, &q.view(), &kv.view(), pos, attn_mask, &w.attn);
    let mut h = e.to_owned();
    h += &attn_out;
    let mlp_in = rms_norm(&h.view(), &w.norm2_weight, cfg.rms_eps);
    let mlp_out = mlp(cfg, &mlp_in.view(), &w.mlp);
    h + &mlp_out
}

pub fn softplus(x: f32, beta: f32, threshold: f32) -> f32 {
    if x * beta > threshold {
        x
    } else {
        (1.0 / beta) * (1.0 + (x * beta).exp()).ln()
    }
}

pub fn dist_attn_bias(
    dist: &ArrayView3<f32>,
    dist_scaling: &[f32],
    dist_scaling_head_direction: &[f32],
    base_mask: &ArrayView4<f32>,
    num_heads: usize,
) -> Array4<f32> {
    let b = dist.len_of(Axis(0));
    let e = dist.len_of(Axis(1));
    let mut out = Array4::<f32>::zeros((b, num_heads, e, e));
    for bi in 0..b {
        for hi in 0..num_heads {
            let scale = dist_scaling_head_direction[hi] * softplus(dist_scaling[hi], 1.0, 20.0);
            for i in 0..e {
                for j in 0..e {
                    out[[bi, hi, i, j]] = base_mask[[bi, 0, i, j]] + scale * dist[[bi, i, j]];
                }
            }
        }
    }
    out
}
