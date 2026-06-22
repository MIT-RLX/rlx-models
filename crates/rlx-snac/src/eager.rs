// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// CPU reference decoder (channels-first `[C, T]` flat row-major). Mirrors the
// HF SNAC decoder math (and rlx-orpheus's validated eager port). Used as the
// oracle for cross-backend graph parity.

use crate::model::{DecoderBlockW, ResidualUnitW, SnacWeights};
use rlx_core::HierarchicalCodes;

/// `[C, T]` flat tensor. `idx(c, t) = c*T + t`.
struct Mat {
    data: Vec<f32>,
    c: usize,
    t: usize,
}

fn snake1d(x: &Mat, alpha: &[f32]) -> Mat {
    let mut out = vec![0f32; x.c * x.t];
    for ci in 0..x.c {
        let a = alpha[ci];
        let inv = 1.0 / (a + 1e-9);
        for ti in 0..x.t {
            let v = x.data[ci * x.t + ti];
            let s = (a * v).sin();
            out[ci * x.t + ti] = v + inv * s * s;
        }
    }
    Mat {
        data: out,
        c: x.c,
        t: x.t,
    }
}

/// Same-length conv1d (symmetric zero pad), weight `[c_out, c_in/groups, k]`.
fn conv1d(
    x: &Mat,
    w: &[f32],
    c_out: usize,
    k: usize,
    pad: usize,
    groups: usize,
    dil: usize,
    bias: Option<&[f32]>,
) -> Mat {
    let (c_in, t) = (x.c, x.t);
    let c_in_g = c_in / groups;
    let c_out_g = c_out / groups;
    let dil = dil.max(1);
    let mut out = vec![0f32; c_out * t];
    for grp in 0..groups {
        for cog in 0..c_out_g {
            let oc = grp * c_out_g + cog;
            for ti in 0..t {
                let mut acc = bias.map(|b| b[oc]).unwrap_or(0.0);
                for cig in 0..c_in_g {
                    let ic = grp * c_in_g + cig;
                    for ki in 0..k {
                        let src = ti + ki * dil;
                        if src >= pad && src < t + pad {
                            acc += x.data[ic * t + (src - pad)] * w[(oc * c_in_g + cig) * k + ki];
                        }
                    }
                }
                out[oc * t + ti] = acc;
            }
        }
    }
    Mat {
        data: out,
        c: c_out,
        t,
    }
}

/// ConvTranspose1d, weight `[c_in, c_out/groups, k]`.
fn conv_transpose1d(
    x: &Mat,
    w: &[f32],
    c_out: usize,
    k: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    bias: Option<&[f32]>,
) -> Mat {
    let (c_in, t_in) = (x.c, x.t);
    let t_out = (t_in - 1) * stride + k - 2 * padding + output_padding;
    let mut out = vec![0f32; c_out * t_out];
    if let Some(b) = bias {
        for oc in 0..c_out {
            for to in 0..t_out {
                out[oc * t_out + to] = b[oc];
            }
        }
    }
    for ti in 0..t_in {
        for ic in 0..c_in {
            let xv = x.data[ic * t_in + ti];
            for ki in 0..k {
                let to = ti * stride + ki;
                if to < padding || to >= t_out + padding {
                    continue;
                }
                let ot = to - padding;
                for oc in 0..c_out {
                    out[oc * t_out + ot] += xv * w[(ic * c_out + oc) * k + ki];
                }
            }
        }
    }
    Mat {
        data: out,
        c: c_out,
        t: t_out,
    }
}

fn residual_add(x: &Mat, y: &Mat) -> Mat {
    if x.t == y.t {
        let data: Vec<f32> = x.data.iter().zip(&y.data).map(|(a, b)| a + b).collect();
        return Mat {
            data,
            c: y.c,
            t: y.t,
        };
    }
    let pad = (x.t - y.t) / 2;
    let mut out = y.data.clone();
    for ci in 0..y.c {
        for ti in 0..y.t {
            out[ci * y.t + ti] += x.data[ci * x.t + pad + ti];
        }
    }
    Mat {
        data: out,
        c: y.c,
        t: y.t,
    }
}

fn noise_block(x: &Mat, noise_w: &[f32], noise: &[f32]) -> Mat {
    let h = conv1d(x, noise_w, x.c, 1, 0, 1, 1, None);
    let mut out = x.data.clone();
    for ti in 0..x.t {
        let n = noise[ti];
        for ci in 0..x.c {
            out[ci * x.t + ti] += h.data[ci * x.t + ti] * n;
        }
    }
    Mat {
        data: out,
        c: x.c,
        t: x.t,
    }
}

fn residual_unit(x: &Mat, ru: &ResidualUnitW) -> Mat {
    let pad = ((7 - 1) * ru.conv1_dilation) / 2;
    let mut h = snake1d(x, &ru.snake1_alpha);
    h = conv1d(
        &h,
        &ru.conv1_w,
        ru.dim,
        7,
        pad,
        ru.groups,
        ru.conv1_dilation,
        Some(&ru.conv1_b),
    );
    h = snake1d(&h, &ru.snake2_alpha);
    h = conv1d(&h, &ru.conv2_w, ru.dim, 1, 0, 1, 1, Some(&ru.conv2_b));
    residual_add(x, &h)
}

fn decoder_block(x: &Mat, b: &DecoderBlockW, noise: Option<&[f32]>) -> Mat {
    let mut h = snake1d(x, &b.snake_alpha);
    let padding = b.stride.div_ceil(2);
    let output_padding = b.stride % 2;
    h = conv_transpose1d(
        &h,
        &b.upsample_w,
        b.out_dim,
        2 * b.stride,
        b.stride,
        padding,
        output_padding,
        Some(&b.upsample_b),
    );
    if let Some(n) = noise {
        h = noise_block(&h, &b.noise_w, n);
    }
    for ru in &b.residual_units {
        h = residual_unit(&h, ru);
    }
    h
}

/// Multi-scale RVQ decode from codes → quantized latent `[latent, T]` flat.
pub fn from_codes(w: &SnacWeights, codes: &HierarchicalCodes) -> anyhow::Result<(Vec<f32>, usize)> {
    let cfg = &w.config;
    let latent = w.latent();
    let cbdim = cfg.codebook_dim;
    anyhow::ensure!(
        codes.num_levels() == w.quantizers.len(),
        "expected {} code levels, got {}",
        w.quantizers.len(),
        codes.num_levels()
    );

    let base_len = codes.levels[0].len();
    let finest = cfg.vq_strides[0];
    let t_base = base_len * finest;

    let mut z_q = vec![0f32; latent * t_base];
    for (q, level) in w.quantizers.iter().zip(&codes.levels) {
        let tl = level.len();
        // embed: [cbdim, tl]
        let mut emb = vec![0f32; cbdim * tl];
        for (ti, &code) in level.iter().enumerate() {
            let row = (code as usize) * cbdim;
            for di in 0..cbdim {
                emb[di * tl + ti] = q.codebook[row + di];
            }
        }
        // out_proj: conv1x1 [latent, cbdim, 1] + bias → [latent, tl]
        let emb_m = Mat {
            data: emb,
            c: cbdim,
            t: tl,
        };
        let proj = conv1d(
            &emb_m,
            &q.out_proj_w,
            latent,
            1,
            0,
            1,
            1,
            Some(&q.out_proj_b),
        );
        // repeat_interleave by stride → [latent, t_base]
        let s = q.stride;
        for ci in 0..latent {
            for ti in 0..tl {
                let v = proj.data[ci * tl + ti];
                for r in 0..s {
                    z_q[ci * t_base + ti * s + r] += v;
                }
            }
        }
    }
    Ok((z_q, t_base))
}

/// Multi-scale RVQ *encode*: latent `[latent, T]` → codes. Mirrors SNAC's
/// `ResidualVectorQuantize.forward` (avg-pool → in_proj → L2-normalized
/// nearest-neighbor → out_proj → repeat-interleave, residual loop). Host-side.
pub fn rvq_encode(w: &SnacWeights, latent: &[f32], t: usize) -> anyhow::Result<HierarchicalCodes> {
    let cfg = &w.config;
    let ld = w.latent();
    let cbdim = cfg.codebook_dim;
    let cbsize = cfg.codebook_size;
    let mut residual = latent.to_vec(); // [ld, t]
    let mut levels = Vec::with_capacity(w.quantizers.len());

    for q in &w.quantizers {
        let s = q.stride;
        let tl = t / s;
        let in_w = q.in_proj_w.as_ref().ok_or_else(|| {
            anyhow::anyhow!("in_proj weights not loaded (encode needs a full checkpoint)")
        })?;
        let in_b = q.in_proj_b.as_ref().unwrap();

        // avg_pool1d(stride): [ld, t] → [ld, tl]
        let pooled = if s > 1 {
            let mut p = vec![0f32; ld * tl];
            for c in 0..ld {
                for ti in 0..tl {
                    let mut acc = 0.0;
                    for r in 0..s {
                        acc += residual[c * t + ti * s + r];
                    }
                    p[c * tl + ti] = acc / s as f32;
                }
            }
            p
        } else {
            residual.clone()
        };

        // in_proj 1×1: z_e[d, ti] = in_b[d] + Σ_c in_w[d,c]·pooled[c,ti]  → [cbdim, tl]
        let mut z_e = vec![0f32; cbdim * tl];
        for d in 0..cbdim {
            for ti in 0..tl {
                let mut acc = in_b[d];
                for c in 0..ld {
                    acc += in_w[d * ld + c] * pooled[c * tl + ti];
                }
                z_e[d * tl + ti] = acc;
            }
        }

        // nearest neighbor on L2-normalized encodings + codebook (cosine).
        let cb_norm: Vec<f32> = (0..cbsize)
            .map(|i| {
                let mut n = 0.0;
                for d in 0..cbdim {
                    let v = q.codebook[i * cbdim + d];
                    n += v * v;
                }
                n.sqrt().max(1e-12)
            })
            .collect();
        let mut indices = vec![0u32; tl];
        for ti in 0..tl {
            let mut enc = vec![0f32; cbdim];
            let mut en = 0.0;
            for d in 0..cbdim {
                enc[d] = z_e[d * tl + ti];
                en += enc[d] * enc[d];
            }
            en = en.sqrt().max(1e-12);
            let mut best = 0usize;
            let mut best_dot = f32::NEG_INFINITY;
            for i in 0..cbsize {
                let mut dot = 0.0;
                for d in 0..cbdim {
                    dot += (enc[d] / en) * (q.codebook[i * cbdim + d] / cb_norm[i]);
                }
                if dot > best_dot {
                    best_dot = dot;
                    best = i;
                }
            }
            indices[ti] = best as u32;
        }

        // out_proj(embed(indices)) → [ld, tl], repeat-interleave → [ld, t], residual -=.
        let mut emb = vec![0f32; cbdim * tl];
        for (ti, &idx) in indices.iter().enumerate() {
            for d in 0..cbdim {
                emb[d * tl + ti] = q.codebook[idx as usize * cbdim + d];
            }
        }
        let emb_m = Mat {
            data: emb,
            c: cbdim,
            t: tl,
        };
        let proj = conv1d(&emb_m, &q.out_proj_w, ld, 1, 0, 1, 1, Some(&q.out_proj_b));
        for c in 0..ld {
            for ti in 0..tl {
                let v = proj.data[c * tl + ti];
                for r in 0..s {
                    residual[c * t + ti * s + r] -= v;
                }
            }
        }
        levels.push(indices);
    }
    Ok(HierarchicalCodes::new(levels))
}

/// CPU reference decode: quantized latent `[latent, T]` + per-block noise planes
/// → waveform. Returns the mono PCM.
pub fn decode(w: &SnacWeights, z_q: &[f32], t: usize, noise: &[Vec<f32>]) -> Vec<f32> {
    let latent = w.latent();
    let z = Mat {
        data: z_q.to_vec(),
        c: latent,
        t,
    };

    // init: depthwise k=7 (same pad 3) → pointwise k=1.
    let mut h = conv1d(
        &z,
        &w.init_dw_w,
        latent,
        7,
        3,
        latent,
        1,
        Some(&w.init_dw_b),
    );
    h = conv1d(
        &h,
        &w.init_pw_w,
        w.config.decoder_dim,
        1,
        0,
        1,
        1,
        Some(&w.init_pw_b),
    );

    for (bi, block) in w.blocks.iter().enumerate() {
        let n = if w.config.noise {
            noise.get(bi).map(|v| v.as_slice())
        } else {
            None
        };
        h = decoder_block(&h, block, n);
    }

    h = snake1d(&h, &w.final_snake_alpha);
    h = conv1d(&h, &w.final_conv_w, 1, 7, 3, 1, 1, Some(&w.final_conv_b));
    // SNAC decoder ends with Tanh.
    h.data.iter().map(|v| v.tanh()).collect()
}

/// Lengths of the per-block noise planes for a latent length `t_latent`.
pub fn noise_plane_lengths(decoder_rates: &[usize], t_latent: usize) -> Vec<usize> {
    let mut t = t_latent;
    let mut out = Vec::new();
    for &stride in decoder_rates {
        // upsample length: (t-1)*stride + 2*stride - 2*ceil(stride/2) + stride%2 = t*stride
        let padding = stride.div_ceil(2);
        let op = stride % 2;
        t = (t - 1) * stride + 2 * stride - 2 * padding + op;
        out.push(t);
    }
    out
}
