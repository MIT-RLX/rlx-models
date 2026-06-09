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

use rlx_cpu::blas;

pub fn silu(x: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o = v / (1.0 + (-v).exp());
    }
}

pub fn conv2d_3x3_pad1(
    input: &[f32],
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_c * h * w];
    for oc in 0..out_c {
        let b = bias[oc];
        for v in out[oc * h * w..(oc + 1) * h * w].iter_mut() {
            *v = b;
        }
    }
    for oc in 0..out_c {
        for ic in 0..in_c {
            let w_oi = &weight[(oc * in_c + ic) * 9..(oc * in_c + ic) * 9 + 9];
            let inp = &input[ic * h * w..(ic + 1) * h * w];
            let oup = &mut out[oc * h * w..(oc + 1) * h * w];
            for oy in 0..h {
                for ox in 0..w {
                    let mut acc = 0.0f32;
                    for ky in 0..3 {
                        let iy = oy as isize + ky as isize - 1;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..3 {
                            let ix = ox as isize + kx as isize - 1;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            acc += inp[iy as usize * w + ix as usize] * w_oi[ky * 3 + kx];
                        }
                    }
                    oup[oy * w + ox] += acc;
                }
            }
        }
    }
    out
}

pub fn conv2d_1x1(
    input: &[f32],
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let n = h * w;
    let mut out = vec![0.0f32; out_c * n];
    blas::sgemm(weight, input, &mut out, out_c, in_c, n);
    for oc in 0..out_c {
        let b = bias[oc];
        for v in out[oc * n..(oc + 1) * n].iter_mut() {
            *v += b;
        }
    }
    out
}

pub fn group_norm(
    x: &[f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    num_groups: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let cpg = channels / num_groups;
    let spatial = h * w;
    let mut out = vec![0.0f32; batch * channels * spatial];
    for b in 0..batch {
        for g in 0..num_groups {
            let c0 = g * cpg;
            let n = (cpg * spatial) as f32;
            let mut mean = 0.0f32;
            for c in 0..cpg {
                let plane = &x
                    [((b * channels + c0 + c) * spatial)..((b * channels + c0 + c + 1) * spatial)];
                mean += plane.iter().sum::<f32>();
            }
            mean /= n;
            let mut var = 0.0f32;
            for c in 0..cpg {
                let plane = &x
                    [((b * channels + c0 + c) * spatial)..((b * channels + c0 + c + 1) * spatial)];
                for &v in plane {
                    let d = v - mean;
                    var += d * d;
                }
            }
            var /= n;
            let inv = 1.0 / (var + eps).sqrt();
            for c in 0..cpg {
                let gi = c0 + c;
                let gamm = gamma[gi];
                let bet = beta[gi];
                let src = &x[((b * channels + gi) * spatial)..((b * channels + gi + 1) * spatial)];
                let dst =
                    &mut out[((b * channels + gi) * spatial)..((b * channels + gi + 1) * spatial)];
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d = (s - mean) * inv * gamm + bet;
                }
            }
        }
    }
    out
}

pub fn downsample_conv2d(
    input: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
) -> (Vec<f32>, usize, usize) {
    // Pad bottom/right by 1 (mflux Flux2Downsample2D), then 3×3 stride-2 conv.
    let h_pad = h + 1;
    let w_pad = w + 1;
    let mut padded = vec![0.0f32; channels * h_pad * w_pad];
    for c in 0..channels {
        let src = &input[c * h * w..(c + 1) * h * w];
        let dst = &mut padded[c * h_pad * w_pad..(c + 1) * h_pad * w_pad];
        for y in 0..h {
            for x in 0..w {
                dst[y * w_pad + x] = src[y * w + x];
            }
        }
    }
    let out_h = (h_pad - 3) / 2 + 1;
    let out_w = (w_pad - 3) / 2 + 1;
    let mut out = vec![0.0f32; channels * out_h * out_w];
    for oc in 0..channels {
        let b = bias[oc];
        for v in out[oc * out_h * out_w..(oc + 1) * out_h * out_w].iter_mut() {
            *v = b;
        }
    }
    for oc in 0..channels {
        for ic in 0..channels {
            let w_oi = &weight[(oc * channels + ic) * 9..(oc * channels + ic) * 9 + 9];
            let inp = &padded[ic * h_pad * w_pad..(ic + 1) * h_pad * w_pad];
            let oup = &mut out[oc * out_h * out_w..(oc + 1) * out_h * out_w];
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = 0.0f32;
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let iy = oy * 2 + ky;
                            let ix = ox * 2 + kx;
                            if iy < h_pad && ix < w_pad {
                                acc += inp[iy * w_pad + ix] * w_oi[ky * 3 + kx];
                            }
                        }
                    }
                    oup[oy * out_w + ox] += acc;
                }
            }
        }
    }
    (out, out_h, out_w)
}

pub fn upsample_nearest_2x(
    input: &[f32],
    channels: usize,
    h: usize,
    w: usize,
) -> (Vec<f32>, usize, usize) {
    let h2 = h * 2;
    let w2 = w * 2;
    let mut out = vec![0.0f32; channels * h2 * w2];
    for c in 0..channels {
        let plane = &input[c * h * w..(c + 1) * h * w];
        let dst = &mut out[c * h2 * w2..(c + 1) * h2 * w2];
        for y in 0..h {
            for x in 0..w {
                let v = plane[y * w + x];
                for dy in 0..2 {
                    for dx in 0..2 {
                        dst[(y * 2 + dy) * w2 + (x * 2 + dx)] = v;
                    }
                }
            }
        }
    }
    (out, h2, w2)
}

/// Spatial self-attention over `H×W` per batch (channels = model dim).
pub fn spatial_attention(
    x: &[f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    to_q_w: &[f32],
    to_q_b: &[f32],
    to_k_w: &[f32],
    to_k_b: &[f32],
    to_v_w: &[f32],
    to_v_b: &[f32],
    to_out_w: &[f32],
    to_out_b: &[f32],
    gn_gamma: &[f32],
    gn_beta: &[f32],
    num_groups: usize,
    eps: f32,
) -> Vec<f32> {
    let seq = h * w;
    let mut out = x.to_vec();
    for b in 0..batch {
        let base = b * channels * seq;
        let xb = &x[base..base + channels * seq];
        let normed = group_norm(xb, 1, channels, h, w, num_groups, gn_gamma, gn_beta, eps);
        let q = conv2d_1x1(&normed, channels, channels, h, w, to_q_w, to_q_b);
        let k = conv2d_1x1(&normed, channels, channels, h, w, to_k_w, to_k_b);
        let v = conv2d_1x1(&normed, channels, channels, h, w, to_v_w, to_v_b);
        let scale = 1.0 / (channels as f32).sqrt();
        let mut fixed = vec![0.0f32; channels * seq];
        for qi in 0..seq {
            let mut scores = vec![0.0f32; seq];
            for kj in 0..seq {
                let mut dot = 0.0f32;
                for c in 0..channels {
                    dot += q[c * seq + qi] * k[c * seq + kj];
                }
                scores[kj] = dot * scale;
            }
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            for kj in 0..seq {
                scores[kj] /= sum;
            }
            for ci in 0..channels {
                let mut acc = 0.0f32;
                for kj in 0..seq {
                    acc += scores[kj] * v[ci * seq + kj];
                }
                fixed[ci * seq + qi] = acc;
            }
        }
        let proj = conv2d_1x1(&fixed, channels, channels, h, w, to_out_w, to_out_b);
        for i in 0..channels * seq {
            out[base + i] = xb[i] + proj[i];
        }
    }
    out
}
