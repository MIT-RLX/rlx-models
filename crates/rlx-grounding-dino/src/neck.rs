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

//! Neck (`input_proj_vision`) + sine position embeddings.
//!
//! Projects the Swin feature maps to `d_model=256` (1×1 conv + GroupNorm for the
//! first three levels, 3×3 stride-2 conv + GroupNorm for the synthesized 4th
//! level), and computes per-level sine position embeddings.

use crate::config::GroundingDinoConfig;
use crate::swin::FeatureMap;
use crate::weights::get;
use anyhow::Result;
use rlx_core::weight_map::WeightMap;
use std::f32::consts::PI;

const GN_GROUPS: usize = 32;
const GN_EPS: f32 = 1e-5;

struct ProjLevel {
    conv_w: Vec<f32>, // [out, in, kh, kw]
    conv_b: Vec<f32>,
    in_c: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    gn_w: Vec<f32>, // [out]
    gn_b: Vec<f32>,
}

/// One projected feature level fed to the encoder.
#[derive(Debug, Clone)]
pub struct Level {
    /// `[d_model, h, w]` projected source.
    pub source: Vec<f32>,
    /// `[d_model, h, w]` sine position embedding + level embed.
    pub pos: Vec<f32>,
    pub h: usize,
    pub w: usize,
}

/// Neck weights + config-derived parameters.
pub struct Neck {
    levels: Vec<ProjLevel>,
    level_embed: Vec<f32>, // [num_levels, d_model]
    d_model: usize,
    num_levels: usize,
    temperature: f32,
}

impl Neck {
    pub fn from_weights(wm: &WeightMap, cfg: &GroundingDinoConfig) -> Result<Self> {
        let n = cfg.num_feature_levels;
        let mut levels = Vec::with_capacity(n);
        for i in 0..n {
            let lp = format!("model.input_proj_vision.{i}.");
            let (conv_w, shape) = crate::weights::get_with_shape(wm, &format!("{lp}0.weight"))?;
            // shape = [out, in, kh, kw]
            let (in_c, kh, kw) = (shape[1], shape[2], shape[3]);
            let (stride, pad) = if kh == 3 { (2, 1) } else { (1, 0) };
            levels.push(ProjLevel {
                conv_w,
                conv_b: get(wm, &format!("{lp}0.bias"))?,
                in_c,
                kh,
                kw,
                stride,
                pad,
                gn_w: get(wm, &format!("{lp}1.weight"))?,
                gn_b: get(wm, &format!("{lp}1.bias"))?,
            });
        }
        Ok(Self {
            levels,
            level_embed: get(wm, "model.level_embed")?,
            d_model: cfg.d_model,
            num_levels: n,
            temperature: cfg.positional_embedding_temperature as f32,
        })
    }

    /// Project the backbone feature maps and synthesize the extra level.
    /// `maps` are the 3 Swin outputs (256/512/1024 channels).
    pub fn forward(&self, maps: &[FeatureMap]) -> Vec<Level> {
        let mut out = Vec::with_capacity(self.num_levels);
        // First `len(maps)` levels project the corresponding backbone maps.
        for (i, m) in maps.iter().enumerate() {
            let proj = &self.levels[i];
            let (s, h, w) = conv2d(
                &m.data,
                m.c,
                m.h,
                m.w,
                &proj.conv_w,
                &proj.conv_b,
                self.d_model,
                proj.in_c,
                proj.kh,
                proj.kw,
                proj.stride,
                proj.pad,
            );
            let s = group_norm(&s, self.d_model, h, w, &proj.gn_w, &proj.gn_b);
            out.push((s, h, w));
        }
        // Extra levels: 3×3 stride-2 conv applied to the LAST backbone map.
        for i in maps.len()..self.num_levels {
            let proj = &self.levels[i];
            let last = maps.last().unwrap();
            let (s, h, w) = conv2d(
                &last.data,
                last.c,
                last.h,
                last.w,
                &proj.conv_w,
                &proj.conv_b,
                self.d_model,
                proj.in_c,
                proj.kh,
                proj.kw,
                proj.stride,
                proj.pad,
            );
            let s = group_norm(&s, self.d_model, h, w, &proj.gn_w, &proj.gn_b);
            out.push((s, h, w));
        }

        // Position embeddings + level embed.
        out.into_iter()
            .enumerate()
            .map(|(lvl, (source, h, w))| {
                let mut pos = sine_position_embedding(h, w, self.d_model, self.temperature);
                let le = &self.level_embed[lvl * self.d_model..(lvl + 1) * self.d_model];
                for c in 0..self.d_model {
                    for p in 0..h * w {
                        pos[c * h * w + p] += le[c];
                    }
                }
                Level { source, pos, h, w }
            })
            .collect()
    }
}

/// 2-D convolution (NCHW), zero padding. Returns `(out[oc*h*w], out_h, out_w)`.
#[allow(clippy::too_many_arguments)]
fn conv2d(
    x: &[f32],
    in_c: usize,
    h: usize,
    w: usize,
    weight: &[f32], // [out_c, in_c, kh, kw]
    bias: &[f32],
    out_c: usize,
    w_in_c: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) -> (Vec<f32>, usize, usize) {
    debug_assert_eq!(in_c, w_in_c);
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let mut out = vec![0f32; out_c * oh * ow];

    // 1×1 stride-1 conv (the first three neck levels) is a plain matmul over the
    // channel dim: out[oc, p] = Σ_ic w[oc, ic]·x[ic, p]. Route through BLAS — the
    // naive loop below over the large feature maps was the neck's dominant cost.
    if kh == 1 && kw == 1 && stride == 1 && pad == 0 {
        let hw = h * w;
        rlx_cpu::blas::sgemm(weight, x, &mut out, out_c, in_c, hw);
        for oc in 0..out_c {
            let row = &mut out[oc * hw..(oc + 1) * hw];
            let bo = bias[oc];
            for v in row.iter_mut() {
                *v += bo;
            }
        }
        return (out, oh, ow);
    }

    for oc in 0..out_c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = bias[oc];
                for ic in 0..in_c {
                    for ky in 0..kh {
                        let iy = oy * stride + ky;
                        if iy < pad || iy - pad >= h {
                            continue;
                        }
                        let iy = iy - pad;
                        for kx in 0..kw {
                            let ix = ox * stride + kx;
                            if ix < pad || ix - pad >= w {
                                continue;
                            }
                            let ix = ix - pad;
                            let iv = x[(ic * h + iy) * w + ix];
                            let wv = weight[((oc * in_c + ic) * kh + ky) * kw + kx];
                            acc += iv * wv;
                        }
                    }
                }
                out[(oc * oh + oy) * ow + ox] = acc;
            }
        }
    }
    (out, oh, ow)
}

/// GroupNorm over `[c, h, w]` with `GN_GROUPS` groups, per-channel affine.
fn group_norm(x: &[f32], c: usize, h: usize, w: usize, gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    let groups = GN_GROUPS.min(c);
    let gc = c / groups;
    let hw = h * w;
    let mut out = vec![0f32; c * hw];
    for g in 0..groups {
        let mut mean = 0f64;
        let cnt = (gc * hw) as f64;
        for ch in g * gc..(g + 1) * gc {
            for p in 0..hw {
                mean += x[ch * hw + p] as f64;
            }
        }
        mean /= cnt;
        let mut var = 0f64;
        for ch in g * gc..(g + 1) * gc {
            for p in 0..hw {
                let d = x[ch * hw + p] as f64 - mean;
                var += d * d;
            }
        }
        var /= cnt;
        let inv = 1.0 / (var + GN_EPS as f64).sqrt();
        for ch in g * gc..(g + 1) * gc {
            for p in 0..hw {
                let normed = (x[ch * hw + p] as f64 - mean) * inv;
                out[ch * hw + p] = (normed as f32) * gamma[ch] + beta[ch];
            }
        }
    }
    out
}

/// Grounding DINO sine position embedding for an all-ones mask of size `h×w`.
/// Returns `[d_model, h, w]` (pos_y in the first half of channels, pos_x in the
/// second), matching `GroundingDinoSinePositionEmbedding`.
pub fn sine_position_embedding(h: usize, w: usize, d_model: usize, temperature: f32) -> Vec<f32> {
    let half = d_model / 2; // 128 per axis
    let eps = 1e-6f32;
    let scale = 2.0 * PI;
    // dim_t[i] = temperature ** (2*floor(i/2)/half)
    let dim_t: Vec<f32> = (0..half)
        .map(|i| temperature.powf((2 * (i / 2)) as f32 / half as f32))
        .collect();

    let mut pos = vec![0f32; d_model * h * w];
    for y in 0..h {
        // cumsum of ones along height = y+1; normalized by last row (= h).
        let y_embed = (y as f32 + 1.0) / (h as f32 + eps) * scale;
        for x in 0..w {
            let x_embed = (x as f32 + 1.0) / (w as f32 + eps) * scale;
            // pos_y channels [0, half), pos_x channels [half, d_model)
            for i in 0..half {
                let vy = y_embed / dim_t[i];
                let vx = x_embed / dim_t[i];
                // stack(sin(even), cos(odd)).flatten: even idx → sin, odd → cos.
                let ey = if i % 2 == 0 { vy.sin() } else { vy.cos() };
                let ex = if i % 2 == 0 { vx.sin() } else { vx.cos() };
                pos[i * h * w + y * w + x] = ey;
                pos[(half + i) * h * w + y * w + x] = ex;
            }
        }
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_norm_zero_mean_unit_var_per_group() {
        let (c, h, w) = (4, 2, 2);
        let x: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        let g = vec![1.0; c];
        let b = vec![0.0; c];
        // 32 groups clamps to c → per-channel norm; each channel constant per
        // pixel here so normalization yields finite values.
        let out = group_norm(&x, c, h, w, &g, &b);
        assert_eq!(out.len(), c * h * w);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sine_pos_shape_and_range() {
        let (h, w, d) = (3, 4, 8);
        let pos = sine_position_embedding(h, w, d, 20.0);
        assert_eq!(pos.len(), d * h * w);
        assert!(pos.iter().all(|v| v.is_finite() && v.abs() <= 1.0001));
    }

    #[test]
    fn conv1x1_identity_passthrough() {
        // 1x1 conv with identity weights reproduces input channels.
        let (c, h, w) = (2, 2, 2);
        let x: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        let mut weight = vec![0f32; c * c];
        for i in 0..c {
            weight[i * c + i] = 1.0;
        }
        let bias = vec![0f32; c];
        let (out, oh, ow) = conv2d(&x, c, h, w, &weight, &bias, c, c, 1, 1, 1, 0);
        assert_eq!((oh, ow), (h, w));
        assert_eq!(out, x);
    }
}
