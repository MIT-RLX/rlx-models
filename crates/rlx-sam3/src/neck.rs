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

//! Native SAM3 detection neck (`Sam3DualViTDetNeck` without the SAM2 head).
//!
//! Per-level branch on the last trunk feature `[B, 1024, 72, 72]`:
//!
//! | scale | branch                                                           |
//! |-------|------------------------------------------------------------------|
//! | 4.0   | dconv2x2(1024→512)·GELU·dconv2x2(512→256)·conv1x1(256→256)·conv3x3 |
//! | 2.0   | dconv2x2(1024→512)·conv1x1(512→256)·conv3x3                       |
//! | 1.0   | conv1x1(1024→256)·conv3x3                                         |
//! | 0.5   | maxpool2x2·conv1x1(1024→256)·conv3x3                              |
//!
//! Each branch also emits a sinusoidal positional encoding of matching
//! shape, computed by `position_encoding_sine_sam3`.

use super::config::SAM3_DET_DIM;
use super::neck_branch_ir::Sam3NeckBranchCompiled;
use super::vision_encoder::Sam3VisionOutput;
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_flow::CompileProfile;
use rlx_runtime::Device;

#[derive(Debug, Clone)]
pub struct Sam3FeatureLevel {
    pub features: Vec<f32>, // NCHW flat [c, h, w]
    pub pos: Vec<f32>,
    pub h: usize,
    pub w: usize,
    pub channels: usize,
}

#[derive(Default)]
pub struct Sam3NeckWeights {
    pub loaded: bool,
    pub branches: Vec<Sam3NeckBranch>,
    /// Per-branch IR graphs (filled by [`compile_neck_branches`]).
    pub ir: Vec<Sam3NeckBranchCompiled>,
}

#[derive(Clone, Default)]
pub struct Sam3NeckBranch {
    pub scale: f32,
    /// First deconv if scale ∈ {2.0, 4.0}.
    pub dconv0_w: Option<Vec<f32>>,
    pub dconv0_b: Option<Vec<f32>>,
    /// Second deconv if scale == 4.0.
    pub dconv1_w: Option<Vec<f32>>,
    pub dconv1_b: Option<Vec<f32>>,
    /// 1x1 conv (after the optional resampling).
    pub c1x1_w: Vec<f32>,
    pub c1x1_b: Vec<f32>,
    pub c1x1_in: usize,
    /// 3x3 conv, in_dim == out_dim == d_model.
    pub c3x3_w: Vec<f32>,
    pub c3x3_b: Vec<f32>,
}

pub fn extract_neck_weights(weights: &mut WeightMap) -> Result<Sam3NeckWeights> {
    let prefixes = [
        "detector.backbone.vision_backbone",
        "backbone.vision_backbone",
        "vision_backbone",
    ];
    let scales = [4.0f32, 2.0, 1.0, 0.5];
    let mut branches = Vec::with_capacity(scales.len());
    for (i, scale) in scales.iter().enumerate() {
        let mut found = None;
        for pref in prefixes {
            let base = format!("{pref}.convs.{i}");
            if weights.has(&format!("{base}.conv_1x1.weight")) {
                found = Some(base);
                break;
            }
        }
        let base = found.ok_or_else(|| {
            anyhow::anyhow!("SAM3 neck branch {i} (scale={scale}) not found in checkpoint")
        })?;

        let (dconv0_w, dconv0_b) = if (*scale - 4.0).abs() < 1e-6 {
            let (w, ws) = weights.take(&format!("{base}.dconv_2x2_0.weight"))?;
            ensure!(
                ws == vec![1024, 512, 2, 2],
                "dconv_2x2_0.weight shape {ws:?}"
            );
            let (b, _) = weights.take(&format!("{base}.dconv_2x2_0.bias"))?;
            (Some(w), Some(b))
        } else if (*scale - 2.0).abs() < 1e-6 {
            let (w, ws) = weights.take(&format!("{base}.dconv_2x2.weight"))?;
            ensure!(ws == vec![1024, 512, 2, 2], "dconv_2x2.weight shape {ws:?}");
            let (b, _) = weights.take(&format!("{base}.dconv_2x2.bias"))?;
            (Some(w), Some(b))
        } else {
            (None, None)
        };
        let (dconv1_w, dconv1_b) = if (*scale - 4.0).abs() < 1e-6 {
            let (w, ws) = weights.take(&format!("{base}.dconv_2x2_1.weight"))?;
            ensure!(
                ws == vec![512, 256, 2, 2],
                "dconv_2x2_1.weight shape {ws:?}"
            );
            let (b, _) = weights.take(&format!("{base}.dconv_2x2_1.bias"))?;
            (Some(w), Some(b))
        } else {
            (None, None)
        };

        let (c1x1_w, c1_shape) = weights.take(&format!("{base}.conv_1x1.weight"))?;
        ensure!(c1_shape.len() == 4 && c1_shape[2] == 1 && c1_shape[3] == 1);
        let c1x1_in = c1_shape[1];
        let (c1x1_b, _) = weights.take(&format!("{base}.conv_1x1.bias"))?;
        let (c3x3_w, c3_shape) = weights.take(&format!("{base}.conv_3x3.weight"))?;
        ensure!(
            c3_shape == vec![SAM3_DET_DIM, SAM3_DET_DIM, 3, 3],
            "conv_3x3.weight shape {c3_shape:?}"
        );
        let (c3x3_b, _) = weights.take(&format!("{base}.conv_3x3.bias"))?;

        branches.push(Sam3NeckBranch {
            scale: *scale,
            dconv0_w,
            dconv0_b,
            dconv1_w,
            dconv1_b,
            c1x1_w,
            c1x1_b,
            c1x1_in,
            c3x3_w,
            c3x3_b,
        });
    }

    // Drop the sam2_convs branch — we don't run the SAM2 head in image-only mode.
    for pref in prefixes {
        let base = format!("{pref}.sam2_convs");
        let keys: Vec<String> = weights
            .keys()
            .filter(|k| k.starts_with(&base))
            .map(|s| s.to_string())
            .collect();
        for k in keys {
            let _ = weights.take(&k);
        }
    }

    Ok(Sam3NeckWeights {
        loaded: true,
        branches,
        ir: Vec::new(),
    })
}

/// Compile each neck branch for the given trunk grid / device.
pub fn compile_neck_branches(
    neck: &mut Sam3NeckWeights,
    in_c: usize,
    grid: usize,
    device: Device,
    profile: &CompileProfile,
) -> Result<()> {
    neck.ir = neck
        .branches
        .iter()
        .map(|b| Sam3NeckBranchCompiled::compile_with_profile(b, in_c, grid, grid, device, profile))
        .collect::<Result<_>>()?;
    Ok(())
}

pub fn apply_neck_native(
    weights: &mut Sam3NeckWeights,
    vision: &Sam3VisionOutput,
) -> Result<Vec<Sam3FeatureLevel>> {
    ensure!(
        weights.loaded,
        "SAM3 neck weights not loaded; call extract_neck_weights()"
    );
    let grid = vision.grid;
    let dim = vision.dim;

    // Vision output is NHWC `[grid*grid, dim]`. Reshape to NCHW for convs.
    let mut x_nchw = vec![0f32; dim * grid * grid];
    for y in 0..grid {
        for xc in 0..grid {
            for c in 0..dim {
                x_nchw[c * grid * grid + y * grid + xc] = vision.tokens[(y * grid + xc) * dim + c];
            }
        }
    }

    let mut levels = Vec::with_capacity(weights.branches.len());
    if weights.ir.len() == weights.branches.len() {
        for compiled in &mut weights.ir {
            let features = compiled.run(&x_nchw, dim, grid, grid)?;
            let pos = position_encoding_sine_sam3(SAM3_DET_DIM, compiled.out_h, compiled.out_w);
            levels.push(Sam3FeatureLevel {
                features,
                pos,
                h: compiled.out_h,
                w: compiled.out_w,
                channels: SAM3_DET_DIM,
            });
        }
    } else {
        for branch in &weights.branches {
            let level = apply_branch_host(branch, &x_nchw, dim, grid, grid)?;
            levels.push(level);
        }
    }
    Ok(levels)
}

fn apply_branch_host(
    branch: &Sam3NeckBranch,
    x: &[f32],
    in_c: usize,
    h: usize,
    w: usize,
) -> Result<Sam3FeatureLevel> {
    let mut cur = x.to_vec();
    let mut cur_c = in_c;
    let mut cur_h = h;
    let mut cur_w = w;

    if (branch.scale - 4.0).abs() < 1e-6 {
        let dw0 = branch.dconv0_w.as_ref().unwrap();
        let db0 = branch.dconv0_b.as_ref().unwrap();
        cur = conv_transpose2d_stride2_k2(&cur, cur_c, 512, cur_h, cur_w, dw0, db0);
        cur_c = 512;
        cur_h *= 2;
        cur_w *= 2;
        gelu_inplace(&mut cur);
        let dw1 = branch.dconv1_w.as_ref().unwrap();
        let db1 = branch.dconv1_b.as_ref().unwrap();
        cur = conv_transpose2d_stride2_k2(&cur, cur_c, 256, cur_h, cur_w, dw1, db1);
        cur_c = 256;
        cur_h *= 2;
        cur_w *= 2;
    } else if (branch.scale - 2.0).abs() < 1e-6 {
        let dw = branch.dconv0_w.as_ref().unwrap();
        let db = branch.dconv0_b.as_ref().unwrap();
        cur = conv_transpose2d_stride2_k2(&cur, cur_c, 512, cur_h, cur_w, dw, db);
        cur_c = 512;
        cur_h *= 2;
        cur_w *= 2;
    } else if (branch.scale - 0.5).abs() < 1e-6 {
        cur = maxpool2x2_stride2(&cur, cur_c, cur_h, cur_w);
        cur_h /= 2;
        cur_w /= 2;
        // cur_c unchanged.
    }
    ensure!(cur_c == branch.c1x1_in, "branch input channels mismatch");

    // 1×1 conv: cur_c → SAM3_DET_DIM.
    cur = conv2d_1x1(
        &cur,
        cur_c,
        SAM3_DET_DIM,
        cur_h,
        cur_w,
        &branch.c1x1_w,
        &branch.c1x1_b,
    );
    cur_c = SAM3_DET_DIM;

    // 3×3 conv with padding=1 stride=1.
    cur = conv2d_3x3_pad1(&cur, cur_c, cur_h, cur_w, &branch.c3x3_w, &branch.c3x3_b);

    let pos = position_encoding_sine_sam3(SAM3_DET_DIM, cur_h, cur_w);
    Ok(Sam3FeatureLevel {
        features: cur,
        pos,
        h: cur_h,
        w: cur_w,
        channels: cur_c,
    })
}

fn gelu_inplace(x: &mut [f32]) {
    // PyTorch's default `nn.GELU()` is the exact (erf-based) form. We
    // approximate `erf` with a high-accuracy Abramowitz-Stegun series so
    // we don't pick up a new dep just for the neck branch.
    let inv_sqrt2 = 1.0f32 / std::f32::consts::SQRT_2;
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf_approx(*v * inv_sqrt2));
    }
}

fn erf_approx(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26. Max abs error ≈ 1.5e-7, plenty for f32.
    let sign = if x < 0.0 { -1.0f32 } else { 1.0 };
    let ax = x.abs();
    let p = 0.3275911f32;
    let a1 = 0.2548296f32;
    let a2 = -0.2844967f32;
    let a3 = 1.4214138f32;
    let a4 = -1.4531521f32;
    let a5 = 1.0614054f32;
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

fn maxpool2x2_stride2(input: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let out_h = h / 2;
    let out_w = w / 2;
    let mut out = vec![0f32; c * out_h * out_w];
    for cc in 0..c {
        let inp = &input[cc * h * w..(cc + 1) * h * w];
        let oup = &mut out[cc * out_h * out_w..(cc + 1) * out_h * out_w];
        for oy in 0..out_h {
            for ox in 0..out_w {
                let iy = oy * 2;
                let ix = ox * 2;
                let a = inp[iy * w + ix];
                let b = inp[iy * w + ix + 1];
                let cv = inp[(iy + 1) * w + ix];
                let d = inp[(iy + 1) * w + ix + 1];
                oup[oy * out_w + ox] = a.max(b).max(cv).max(d);
            }
        }
    }
    out
}

fn conv2d_1x1(
    input: &[f32],
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weight: &[f32], // [out_c, in_c, 1, 1] → row-major [out_c, in_c]
    bias: &[f32],
) -> Vec<f32> {
    // Treat the spatial dims as the GEMM K-axis-free batch; weights map
    // channels through a single matmul: out[oc, n] = sum_ic w[oc, ic] * in[ic, n].
    // Use sgemm: A = weight [out_c, in_c], B = input [in_c, hw], C [out_c, hw].
    let n = h * w;
    let mut out = vec![0f32; out_c * n];
    rlx_cpu::blas::sgemm(weight, input, &mut out, out_c, in_c, n);
    // Add bias.
    for oc in 0..out_c {
        let b = bias[oc];
        let row = &mut out[oc * n..(oc + 1) * n];
        for v in row {
            *v += b;
        }
    }
    out
}

fn conv2d_3x3_pad1(
    input: &[f32],
    c: usize,
    h: usize,
    w: usize,
    weight: &[f32], // [out_c=c, in_c=c, 3, 3]
    bias: &[f32],
) -> Vec<f32> {
    let mut out = vec![0f32; c * h * w];
    for oc in 0..c {
        let b = bias[oc];
        let oup = &mut out[oc * h * w..(oc + 1) * h * w];
        for v in oup.iter_mut() {
            *v = b;
        }
    }
    for oc in 0..c {
        for ic in 0..c {
            let w_oi = &weight[((oc * c + ic) * 9)..((oc * c + ic) * 9 + 9)];
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

fn conv_transpose2d_stride2_k2(
    input: &[f32],
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weight: &[f32], // PyTorch ConvTranspose2d weight: [in_c, out_c, k, k]
    bias: &[f32],
) -> Vec<f32> {
    let out_h = h * 2;
    let out_w = w * 2;
    let mut out = vec![0f32; out_c * out_h * out_w];
    for oc in 0..out_c {
        let b = bias[oc];
        let plane = &mut out[oc * out_h * out_w..(oc + 1) * out_h * out_w];
        for v in plane.iter_mut() {
            *v = b;
        }
    }
    for ic in 0..in_c {
        for iy in 0..h {
            for ix in 0..w {
                let v = input[ic * h * w + iy * w + ix];
                if v == 0.0 {
                    continue;
                }
                for ky in 0..2 {
                    let oy = iy * 2 + ky;
                    for kx in 0..2 {
                        let ox = ix * 2 + kx;
                        for oc in 0..out_c {
                            let w_idx = ((ic * out_c + oc) * 2 + ky) * 2 + kx;
                            out[oc * out_h * out_w + oy * out_w + ox] += v * weight[w_idx];
                        }
                    }
                }
            }
        }
    }
    out
}

/// SAM3-flavour 2D sinusoidal positional encoding (matches
/// `sam3.model.position_encoding.PositionEmbeddingSine`). Output shape
/// `[d_model, h, w]` (NCHW) with `d_model = 2 * num_pos_feats`.
pub fn position_encoding_sine_sam3(d_model: usize, h: usize, w: usize) -> Vec<f32> {
    assert!(d_model.is_multiple_of(2), "d_model must be even");
    let num_pos_feats = d_model / 2;
    let scale = 2.0 * std::f32::consts::PI;
    let eps = 1e-6f32;
    let temperature = 10000.0f32;

    let mut dim_t = vec![0.0f32; num_pos_feats];
    for i in 0..num_pos_feats {
        let exp = 2.0 * ((i / 2) as f32) / num_pos_feats as f32;
        dim_t[i] = temperature.powf(exp);
    }

    let mut out = vec![0.0f32; d_model * h * w];
    let y_denom = h as f32 + eps; // last row index after +1 is h
    let x_denom = w as f32 + eps;

    for y in 0..h {
        let y_norm = ((y + 1) as f32) / y_denom * scale;
        for x in 0..w {
            let x_norm = ((x + 1) as f32) / x_denom * scale;
            // pos_y in the first num_pos_feats channels, pos_x in the second.
            for i in 0..num_pos_feats {
                let py = y_norm / dim_t[i];
                let v = if i % 2 == 0 { py.sin() } else { py.cos() };
                out[i * h * w + y * w + x] = v;
            }
            for i in 0..num_pos_feats {
                let px = x_norm / dim_t[i];
                let v = if i % 2 == 0 { px.sin() } else { px.cos() };
                out[(num_pos_feats + i) * h * w + y * w + x] = v;
            }
        }
    }
    out
}
