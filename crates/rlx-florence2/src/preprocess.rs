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

//! CLIP image preprocessing (resize → rescale → normalize) producing
//! `pixel_values` NCHW `[1,3,size,size]`.
//!
//! The resize reproduces Pillow's bicubic resampler (`resample=3`, the same
//! convolution PIL/`CLIPImageProcessor` use), so `pixel_values` match the HF
//! processor to floating-point precision.

use crate::config::{IMAGE_MEAN, IMAGE_STD};

/// Resize an RGB-HWC `u8` image to `out×out` using Pillow bicubic, then
/// rescale (`/255`) and normalize, returning NCHW `[3·out·out]` f32.
pub fn preprocess_rgb(rgb: &[u8], in_h: usize, in_w: usize, out: usize) -> Vec<f32> {
    // Per-channel f32 planes [in_h, in_w].
    let mut planes = [
        vec![0f32; in_h * in_w],
        vec![0f32; in_h * in_w],
        vec![0f32; in_h * in_w],
    ];
    for i in 0..in_h * in_w {
        for (c, plane) in planes.iter_mut().enumerate() {
            plane[i] = rgb[i * 3 + c] as f32;
        }
    }

    // Separable bicubic: horizontal then vertical.
    let mut out_planes = Vec::with_capacity(3);
    for plane in &planes {
        let tmp = resample_axis(plane, in_h, in_w, out, true); // [in_h, out]
        let res = resample_axis(&tmp, in_h, out, out, false); // [out, out]
        out_planes.push(res);
    }

    // Rescale + normalize, lay out NCHW.
    let mut pixel = vec![0f32; 3 * out * out];
    for c in 0..3 {
        let mean = IMAGE_MEAN[c];
        let std = IMAGE_STD[c];
        for i in 0..out * out {
            let v = out_planes[c][i] / 255.0;
            pixel[c * out * out + i] = (v - mean) / std;
        }
    }
    pixel
}

/// Pillow cubic kernel with `a = -0.5`.
fn cubic(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// Resample a `[rows, cols]` row-major plane along one axis, matching Pillow's
/// `ImagingResampleHorizontal`/`Vertical` convolution. `horizontal=true`
/// resamples the column axis (`cols → out_size`, output `[rows, out_size]`);
/// `horizontal=false` resamples the row axis (`rows → out_size`, output
/// `[out_size, cols]`).
fn resample_axis(
    data: &[f32],
    rows: usize,
    cols: usize,
    out_size: usize,
    horizontal: bool,
) -> Vec<f32> {
    const SUPPORT: f64 = 2.0; // cubic
    let in_size = if horizontal { cols } else { rows };
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = SUPPORT * filterscale;

    // Per-output bounds + normalized weights.
    let mut starts = Vec::with_capacity(out_size);
    let mut weights: Vec<Vec<f64>> = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let mut xmin = (center - support + 0.5).floor() as isize;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5).floor() as isize;
        if xmax > in_size as isize {
            xmax = in_size as isize;
        }
        let (xmin, xmax) = (xmin as usize, xmax as usize);
        let mut w = Vec::with_capacity(xmax - xmin);
        let mut sum = 0f64;
        for x in xmin..xmax {
            let val = cubic((x as f64 - center + 0.5) / filterscale);
            w.push(val);
            sum += val;
        }
        if sum != 0.0 {
            for v in &mut w {
                *v /= sum;
            }
        }
        starts.push(xmin);
        weights.push(w);
    }

    if horizontal {
        let mut out = vec![0f32; rows * out_size];
        for r in 0..rows {
            let src = &data[r * cols..(r + 1) * cols];
            for xx in 0..out_size {
                let xmin = starts[xx];
                let mut acc = 0f64;
                for (k, &wk) in weights[xx].iter().enumerate() {
                    acc += src[xmin + k] as f64 * wk;
                }
                out[r * out_size + xx] = acc as f32;
            }
        }
        out
    } else {
        let mut out = vec![0f32; out_size * cols];
        for yy in 0..out_size {
            let ymin = starts[yy];
            for col in 0..cols {
                let mut acc = 0f64;
                for (k, &wk) in weights[yy].iter().enumerate() {
                    acc += data[(ymin + k) * cols + col] as f64 * wk;
                }
                out[yy * cols + col] = acc as f32;
            }
        }
        out
    }
}
