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

//! SigLIP2's position table resampled to a patch grid, as
//! `F.interpolate(mode="bilinear", align_corners=False, antialias=True)`
//! does it: plain host math, which the packages built on it (LFM2-VL's
//! programs) build too.

/// Separable float bilinear-antialias resize of a `[side·side·hidden]` grid to
/// `[h·w·hidden]`. Matches PyTorch `F.interpolate(mode="bilinear",
/// align_corners=False, antialias=True)` via Pillow's reducing-filter
/// coefficients (no inter-pass 8-bit rounding).
///
/// This is SigLIP2's `resize_positional_embeddings` for one image; models that
/// embed NaFlex patches themselves (LFM2-VL) call it directly.
pub fn resize_position_grid(
    grid: &[f32],
    side: usize,
    hidden: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    resize_pos_grid(grid, side, hidden, h, w)
}

pub(crate) fn resize_pos_grid(
    grid: &[f32],
    side: usize,
    hidden: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let coeff_h = precompute_coeffs(side, h);
    let coeff_w = precompute_coeffs(side, w);
    let mut out = vec![0f32; h * w * hidden];
    for (oy, (iy0, wy)) in coeff_h.iter().enumerate() {
        for (ox, (ix0, wx)) in coeff_w.iter().enumerate() {
            let o = (oy * w + ox) * hidden;
            for (i, &why) in wy.iter().enumerate() {
                let iy = iy0 + i;
                for (j, &wxx) in wx.iter().enumerate() {
                    let ix = ix0 + j;
                    let coef = why * wxx;
                    if coef == 0.0 {
                        continue;
                    }
                    let src = (iy * side + ix) * hidden;
                    for e in 0..hidden {
                        out[o + e] += coef * grid[src + e];
                    }
                }
            }
        }
    }
    out
}

/// Pillow `precompute_coeffs` (triangle/bilinear filter), antialiased for
/// downscale (`filterscale = max(1, in/out)`). Returns `(window_start, weights)`.
fn precompute_coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 1.0 * filterscale; // bilinear support = 1.0
    let inv = 1.0 / filterscale;
    let kernel = |x: f64| -> f64 {
        let x = x.abs();
        if x < 1.0 { 1.0 - x } else { 0.0 }
    };
    let mut coeffs = Vec::with_capacity(out_size);
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
        let xmin = xmin as usize;
        let n = (xmax as usize).saturating_sub(xmin);
        let mut weights = Vec::with_capacity(n);
        let mut total = 0.0f64;
        for i in 0..n {
            let wv = kernel(((xmin + i) as f64 - center + 0.5) * inv);
            weights.push(wv);
            total += wv;
        }
        if total != 0.0 {
            for wv in &mut weights {
                *wv /= total;
            }
        }
        coeffs.push((xmin, weights.into_iter().map(|x| x as f32).collect()));
    }
    coeffs
}
