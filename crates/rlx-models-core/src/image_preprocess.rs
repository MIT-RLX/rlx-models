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

//! Pure-Rust image preprocessing matching PIL / `torchvision` — no Python.
//!
//! Vision checkpoints (CLIP, DINOv2, SigLIP, …) are trained against
//! PIL's resampler. To reproduce their inference numerics we need the
//! *same* resize, not just "a bicubic resize": PIL convolves with a
//! cubic kernel whose support is widened by the downscale factor
//! (antialiasing), normalizes the per-output-pixel weights, and runs two
//! 8-bit passes (horizontal then vertical). This module re-implements
//! that algorithm (`precompute_coeffs` + `ResampleHorizontal/Vertical`
//! from Pillow's `src/libImaging/Resample.c`) in safe Rust.
//!
//! [`ImagePreprocessor`] wraps the common
//! resize-(shortest|longest)-side → center-crop → normalize → NCHW
//! pipeline so any model crate can share it:
//!
//! ```no_run
//! use rlx_models_core::image_preprocess::ImagePreprocessor;
//! let pp = ImagePreprocessor::clip(
//!     224,
//!     [0.481_454_66, 0.457_827_5, 0.408_210_73],
//!     [0.268_629_54, 0.261_302_6, 0.275_777_1],
//! );
//! let nchw = pp.load("photo.jpg")?; // [3 * 224 * 224] f32, CLIP-normalized
//! # anyhow::Ok(())
//! ```

use anyhow::{Context, Result};
use std::path::Path;

/// Resampling filter (PIL `Resampling`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// Cubic convolution, `a = -0.5` (PIL `BICUBIC`). Support 2.0.
    Bicubic,
    /// Triangle (PIL `BILINEAR`). Support 1.0.
    Bilinear,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Bicubic => 2.0,
            Filter::Bilinear => 1.0,
        }
    }

    /// Filter kernel evaluated at `x` (already in filter space).
    fn kernel(self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            Filter::Bilinear => {
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            Filter::Bicubic => {
                // a = -0.5 (Pillow's `bicubic_filter`).
                const A: f64 = -0.5;
                if x < 1.0 {
                    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * A
                } else {
                    0.0
                }
            }
        }
    }
}

/// How the source is scaled relative to the crop target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeMode {
    /// Resize the *shortest* side to `size` (preserve aspect), then crop.
    /// This is `torchvision.Resize(size: int)` / open_clip "shortest".
    Shortest,
    /// Resize the *longest* side to `size` (preserve aspect), then crop.
    Longest,
    /// Resize to exactly `size × size` (ignore aspect ratio).
    Exact,
}

/// Reusable image → normalized NCHW tensor preprocessor.
#[derive(Clone, Debug)]
pub struct ImagePreprocessor {
    /// Output square side length.
    pub size: usize,
    /// Per-channel mean (RGB), applied to pixels scaled to `[0, 1]`.
    pub mean: [f32; 3],
    /// Per-channel std (RGB).
    pub std: [f32; 3],
    /// Resampling filter used for the resize step.
    pub filter: Filter,
    /// How the source is scaled relative to the crop target.
    pub resize_mode: ResizeMode,
    /// Center-crop to `size × size` after resizing (PIL `CenterCrop`).
    pub center_crop: bool,
}

impl ImagePreprocessor {
    /// CLIP / OpenCLIP inference preprocessing: bicubic resize of the
    /// shortest side, center crop, normalize.
    pub fn clip(size: usize, mean: [f32; 3], std: [f32; 3]) -> Self {
        Self {
            size,
            mean,
            std,
            filter: Filter::Bicubic,
            resize_mode: ResizeMode::Shortest,
            center_crop: true,
        }
    }

    /// Load an image from disk and produce a normalized NCHW tensor
    /// `[3 · size · size]`.
    pub fn load(&self, path: impl AsRef<Path>) -> Result<Vec<f32>> {
        let path = path.as_ref();
        let img = image::open(path).with_context(|| format!("opening image {path:?}"))?;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        Ok(self.from_rgb(rgb.as_raw(), w, h))
    }

    /// Preprocess an in-memory HWC RGB8 buffer (`len == w · h · 3`).
    pub fn from_rgb(&self, rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
        let (cropped, cw, ch) = self.resize_and_crop(rgb, w, h);
        debug_assert_eq!((cw, ch), (self.size, self.size));
        let s = self.size;
        let mut out = vec![0f32; 3 * s * s];
        for y in 0..s {
            for x in 0..s {
                for c in 0..3 {
                    let v = cropped[(y * cw + x) * 3 + c] as f32 / 255.0;
                    out[c * s * s + y * s + x] = (v - self.mean[c]) / self.std[c];
                }
            }
        }
        out
    }

    /// Resize then center-crop to the target square, returning the HWC
    /// RGB8 crop and its dimensions.
    pub fn resize_and_crop(&self, rgb: &[u8], w: usize, h: usize) -> (Vec<u8>, usize, usize) {
        let (dw, dh) = self.resized_dims(w, h);
        let resized = pil_resize_rgb8(rgb, w, h, dw, dh, self.filter);
        if !self.center_crop {
            return (resized, dw, dh);
        }
        let s = self.size;
        let cx = dw.saturating_sub(s) / 2;
        let cy = dh.saturating_sub(s) / 2;
        let mut out = vec![0u8; s * s * 3];
        for y in 0..s {
            let sy = (cy + y).min(dh - 1);
            for x in 0..s {
                let sx = (cx + x).min(dw - 1);
                let src = (sy * dw + sx) * 3;
                let dst = (y * s + x) * 3;
                out[dst..dst + 3].copy_from_slice(&resized[src..src + 3]);
            }
        }
        (out, s, s)
    }

    /// Target resize dimensions for the configured mode.
    fn resized_dims(&self, w: usize, h: usize) -> (usize, usize) {
        let size = self.size as f64;
        let round = |v: f64| v.round().max(1.0) as usize;
        match self.resize_mode {
            ResizeMode::Exact => (self.size, self.size),
            ResizeMode::Shortest => {
                if w <= h {
                    (self.size, round(h as f64 * size / w as f64))
                } else {
                    (round(w as f64 * size / h as f64), self.size)
                }
            }
            ResizeMode::Longest => {
                if w >= h {
                    (self.size, round(h as f64 * size / w as f64))
                } else {
                    (round(w as f64 * size / h as f64), self.size)
                }
            }
        }
    }
}

/// Per-output-pixel resample coefficients: `(window_start, weights)`.
/// Faithful port of Pillow's `precompute_coeffs` (antialiased: the
/// filter support is scaled by the downscale ratio).
fn precompute_coeffs(in_size: usize, out_size: usize, filter: Filter) -> Vec<(usize, Vec<f32>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let inv = 1.0 / filterscale;

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
            let w = filter.kernel(((xmin + i) as f64 - center + 0.5) * inv);
            weights.push(w);
            total += w;
        }
        if total != 0.0 {
            for w in &mut weights {
                *w /= total;
            }
        }
        coeffs.push((xmin, weights.into_iter().map(|w| w as f32).collect()));
    }
    coeffs
}

/// Pillow's `Image.resize(size, filter)` on an RGB image, bit-exact.
///
/// [`pil_resize_rgb8`] accumulates in `f32`; Pillow quantizes the filter
/// weights to 22-bit fixed point and accumulates in integers, and the two
/// disagree by one level on a few pixels in 10^4 on a large downscale. This
/// is Pillow's arithmetic (`normalize_coeffs_8bpc` + `ImagingResample*_8bpc`),
/// for callers whose reference is Pillow's exact output — LFM2-VL's
/// `cap_pixels`, for one.
pub fn pil_resize_rgb8_exact(
    src: &[u8],
    w: usize,
    h: usize,
    dw: usize,
    dh: usize,
    filter: Filter,
) -> Vec<u8> {
    fixed_point_resize_rgb8(src, w, h, dw, dh, filter, WeightPrecision::Pillow)
}

/// PyTorch's antialiased resize of a `uint8` RGB image —
/// `torchvision.transforms.v2.functional.resize(t, [dh, dw], interpolation,
/// antialias=True)` on a `uint8` tensor, as HF image processors built on the
/// torchvision backend (LFM2-VL, Qwen-VL fast processors, …) call it.
///
/// Same separable structure as Pillow (horizontal pass, then vertical, with a
/// rounded 8-bit intermediate) and the same filter windows, but torch
/// quantizes the weights to **int16 at a per-axis precision**: the largest
/// `p < 22` with `round(w_max · 2^(p+1)) < 2^15`. Pillow uses a fixed 22-bit
/// precision instead, which disagrees with torch by one level on ~0.2% of
/// pixels. Bit-exact against torch 2.14 / torchvision 0.29 on downscales and
/// upscales.
pub fn torch_resize_rgb8_aa(
    src: &[u8],
    w: usize,
    h: usize,
    dw: usize,
    dh: usize,
    filter: Filter,
) -> Vec<u8> {
    fixed_point_resize_rgb8(src, w, h, dw, dh, filter, WeightPrecision::TorchInt16)
}

/// How filter weights are quantized before the integer accumulation.
#[derive(Clone, Copy)]
enum WeightPrecision {
    /// Fixed 22 bits (Pillow's `PRECISION_BITS = 32 - 8 - 2`).
    Pillow,
    /// One precision per axis, the most that keeps the largest weight in
    /// int16 (torch's `_compute_int16_weights`).
    TorchInt16,
}

/// Separable two-pass integer resample (horizontal, rounded to 8 bits,
/// then vertical) shared by Pillow's and torch's 8-bit paths; they differ
/// only in [`WeightPrecision`].
fn fixed_point_resize_rgb8(
    src: &[u8],
    w: usize,
    h: usize,
    dw: usize,
    dh: usize,
    filter: Filter,
    precision: WeightPrecision,
) -> Vec<u8> {
    if w == dw && h == dh {
        return src.to_vec();
    }
    let (hcoeffs, hp) = fixed_point_coeffs(w, dw, filter, precision);
    let mut tmp = vec![0u8; dw * h * 3];
    for y in 0..h {
        let row = y * w * 3;
        for (ox, (xmin, ws)) in hcoeffs.iter().enumerate() {
            let mut acc = [1i64 << (hp - 1); 3];
            for (i, &wt) in ws.iter().enumerate() {
                let p = row + (xmin + i) * 3;
                for (c, a) in acc.iter_mut().enumerate() {
                    *a += wt * src[p + c] as i64;
                }
            }
            let d = (y * dw + ox) * 3;
            for (c, a) in acc.iter().enumerate() {
                tmp[d + c] = (a >> hp).clamp(0, 255) as u8;
            }
        }
    }
    let (vcoeffs, vp) = fixed_point_coeffs(h, dh, filter, precision);
    let mut out = vec![0u8; dw * dh * 3];
    for (oy, (ymin, ws)) in vcoeffs.iter().enumerate() {
        for ox in 0..dw {
            let mut acc = [1i64 << (vp - 1); 3];
            for (i, &wt) in ws.iter().enumerate() {
                let p = ((ymin + i) * dw + ox) * 3;
                for (c, a) in acc.iter_mut().enumerate() {
                    *a += wt * tmp[p + c] as i64;
                }
            }
            let d = (oy * dw + ox) * 3;
            for (c, a) in acc.iter().enumerate() {
                out[d + c] = (a >> vp).clamp(0, 255) as u8;
            }
        }
    }
    out
}

/// Filter windows as [`precompute_coeffs`] (in `f64`), quantized at
/// `precision` with rounding half away from zero. Returns the windows and
/// the precision in bits.
fn fixed_point_coeffs(
    in_size: usize,
    out_size: usize,
    filter: Filter,
    precision: WeightPrecision,
) -> (Vec<(usize, Vec<i64>)>, u32) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let inv = 1.0 / filterscale;
    let mut windows = Vec::with_capacity(out_size);
    let mut w_max = 0.0f64;
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5).floor() as isize).max(0) as usize;
        let xmax = ((center + support + 0.5).floor() as isize).min(in_size as isize) as usize;
        let mut ws: Vec<f64> = (0..xmax.saturating_sub(xmin))
            .map(|i| filter.kernel(((xmin + i) as f64 - center + 0.5) * inv))
            .collect();
        let total: f64 = ws.iter().sum();
        if total != 0.0 {
            for v in &mut ws {
                *v /= total;
            }
        }
        w_max = ws.iter().copied().fold(w_max, f64::max);
        windows.push((xmin, ws));
    }
    let bits = match precision {
        WeightPrecision::Pillow => 22,
        WeightPrecision::TorchInt16 => {
            let mut p = 0u32;
            while p < 22 {
                if (0.5 + w_max * (1u64 << (p + 1)) as f64) as i64 >= (1 << 15) {
                    break;
                }
                p += 1;
            }
            p
        }
    };
    let one = (1u64 << bits) as f64;
    let quantized = windows
        .into_iter()
        .map(|(xmin, ws)| {
            let q = ws
                .into_iter()
                .map(|v| {
                    let s = v * one;
                    (if v < 0.0 { s - 0.5 } else { s + 0.5 }) as i64
                })
                .collect();
            (xmin, q)
        })
        .collect();
    (quantized, bits)
}

#[inline]
fn clip8(v: f32) -> u8 {
    // Pillow rounds half up then clamps to the u8 range.
    let r = (v + 0.5).floor();
    if r <= 0.0 {
        0
    } else if r >= 255.0 {
        255
    } else {
        r as u8
    }
}

/// PIL-equivalent two-pass (horizontal then vertical) resample of an HWC
/// RGB8 image, with 8-bit rounding between passes.
pub fn pil_resize_rgb8(
    src: &[u8],
    w: usize,
    h: usize,
    dw: usize,
    dh: usize,
    filter: Filter,
) -> Vec<u8> {
    if w == dw && h == dh {
        return src.to_vec();
    }

    // Horizontal pass: (w × h) → (dw × h).
    let hcoeffs = precompute_coeffs(w, dw, filter);
    let mut tmp = vec![0u8; dw * h * 3];
    for y in 0..h {
        let row = y * w * 3;
        for (ox, (xmin, ws)) in hcoeffs.iter().enumerate() {
            let mut acc = [0f32; 3];
            for (i, &wt) in ws.iter().enumerate() {
                let p = row + (xmin + i) * 3;
                acc[0] += wt * src[p] as f32;
                acc[1] += wt * src[p + 1] as f32;
                acc[2] += wt * src[p + 2] as f32;
            }
            let d = (y * dw + ox) * 3;
            tmp[d] = clip8(acc[0]);
            tmp[d + 1] = clip8(acc[1]);
            tmp[d + 2] = clip8(acc[2]);
        }
    }

    // Vertical pass: (dw × h) → (dw × dh).
    let vcoeffs = precompute_coeffs(h, dh, filter);
    let mut out = vec![0u8; dw * dh * 3];
    for (oy, (ymin, ws)) in vcoeffs.iter().enumerate() {
        for ox in 0..dw {
            let mut acc = [0f32; 3];
            for (i, &wt) in ws.iter().enumerate() {
                let p = ((ymin + i) * dw + ox) * 3;
                acc[0] += wt * tmp[p] as f32;
                acc[1] += wt * tmp[p + 1] as f32;
                acc[2] += wt * tmp[p + 2] as f32;
            }
            let d = (oy * dw + ox) * 3;
            out[d] = clip8(acc[0]);
            out[d + 1] = clip8(acc[1]);
            out[d + 2] = clip8(acc[2]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_resize_is_noop() {
        let src: Vec<u8> = (0..4 * 4 * 3).map(|i| (i % 256) as u8).collect();
        let out = pil_resize_rgb8(&src, 4, 4, 4, 4, Filter::Bicubic);
        assert_eq!(src, out);
    }

    #[test]
    fn coeffs_sum_to_one() {
        for (_, ws) in precompute_coeffs(100, 37, Filter::Bicubic) {
            let s: f32 = ws.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "weights sum {s}");
        }
    }

    #[test]
    fn upscale_preserves_flat_color() {
        // A constant image must stay constant through resize (weights
        // normalize to 1) regardless of scale direction.
        let src = vec![123u8; 8 * 6 * 3];
        let up = pil_resize_rgb8(&src, 8, 6, 20, 15, Filter::Bicubic);
        assert!(up.iter().all(|&v| v == 123));
        let down = pil_resize_rgb8(&src, 8, 6, 3, 2, Filter::Bicubic);
        assert!(down.iter().all(|&v| v == 123));
    }

    #[test]
    fn center_crop_dims() {
        let pp = ImagePreprocessor::clip(4, [0.0; 3], [1.0; 3]);
        let src = vec![10u8; 10 * 7 * 3];
        let (crop, cw, ch) = pp.resize_and_crop(&src, 10, 7);
        assert_eq!((cw, ch), (4, 4));
        assert_eq!(crop.len(), 4 * 4 * 3);
    }
}
