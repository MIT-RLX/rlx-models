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

//! Host-side image-tensor ops (pad + bilinear resize) for the native RLX OCR
//! path — drop-in replacements for the `rten::Operators` `pad` / `resize_image`
//! used during detection/recognition pre- and post-processing.
//!
//! These exist so the default build pulls **no RTen inference engine**: the only
//! reason `rlx-ocr` linked `rten` by default was these two host ops. The bilinear
//! resize uses ONNX/OpenCV **half-pixel** sample centres with no antialiasing, to
//! stay numerically equivalent to the previous `rten::resize_image` path
//! (cross-checked against the `rten-inference` oracle in the parity tests).

use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

/// Pad an NCHW tensor at the **end** of the H and W axes (bottom / right) with
/// `value`. Equivalent to `image.pad([0,0,0,0, 0,0,pad_h,pad_w], value)`.
pub fn pad_hw_end(
    img: NdTensorView<f32, 4>,
    pad_h: usize,
    pad_w: usize,
    value: f32,
) -> NdTensor<f32, 4> {
    let [n, c, h, w] = img.shape();
    let (oh, ow) = (h + pad_h, w + pad_w);
    let mut out = vec![value; n * c * oh * ow];
    for ni in 0..n {
        for ci in 0..c {
            for y in 0..h {
                let row = ((ni * c + ci) * oh + y) * ow;
                for x in 0..w {
                    out[row + x] = img[[ni, ci, y, x]];
                }
            }
        }
    }
    NdTensor::from_data([n, c, oh, ow], out)
}

/// Bilinear resize of an NCHW tensor over its H/W axes (half-pixel centres, no
/// antialiasing — matches ONNX `Resize` linear / the prior `rten::resize_image`).
pub fn resize_bilinear(img: NdTensorView<f32, 4>, out_h: usize, out_w: usize) -> NdTensor<f32, 4> {
    let [n, c, h, w] = img.shape();
    if out_h == h && out_w == w {
        return img.to_tensor();
    }
    let mut out = vec![0f32; n * c * out_h * out_w];
    let scale_y = h as f32 / out_h as f32;
    let scale_x = w as f32 / out_w as f32;
    let max_y = h.saturating_sub(1);
    let max_x = w.saturating_sub(1);
    for ni in 0..n {
        for ci in 0..c {
            for oy in 0..out_h {
                // half-pixel: src = (dst + 0.5) * scale - 0.5, clamped to the edge.
                let fy = ((oy as f32 + 0.5) * scale_y - 0.5).clamp(0.0, max_y as f32);
                let y0 = fy.floor() as usize;
                let y1 = (y0 + 1).min(max_y);
                let wy = fy - y0 as f32;
                let dst_row = ((ni * c + ci) * out_h + oy) * out_w;
                for ox in 0..out_w {
                    let fx = ((ox as f32 + 0.5) * scale_x - 0.5).clamp(0.0, max_x as f32);
                    let x0 = fx.floor() as usize;
                    let x1 = (x0 + 1).min(max_x);
                    let wx = fx - x0 as f32;
                    let v00 = img[[ni, ci, y0, x0]];
                    let v01 = img[[ni, ci, y0, x1]];
                    let v10 = img[[ni, ci, y1, x0]];
                    let v11 = img[[ni, ci, y1, x1]];
                    let top = v00 + (v01 - v00) * wx;
                    let bot = v10 + (v11 - v10) * wx;
                    out[dst_row + ox] = top + (bot - top) * wy;
                }
            }
        }
    }
    NdTensor::from_data([n, c, out_h, out_w], out)
}

#[cfg(test)]
mod timing {
    //! `cargo test -p rlx-ocr --release --lib time_host_resize -- --ignored --nocapture`
    use super::*;
    #[test]
    #[ignore]
    fn time_host_resize() {
        let (h, w) = (2048usize, 2048usize);
        let img = NdTensor::from_data([1, 1, h, w], vec![0.5f32; h * w]);
        let t = std::time::Instant::now();
        let _ = resize_bilinear(img.view(), 1024, 1024);
        eprintln!("contiguous 2048x2048 -> 1024x1024: {:?}", t.elapsed());
        // strided view, like the detection mask-resize-back path
        let sliced = img.slice((.., .., ..1500, ..1500));
        let t2 = std::time::Instant::now();
        let _ = resize_bilinear(sliced, 1654, 2339);
        eprintln!("strided 1500x1500 -> 1654x2339: {:?}", t2.elapsed());
        let t3 = std::time::Instant::now();
        let _ = pad_hw_end(img.view(), 200, 200, -0.5);
        eprintln!("pad 2048x2048 +200,200: {:?}", t3.elapsed());
    }
}

#[cfg(all(test, feature = "tensor-ops"))]
mod parity_tests {
    //! Parity vs the `rten::Operators` ops these functions replace. Run with:
    //! `cargo test -p rlx-ocr --features tensor-ops`.
    use super::*;
    use rten::{FloatOperators, Operators};

    fn ramp(h: usize, w: usize) -> NdTensor<f32, 4> {
        let data: Vec<f32> = (0..h * w)
            .map(|i| ((i * 31 % 251) as f32) / 251.0)
            .collect();
        NdTensor::from_data([1, 1, h, w], data)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(
            a.len(),
            b.len(),
            "shape mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn resize_matches_rten() {
        let img = ramp(17, 23);
        for (oh, ow) in [(32, 40), (8, 11), (17, 23), (40, 7)] {
            let mine: Vec<f32> = resize_bilinear(img.view(), oh, ow)
                .iter()
                .copied()
                .collect();
            let theirs: Vec<f32> = img
                .view()
                .resize_image([oh, ow])
                .unwrap()
                .iter()
                .copied()
                .collect();
            let d = max_abs_diff(&mine, &theirs);
            assert!(d < 1e-4, "resize {oh}x{ow}: max abs diff {d}");
        }
    }

    #[test]
    fn pad_matches_rten() {
        let img = ramp(13, 19);
        let (pb, pr) = (6, 4);
        let val = super::super::preprocess::BLACK_VALUE;
        let mine: Vec<f32> = pad_hw_end(img.view(), pb, pr, val)
            .iter()
            .copied()
            .collect();
        let pads = &[0, 0, 0, 0, 0, 0, pb as i32, pr as i32];
        let theirs: Vec<f32> = img
            .view()
            .pad(pads.into(), val)
            .unwrap()
            .iter()
            .copied()
            .collect();
        let d = max_abs_diff(&mine, &theirs);
        assert!(d < 1e-6, "pad: max abs diff {d}");
    }
}
