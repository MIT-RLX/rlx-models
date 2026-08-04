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

//! Host-side image preprocessing for VLASH, matching the reference
//! `resize_with_pad` (`policies/pi0/utils.py`) + `img * 2 - 1` normalization.
//!
//! The reference pipeline (`PI0Policy.prepare_images`):
//! ```text
//!   img [C,H,W] in [0,1]
//!     → resize_with_pad(224, 224, pad_value=0)   # aspect-preserving, pad top/left
//!     → img * 2 - 1                              # → [-1, 1]  (== SigLIP mean=std=0.5)
//! ```
//!
//! `resize_with_pad` scales by `ratio = max(W/224, H/224)`, resizes with
//! **bilinear, align_corners=False** (PyTorch default), then left/top-pads to
//! `224×224` with 0 so the image sits bottom-right. Bilinear is implemented
//! here to match `torch.nn.functional.interpolate` exactly (half-pixel
//! coordinate transform + edge clamping).

/// Bilinear resize of a single-channel plane, matching PyTorch
/// `F.interpolate(mode="bilinear", align_corners=False)`.
///
/// `src` is `h_in × w_in` row-major. Output is `h_out × w_out`.
fn resize_plane_bilinear(
    src: &[f32],
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; h_out * w_out];
    if h_out == 0 || w_out == 0 || h_in == 0 || w_in == 0 {
        return out;
    }
    let scale_y = h_in as f64 / h_out as f64;
    let scale_x = w_in as f64 / w_out as f64;
    for oy in 0..h_out {
        // half-pixel: src = (dst + 0.5) * scale - 0.5, clamped to [0, in-1].
        let sy = ((oy as f64 + 0.5) * scale_y - 0.5).max(0.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(h_in - 1);
        let wy = (sy - y0 as f64) as f32;
        for ox in 0..w_out {
            let sx = ((ox as f64 + 0.5) * scale_x - 0.5).max(0.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(w_in - 1);
            let wx = (sx - x0 as f64) as f32;
            let v00 = src[y0 * w_in + x0];
            let v01 = src[y0 * w_in + x1];
            let v10 = src[y1 * w_in + x0];
            let v11 = src[y1 * w_in + x1];
            let top = v00 * (1.0 - wx) + v01 * wx;
            let bot = v10 * (1.0 - wx) + v11 * wx;
            out[oy * w_out + ox] = top * (1.0 - wy) + bot * wy;
        }
    }
    out
}

/// Aspect-preserving resize + left/top pad to `target×target`, then `×2 − 1`.
///
/// `chw` is a `3 × h_in × w_in` row-major image in `[0, 1]`. Returns a
/// SigLIP-normalized NCHW tensor `[3 · target · target]` in `[-1, 1]`.
pub fn resize_with_pad_normalize(chw: &[f32], h_in: usize, w_in: usize, target: usize) -> Vec<f32> {
    assert_eq!(chw.len(), 3 * h_in * w_in, "chw length mismatch");
    // ratio = max(W/target, H/target); resized dims truncate toward zero (Python int()).
    let ratio = (w_in as f64 / target as f64).max(h_in as f64 / target as f64);
    let resized_h = ((h_in as f64) / ratio) as usize;
    let resized_w = ((w_in as f64) / ratio) as usize;
    let resized_h = resized_h.min(target);
    let resized_w = resized_w.min(target);
    let pad_top = target - resized_h;
    let pad_left = target - resized_w;

    let mut out = vec![0f32; 3 * target * target]; // pad value = 0 → after ×2−1 becomes −1
    for c in 0..3 {
        let plane = &chw[c * h_in * w_in..(c + 1) * h_in * w_in];
        let resized = resize_plane_bilinear(plane, h_in, w_in, resized_h, resized_w);
        for ry in 0..resized_h {
            for rx in 0..resized_w {
                let dst = c * target * target + (pad_top + ry) * target + (pad_left + rx);
                out[dst] = resized[ry * resized_w + rx];
            }
        }
    }
    // ×2 − 1
    for v in out.iter_mut() {
        *v = *v * 2.0 - 1.0;
    }
    out
}

/// Convenience: RGB8 (HWC, `[0,255]`) → normalized NCHW `[-1,1]` at `target×target`.
pub fn rgb8_to_nchw_normalized(rgb: &[u8], h_in: usize, w_in: usize, target: usize) -> Vec<f32> {
    assert_eq!(
        rgb.len(),
        3 * h_in * w_in,
        "rgb length mismatch (expect HWC)"
    );
    // HWC u8 → CHW f32 in [0,1].
    let mut chw = vec![0f32; 3 * h_in * w_in];
    for y in 0..h_in {
        for x in 0..w_in {
            for c in 0..3 {
                chw[c * h_in * w_in + y * w_in + x] = rgb[(y * w_in + x) * 3 + c] as f32 / 255.0;
            }
        }
    }
    resize_with_pad_normalize(&chw, h_in, w_in, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_input_is_exact_resize_no_pad() {
        // A square image needs no padding (ratio uses both dims equally).
        let n = 28usize;
        let chw: Vec<f32> = (0..3 * n * n).map(|i| (i % 7) as f32 / 7.0).collect();
        let out = resize_with_pad_normalize(&chw, n, n, 14);
        assert_eq!(out.len(), 3 * 14 * 14);
        assert!(out.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    #[test]
    fn wide_input_pads_left_top() {
        // 2:1 image → resized to 224×112, padded on the left by 112.
        let (h, w, t) = (112usize, 224usize, 224usize);
        let chw = vec![0.5f32; 3 * h * w];
        let out = resize_with_pad_normalize(&chw, h, w, t);
        // Top rows are padded (value 0 → −1 after ×2−1).
        let c0 = &out[0..t * t];
        assert!(
            (c0[0] - (-1.0)).abs() < 1e-6,
            "top-left should be padded to -1"
        );
        // Bottom-right region holds resized content (0.5 → 0.0 after ×2−1).
        let br = c0[(t - 1) * t + (t - 1)];
        assert!(br.abs() < 1e-4, "content region should be ~0 (0.5*2-1)");
    }
}
