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

//! Image preprocessing for Grounding DINO, mirroring HF `GroundingDinoImageProcessor`:
//! aspect-preserving resize (shortest edge → 800, longest edge ≤ 1333), rescale to
//! `[0, 1]`, ImageNet normalize, and (for a single image) an all-ones pixel mask.

use crate::config::{DEFAULT_LONGEST_EDGE, DEFAULT_SHORTEST_EDGE, IMAGENET_MEAN, IMAGENET_STD};

/// Result of preprocessing a single RGB image.
#[derive(Debug, Clone)]
pub struct Preprocessed {
    /// Normalized pixel values, NCHW (`[1, 3, h, w]`), row-major.
    pub pixel_values: Vec<f32>,
    /// Pixel mask, `[1, h, w]` (1 = real pixel, 0 = padding). All ones for a single image.
    pub pixel_mask: Vec<u8>,
    /// Height/width fed to the model (post-resize, post-pad).
    pub height: usize,
    pub width: usize,
    /// Original image dimensions (needed to rescale boxes back).
    pub orig_height: usize,
    pub orig_width: usize,
}

/// HF `get_size_with_aspect_ratio`: pick `(out_h, out_w)` so the shortest edge equals
/// `size` while the longest edge does not exceed `max_size`, preserving aspect ratio.
pub fn size_with_aspect_ratio(
    height: usize,
    width: usize,
    size: usize,
    max_size: usize,
) -> (usize, usize) {
    let (h, w) = (height as f64, width as f64);
    let min_orig = h.min(w);
    let max_orig = h.max(w);
    let mut size = size as f64;
    if max_orig / min_orig * size > max_size as f64 {
        size = (max_size as f64 * min_orig / max_orig).round();
    }
    if (height <= width && (h - size).abs() < 0.5) || (width <= height && (w - size).abs() < 0.5) {
        return (height, width);
    }
    if width < height {
        let ow = size;
        let oh = (size * h / w).round();
        (oh as usize, ow as usize)
    } else {
        let oh = size;
        let ow = (size * w / h).round();
        (oh as usize, ow as usize)
    }
}

/// Bilinear resize of an interleaved RGB f32 image (values in `[0, 1]`), using
/// half-pixel-center sampling (`align_corners = false`), matching torch/PIL bilinear.
fn bilinear_resize_rgb(
    src: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    let mut dst = vec![0f32; dst_h * dst_w * 3];
    let scale_h = src_h as f32 / dst_h as f32;
    let scale_w = src_w as f32 / dst_w as f32;
    for y in 0..dst_h {
        let sy = ((y as f32 + 0.5) * scale_h - 0.5).max(0.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = sy - y0 as f32;
        for x in 0..dst_w {
            let sx = ((x as f32 + 0.5) * scale_w - 0.5).max(0.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = sx - x0 as f32;
            for c in 0..3 {
                let p00 = src[(y0 * src_w + x0) * 3 + c];
                let p01 = src[(y0 * src_w + x1) * 3 + c];
                let p10 = src[(y1 * src_w + x0) * 3 + c];
                let p11 = src[(y1 * src_w + x1) * 3 + c];
                let top = p00 + (p01 - p00) * wx;
                let bot = p10 + (p11 - p10) * wx;
                dst[(y * dst_w + x) * 3 + c] = top + (bot - top) * wy;
            }
        }
    }
    dst
}

/// Preprocess a single interleaved RGB `u8` image (`[h*w*3]`).
pub fn preprocess_rgb(rgb: &[u8], height: usize, width: usize) -> Preprocessed {
    preprocess_rgb_sized(
        rgb,
        height,
        width,
        DEFAULT_SHORTEST_EDGE,
        DEFAULT_LONGEST_EDGE,
    )
}

/// Preprocess with explicit resize bounds.
pub fn preprocess_rgb_sized(
    rgb: &[u8],
    height: usize,
    width: usize,
    shortest_edge: usize,
    longest_edge: usize,
) -> Preprocessed {
    assert_eq!(rgb.len(), height * width * 3, "rgb len mismatch");
    // u8 → f32 in [0, 1].
    let src: Vec<f32> = rgb.iter().map(|&v| v as f32 / 255.0).collect();
    let (oh, ow) = size_with_aspect_ratio(height, width, shortest_edge, longest_edge);
    let resized = bilinear_resize_rgb(&src, height, width, oh, ow);

    // Normalize into NCHW.
    let mut pixel_values = vec![0f32; 3 * oh * ow];
    for c in 0..3 {
        let mean = IMAGENET_MEAN[c];
        let std = IMAGENET_STD[c];
        let plane = c * oh * ow;
        for y in 0..oh {
            for x in 0..ow {
                let v = resized[(y * ow + x) * 3 + c];
                pixel_values[plane + y * ow + x] = (v - mean) / std;
            }
        }
    }
    Preprocessed {
        pixel_values,
        pixel_mask: vec![1u8; oh * ow],
        height: oh,
        width: ow,
        orig_height: height,
        orig_width: width,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_ratio_clamps_longest_edge() {
        // 360x640 (h x w), shortest 800, longest 1333.
        let (oh, ow) = size_with_aspect_ratio(360, 640, 800, 1333);
        // shortest edge would scale to 800 → longest = 800*640/360 ≈ 1422 > 1333,
        // so it is clamped by the longest-edge rule.
        assert!(ow <= 1333 && oh <= 1333);
        assert!(ow >= oh);
    }

    #[test]
    fn preprocess_shapes_and_mask() {
        let (h, w) = (60, 80);
        let rgb = vec![128u8; h * w * 3];
        let p = preprocess_rgb(&rgb, h, w);
        assert_eq!(p.pixel_values.len(), 3 * p.height * p.width);
        assert_eq!(p.pixel_mask.len(), p.height * p.width);
        assert!(p.pixel_mask.iter().all(|&m| m == 1));
        assert_eq!(p.orig_height, h);
        assert_eq!(p.orig_width, w);
    }
}
