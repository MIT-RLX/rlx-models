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

//! Kraken's line transform for baseline-segmented input
//! (`ImageInputTransforms(…, valid_norm=False)`): convert to the model's
//! mode, Lanczos-resize to the input height keeping the aspect ratio
//! (`width = ⌊w · H / h⌋`), pad `pad` white columns left and right, scale
//! to `[0, 1]`, and invert as `max − x`.
//!
//! The resize is Pillow's own fixed-point Lanczos
//! ([`rlx_core::image_preprocess::pil_resize_rgb8_exact`]), so the input
//! matches Kraken's pixel for pixel.

use image::{DynamicImage, GrayImage};
use rlx_core::image_preprocess::{Filter, pil_resize_rgb8_exact};

/// Grayscale the way Pillow's `convert("L")` does.
pub fn to_gray(img: &DynamicImage) -> GrayImage {
    match img {
        DynamicImage::ImageLuma8(g) => g.clone(),
        other => {
            let rgb = other.to_rgb8();
            let (w, h) = rgb.dimensions();
            let data = rgb
                .pixels()
                .map(|p| {
                    let [r, g, b] = p.0.map(u32::from);
                    ((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16) as u8
                })
                .collect();
            GrayImage::from_raw(w, h, data).expect("gray buffer size")
        }
    }
}

/// `[C, H, W]` input for a model with `channels` (1 or 3) and input height
/// `height` (`0` keeps the image height), padded by `pad` columns a side.
pub fn line_tensor(
    img: &DynamicImage,
    channels: usize,
    height: usize,
    pad: usize,
) -> (Vec<f32>, usize, usize) {
    // Pillow's 8-bit resample treats channels independently, so a gray
    // line goes through the RGB path as three equal planes.
    let (rgb, iw, ih) = if channels == 3 {
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        (rgb.into_raw(), w as usize, h as usize)
    } else {
        let g = to_gray(img);
        let (w, h) = g.dimensions();
        let rgb = g.as_raw().iter().flat_map(|&v| [v, v, v]).collect();
        (rgb, w as usize, h as usize)
    };
    let (w, h) = if height == 0 {
        (iw, ih)
    } else {
        // `_fixed_resize`: width = int(w · H / h).
        ((iw * height / ih).max(1), height)
    };
    let rgb = pil_resize_rgb8_exact(&rgb, iw, ih, w, h, Filter::Lanczos);
    let planes: Vec<Vec<u8>> = (0..channels.min(3))
        .map(|c| rgb.iter().skip(c).step_by(3).copied().collect())
        .collect();
    let pw = w + 2 * pad;
    let mut out = vec![1f32; planes.len() * h * pw];
    for (c, plane) in planes.iter().enumerate() {
        for y in 0..h {
            for x in 0..w {
                out[(c * h + y) * pw + pad + x] = f32::from(plane[y * w + x]) / 255.0;
            }
        }
    }
    // `tensor_invert`: max − x over the whole tensor.
    let max = out.iter().copied().fold(f32::MIN, f32::max);
    for v in &mut out {
        *v = max - *v;
    }
    (out, h, pw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pads_white_then_inverts() {
        let g = GrayImage::from_raw(2, 1, vec![0, 255]).unwrap();
        let (x, h, w) = line_tensor(&DynamicImage::ImageLuma8(g), 1, 0, 1);
        assert_eq!((h, w), (1, 4));
        // [pad=white, black, white, pad=white] → inverted.
        assert_eq!(x, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn invert_is_relative_to_the_max() {
        // No padding and no pure white: max − x, not 1 − x.
        let g = GrayImage::from_raw(2, 1, vec![0, 51]).unwrap();
        let (x, _, _) = line_tensor(&DynamicImage::ImageLuma8(g), 1, 0, 0);
        assert_eq!(x, vec![0.2, 0.0]);
    }
}
