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

//! PyLaia's prediction transform: `ToImageTensor(invert=True, mode="L",
//! min_width=…)` — grayscale, invert (ink becomes bright), scale to
//! `[0, 1]`, and centre-pad with zeros up to the model's minimum width.
//!
//! PyLaia does **not** resize at prediction time: a model expects lines at
//! the height it was trained on (128 px for the Teklia releases).
//! [`resize_to_height`](crate::preprocess::resize_to_height) is offered for
//! convenience, with Pillow's Lanczos as PyLaia's data preparation uses.

use image::{DynamicImage, GrayImage};
use rlx_core::image_preprocess::{Filter, pil_resize_rgb8_exact};

/// Grayscale the way Pillow's `convert("L")` does (ITU-R 601-2 luma in
/// 16-bit fixed point), so an RGB line matches the reference pixel for
/// pixel. Already-grayscale images pass through.
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

/// Scale a line to `height` keeping the aspect ratio (`w · height / h`,
/// floored), with Pillow's Lanczos.
pub fn resize_to_height(img: &GrayImage, height: u32) -> GrayImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let height = height as usize;
    if h == height {
        return img.clone();
    }
    let nw = (w * height / h).max(1);
    // Pillow resamples each channel alike: run gray through the RGB path.
    let rgb: Vec<u8> = img.as_raw().iter().flat_map(|&v| [v, v, v]).collect();
    let out = pil_resize_rgb8_exact(&rgb, w, h, nw, height, Filter::Lanczos);
    let gray = out.iter().step_by(3).copied().collect();
    GrayImage::from_raw(nw as u32, height as u32, gray).expect("gray buffer size")
}

/// `[1, H, W']` network input: `(255 − v) / 255`, centre-padded with zeros
/// to at least `min_width` columns (`pad_transform`: the odd column goes
/// right). Returns the tensor and its `(height, width)`.
pub fn to_input(img: &GrayImage, min_width: usize) -> (Vec<f32>, usize, usize) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let nw = w.max(min_width);
    let left = (nw - w) / 2;
    let mut out = vec![0f32; h * nw];
    for (y, row) in img.as_raw().chunks_exact(w).enumerate() {
        for (x, &v) in row.iter().enumerate() {
            out[y * nw + left + x] = f32::from(255 - v) / 255.0;
        }
    }
    (out, h, nw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverts_and_pads_centred() {
        let g = GrayImage::from_raw(3, 1, vec![255, 0, 51]).unwrap();
        let (x, h, w) = to_input(&g, 6);
        assert_eq!((h, w), (1, 6));
        // 3 columns of padding: 1 left, 2 right.
        assert_eq!(x, vec![0.0, 0.0, 1.0, 0.8, 0.0, 0.0]);
    }

    #[test]
    fn pillow_luma() {
        let img = DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(2, 1, vec![255, 0, 0, 10, 200, 30]).unwrap(),
        );
        // PIL: Image.new("RGB", …).convert("L") gives 76 and 124.
        assert_eq!(to_gray(&img).into_raw(), vec![76, 124]);
    }
}
