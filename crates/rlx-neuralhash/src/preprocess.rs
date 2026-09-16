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

//! Image → `[1, 3, 360, 360]` NCHW input tensor.
//!
//! The reference preprocessing is:
//!
//! ```python
//! image = Image.open(path).convert('RGB')
//! image = image.resize([360, 360])
//! arr   = np.array(image).astype(np.float32) / 255.0
//! arr   = arr * 2.0 - 1.0
//! arr   = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])
//! ```
//!
//! Two details drive hash parity and are easy to get wrong:
//!
//! * `Image.resize` **ignores aspect ratio** — a non-square source is squashed
//!   to 360×360, not letterboxed or center-cropped.
//! * `Image.resize` with no `resample=` argument defaults to `BICUBIC` for RGB
//!   images, and Pillow's resampler is *antialiased*: when downscaling it
//!   widens the filter support by the scale ratio. A plain non-antialiased
//!   bicubic (or the wrong `a` coefficient) shifts pixels enough to flip hash
//!   bits. [`rlx_core::image_preprocess::pil_resize_rgb8`] is a faithful port
//!   of Pillow's two-pass `ImagingResample` including the 8-bit round-trip
//!   between the horizontal and vertical passes.
//!
//! `x / 255 * 2 - 1` is `(x / 255 - 0.5) / 0.5`, i.e. mean = std = 0.5, so the
//! shared [`ImagePreprocessor`] expresses the whole pipeline.

use anyhow::{Context, Result, ensure};
use rlx_core::image_preprocess::{Filter, ImagePreprocessor, ResizeMode};
use std::path::Path;

/// Square input side length baked into the NeuralHash model.
pub const INPUT_SIZE: usize = 360;
/// Elements in one `[3, 360, 360]` input tensor.
pub const INPUT_ELEMS: usize = 3 * INPUT_SIZE * INPUT_SIZE;

/// The `nnhash.py` preprocessing chain as a reusable descriptor.
pub fn preprocessor() -> ImagePreprocessor {
    ImagePreprocessor {
        size: INPUT_SIZE,
        // (x/255 - 0.5) / 0.5  ==  x/255 * 2 - 1
        mean: [0.5; 3],
        std: [0.5; 3],
        // PIL `Image.resize` defaults to BICUBIC for RGB.
        filter: Filter::Bicubic,
        // PIL `Image.resize([360, 360])` does not preserve aspect ratio.
        resize_mode: ResizeMode::Exact,
        center_crop: false,
    }
}

/// Load an image file and preprocess it to `[3, 360, 360]` NCHW f32.
pub fn load_image(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let t = preprocessor()
        .load(path)
        .with_context(|| format!("preprocessing {}", path.display()))?;
    debug_assert_eq!(t.len(), INPUT_ELEMS);
    Ok(t)
}

/// Preprocess an in-memory HWC RGB8 buffer (`rgb.len() == w * h * 3`).
pub fn from_rgb8(rgb: &[u8], w: usize, h: usize) -> Result<Vec<f32>> {
    ensure!(w > 0 && h > 0, "neuralhash: empty image ({w}x{h})");
    ensure!(
        rgb.len() == w * h * 3,
        "neuralhash: RGB8 buffer is {} bytes, expected {} for {w}x{h}",
        rgb.len(),
        w * h * 3
    );
    Ok(preprocessor().from_rgb(rgb, w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_is_nchw_and_normalized() {
        // Flat mid-gray: 128/255*2 - 1.
        let rgb = vec![128u8; 8 * 5 * 3];
        let t = from_rgb8(&rgb, 8, 5).unwrap();
        assert_eq!(t.len(), INPUT_ELEMS);
        let expect = 128.0f32 / 255.0 * 2.0 - 1.0;
        for v in &t {
            assert!((v - expect).abs() < 1e-6, "{v} != {expect}");
        }
    }

    #[test]
    fn range_endpoints_map_to_minus_one_and_one() {
        let black = from_rgb8(&[0u8; 4 * 4 * 3], 4, 4).unwrap();
        assert!(black.iter().all(|v| (*v + 1.0).abs() < 1e-6));
        let white = from_rgb8(&[255u8; 4 * 4 * 3], 4, 4).unwrap();
        assert!(white.iter().all(|v| (*v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn channels_are_planar_not_interleaved() {
        // Pure red source → plane 0 is +1, planes 1 and 2 are -1.
        let mut rgb = vec![0u8; 6 * 6 * 3];
        for px in rgb.chunks_exact_mut(3) {
            px[0] = 255;
        }
        let t = from_rgb8(&rgb, 6, 6).unwrap();
        let plane = INPUT_SIZE * INPUT_SIZE;
        assert!(t[..plane].iter().all(|v| (*v - 1.0).abs() < 1e-6));
        assert!(t[plane..2 * plane].iter().all(|v| (*v + 1.0).abs() < 1e-6));
        assert!(t[2 * plane..].iter().all(|v| (*v + 1.0).abs() < 1e-6));
    }

    #[test]
    fn non_square_input_is_squashed_not_cropped() {
        // Left half red, right half blue in a wide image. `Image.resize`
        // squashes, so the split must land at x = 180 in the output — a
        // center-crop or aspect-preserving resize would not.
        let (w, h) = (400usize, 100usize);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 3;
                if x < w / 2 {
                    rgb[p] = 255;
                } else {
                    rgb[p + 2] = 255;
                }
            }
        }
        let t = from_rgb8(&rgb, w, h).unwrap();
        let plane = INPUT_SIZE * INPUT_SIZE;
        let red = |x: usize, y: usize| t[y * INPUT_SIZE + x];
        let blue = |x: usize, y: usize| t[2 * plane + y * INPUT_SIZE + x];
        // Well inside each half (away from the resampled boundary).
        assert!(red(10, 180) > 0.9 && blue(10, 180) < -0.9, "left is red");
        assert!(
            red(350, 180) < -0.9 && blue(350, 180) > 0.9,
            "right is blue"
        );
        // The transition sits at the midpoint, not at a cropped edge.
        assert!(red(170, 180) > 0.9, "still red just left of centre");
        assert!(red(190, 180) < -0.9, "already blue just right of centre");
    }

    #[test]
    fn rejects_malformed_buffers() {
        assert!(from_rgb8(&[0u8; 10], 4, 4).is_err());
        assert!(from_rgb8(&[], 0, 0).is_err());
    }
}
