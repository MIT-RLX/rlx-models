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

//! Image preprocessing for TRELLIS.2 (`Trellis2ImageTo3DPipeline.preprocess_image`).
//!
//! Upstream optionally runs BiRefNet / RMBG-2.0 when the input has no alpha.
//! This crate keeps a host path that:
//!   * uses an existing non-opaque alpha channel when present;
//!   * otherwise accepts RGB and treats the whole frame as opaque foreground
//!     (`--no-rembg` / `PreprocessOptions::allow_rgb_fallback`).
//! BiRefNet itself is **not** ported here.

use anyhow::{Result, bail};
use image::DynamicImage;

/// Options for [`preprocess_image`].
#[derive(Debug, Clone, Copy)]
pub struct PreprocessOptions {
    /// Allow RGB inputs without alpha (skip rembg). Default `false` so callers
    /// must opt in when they have not removed the background themselves.
    pub allow_rgb_fallback: bool,
    /// Max side length before crop (upstream uses 1024).
    pub max_side: u32,
}

impl Default for PreprocessOptions {
    fn default() -> Self {
        Self {
            allow_rgb_fallback: false,
            max_side: 1024,
        }
    }
}

/// Preprocessed RGB image (black-composited foreground), HWC u8.
#[derive(Clone, Debug)]
pub struct PreprocessedImage {
    pub rgb: Vec<u8>,
    pub height: usize,
    pub width: usize,
}

/// Load an image from disk and run [`preprocess_image`].
pub fn load_and_preprocess(
    path: impl AsRef<std::path::Path>,
    opts: PreprocessOptions,
) -> Result<PreprocessedImage> {
    let img = image::open(path.as_ref())?;
    preprocess_image(&img, opts)
}

/// Match upstream crop + black composite.
pub fn preprocess_image(
    input: &DynamicImage,
    opts: PreprocessOptions,
) -> Result<PreprocessedImage> {
    let mut rgba = input.to_rgba8();
    let (mut w, mut h) = (rgba.width(), rgba.height());
    let max_side = w.max(h);
    if max_side > opts.max_side {
        let scale = opts.max_side as f32 / max_side as f32;
        let nw = ((w as f32 * scale).round() as u32).max(1);
        let nh = ((h as f32 * scale).round() as u32).max(1);
        let resized = image::imageops::resize(&rgba, nw, nh, image::imageops::FilterType::Lanczos3);
        rgba = resized;
        w = nw;
        h = nh;
    }

    let has_alpha = rgba.pixels().any(|p| p[3] < 255);
    if !has_alpha && !opts.allow_rgb_fallback {
        bail!(
            "input image has no alpha channel; provide an RGBA cutout or pass \
             --no-rembg to treat the full frame as opaque foreground (BiRefNet \
             rembg is not bundled in rlx-trellis2)"
        );
    }

    // Alpha threshold + tight square crop around the foreground.
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut any = false;
    for y in 0..h {
        for x in 0..w {
            let a = rgba.get_pixel(x, y)[3];
            if a > ((0.8 * 255.0) as u8) {
                any = true;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    if !any {
        // Fully opaque RGB fallback: use the whole frame.
        min_x = 0;
        min_y = 0;
        max_x = w.saturating_sub(1);
        max_y = h.saturating_sub(1);
    }

    let cx = (min_x + max_x) as f32 / 2.0;
    let cy = (min_y + max_y) as f32 / 2.0;
    let size = (max_x - min_x).max(max_y - min_y) as f32;
    let half = size / 2.0;
    let x0 = (cx - half).floor() as i32;
    let y0 = (cy - half).floor() as i32;
    let x1 = (cx + half).ceil() as i32;
    let y1 = (cy + half).ceil() as i32;
    let crop_w = (x1 - x0).max(1) as u32;
    let crop_h = (y1 - y0).max(1) as u32;

    let mut rgb = vec![0u8; (crop_w * crop_h * 3) as usize];
    for dy in 0..crop_h {
        for dx in 0..crop_w {
            let sx = x0 + dx as i32;
            let sy = y0 + dy as i32;
            let (r, g, b, a) = if sx >= 0 && sy >= 0 && (sx as u32) < w && (sy as u32) < h {
                let p = rgba.get_pixel(sx as u32, sy as u32).0;
                (p[0], p[1], p[2], p[3])
            } else {
                (0, 0, 0, 0)
            };
            let af = a as f32 / 255.0;
            let i = ((dy * crop_w + dx) * 3) as usize;
            rgb[i] = (r as f32 * af).round() as u8;
            rgb[i + 1] = (g as f32 * af).round() as u8;
            rgb[i + 2] = (b as f32 * af).round() as u8;
        }
    }

    Ok(PreprocessedImage {
        rgb,
        height: crop_h as usize,
        width: crop_w as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    #[test]
    fn alpha_cutout_composites_to_black() {
        let mut img = RgbaImage::from_pixel(8, 8, Rgba([0, 0, 0, 0]));
        for y in 2..6 {
            for x in 2..6 {
                img.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            }
        }
        let out =
            preprocess_image(&DynamicImage::ImageRgba8(img), PreprocessOptions::default()).unwrap();
        assert!(out.width >= 3 && out.height >= 3);
        // center should be red
        let cx = out.width / 2;
        let cy = out.height / 2;
        let i = (cy * out.width + cx) * 3;
        assert_eq!(out.rgb[i], 255);
        assert_eq!(out.rgb[i + 1], 0);
    }
}
