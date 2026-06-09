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

//! RGB image load for FLUX.2 img2img / edit (`flux2-image` feature).

use anyhow::{Context, Result};
use std::path::Path;

/// Load an image, resize to `(width, height)`, return planar NCHW f32 in `[-1, 1]`.
#[cfg(feature = "flux2-image")]
pub fn load_rgb_planar(path: &Path, width: usize, height: usize) -> Result<Vec<f32>> {
    use image::imageops::FilterType;

    let img = image::open(path).with_context(|| format!("opening image {path:?}"))?;
    let rgb = img.to_rgb8();
    let resized = image::imageops::resize(&rgb, width as u32, height as u32, FilterType::Lanczos3);
    let mut out = vec![0.0f32; 3 * width * height];
    for y in 0..height {
        for x in 0..width {
            let p = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let v = p[c] as f32 / 255.0;
                out[c * width * height + y * width + x] = 2.0 * v - 1.0;
            }
        }
    }
    Ok(out)
}

#[cfg(not(feature = "flux2-image"))]
pub fn load_rgb_planar(path: &Path, _width: usize, _height: usize) -> Result<Vec<f32>> {
    anyhow::bail!(
        "FLUX.2 image load requires `flux2-image` feature (rebuild rlx-models with \
         `features = [\"flux2-image\"]`); path was {path:?}"
    );
}
