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

//! Image preprocessing — HF `LocateAnythingImageProcessor` (native resolution).

use crate::config::{LocateAnythingConfig, LocateAnythingPreprocessorConfig, MoonVitConfig};
use anyhow::{Result, ensure};
use image::DynamicImage;
use image::imageops::FilterType;
use std::path::Path;

/// Flattened patch tensor `[num_patches, C * patch_h * patch_w]` and grid `(h_patches, w_patches)`.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    pub patches: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    pub patch_dim: usize,
    /// Width/height in pixels after rescale + pad (used for box coordinate scaling).
    pub pixel_w: u32,
    pub pixel_h: u32,
}

impl PreprocessedImage {
    pub fn num_patches(&self) -> usize {
        self.grid_h * self.grid_w
    }
}

pub fn preprocess_image(
    img: &DynamicImage,
    cfg: &LocateAnythingConfig,
) -> Result<PreprocessedImage> {
    preprocess_image_with_limit(img, &cfg.vision_config, &cfg.preprocessor)
}

pub fn preprocess_path(path: &Path, cfg: &LocateAnythingConfig) -> Result<PreprocessedImage> {
    let img = image::open(path)?;
    preprocess_image(&img, cfg)
}

fn preprocess_image_with_limit(
    img: &DynamicImage,
    vit: &MoonVitConfig,
    pre: &LocateAnythingPreprocessorConfig,
) -> Result<PreprocessedImage> {
    let patch_size = vit.patch_size;
    let in_token_limit = pre.in_token_limit;
    let merge_kernel = vit.merge_kernel_size;
    let mean = pre.image_mean;
    let std = pre.image_std;

    let mut rgb = img.to_rgb8();
    let (mut w, mut h) = rgb.dimensions();

    let patches_before_merge = (w as usize / patch_size) * (h as usize / patch_size);
    if patches_before_merge > in_token_limit {
        let scale = (in_token_limit as f32 / patches_before_merge as f32).sqrt();
        let new_w = (w as f32 * scale) as u32;
        let new_h = (h as f32 * scale) as u32;
        rgb = image::DynamicImage::ImageRgb8(rgb)
            .resize_exact(new_w.max(1), new_h.max(1), FilterType::CatmullRom)
            .to_rgb8();
        w = rgb.width();
        h = rgb.height();
    }

    let pad_h = merge_kernel[0] * patch_size;
    let pad_w = merge_kernel[1] * patch_size;
    let target_w = (w as usize).div_ceil(pad_w) * pad_w;
    let target_h = (h as usize).div_ceil(pad_h) * pad_h;

    if target_w != w as usize || target_h != h as usize {
        rgb = image::DynamicImage::ImageRgb8(rgb)
            .resize_exact(target_w as u32, target_h as u32, FilterType::CatmullRom)
            .to_rgb8();
        w = rgb.width();
        h = rgb.height();
    }

    let grid_h = h as usize / patch_size;
    let grid_w = w as usize / patch_size;
    ensure!(
        grid_h < 512 && grid_w < 512,
        "grid {grid_h}x{grid_w} exceeds position embedding limit"
    );
    let mut tensor = vec![0f32; 3 * h as usize * w as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let p = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let v = p[c] as f32 / 255.0;
                tensor[c * h as usize * w as usize + y * w as usize + x] = (v - mean[c]) / std[c];
            }
        }
    }

    let patch_dim = 3 * patch_size * patch_size;
    let num_patches = grid_h * grid_w;
    let mut patches = vec![0f32; num_patches * patch_dim];

    for py in 0..grid_h {
        for px in 0..grid_w {
            let out_patch = (py * grid_w + px) * patch_dim;
            for c in 0..3 {
                for dy in 0..patch_size {
                    for dx in 0..patch_size {
                        let y = py * patch_size + dy;
                        let x = px * patch_size + dx;
                        let src = c * h as usize * w as usize + y * w as usize + x;
                        let dst = out_patch + c * patch_size * patch_size + dy * patch_size + dx;
                        patches[dst] = tensor[src];
                    }
                }
            }
        }
    }

    Ok(PreprocessedImage {
        patches,
        grid_h,
        grid_w,
        patch_dim,
        pixel_w: w,
        pixel_h: h,
    })
}
