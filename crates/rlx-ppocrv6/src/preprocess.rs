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

//! Image load + PP-OCR normalize / resize.

use crate::config::{DET_MEAN, DET_STD, REC_MEAN, REC_SCALE, REC_STD};
use anyhow::{Context, Result, bail};
use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};
use std::path::Path;

/// Loaded RGB8 page plus original size.
#[derive(Debug, Clone)]
pub struct RgbPage {
    pub width: u32,
    pub height: u32,
    pub rgb: RgbImage,
}

pub fn load_rgb(path: &Path) -> Result<RgbPage> {
    let img = image::open(path)
        .with_context(|| format!("open image {}", path.display()))?
        .to_rgb8();
    let (width, height) = img.dimensions();
    Ok(RgbPage {
        width,
        height,
        rgb: img,
    })
}

pub fn from_dynamic(img: DynamicImage) -> RgbPage {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    RgbPage {
        width,
        height,
        rgb,
    }
}

/// Aspect-preserving resize so max(H,W) == `limit_side_len`, then pad to multiple of 32.
pub fn det_resize(page: &RgbPage, limit_side_len: usize) -> (RgbImage, f32, f32, usize, usize) {
    let (ow, oh) = (page.width as f32, page.height as f32);
    let limit = limit_side_len as f32;
    let ratio = if ow.max(oh) > limit {
        limit / ow.max(oh)
    } else {
        1.0
    };
    let mut rw = (ow * ratio).round().max(1.0) as u32;
    let mut rh = (oh * ratio).round().max(1.0) as u32;
    // round up to multiple of 32
    rw = ((rw + 31) / 32) * 32;
    rh = ((rh + 31) / 32) * 32;
    let resized = image::imageops::resize(&page.rgb, rw, rh, FilterType::Triangle);
    let sx = rw as f32 / page.width as f32;
    let sy = rh as f32 / page.height as f32;
    (resized, sx, sy, rh as usize, rw as usize)
}

/// NCHW f32 ImageNet normalize from RGB8 (channel order RGB).
pub fn det_nchw(img: &RgbImage) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let n = (w * h) as usize;
    let mut out = vec![0f32; 3 * n];
    for (i, p) in img.pixels().enumerate() {
        let r = p[0] as f32 / 255.0;
        let g = p[1] as f32 / 255.0;
        let b = p[2] as f32 / 255.0;
        out[i] = (r - DET_MEAN[0]) / DET_STD[0];
        out[n + i] = (g - DET_MEAN[1]) / DET_STD[1];
        out[2 * n + i] = (b - DET_MEAN[2]) / DET_STD[2];
    }
    out
}

/// Resize a line crop to height `target_h`, width proportional (capped), then pad.
pub fn rec_resize_pad(
    crop: &RgbImage,
    target_h: usize,
    max_w: usize,
) -> Result<(Vec<f32>, usize, usize)> {
    let (cw, ch) = crop.dimensions();
    if cw == 0 || ch == 0 {
        bail!("empty recognition crop");
    }
    let ratio = cw as f32 / ch as f32;
    let mut tw = (target_h as f32 * ratio).round().max(1.0) as usize;
    tw = tw.min(max_w);
    // keep width multiple of 8 for stride stack
    tw = ((tw + 7) / 8) * 8;
    tw = tw.max(8);
    let resized = image::imageops::resize(
        crop,
        tw as u32,
        target_h as u32,
        FilterType::Triangle,
    );
    let n = tw * target_h;
    let mut out = vec![0f32; 3 * n];
    for (i, p) in resized.pixels().enumerate() {
        for c in 0..3 {
            let v = p[c] as f32 * REC_SCALE;
            out[c * n + i] = (v - REC_MEAN) / REC_STD;
        }
    }
    Ok((out, target_h, tw))
}

/// Perspective-ish axis-aligned crop from a 4-point box (min/max).
pub fn crop_quad(page: &RgbPage, pts: &[[f32; 2]; 4]) -> RgbImage {
    let xs: Vec<f32> = pts.iter().map(|p| p[0]).collect();
    let ys: Vec<f32> = pts.iter().map(|p| p[1]).collect();
    let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min).floor() as i32;
    let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil() as i32;
    let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min).floor() as i32;
    let max_y = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil() as i32;
    let x0 = min_x.clamp(0, page.width as i32 - 1) as u32;
    let y0 = min_y.clamp(0, page.height as i32 - 1) as u32;
    let x1 = max_x.clamp(1, page.width as i32) as u32;
    let y1 = max_y.clamp(1, page.height as i32) as u32;
    let w = (x1 - x0).max(1);
    let h = (y1 - y0).max(1);
    image::imageops::crop_imm(&page.rgb, x0, y0, w, h).to_image()
}
