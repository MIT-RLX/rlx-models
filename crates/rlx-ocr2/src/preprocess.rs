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

//! Host-side image → recognizer line input, and detector letterbox + line cropping.

use anyhow::Result;
use image::imageops::FilterType;
use image::{GrayImage, Rgb, RgbImage};
use std::path::Path;

pub const DET_SIZE: u32 = 480;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_INV_STD: [f32; 3] = [4.366_812, 4.464_285, 4.444_444];

/// Letterbox mapping from the 480×480 detector canvas back to original pixels.
#[derive(Clone, Copy, Debug)]
pub struct Letterbox {
    pub scale: f32,
    pub ox: f32,
    pub oy: f32,
    pub orig_w: u32,
    pub orig_h: u32,
}

/// Load an image, letterbox into 480×480 (white pad), imagenet-normalize to
/// `[1,3,480,480]`. Also returns the letterbox map and the original grayscale image
/// (used for line crops).
pub fn detector_input(path: &Path) -> Result<(Vec<f32>, Letterbox, GrayImage)> {
    let rgb = image::open(path)?.to_rgb8();
    let (iw, ih) = rgb.dimensions();
    let s = (DET_SIZE as f32 / iw as f32).min(DET_SIZE as f32 / ih as f32);
    let (dw, dh) = ((iw as f32 * s) as u32, (ih as f32 * s) as u32);
    let resized = image::imageops::resize(&rgb, dw.max(1), dh.max(1), FilterType::Triangle);
    let (ox, oy) = ((DET_SIZE - dw.max(1)) / 2, (DET_SIZE - dh.max(1)) / 2);
    let mut canvas = RgbImage::from_pixel(DET_SIZE, DET_SIZE, Rgb([255, 255, 255]));
    image::imageops::overlay(&mut canvas, &resized, ox as i64, oy as i64);

    let n = (DET_SIZE * DET_SIZE) as usize;
    let mut out = vec![0f32; 3 * n];
    for y in 0..DET_SIZE {
        for x in 0..DET_SIZE {
            let p = canvas.get_pixel(x, y);
            for c in 0..3 {
                out[c * n + y as usize * DET_SIZE as usize + x as usize] =
                    (f32::from(p[c]) / 255.0 - IMAGENET_MEAN[c]) * IMAGENET_INV_STD[c];
            }
        }
    }
    let lb = Letterbox { scale: s, ox: ox as f32, oy: oy as f32, orig_w: iw, orig_h: ih };
    Ok((out, lb, image::open(path)?.to_luma8()))
}

/// Crop a line box (original-image pixel coords) from `gray`, resize to height 32,
/// pad width to a multiple of 4, normalize to `[0,1]` (background high). `(luma, width)`.
pub fn crop_line_luma(gray: &GrayImage, x0: u32, y0: u32, x1: u32, y1: u32) -> Option<(Vec<f32>, usize)> {
    let (cw, ch) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    if cw < 4 || ch < 4 {
        return None;
    }
    let sub = image::imageops::crop_imm(gray, x0, y0, cw, ch).to_image();
    let new_w = (((cw as f32) * 32.0 / (ch as f32)).round() as u32).max(8);
    let resized = image::imageops::resize(&sub, new_w, 32, FilterType::Triangle);
    let pad_w = (new_w as usize).div_ceil(4) * 4; // round width up to a multiple of 4
    let mut out = vec![1.0f32; 32 * pad_w];
    for y in 0..32u32 {
        for x in 0..new_w {
            out[y as usize * pad_w + x as usize] = f32::from(resized.get_pixel(x, y)[0]) / 255.0;
        }
    }
    Some((out, pad_w))
}

/// Load an image as a recognizer line input: grayscale, resized to height 32,
/// width padded to a multiple of 4, normalized to `[0,1]` (background high, ink low
/// — the recognizer's polarity). Returns `(luma[32*width], width)`.
pub fn luma_line(path: &Path) -> Result<(Vec<f32>, usize)> {
    let img = image::open(path)?.to_luma8();
    let (w0, h0) = (img.width().max(1), img.height().max(1));
    let new_w = (((w0 as f32) * 32.0 / (h0 as f32)).round() as usize).max(4);
    let resized = image::imageops::resize(&img, new_w as u32, 32, FilterType::Triangle);
    let pad_w = new_w.next_multiple_of(4);
    let mut out = vec![1.0f32; 32 * pad_w]; // pad columns with background (white)
    for y in 0..32usize {
        for x in 0..new_w {
            out[y * pad_w + x] = f32::from(resized.get_pixel(x as u32, y as u32)[0]) / 255.0;
        }
    }
    Ok((out, pad_w))
}
