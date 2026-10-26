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

//! SAM v1's image preprocess: the long side resized to `SAM_IMG_SIZE`,
//! normalized, zero-padded square. Plain host math, which the exported
//! package's program builds too.

use super::config::{SAM_IMG_SIZE, SAM_PIXEL_MEAN, SAM_PIXEL_STD};

/// Resize an RGB u8 image to fit within `SAM_IMG_SIZE` on the long
/// side (aspect-ratio preserved), normalize with SAM's pixel stats,
/// and zero-pad to a square `[3, 1024, 1024]` NCHW f32 tensor.
///
/// `rgb` is `H_in · W_in · 3` row-major (u8). Returns `(nchw, (h, w))`
/// where `(h, w)` are the resized (pre-pad) dimensions — needed at the
/// decoder to crop predicted masks back to the original aspect ratio.
pub fn preprocess_image(rgb: &[u8], h_in: usize, w_in: usize) -> (Vec<f32>, (usize, usize)) {
    let scale = (SAM_IMG_SIZE as f32) / (h_in.max(w_in) as f32);
    let new_h = ((h_in as f32) * scale).round() as usize;
    let new_w = ((w_in as f32) * scale).round() as usize;
    // Bilinear resize.
    let mut resized = vec![0f32; 3 * new_h * new_w];
    let sx = (w_in as f32 - 1.0) / (new_w.max(1) as f32 - 1.0).max(1.0);
    let sy = (h_in as f32 - 1.0) / (new_h.max(1) as f32 - 1.0).max(1.0);
    for y in 0..new_h {
        let fy = y as f32 * sy;
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(h_in - 1);
        let dy = fy - y0 as f32;
        for x in 0..new_w {
            let fx = x as f32 * sx;
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(w_in - 1);
            let dx = fx - x0 as f32;
            for c in 0..3 {
                let p00 = rgb[(y0 * w_in + x0) * 3 + c] as f32;
                let p01 = rgb[(y0 * w_in + x1) * 3 + c] as f32;
                let p10 = rgb[(y1 * w_in + x0) * 3 + c] as f32;
                let p11 = rgb[(y1 * w_in + x1) * 3 + c] as f32;
                let top = p00 * (1.0 - dx) + p01 * dx;
                let bot = p10 * (1.0 - dx) + p11 * dx;
                let v = top * (1.0 - dy) + bot * dy;
                // SAM normalises raw pixel values (NOT /255 first).
                resized[c * new_h * new_w + y * new_w + x] =
                    (v - SAM_PIXEL_MEAN[c]) / SAM_PIXEL_STD[c];
            }
        }
    }
    // Zero-pad to [3, 1024, 1024].
    let mut padded = vec![0f32; 3 * SAM_IMG_SIZE * SAM_IMG_SIZE];
    for c in 0..3 {
        for y in 0..new_h {
            let src_row = c * new_h * new_w + y * new_w;
            let dst_row = c * SAM_IMG_SIZE * SAM_IMG_SIZE + y * SAM_IMG_SIZE;
            padded[dst_row..dst_row + new_w].copy_from_slice(&resized[src_row..src_row + new_w]);
        }
    }
    (padded, (new_h, new_w))
}
