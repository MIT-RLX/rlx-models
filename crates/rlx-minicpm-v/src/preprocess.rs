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

//! Image slicing — a port of `MiniCPMV4_6ImageProcessor`.
//!
//! An image becomes a **thumbnail** (the whole picture resized to about
//! `scale_resolution²` pixels) plus, when it is large enough, a grid of
//! **slices** cut from a higher-resolution resize. Every side is a multiple
//! of `4 · patch_size`, so each crop's patch grid survives both 2×2 merges.
//!
//! The arithmetic follows the Python exactly: `round` is banker's rounding
//! and `int()` truncates, and both decide crop sizes, so a "close enough"
//! port gives a different number of image tokens.

use crate::config::SliceConfig;
use anyhow::{Result, ensure};
use rlx_core::image_preprocess::{Filter, torch_resize_rgb8_aa};

/// Python's `max(round(length / divisor) * divisor, divisor)`.
fn ensure_divide(length: f64, divisor: usize) -> usize {
    let d = divisor as f64;
    (((length / d).round_ties_even()) * d).max(d) as usize
}

/// `find_best_resize`: `(height, width)` with both sides a multiple of
/// `4 · patch_size`.
pub fn find_best_resize(
    height: f64,
    width: f64,
    scale_resolution: usize,
    patch_size: usize,
    allow_upscale: bool,
) -> (usize, usize) {
    let (mut h, mut w) = (height, width);
    let s = scale_resolution as f64;
    if h * w > s * s || allow_upscale {
        let aspect = w / h;
        h = (s / aspect.sqrt()).trunc();
        w = (h * aspect).trunc();
    }
    (
        ensure_divide(h, patch_size * 4),
        ensure_divide(w, patch_size * 4),
    )
}

/// `get_refine_size`: the size the image is resized to before slicing.
pub fn get_refine_size(
    height: usize,
    width: usize,
    grid: (usize, usize),
    scale_resolution: usize,
    patch_size: usize,
) -> (usize, usize) {
    let (gy, gx) = grid;
    let refine_w = ensure_divide(width as f64, gx);
    let refine_h = ensure_divide(height as f64, gy);
    let (bh, bw) = find_best_resize(
        refine_h as f64 / gy as f64,
        refine_w as f64 / gx as f64,
        scale_resolution,
        patch_size,
        true,
    );
    (bh * gy, bw * gx)
}

/// `get_sliced_grid`: `(rows, cols)` of detail slices, or `None` when the
/// image is small enough for the thumbnail alone.
pub fn get_sliced_grid(
    height: usize,
    width: usize,
    max_slice_nums: usize,
    scale_resolution: usize,
) -> Option<(usize, usize)> {
    let (h, w) = (height as f64, width as f64);
    let log_ratio = (w / h).ln();
    let ratio = w * h / (scale_resolution * scale_resolution) as f64;
    let multiple = (ratio.ceil() as usize).min(max_slice_nums);
    if multiple <= 1 {
        return None;
    }
    let mut best = (1usize, 1usize);
    let mut min_error = f64::INFINITY;
    for num_slices in [multiple - 1, multiple, multiple + 1] {
        if num_slices == 1 || num_slices > max_slice_nums {
            continue;
        }
        for rows in 1..=num_slices {
            if num_slices % rows != 0 {
                continue;
            }
            let cols = num_slices / rows;
            let error = (log_ratio - (cols as f64 / rows as f64).ln()).abs();
            if error < min_error {
                best = (rows, cols);
                min_error = error;
            } else if error == min_error && rows > best.0 {
                best = (rows, cols);
            }
        }
    }
    Some(best)
}

/// One crop, patchified for the vision tower.
#[derive(Debug, Clone)]
pub struct Crop {
    /// Patch grid `(rows, cols)` — HF `target_sizes`.
    pub grid_h: usize,
    pub grid_w: usize,
    /// `[grid_h · grid_w, 3 · patch²]`, patches row-major, each patch in
    /// the conv weight's `(channel, ky, kx)` order.
    pub patches: Vec<f32>,
}

impl Crop {
    pub fn num_patches(&self) -> usize {
        self.grid_h * self.grid_w
    }
}

/// An image after slicing: the thumbnail first, then the slices row-major.
#[derive(Debug, Clone)]
pub struct SlicedImage {
    pub crops: Vec<Crop>,
    /// `(rows, cols)` of the slice grid, `None` for a thumbnail-only image.
    pub grid: Option<(usize, usize)>,
}

impl SlicedImage {
    pub fn thumbnail(&self) -> &Crop {
        &self.crops[0]
    }

    pub fn slices(&self) -> &[Crop] {
        &self.crops[1..]
    }
}

/// Slice and patchify an interleaved RGB8 image of `width × height`.
pub fn slice_image(
    rgb: &[u8],
    width: usize,
    height: usize,
    cfg: &SliceConfig,
    patch_size: usize,
) -> Result<SlicedImage> {
    ensure!(
        rgb.len() == width * height * 3,
        "minicpm-v: RGB buffer is {} bytes, expected {}",
        rgb.len(),
        width * height * 3
    );
    ensure!(width > 0 && height > 0, "minicpm-v: empty image");
    let grid = if cfg.slice_mode {
        get_sliced_grid(height, width, cfg.max_slice_nums, cfg.scale_resolution)
    } else {
        None
    };

    let (th, tw) = find_best_resize(
        height as f64,
        width as f64,
        cfg.scale_resolution,
        patch_size,
        grid.is_none(),
    );
    let thumb = torch_resize_rgb8_aa(rgb, width, height, tw, th, Filter::Bicubic);
    let mut crops = vec![patchify(&thumb, tw, th, cfg, patch_size)];

    if let Some((gy, gx)) = grid {
        let (rh, rw) = get_refine_size(height, width, (gy, gx), cfg.scale_resolution, patch_size);
        let refine = torch_resize_rgb8_aa(rgb, width, height, rw, rh, Filter::Bicubic);
        let (ph, pw) = (rh / gy, rw / gx);
        for r in 0..gy {
            for c in 0..gx {
                let mut tile = Vec::with_capacity(ph * pw * 3);
                for y in r * ph..(r + 1) * ph {
                    let row = (y * rw + c * pw) * 3;
                    tile.extend_from_slice(&refine[row..row + pw * 3]);
                }
                crops.push(patchify(&tile, pw, ph, cfg, patch_size));
            }
        }
    }
    Ok(SlicedImage { crops, grid })
}

/// Slicing of the resampler generation (2.x–4.5, `MiniCPMVImageProcessor`
/// in the checkpoint's remote code). Same plan as [`slice_image`], with its
/// own arithmetic: sides are multiples of `patch_size` (no 2×2 merges to
/// survive), the grid is chosen first-best (ties keep the earlier
/// candidate), Pillow does the bicubic resizes, and pixels are normalized as
/// `(x / 255 − mean) / std`.
pub mod resampler_gen {
    use super::{Crop, SlicedImage, ensure_divide};
    use crate::config::SliceConfig;
    use anyhow::{Result, ensure};
    use rlx_core::image_preprocess::{Filter, pil_resize_rgb8_exact};

    /// `find_best_resize`: `(width, height)`, multiples of `patch_size`.
    pub fn find_best_resize(
        width: f64,
        height: f64,
        scale_resolution: usize,
        patch_size: usize,
        allow_upscale: bool,
    ) -> (usize, usize) {
        let (mut w, mut h) = (width, height);
        let s = scale_resolution as f64;
        if w * h > s * s || allow_upscale {
            let r = w / h;
            h = (s / r.sqrt()).trunc();
            w = (h * r).trunc();
        }
        (ensure_divide(w, patch_size), ensure_divide(h, patch_size))
    }

    /// `get_refine_size` for a `(cols, rows)` grid: `(width, height)`.
    pub fn get_refine_size(
        width: usize,
        height: usize,
        grid: (usize, usize),
        scale_resolution: usize,
        patch_size: usize,
    ) -> (usize, usize) {
        let (gx, gy) = grid;
        let rw = ensure_divide(width as f64, gx);
        let rh = ensure_divide(height as f64, gy);
        let (bw, bh) = find_best_resize(
            rw as f64 / gx as f64,
            rh as f64 / gy as f64,
            scale_resolution,
            patch_size,
            true,
        );
        (bw * gx, bh * gy)
    }

    /// `get_sliced_grid`: `(cols, rows)`, or `None` for thumbnail only.
    pub fn get_sliced_grid(
        width: usize,
        height: usize,
        max_slice_nums: usize,
        scale_resolution: usize,
    ) -> Option<(usize, usize)> {
        let (w, h) = (width as f64, height as f64);
        let log_ratio = (w / h).ln();
        let ratio = w * h / (scale_resolution * scale_resolution) as f64;
        let multiple = (ratio.ceil() as usize).min(max_slice_nums);
        if multiple <= 1 {
            return None;
        }
        let mut best = (1usize, 1usize);
        let mut min_error = f64::INFINITY;
        for n in [multiple - 1, multiple, multiple + 1] {
            if n == 1 || n > max_slice_nums {
                continue;
            }
            for m in 1..=n {
                if n % m != 0 {
                    continue;
                }
                let error = (log_ratio - (m as f64 / (n / m) as f64).ln()).abs();
                if error < min_error {
                    best = (m, n / m);
                    min_error = error;
                }
            }
        }
        Some(best)
    }

    /// Slice and patchify an RGB8 image. `SlicedImage::grid` is
    /// `(rows, cols)`, as for the newer generation.
    pub fn slice_image(
        rgb: &[u8],
        width: usize,
        height: usize,
        cfg: &SliceConfig,
        patch_size: usize,
    ) -> Result<SlicedImage> {
        ensure!(
            rgb.len() == width * height * 3 && width > 0 && height > 0,
            "minicpm-v: RGB buffer of {} bytes for {width}x{height}",
            rgb.len()
        );
        let grid = if cfg.slice_mode {
            get_sliced_grid(width, height, cfg.max_slice_nums, cfg.scale_resolution)
        } else {
            None
        };
        let (tw, th) = find_best_resize(
            width as f64,
            height as f64,
            cfg.scale_resolution,
            patch_size,
            grid.is_none(),
        );
        let thumb = pil_resize_rgb8_exact(rgb, width, height, tw, th, Filter::Bicubic);
        let mut crops = vec![patchify(&thumb, tw, th, cfg, patch_size)];
        if let Some((gx, gy)) = grid {
            let (rw, rh) =
                get_refine_size(width, height, (gx, gy), cfg.scale_resolution, patch_size);
            let refine = pil_resize_rgb8_exact(rgb, width, height, rw, rh, Filter::Bicubic);
            // `split_to_patches`: boxes stepped by `int(width / cols)`.
            let (pw, ph) = (rw / gx, rh / gy);
            for r in 0..gy {
                for c in 0..gx {
                    let mut tile = Vec::with_capacity(ph * pw * 3);
                    for y in r * ph..(r + 1) * ph {
                        let row = (y * rw + c * pw) * 3;
                        tile.extend_from_slice(&refine[row..row + pw * 3]);
                    }
                    crops.push(patchify(&tile, pw, ph, cfg, patch_size));
                }
            }
        }
        Ok(SlicedImage {
            crops,
            grid: grid.map(|(c, r)| (r, c)),
        })
    }

    /// `(x / 255 − mean) / std` in f32, cut into `[n_patches, 3·p²]` rows.
    fn patchify(rgb: &[u8], width: usize, height: usize, cfg: &SliceConfig, p: usize) -> Crop {
        let (gh, gw) = (height / p, width / p);
        let dim = 3 * p * p;
        let mut patches = vec![0f32; gh * gw * dim];
        for py in 0..gh {
            for px in 0..gw {
                let row = &mut patches[(py * gw + px) * dim..][..dim];
                for c in 0..3 {
                    for ky in 0..p {
                        for kx in 0..p {
                            let (y, x) = (py * p + ky, px * p + kx);
                            let v = rgb[(y * width + x) * 3 + c] as f32 / 255.0;
                            row[(c * p + ky) * p + kx] = (v - cfg.image_mean[c]) / cfg.image_std[c];
                        }
                    }
                }
            }
        }
        Crop {
            grid_h: gh,
            grid_w: gw,
            patches,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn grids_match_the_v4_processor_for_640x360() {
            // MiniCPM-V-4's processor: tgt_sizes [[24, 43], [34, 30], [34, 30]].
            assert_eq!(get_sliced_grid(640, 360, 9, 448), Some((2, 1)));
            assert_eq!(find_best_resize(640.0, 360.0, 448, 14, false), (602, 336));
            assert_eq!(get_refine_size(640, 360, (2, 1), 448, 14), (840, 476));
        }
    }
}

/// Normalize and cut an RGB8 crop into `[n_patches, 3·p²]` rows.
///
/// HF fuses rescale and normalize into `(x − 255·mean) / (255·std)`; the
/// same expression keeps the f32 values identical.
fn patchify(rgb: &[u8], width: usize, height: usize, cfg: &SliceConfig, p: usize) -> Crop {
    let (gh, gw) = (height / p, width / p);
    let mean = cfg.image_mean.map(|m| m * 255.0);
    let std = cfg.image_std.map(|s| s * 255.0);
    let dim = 3 * p * p;
    let mut patches = vec![0f32; gh * gw * dim];
    for py in 0..gh {
        for px in 0..gw {
            let row = &mut patches[(py * gw + px) * dim..][..dim];
            for c in 0..3 {
                for ky in 0..p {
                    for kx in 0..p {
                        let (y, x) = (py * p + ky, px * p + kx);
                        let v = rgb[(y * width + x) * 3 + c] as f32;
                        row[(c * p + ky) * p + kx] = (v - mean[c]) / std[c];
                    }
                }
            }
        }
    }
    Crop {
        grid_h: gh,
        grid_w: gw,
        patches,
    }
}

/// Unpack HF's NaViT-packed `pixel_values [1, 3, p, L]` back into one
/// `[n, 3·p²]` patch matrix per crop (for parity tests that feed the
/// reference processor's output).
pub fn unpack_navit(
    pixel_values: &[f32],
    total_len: usize,
    p: usize,
    grids: &[(usize, usize)],
) -> Vec<Crop> {
    // Layout: value(c, ky, col) at ((c·p + ky)·L + col); patch n occupies
    // columns n·p .. n·p + p of the packed row.
    let dim = 3 * p * p;
    let mut out = Vec::with_capacity(grids.len());
    let mut first_patch = 0usize;
    for &(gh, gw) in grids {
        let n = gh * gw;
        let mut patches = vec![0f32; n * dim];
        for i in 0..n {
            let col0 = (first_patch + i) * p;
            for c in 0..3 {
                for ky in 0..p {
                    for kx in 0..p {
                        patches[i * dim + (c * p + ky) * p + kx] =
                            pixel_values[(c * p + ky) * total_len + col0 + kx];
                    }
                }
            }
        }
        first_patch += n;
        out.push(Crop {
            grid_h: gh,
            grid_w: gw,
            patches,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_and_sizes_match_hf_for_640x360() {
        // The HF processor gives target_sizes [[24, 44], [32, 32], [32, 32]].
        let g = get_sliced_grid(360, 640, 9, 448);
        assert_eq!(g, Some((1, 2)));
        assert_eq!(find_best_resize(360.0, 640.0, 448, 14, false), (336, 616));
        assert_eq!(get_refine_size(360, 640, (1, 2), 448, 14), (448, 896));
    }

    #[test]
    fn small_image_is_thumbnail_only_and_upscaled() {
        assert_eq!(get_sliced_grid(200, 300, 9, 448), None);
        let (h, w) = find_best_resize(200.0, 300.0, 448, 14, true);
        assert_eq!((h % 56, w % 56), (0, 0));
        assert!(h * w > 200 * 300);
    }

    #[test]
    fn ensure_divide_rounds_half_to_even() {
        // 84 / 56 = 1.5 → 2, 140 / 56 = 2.5 → 2 (banker's rounding).
        assert_eq!(ensure_divide(84.0, 56), 112);
        assert_eq!(ensure_divide(140.0, 56), 112);
        assert_eq!(ensure_divide(10.0, 56), 56);
    }

    #[test]
    fn slices_are_row_major_tiles() {
        let (w, h) = (1200usize, 500usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 253) as u8).collect();
        let s = slice_image(&rgb, w, h, &SliceConfig::default(), 14).unwrap();
        let (rows, cols) = s.grid.unwrap();
        assert_eq!(s.crops.len(), 1 + rows * cols);
        for c in &s.crops {
            assert_eq!(c.patches.len(), c.num_patches() * 588);
            assert_eq!((c.grid_h % 4, c.grid_w % 4), (0, 0));
        }
    }

    #[test]
    fn navit_unpack_inverts_hf_packing() {
        // Pack two crops the way `reshape_by_patch` + concat does, unpack.
        let p = 2;
        let grids = [(1, 2), (2, 1)];
        let total: usize = grids.iter().map(|(h, w)| h * w).sum();
        let len = total * p;
        let packed: Vec<f32> = (0..3 * p * len).map(|i| i as f32).collect();
        let crops = unpack_navit(&packed, len, p, &grids);
        // Patch 2 (first of crop 1), channel 1, ky 1, kx 0.
        let want = packed[(p + 1) * len + 2 * p];
        assert_eq!(crops[1].patches[(p + 1) * p], want);
    }
}
