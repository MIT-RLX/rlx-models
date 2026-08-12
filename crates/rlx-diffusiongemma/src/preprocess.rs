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

//! `Gemma4ImageProcessor` — turning an RGB image into everything
//! [`crate::vision::build_vision_flow`] needs.
//!
//! The pipeline is aspect-ratio-preserving and variable-resolution: an image is
//! resized to the largest size that both fits the patch budget
//! (`max_soft_tokens · pooling_kernel_size²` patches) and is divisible by
//! `pooling_kernel_size · patch_size` on each side, then rescaled to `[0,1]`,
//! patchified, and padded out to the fixed budget. Note "largest that fits"
//! means small images are resized *up* — a 48×48 input becomes 384×384 at the
//! 70-soft-token budget — which is faithful to the reference. There is no mean/std
//! normalization — Gemma 4 was trained on `[0,1]` pixels and the tower rescales
//! to `[-1,1]` itself.
//!
//! Two ordering details matter and are easy to get backwards:
//!
//! * **Patch element order is `[row][col][channel]`**, not `[channel][row][col]`
//!   — HF reshapes to `(C, nph, ps, npw, ps)` and then transposes
//!   `(1, 3, 2, 4, 0)`, putting channel innermost.
//! * **Resize happens before rescale**, on `u8` pixels, through PIL. PIL runs a
//!   separable antialiased filter with the support widened by the downscale
//!   factor, and rounds to `u8` *between* the horizontal and vertical passes —
//!   so this reproduces both passes rather than doing one f32 2-D convolution.

use anyhow::{Result, anyhow};

use crate::vision::vision_pool_matrix;

/// Soft-token budgets the reference processor accepts.
pub const SUPPORTED_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];

/// Processor geometry, from `processor_config.json`'s `image_processor`.
#[derive(Debug, Clone, Copy)]
pub struct ImagePreprocessConfig {
    pub patch_size: usize,
    pub max_soft_tokens: usize,
    pub pooling_kernel_size: usize,
}

impl Default for ImagePreprocessConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            max_soft_tokens: 280,
            pooling_kernel_size: 3,
        }
    }
}

impl ImagePreprocessConfig {
    /// Patch budget: `max_soft_tokens · pooling_kernel_size²`.
    pub fn max_patches(&self) -> usize {
        self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size
    }

    /// Width of one flattened patch: `3 · patch_size²`.
    pub fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            SUPPORTED_SOFT_TOKENS.contains(&self.max_soft_tokens),
            "max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS:?}, got {}",
            self.max_soft_tokens
        );
        anyhow::ensure!(self.patch_size > 0 && self.pooling_kernel_size > 0);
        Ok(())
    }
}

/// The largest aspect-preserving size that fits the patch budget and is
/// divisible by `pooling_kernel_size · patch_size` on both sides.
pub fn target_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize)> {
    anyhow::ensure!(height > 0 && width > 0, "image must be non-empty");
    let total_px = (height * width) as f64;
    let target_px = (max_patches * patch_size * patch_size) as f64;
    let factor = (target_px / total_px).sqrt();
    let side_mult = pooling_kernel_size * patch_size;

    let mut th = ((factor * height as f64) / side_mult as f64).floor() as usize * side_mult;
    let mut tw = ((factor * width as f64) / side_mult as f64).floor() as usize * side_mult;

    if th == 0 && tw == 0 {
        return Err(anyhow!(
            "image {height}x{width} resizes to 0x0; each side must reach \
             pooling_kernel_size · patch_size = {side_mult}"
        ));
    }
    let max_side = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if th == 0 {
        th = side_mult;
        tw = ((width / height) * side_mult).min(max_side);
    } else if tw == 0 {
        tw = side_mult;
        th = ((height / width) * side_mult).min(max_side);
    }
    anyhow::ensure!(
        (th * tw) as f64 <= target_px,
        "resizing {height}x{width} to {th}x{tw} exceeds {max_patches} patches"
    );
    Ok((th, tw))
}

/// PIL's bicubic kernel (Catmull-Rom style with `a = -0.5`).
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Per-output-pixel filter taps, matching PIL's `precompute_coeffs`.
///
/// The support widens by the downscale factor (this is the antialiasing), and
/// the taps are normalized to sum to 1.
fn coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f64>)> {
    const SUPPORT: f64 = 2.0; // bicubic
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = SUPPORT * filterscale;
    let mut out = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5).floor() as isize).max(0) as usize;
        let xmax = ((center + support + 0.5).floor() as isize).min(in_size as isize) as usize;
        let mut ws: Vec<f64> = (xmin..xmax)
            .map(|x| bicubic((x as f64 - center + 0.5) / filterscale))
            .collect();
        let sum: f64 = ws.iter().sum();
        if sum != 0.0 {
            for w in &mut ws {
                *w /= sum;
            }
        }
        out.push((xmin, ws));
    }
    out
}

fn clip8(v: f64) -> u8 {
    // PIL rounds half away from zero, then clamps.
    let r = (v + 0.5).floor();
    r.clamp(0.0, 255.0) as u8
}

/// PIL-compatible antialiased bicubic resize of an interleaved RGB `u8` image.
///
/// Horizontal pass then vertical pass, each rounded back to `u8` — the same two
/// stages PIL runs, so intermediate quantization matches.
pub fn resize_bicubic_u8(
    src: &[u8],
    height: usize,
    width: usize,
    channels: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<u8>> {
    anyhow::ensure!(
        src.len() == height * width * channels,
        "image buffer is {} bytes, expected {}",
        src.len(),
        height * width * channels
    );
    if height == out_h && width == out_w {
        return Ok(src.to_vec());
    }

    // Horizontal: [h, w, c] -> [h, out_w, c]
    let hc = coeffs(width, out_w);
    let mut tmp = vec![0u8; height * out_w * channels];
    for y in 0..height {
        for (xx, (xmin, ws)) in hc.iter().enumerate() {
            for c in 0..channels {
                let mut acc = 0f64;
                for (i, w) in ws.iter().enumerate() {
                    acc += w * src[(y * width + xmin + i) * channels + c] as f64;
                }
                tmp[(y * out_w + xx) * channels + c] = clip8(acc);
            }
        }
    }

    // Vertical: [h, out_w, c] -> [out_h, out_w, c]
    let vc = coeffs(height, out_h);
    let mut dst = vec![0u8; out_h * out_w * channels];
    for (yy, (ymin, ws)) in vc.iter().enumerate() {
        for x in 0..out_w {
            for c in 0..channels {
                let mut acc = 0f64;
                for (i, w) in ws.iter().enumerate() {
                    acc += w * tmp[((ymin + i) * out_w + x) * channels + c] as f64;
                }
                dst[(yy * out_w + x) * channels + c] = clip8(acc);
            }
        }
    }
    Ok(dst)
}

/// Everything one image contributes to the vision graph.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    /// `[max_patches · 3·patch²]` in `[0,1]`, padded with zeros.
    pub pixels: Vec<f32>,
    /// Per-patch x index, `[max_patches]`; padded entries are 0 (and masked).
    pub pos_x: Vec<f32>,
    /// Per-patch y index, `[max_patches]`.
    pub pos_y: Vec<f32>,
    /// `1.0` for a real patch, `0.0` for padding, `[max_patches]`.
    pub valid: Vec<f32>,
    /// Average-pooling matrix `[max_soft_tokens, max_patches]`.
    pub pool: Vec<f32>,
    /// Patches this image actually occupies (before padding).
    pub num_patches: usize,
    /// Soft tokens this image actually occupies.
    pub num_soft_tokens: usize,
    /// Resized patch grid, `(cols, rows)`.
    pub grid: (usize, usize),
    /// Resized pixel size, `(height, width)`.
    pub size: (usize, usize),
}

/// Run the full processor over an interleaved RGB `u8` image.
///
/// Returns tensors sized to the *fixed* budget (`max_patches` /
/// `max_soft_tokens`), so one compiled vision graph serves every image
/// regardless of aspect ratio.
pub fn preprocess_image(
    rgb: &[u8],
    height: usize,
    width: usize,
    cfg: ImagePreprocessConfig,
) -> Result<PreprocessedImage> {
    cfg.validate()?;
    let max_patches = cfg.max_patches();
    let (th, tw) = target_size(
        height,
        width,
        cfg.patch_size,
        max_patches,
        cfg.pooling_kernel_size,
    )?;
    let resized = resize_bicubic_u8(rgb, height, width, 3, th, tw)?;

    let ps = cfg.patch_size;
    let (nph, npw) = (th / ps, tw / ps);
    let num_patches = nph * npw;
    anyhow::ensure!(
        num_patches <= max_patches,
        "{num_patches} patches exceeds the budget of {max_patches}"
    );
    let pool_k = cfg.pooling_kernel_size;
    let num_soft = num_patches / (pool_k * pool_k);

    // Patchify. Element order within a patch is [row][col][channel], matching
    // HF's transpose(1, 3, 2, 4, 0).
    let patch_dim = cfg.patch_dim();
    let mut pixels = vec![0f32; max_patches * patch_dim];
    let scale = 1.0 / 255.0;
    for py in 0..nph {
        for px in 0..npw {
            let patch = py * npw + px;
            let dst = &mut pixels[patch * patch_dim..(patch + 1) * patch_dim];
            for r in 0..ps {
                for c in 0..ps {
                    let sy = py * ps + r;
                    let sx = px * ps + c;
                    for ch in 0..3 {
                        dst[(r * ps + c) * 3 + ch] =
                            resized[(sy * tw + sx) * 3 + ch] as f32 * scale;
                    }
                }
            }
        }
    }

    let mut pos_x = vec![0f32; max_patches];
    let mut pos_y = vec![0f32; max_patches];
    let mut valid = vec![0f32; max_patches];
    let mut real_positions = Vec::with_capacity(num_patches);
    for y in 0..nph {
        for x in 0..npw {
            let i = y * npw + x;
            pos_x[i] = x as f32;
            pos_y[i] = y as f32;
            valid[i] = 1.0;
            real_positions.push((x as u32, y as u32));
        }
    }

    // Pooling matrix over the *real* patches; padded columns stay zero, which
    // is equivalent to HF zeroing padded hidden states before the average.
    let real_pool = vision_pool_matrix(&real_positions, pool_k, cfg.max_soft_tokens);
    let mut pool = vec![0f32; cfg.max_soft_tokens * max_patches];
    for b in 0..cfg.max_soft_tokens {
        let src = &real_pool[b * num_patches..(b + 1) * num_patches];
        pool[b * max_patches..b * max_patches + num_patches].copy_from_slice(src);
    }

    Ok(PreprocessedImage {
        pixels,
        pos_x,
        pos_y,
        valid,
        pool,
        num_patches,
        num_soft_tokens: num_soft,
        grid: (npw, nph),
        size: (th, tw),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ImagePreprocessConfig {
        ImagePreprocessConfig::default()
    }

    #[test]
    fn budget_is_soft_tokens_times_kernel_squared() {
        let c = cfg();
        assert_eq!(c.max_patches(), 280 * 9);
        assert_eq!(c.patch_dim(), 3 * 16 * 16);
    }

    #[test]
    fn target_size_is_divisible_and_within_budget() {
        let c = cfg();
        let side = c.pooling_kernel_size * c.patch_size; // 48
        for (h, w) in [(1024, 768), (640, 480), (100, 3000), (57, 57), (4000, 12)] {
            let (th, tw) = target_size(h, w, c.patch_size, c.max_patches(), c.pooling_kernel_size)
                .unwrap_or_else(|e| panic!("{h}x{w}: {e}"));
            assert_eq!(th % side, 0, "{h}x{w} -> {th}x{tw} height not divisible");
            assert_eq!(tw % side, 0, "{h}x{w} -> {th}x{tw} width not divisible");
            let patches = (th / c.patch_size) * (tw / c.patch_size);
            assert!(
                patches <= c.max_patches(),
                "{h}x{w} -> {th}x{tw} = {patches} patches over budget"
            );
            assert!(th > 0 && tw > 0);
        }
    }

    #[test]
    fn target_size_preserves_aspect_ratio_roughly() {
        let c = cfg();
        let (th, tw) = target_size(
            1000,
            500,
            c.patch_size,
            c.max_patches(),
            c.pooling_kernel_size,
        )
        .unwrap();
        // 2:1 input stays close to 2:1 after rounding down to multiples of 48.
        let ratio = th as f64 / tw as f64;
        assert!((ratio - 2.0).abs() < 0.35, "ratio {ratio} for {th}x{tw}");
    }

    #[test]
    fn identity_resize_is_a_passthrough() {
        let src: Vec<u8> = (0..(4 * 6 * 3)).map(|i| (i % 251) as u8).collect();
        let out = resize_bicubic_u8(&src, 4, 6, 3, 4, 6).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn resize_of_a_constant_image_is_constant() {
        // Normalized taps mean a flat image survives any rescale exactly.
        let src = vec![137u8; 12 * 9 * 3];
        let out = resize_bicubic_u8(&src, 12, 9, 3, 6, 3).unwrap();
        assert_eq!(out.len(), 6 * 3 * 3);
        assert!(out.iter().all(|&v| v == 137), "got {:?}", &out[..9]);
    }

    #[test]
    fn resize_rejects_a_mis_sized_buffer() {
        assert!(resize_bicubic_u8(&[0u8; 10], 4, 6, 3, 2, 2).is_err());
    }

    /// 384×384 is already the budget-filling square for `max_soft_tokens = 70`,
    /// so the resize is a no-op and the patch layout is directly checkable.
    #[test]
    fn preprocess_lays_out_patches_row_major_with_channels_innermost() {
        let c = ImagePreprocessConfig {
            patch_size: 16,
            max_soft_tokens: 70,
            pooling_kernel_size: 3,
        };
        // Distinct value per (y, x, channel) so ordering errors are visible.
        let (h, w) = (384usize, 384usize);
        let mut img = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3 {
                    img[(y * w + x) * 3 + ch] = ((y * 7 + x * 3 + ch * 53) % 256) as u8;
                }
            }
        }
        let p = preprocess_image(&img, h, w, c).unwrap();
        assert_eq!(
            p.size,
            (384, 384),
            "already budget-sized: resize is identity"
        );
        assert_eq!(p.grid, (24, 24));
        assert_eq!(p.num_patches, 576);
        assert_eq!(p.num_soft_tokens, 64);
        assert_eq!(p.pixels.len(), c.max_patches() * c.patch_dim());

        // Patch 0 covers rows 0..16, cols 0..16. Element (r, c, ch) sits at
        // ((r*16 + c)*3 + ch), i.e. channel innermost.
        let d = c.patch_dim();
        for (r, cc, ch) in [(0usize, 0usize, 0usize), (0, 1, 2), (3, 5, 1), (15, 15, 2)] {
            let want = img[((r) * w + cc) * 3 + ch] as f32 / 255.0;
            let got = p.pixels[(r * 16 + cc) * 3 + ch];
            assert!((got - want).abs() < 1e-6, "patch0 ({r},{cc},{ch})");
        }
        // Patch 1 is the next 16 columns.
        let want = img[16 * 3] as f32 / 255.0;
        assert!((p.pixels[d] - want).abs() < 1e-6);
        // Patch `grid.0` starts the second patch row.
        let want = img[(16 * w) * 3] as f32 / 255.0;
        assert!((p.pixels[p.grid.0 * d] - want).abs() < 1e-6);
    }

    /// A wide image leaves part of the fixed budget unused; those slots must be
    /// zeroed, masked, and excluded from the pooling matrix.
    #[test]
    fn preprocess_pads_and_masks_the_unused_budget() {
        let c = ImagePreprocessConfig {
            patch_size: 16,
            max_soft_tokens: 70,
            pooling_kernel_size: 3,
        };
        let (h, w) = (240usize, 720usize);
        let img = vec![200u8; h * w * 3];
        let p = preprocess_image(&img, h, w, c).unwrap();
        let (cols, rows) = p.grid;
        let n = p.num_patches;
        assert_eq!(n, cols * rows);
        assert!(n < c.max_patches(), "test needs a partially filled budget");
        assert_eq!(p.num_soft_tokens, n / 9);

        assert_eq!(p.valid.len(), c.max_patches());
        assert!(p.valid[..n].iter().all(|&v| v == 1.0));
        assert!(p.valid[n..].iter().all(|&v| v == 0.0));
        // Padded patch pixels are zero.
        assert!(p.pixels[n * c.patch_dim()..].iter().all(|&v| v == 0.0));
        // Positions are row-major over the grid.
        assert_eq!((p.pos_x[0], p.pos_y[0]), (0.0, 0.0));
        assert_eq!(
            (p.pos_x[cols - 1], p.pos_y[cols - 1]),
            ((cols - 1) as f32, 0.0)
        );
        assert_eq!((p.pos_x[cols], p.pos_y[cols]), (0.0, 1.0));

        // The pool matrix only touches real patches, and each used bucket
        // averages exactly k² of them.
        assert_eq!(p.pool.len(), c.max_soft_tokens * c.max_patches());
        for b in 0..p.num_soft_tokens {
            let row = &p.pool[b * c.max_patches()..(b + 1) * c.max_patches()];
            let hits = row.iter().filter(|&&x| x != 0.0).count();
            assert_eq!(hits, 9, "bucket {b}");
            assert!(row[n..].iter().all(|&x| x == 0.0), "padding must stay zero");
        }
        for b in p.num_soft_tokens..c.max_soft_tokens {
            let row = &p.pool[b * c.max_patches()..(b + 1) * c.max_patches()];
            assert!(row.iter().all(|&x| x == 0.0), "unused bucket {b}");
        }
    }

    /// Surprising but faithful: the processor resizes *up* as well as down —
    /// the target is the largest size fitting the budget, not `min(orig, ..)`.
    #[test]
    fn small_images_are_upscaled_to_fill_the_budget() {
        let c = ImagePreprocessConfig {
            patch_size: 16,
            max_soft_tokens: 70,
            pooling_kernel_size: 3,
        };
        let p = preprocess_image(&vec![10u8; 48 * 48 * 3], 48, 48, c).unwrap();
        assert_eq!(p.size, (384, 384));
        assert_eq!(p.num_patches, 576);
        // A flat image survives the resample exactly.
        let v = 10.0f32 / 255.0;
        assert!(
            p.pixels[..p.num_patches * c.patch_dim()]
                .iter()
                .all(|&x| (x - v).abs() < 1e-6)
        );
    }

    #[test]
    fn rejects_an_unsupported_soft_token_budget() {
        let c = ImagePreprocessConfig {
            patch_size: 16,
            max_soft_tokens: 100,
            pooling_kernel_size: 3,
        };
        assert!(preprocess_image(&[0u8; 48 * 48 * 3], 48, 48, c).is_err());
    }
}
