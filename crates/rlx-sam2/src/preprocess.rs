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

//! SAM 2 host-side preprocessing.
//!
//! Three pieces live on the host (outside the IR graph):
//!   1. **Image preprocess** — square resize to 1024×1024 (bilinear, no
//!      aspect-ratio preservation), /255, ImageNet normalize. Differs
//!      from SAM v1 in two ways: SAM 2 *does* divide by 255 first, and
//!      *does not* keep aspect ratio (the reference's
//!      `SAM2Transforms.__call__` is just `Resize((1024,1024))` +
//!      `Normalize`).
//!   2. **Patch embedding** — Conv2d(in=3, out=embed_dim, k=7, s=4,
//!      p=3). Overlapping kernel (k > s) so we can't reduce to a plain
//!      per-patch matmul like SAM v1. Runs as a direct host-side
//!      Conv2d; cheap (once per image vs. per-block).
//!   3. **Stage-0 position embedding** — bicubic-interpolated
//!      `pos_embed` table + tiled `pos_embed_window`, summed into the
//!      patch tokens before they enter the encoder body. Materialised
//!      host-side because IR has no bicubic-resample op.

use super::config::{
    SAM2_IMG_SIZE, SAM2_PATCH_GRID, SAM2_PATCH_KERNEL, SAM2_PATCH_PADDING, SAM2_PATCH_STRIDE,
    SAM2_PIXEL_MEAN, SAM2_PIXEL_STD, Sam2HieraConfig,
};
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

/// Weights extracted from the safetensors checkpoint that the host
/// uses *before* the encoder graph runs.
pub struct Sam2PreprocessWeights {
    /// Patch projection weight in raw `[E, 3, k, k]` NCHW layout. Kept
    /// raw (not transposed) because the host-side conv2d needs index
    /// access; see `assemble_patch_tokens`.
    pub patch_proj_w: Vec<f32>,
    /// Patch projection bias `[E]`.
    pub patch_proj_b: Vec<f32>,
    /// Stage-0 position embedding, already interpolated + tiled to
    /// `[grid · grid · E]` BHWC. Added to patch tokens before the
    /// encoder body.
    pub pos_embed_full: Vec<f32>,
    pub embed_dim: usize,
    pub grid: usize, // = 256 (SAM2_PATCH_GRID)
}

pub(super) fn extract_preprocess_weights(
    weights: &mut WeightMap,
    cfg: &Sam2HieraConfig,
) -> Result<Sam2PreprocessWeights> {
    let e = cfg.embed_dim;
    let k = SAM2_PATCH_KERNEL;
    let grid = SAM2_PATCH_GRID;

    // image_encoder.trunk.patch_embed.proj.weight [E, 3, k, k]
    let (patch_proj_w, w_shape) = weights.take("image_encoder.trunk.patch_embed.proj.weight")?;
    ensure!(
        w_shape == vec![e, 3, k, k],
        "patch_embed.proj.weight expected [{e}, 3, {k}, {k}], got {w_shape:?}"
    );
    let (patch_proj_b, _) = weights.take("image_encoder.trunk.patch_embed.proj.bias")?;

    // image_encoder.trunk.pos_embed         [1, E, Ph, Pw]
    // image_encoder.trunk.pos_embed_window  [1, E, mu, mu]
    let (pe_raw, pe_shape) = weights.take("image_encoder.trunk.pos_embed")?;
    let [ph, pw] = cfg.window_pos_embed_bkg_spatial_size;
    ensure!(
        pe_shape == vec![1, e, ph, pw],
        "pos_embed expected [1, {e}, {ph}, {pw}], got {pe_shape:?}"
    );

    let mu = cfg.window_size_at_stage(0);
    let (pew_raw, pew_shape) = weights.take("image_encoder.trunk.pos_embed_window")?;
    ensure!(
        pew_shape == vec![1, e, mu, mu],
        "pos_embed_window expected [1, {e}, {mu}, {mu}], got {pew_shape:?}"
    );

    let pos_embed_full = build_full_pos_embed(&pe_raw, &pew_raw, e, ph, pw, mu, grid);

    Ok(Sam2PreprocessWeights {
        patch_proj_w,
        patch_proj_b,
        pos_embed_full,
        embed_dim: e,
        grid,
    })
}

/// Replicates the reference's `Hiera._get_pos_embed`:
///   - bicubic-interpolate `pos_embed` from `[Ph, Pw]` to `[grid, grid]`
///   - tile `pos_embed_window` to `[grid, grid]`
///   - sum, permute to NHWC (`[grid, grid, E]`), flatten
fn build_full_pos_embed(
    pe: &[f32],
    pew: &[f32],
    e: usize,
    ph: usize,
    pw: usize,
    mu: usize,
    grid: usize,
) -> Vec<f32> {
    debug_assert_eq!(pe.len(), e * ph * pw);
    debug_assert_eq!(pew.len(), e * mu * mu);
    debug_assert_eq!(
        grid % mu,
        0,
        "Hiera pos_embed_window must tile grid evenly (grid={grid}, mu={mu})"
    );

    // 1) bicubic-interpolate pe per channel into `interp_pe` [E, grid, grid].
    let mut interp_pe = vec![0f32; e * grid * grid];
    for c in 0..e {
        let src = &pe[c * ph * pw..(c + 1) * ph * pw];
        let dst = &mut interp_pe[c * grid * grid..(c + 1) * grid * grid];
        bicubic_resize_2d(src, ph, pw, dst, grid, grid);
    }

    // 2) Tile pew across grid (it tiles by integer factor since
    //    grid is a multiple of mu) and sum.
    let mut out_nchw = interp_pe; // reuse
    for c in 0..e {
        for y in 0..grid {
            let ty = y % mu;
            for x in 0..grid {
                let tx = x % mu;
                let w_val = pew[c * mu * mu + ty * mu + tx];
                out_nchw[c * grid * grid + y * grid + x] += w_val;
            }
        }
    }

    // 3) Permute NCHW → BHWC (just a single sample, B=1) and flatten.
    let mut out_bhwc = vec![0f32; grid * grid * e];
    for y in 0..grid {
        for x in 0..grid {
            for c in 0..e {
                out_bhwc[(y * grid + x) * e + c] = out_nchw[c * grid * grid + y * grid + x];
            }
        }
    }
    out_bhwc
}

/// Catmull-Rom bicubic resize of a single-channel `[h_in, w_in]` image
/// into `[h_out, w_out]`. Uses the OpenCV / PyTorch default
/// `align_corners=False` convention.
///
/// Only used for the 14×14 → 256×256 `pos_embed` interpolation (i.e.
/// once per model load, not per inference) so the simple loop is fine.
fn bicubic_resize_2d(
    src: &[f32],
    h_in: usize,
    w_in: usize,
    dst: &mut [f32],
    h_out: usize,
    w_out: usize,
) {
    fn cubic(t: f32) -> f32 {
        // Standard Catmull-Rom kernel (a = -0.75, PyTorch / cv2 default).
        let a = -0.75_f32;
        let t = t.abs();
        if t < 1.0 {
            ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0
        } else if t < 2.0 {
            (((t - 5.0) * t + 8.0) * t - 4.0) * a
        } else {
            0.0
        }
    }
    fn idx(i: isize, max: isize) -> usize {
        // Replicate-edge (clamped) indexing.
        i.clamp(0, max - 1) as usize
    }

    let sx = (w_in as f32) / (w_out as f32);
    let sy = (h_in as f32) / (h_out as f32);

    for y_o in 0..h_out {
        // align_corners=False: src y = (y_o + 0.5) * sy - 0.5
        let yf = (y_o as f32 + 0.5) * sy - 0.5;
        let yi = yf.floor();
        let dy = yf - yi;
        let wy = [cubic(1.0 + dy), cubic(dy), cubic(1.0 - dy), cubic(2.0 - dy)];
        for x_o in 0..w_out {
            let xf = (x_o as f32 + 0.5) * sx - 0.5;
            let xi = xf.floor();
            let dx = xf - xi;
            let wx = [cubic(1.0 + dx), cubic(dx), cubic(1.0 - dx), cubic(2.0 - dx)];

            let mut acc = 0f32;
            for jy in 0..4 {
                let iy = idx(yi as isize - 1 + jy, h_in as isize);
                for jx in 0..4 {
                    let ix = idx(xi as isize - 1 + jx as isize, w_in as isize);
                    acc += src[iy * w_in + ix] * wy[jy as usize] * wx[jx];
                }
            }
            dst[y_o * w_out + x_o] = acc;
        }
    }
}

/// Square-resize an RGB u8 image to 1024×1024 (bilinear, no aspect-
/// ratio preservation), /255, then ImageNet-normalise. Returns a
/// contiguous `[3, 1024, 1024]` NCHW f32 buffer.
///
/// Matches `SAM2Transforms` in the reference exactly:
/// `Resize((1024, 1024))` (PIL bilinear) → `ToTensor` (/255) →
/// `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`.
pub fn preprocess_image(rgb: &[u8], h_in: usize, w_in: usize) -> Vec<f32> {
    debug_assert_eq!(rgb.len(), h_in * w_in * 3);
    let out_size = SAM2_IMG_SIZE;
    let mut nchw = vec![0f32; 3 * out_size * out_size];

    // PIL Resize uses `align_corners=False` bilinear.
    let sx = (w_in as f32) / (out_size as f32);
    let sy = (h_in as f32) / (out_size as f32);

    for y_o in 0..out_size {
        let yf = (y_o as f32 + 0.5) * sy - 0.5;
        let y0 = yf.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(h_in - 1);
        let dy = (yf - yf.floor()).clamp(0.0, 1.0);
        for x_o in 0..out_size {
            let xf = (x_o as f32 + 0.5) * sx - 0.5;
            let x0 = xf.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(w_in - 1);
            let dx = (xf - xf.floor()).clamp(0.0, 1.0);
            for c in 0..3 {
                let p00 = rgb[(y0 * w_in + x0) * 3 + c] as f32;
                let p01 = rgb[(y0 * w_in + x1) * 3 + c] as f32;
                let p10 = rgb[(y1 * w_in + x0) * 3 + c] as f32;
                let p11 = rgb[(y1 * w_in + x1) * 3 + c] as f32;
                let top = p00 * (1.0 - dx) + p01 * dx;
                let bot = p10 * (1.0 - dx) + p11 * dx;
                let v01 = (top * (1.0 - dy) + bot * dy) / 255.0;
                nchw[c * out_size * out_size + y_o * out_size + x_o] =
                    (v01 - SAM2_PIXEL_MEAN[c]) / SAM2_PIXEL_STD[c];
            }
        }
    }
    nchw
}

/// Run Hiera's patch embedding (Conv2d k=7 s=4 p=3) on the host, then
/// add the stage-0 position embedding. Output is `[grid, grid, E]`
/// BHWC (the layout Hiera operates on internally), flattened.
///
/// `image_nchw` is the `[3, 1024, 1024]` tensor from `preprocess_image`.
pub fn assemble_patch_tokens(pre: &Sam2PreprocessWeights, image_nchw: &[f32]) -> Result<Vec<f32>> {
    let e = pre.embed_dim;
    let grid = pre.grid;
    let k = SAM2_PATCH_KERNEL;
    let s = SAM2_PATCH_STRIDE;
    let pad = SAM2_PATCH_PADDING;
    ensure!(
        image_nchw.len() == 3 * SAM2_IMG_SIZE * SAM2_IMG_SIZE,
        "image must be [3, {}, {}] NCHW, got len {}",
        SAM2_IMG_SIZE,
        SAM2_IMG_SIZE,
        image_nchw.len()
    );

    let h = SAM2_IMG_SIZE;
    let w = SAM2_IMG_SIZE;
    let mut out = vec![0f32; grid * grid * e];

    // Direct Conv2d. Per-output-pixel cost is k·k·in_c·E = 7·7·3·E.
    // For E=112, grid=256 this is ~256² · 7² · 3 · 112 ≈ 1.1 G fmas —
    // about the same as a single transformer block, run once.
    for py in 0..grid {
        for px in 0..grid {
            // dst row in BHWC
            let dst = &mut out[(py * grid + px) * e..(py * grid + px + 1) * e];
            // Start with bias.
            dst.copy_from_slice(&pre.patch_proj_b);
            // Convolve.
            for ky in 0..k {
                let iy = (py * s) as isize + ky as isize - pad as isize;
                if iy < 0 || iy >= h as isize {
                    continue;
                }
                let iy = iy as usize;
                for kx in 0..k {
                    let ix = (px * s) as isize + kx as isize - pad as isize;
                    if ix < 0 || ix >= w as isize {
                        continue;
                    }
                    let ix = ix as usize;
                    for c in 0..3 {
                        let v = image_nchw[c * h * w + iy * w + ix];
                        // weight is [E, 3, k, k]: row-major
                        let w_base = c * k * k + ky * k + kx;
                        let stride = 3 * k * k;
                        for ei in 0..e {
                            dst[ei] += v * pre.patch_proj_w[ei * stride + w_base];
                        }
                    }
                }
            }
        }
    }

    // Add stage-0 position embedding (already in BHWC).
    ensure!(
        pre.pos_embed_full.len() == grid * grid * e,
        "pos_embed_full size mismatch"
    );
    for i in 0..grid * grid * e {
        out[i] += pre.pos_embed_full[i];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_shape_and_range() {
        // 50×30 RGB → 1024×1024 NCHW.
        let img = vec![128u8; 50 * 30 * 3];
        let nchw = preprocess_image(&img, 50, 30);
        assert_eq!(nchw.len(), 3 * 1024 * 1024);
        // 128/255 ≈ 0.5019; per-channel normalised ≈ (0.502 - mean) / std.
        for c in 0..3 {
            let expected = (128.0 / 255.0 - SAM2_PIXEL_MEAN[c]) / SAM2_PIXEL_STD[c];
            let mid = nchw[c * 1024 * 1024 + 512 * 1024 + 512];
            assert!(
                (mid - expected).abs() < 1e-4,
                "channel {c}: {mid} vs {expected}"
            );
        }
    }

    #[test]
    fn bicubic_identity() {
        // 8×8 → 8×8 should be (close to) identity for bicubic with
        // align_corners=False.
        let src: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let mut dst = vec![0f32; 64];
        bicubic_resize_2d(&src, 8, 8, &mut dst, 8, 8);
        for i in 0..64 {
            assert!(
                (src[i] - dst[i]).abs() < 1e-4,
                "identity broken at {i}: {} vs {}",
                src[i],
                dst[i]
            );
        }
    }
}
