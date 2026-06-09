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

//! SAM v1 host-side preprocessing.
//!
//! Two host-side tensor manipulations live here instead of in the IR
//! graph:
//!   1. **Image preprocess** — resize the long side to 1024 (preserve
//!      aspect ratio), normalize, zero-pad to 1024×1024 NCHW. Matches
//!      `sam.rs::preprocess()` in candle exactly.
//!   2. **Patch embedding** — Conv2d(in=3, out=embed_dim, k=16, s=16)
//!      with no padding, equivalent to per-patch matmul. We do it on
//!      the CPU for the same reason as DINOv2: rlx-ir has no f32
//!      forward Conv2d. The output is the input to the encoder graph,
//!      already in `[B, H, W, C]` BHWC layout (matching SAM's internal
//!      convention).

use super::config::{
    SAM_IMG_SIZE, SAM_PATCH_SIZE, SAM_PIXEL_MEAN, SAM_PIXEL_STD, SamEncoderConfig,
};
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

/// Weights extracted from the safetensors checkpoint that the host
/// uses *before* the encoder graph runs.
pub struct SamPreprocessWeights {
    /// Patch projection weight `[E, 3, 16, 16]` flattened+transposed to
    /// `[3·16·16, E]` for row-major sgemm.
    pub patch_proj_w: Vec<f32>,
    /// Patch projection bias `[E]`.
    pub patch_proj_b: Vec<f32>,
    /// Optional absolute positional embedding `[1, hw, hw, E]`
    /// flattened to `[hw · hw · E]`. Added to the patch embeddings
    /// before they enter the IR graph.
    pub pos_embed: Option<Vec<f32>>,
    pub embed_dim: usize,
    pub hw: usize,
}

pub(super) fn extract_preprocess_weights(
    weights: &mut WeightMap,
    cfg: &SamEncoderConfig,
) -> Result<SamPreprocessWeights> {
    let e = cfg.embed_dim;
    let hw = cfg.num_patches_per_side();
    let pd = 3 * SAM_PATCH_SIZE * SAM_PATCH_SIZE;

    // image_encoder.patch_embed.proj.weight  [E, 3, 16, 16]
    let (proj_raw, proj_shape) = weights.take("image_encoder.patch_embed.proj.weight")?;
    ensure!(
        proj_shape == vec![e, 3, SAM_PATCH_SIZE, SAM_PATCH_SIZE],
        "patch_embed.proj.weight expected [{e}, 3, {SAM_PATCH_SIZE}, {SAM_PATCH_SIZE}], got {proj_shape:?}"
    );
    // Flatten [E, 3, 16, 16] → [E, patch_dim] (already contiguous) then
    // transpose to [patch_dim, E].
    let mut patch_proj_w = vec![0f32; e * pd];
    for ei in 0..e {
        for d in 0..pd {
            patch_proj_w[d * e + ei] = proj_raw[ei * pd + d];
        }
    }
    let (patch_proj_b, _) = weights.take("image_encoder.patch_embed.proj.bias")?;

    let pos_embed = if cfg.use_abs_pos {
        let (data, shape) = weights.take("image_encoder.pos_embed")?;
        ensure!(
            shape == vec![1, hw, hw, e],
            "pos_embed expected [1, {hw}, {hw}, {e}], got {shape:?}"
        );
        Some(data)
    } else {
        None
    };

    Ok(SamPreprocessWeights {
        patch_proj_w,
        patch_proj_b,
        pos_embed,
        embed_dim: e,
        hw,
    })
}

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

/// Run the patch embedding (Conv2d k=16 s=16 no padding) on the host
/// and add the absolute positional embedding. Output is `[1, hw, hw,
/// E]` BHWC (SAM's internal convention) flattened to a contiguous
/// f32 buffer for the encoder graph.
pub fn assemble_patch_tokens(pre: &SamPreprocessWeights, image_nchw: &[f32]) -> Result<Vec<f32>> {
    let e = pre.embed_dim;
    let hw = pre.hw;
    let pd = 3 * SAM_PATCH_SIZE * SAM_PATCH_SIZE;
    ensure!(
        image_nchw.len() == 3 * SAM_IMG_SIZE * SAM_IMG_SIZE,
        "image must be [3, {SAM_IMG_SIZE}, {SAM_IMG_SIZE}] NCHW, got len {}",
        image_nchw.len()
    );

    let mut out = vec![0f32; hw * hw * e];
    let mut patch_buf = vec![0f32; pd];
    for py in 0..hw {
        for px in 0..hw {
            // Fill patch_buf in CHW order matching the Conv2d weight
            // layout `[E, C=3, ph, pw]` that we flattened earlier.
            for c in 0..3 {
                for ry in 0..SAM_PATCH_SIZE {
                    let src_y = py * SAM_PATCH_SIZE + ry;
                    for rx in 0..SAM_PATCH_SIZE {
                        let src_x = px * SAM_PATCH_SIZE + rx;
                        let src = c * SAM_IMG_SIZE * SAM_IMG_SIZE + src_y * SAM_IMG_SIZE + src_x;
                        let dst = c * SAM_PATCH_SIZE * SAM_PATCH_SIZE + ry * SAM_PATCH_SIZE + rx;
                        patch_buf[dst] = image_nchw[src];
                    }
                }
            }
            // patch_buf @ proj_w + proj_b → embed_dim vector.
            let row = py * hw + px;
            let dst = &mut out[row * e..(row + 1) * e];
            dst.copy_from_slice(&pre.patch_proj_b);
            for d in 0..pd {
                let v = patch_buf[d];
                if v == 0.0 {
                    continue;
                }
                let w_row = &pre.patch_proj_w[d * e..(d + 1) * e];
                for k in 0..e {
                    dst[k] += v * w_row[k];
                }
            }
        }
    }

    // Add absolute positional embedding (broadcast over batch).
    if let Some(pos) = &pre.pos_embed {
        ensure!(pos.len() == hw * hw * e, "pos_embed size mismatch");
        for i in 0..hw * hw * e {
            out[i] += pos[i];
        }
    }
    Ok(out)
}
