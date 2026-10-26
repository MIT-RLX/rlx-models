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
//! Two host-side tensor manipulations happen before the IR graph:
//!   1. **Image preprocess** ([`super::image`]) — resize the long side
//!      to 1024 (preserve aspect ratio), normalize, zero-pad to
//!      1024×1024 NCHW. Matches `sam.rs::preprocess()` in candle exactly.
//!   2. **Patch embedding** (here) — Conv2d(in=3, out=embed_dim, k=16, s=16)
//!      with no padding, equivalent to per-patch matmul. We do it on
//!      the CPU for the same reason as DINOv2: rlx-ir has no f32
//!      forward Conv2d. The output is the input to the encoder graph,
//!      already in `[B, H, W, C]` BHWC layout (matching SAM's internal
//!      convention).

use super::config::{SAM_IMG_SIZE, SAM_PATCH_SIZE, SamEncoderConfig};
pub use super::image::preprocess_image;
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
