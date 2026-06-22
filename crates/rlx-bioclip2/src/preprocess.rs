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

//! Host-side BioCLIP-2 vision preprocessing: conv1 patch projection +
//! token assembly.
//!
//! rlx-ir has no f32 forward Conv2d today, so `visual.conv1` (a
//! stride-`patch_size`, bias-free Conv2d in OpenCLIP) is performed on the
//! CPU using a single matmul over unfolded patches — the same trick the
//! DINOv2 runner uses. The output is the `"hidden"` tensor `[B, seq,
//! width]` consumed by the vision IR graph.
//!
//! ## Pipeline
//! ```text
//!   image [B, 3, H, W] (CLIP-normalized, NCHW f32)
//!     → unfold to [B, np, 3·ps·ps]
//!     → matmul conv1_w → [B, np, width]
//!     → prepend class_embedding → [B, 1+np, width]
//!     → add positional_embedding → [B, seq, width]
//! ```

use crate::config::{BioClip2Config, CLIP_MEAN, CLIP_STD};
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

/// Vision-embed weights extracted from the checkpoint (consumed on host).
pub struct VisionEmbedWeights {
    /// `visual.conv1.weight` `[width, 3, ps, ps]` reshaped+transposed to
    /// `[3·ps·ps, width]` (row-major sgemm-friendly). No bias.
    pub conv1_w: Vec<f32>,
    /// `visual.class_embedding` `[width]`.
    pub class_embedding: Vec<f32>,
    /// `visual.positional_embedding` flattened `[seq · width]`.
    pub positional_embedding: Vec<f32>,
    pub width: usize,
    pub patch_dim: usize,
    pub num_patches: usize,
    pub seq: usize,
}

pub(crate) fn extract_vision_embed_weights(
    weights: &mut WeightMap,
    cfg: &BioClip2Config,
) -> Result<VisionEmbedWeights> {
    let v = &cfg.vision;
    let width = v.width;
    let patch_dim = v.patch_dim();
    let num_patches = v.num_patches();
    let seq = v.seq_len();

    // Conv2d [width, 3, ps, ps] → flatten to [width, patch_dim] → transpose
    // to [patch_dim, width].
    let (conv_raw, conv_shape) = weights.take("visual.conv1.weight")?;
    ensure!(
        conv_shape.len() == 4
            && conv_shape[0] == width
            && conv_shape[1] * conv_shape[2] * conv_shape[3] == patch_dim,
        "visual.conv1.weight expected [width={width}, 3, ps, ps] (patch_dim={patch_dim}), got {conv_shape:?}"
    );
    let mut conv1_w = vec![0f32; width * patch_dim];
    for e in 0..width {
        for d in 0..patch_dim {
            conv1_w[d * width + e] = conv_raw[e * patch_dim + d];
        }
    }

    let (class_embedding, _) = weights.take("visual.class_embedding")?;
    ensure!(
        class_embedding.len() == width,
        "visual.class_embedding length {} != width {width}",
        class_embedding.len()
    );

    let (positional_embedding, pos_shape) = weights.take("visual.positional_embedding")?;
    ensure!(
        positional_embedding.len() == seq * width,
        "visual.positional_embedding length {} != seq*width ({seq}*{width}); shape={pos_shape:?}",
        positional_embedding.len()
    );

    Ok(VisionEmbedWeights {
        conv1_w,
        class_embedding,
        positional_embedding,
        width,
        patch_dim,
        num_patches,
        seq,
    })
}

/// Image → vision hidden tensor for the encoder graph.
///
/// `image`: NCHW float32, length `batch · 3 · img · img`, already
/// CLIP-normalized (see [`clip_normalize_nchw`]). Returns
/// `[batch · seq · width]` flat row-major.
pub fn assemble_vision_hidden(
    pre: &VisionEmbedWeights,
    image: &[f32],
    batch: usize,
    patch_size: usize,
    img_size: usize,
) -> Result<Vec<f32>> {
    let w = pre.width;
    let np = pre.num_patches;
    let seq = pre.seq;
    let pd = pre.patch_dim;
    let n_side = img_size / patch_size;

    ensure!(
        image.len() == batch * 3 * img_size * img_size,
        "image length {} != B·3·H·W ({}·3·{}·{})",
        image.len(),
        batch,
        img_size,
        img_size
    );
    ensure!(
        np == n_side * n_side,
        "num_patches mismatch — img_size/patch_size inconsistent"
    );

    let mut hidden = vec![0f32; batch * seq * w];

    for b in 0..batch {
        let img_off = b * 3 * img_size * img_size;
        let out_off = b * seq * w;

        // Row 0 = class embedding.
        hidden[out_off..out_off + w].copy_from_slice(&pre.class_embedding);

        // Patch tokens — unfold + project. Patch (py, px) → row 1 + py*n_side + px.
        let mut patch_buf = vec![0f32; pd];
        for py in 0..n_side {
            for px in 0..n_side {
                // CHW order to match the Conv2d weight layout [width, C=3, ph, pw].
                for c in 0..3 {
                    for ry in 0..patch_size {
                        let src_y = py * patch_size + ry;
                        for rx in 0..patch_size {
                            let src_x = px * patch_size + rx;
                            let src_idx =
                                img_off + c * img_size * img_size + src_y * img_size + src_x;
                            let dst_idx = c * patch_size * patch_size + ry * patch_size + rx;
                            patch_buf[dst_idx] = image[src_idx];
                        }
                    }
                }
                let row = 1 + py * n_side + px;
                let out_slice = &mut hidden[out_off + row * w..out_off + (row + 1) * w];
                // No conv bias in OpenCLIP.
                for d in 0..pd {
                    let v = patch_buf[d];
                    if v == 0.0 {
                        continue;
                    }
                    let w_row = &pre.conv1_w[d * w..(d + 1) * w];
                    for k in 0..w {
                        out_slice[k] += v * w_row[k];
                    }
                }
            }
        }

        // Add positional embedding (broadcast over batch).
        for i in 0..seq * w {
            hidden[out_off + i] += pre.positional_embedding[i];
        }
    }

    Ok(hidden)
}

/// Convert an RGB u8 image (HWC) of arbitrary size to a CLIP-normalized
/// NCHW f32 tensor at `img_size×img_size`, matching OpenCLIP / PIL
/// inference preprocessing: antialiased bicubic resize of the shortest
/// side to `img_size`, then a center crop.
///
/// Delegates to the shared, PIL-faithful resampler in
/// [`rlx_core::image_preprocess`] so every vision model gets identical
/// (and reusable) preprocessing. The PIL bicubic numerics are reproduced
/// in pure Rust — no Python dependency.
pub fn clip_normalize_nchw(rgb: &[u8], h_in: usize, w_in: usize, img_size: usize) -> Vec<f32> {
    rlx_core::image_preprocess::ImagePreprocessor::clip(img_size, CLIP_MEAN, CLIP_STD)
        .from_rgb(rgb, w_in, h_in)
}
