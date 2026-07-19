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

//! Host-side SigLIP 2 vision preprocessing (fixed-resolution family) and
//! the MAP-head weight split.
//!
//! The `patch_embedding` Conv2d (stride = patch, **with bias**) is applied
//! on the CPU via a single matmul over unfolded patches — rlx-ir has no
//! f32 forward Conv2d, and this keeps the graph a pure float pipeline
//! (matching the bioclip / dinov2 runners). SigLIP has **no CLS token**, so
//! the assembled tensor is `[B, num_patches, width]`.
//!
//! ```text
//!   image [B, 3, H, W] (SigLIP-normalized, NCHW f32)
//!     → unfold to [B, np, 3·ps·ps]
//!     → matmul patch_wᵀ + patch_b → [B, np, width]
//!     → add position_embedding → [B, np, width]
//! ```

use crate::config::{SIGLIP_MEAN, SIGLIP_STD, Siglip2Config};
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

/// Vision-embed weights extracted from the checkpoint (consumed on host).
pub struct VisionEmbedWeights {
    /// `patch_embedding.weight` `[width,3,ps,ps]` reshaped+transposed to
    /// `[3·ps·ps, width]` (row-major, sgemm-friendly).
    pub patch_w: Vec<f32>,
    /// `patch_embedding.bias` `[width]`.
    pub patch_b: Vec<f32>,
    /// `position_embedding.weight` flattened `[num_patches · width]`.
    pub pos_embed: Vec<f32>,
    pub width: usize,
    pub patch_dim: usize,
    pub num_patches: usize,
}

/// Pre-split MAP-head projections + tiled probe for the pooling attention.
/// The packed `nn.MultiheadAttention` `in_proj_weight` `[3W,W]` is split
/// host-side into transposed `[W,W]` q/k/v matrices (so `x @ M` computes
/// the projection directly) plus biases; the probe is tiled to the compiled
/// batch. Injected via `Emit::synth_param` — never narrowed in-graph.
#[derive(Clone)]
pub struct PoolingWeights {
    pub q_w: Vec<f32>,
    pub k_w: Vec<f32>,
    pub v_w: Vec<f32>,
    pub q_b: Vec<f32>,
    pub k_b: Vec<f32>,
    pub v_b: Vec<f32>,
    /// Probe tiled to `[batch · width]`.
    pub probe: Vec<f32>,
}

pub(crate) fn extract_vision_embed_weights(
    weights: &mut WeightMap,
    cfg: &Siglip2Config,
) -> Result<VisionEmbedWeights> {
    let v = &cfg.vision;
    let width = v.width;
    let patch_dim = v.patch_dim();
    let num_patches = v.num_patches();

    // Conv2d [width, 3, ps, ps] → flatten [width, patch_dim] → transpose
    // [patch_dim, width].
    let (conv_raw, conv_shape) = weights.take("vision_model.embeddings.patch_embedding.weight")?;
    ensure!(
        conv_shape.len() == 4
            && conv_shape[0] == width
            && conv_shape[1] * conv_shape[2] * conv_shape[3] == patch_dim,
        "patch_embedding.weight expected [width={width}, 3, ps, ps] (patch_dim={patch_dim}), got {conv_shape:?}"
    );
    let mut patch_w = vec![0f32; width * patch_dim];
    for e in 0..width {
        for d in 0..patch_dim {
            patch_w[d * width + e] = conv_raw[e * patch_dim + d];
        }
    }

    let (patch_b, _) = weights.take("vision_model.embeddings.patch_embedding.bias")?;
    ensure!(patch_b.len() == width, "patch_embedding.bias != width");

    let (pos_embed, pos_shape) =
        weights.take("vision_model.embeddings.position_embedding.weight")?;
    ensure!(
        pos_embed.len() == num_patches * width,
        "position_embedding length {} != num_patches*width ({num_patches}*{width}); shape={pos_shape:?}",
        pos_embed.len()
    );

    Ok(VisionEmbedWeights {
        patch_w,
        patch_b,
        pos_embed,
        width,
        patch_dim,
        num_patches,
    })
}

/// Split the packed MAP-head `in_proj_weight`/`in_proj_bias` and tile the probe.
pub(crate) fn extract_pooling_weights(
    weights: &mut WeightMap,
    cfg: &Siglip2Config,
    batch: usize,
) -> Result<PoolingWeights> {
    let w = cfg.vision.width;
    let (inw, in_shape) = weights.take("vision_model.head.attention.in_proj_weight")?;
    ensure!(
        in_shape == vec![3 * w, w],
        "head in_proj_weight shape {in_shape:?} != [{}, {w}]",
        3 * w
    );
    let (inb, _) = weights.take("vision_model.head.attention.in_proj_bias")?;
    ensure!(
        inb.len() == 3 * w,
        "head in_proj_bias len {} != 3W",
        inb.len()
    );

    // in_proj_weight row-major [3W, W]: block r∈[0,W) = Wq, [W,2W) = Wk,
    // [2W,3W) = Wv, each [out, in]. Transpose each block → [in, out] so
    // `x @ M` = `x @ Wᵀ`.
    let block = |off: usize| -> Vec<f32> {
        let mut m = vec![0f32; w * w];
        for o in 0..w {
            let src = (off + o) * w;
            for i in 0..w {
                m[i * w + o] = inw[src + i];
            }
        }
        m
    };
    let q_w = block(0);
    let k_w = block(w);
    let v_w = block(2 * w);
    let q_b = inb[0..w].to_vec();
    let k_b = inb[w..2 * w].to_vec();
    let v_b = inb[2 * w..3 * w].to_vec();

    let (probe, probe_shape) = weights.take("vision_model.head.probe")?;
    ensure!(
        probe.len() == w,
        "head.probe len {} != width {w}; shape={probe_shape:?}",
        probe.len()
    );
    let mut tiled = Vec::with_capacity(batch * w);
    for _ in 0..batch {
        tiled.extend_from_slice(&probe);
    }

    Ok(PoolingWeights {
        q_w,
        k_w,
        v_w,
        q_b,
        k_b,
        v_b,
        probe: tiled,
    })
}

/// Image → vision hidden tensor `[batch · num_patches · width]` (no CLS).
///
/// `image`: NCHW f32, length `batch·3·img·img`, already SigLIP-normalized.
pub fn assemble_vision_hidden(
    pre: &VisionEmbedWeights,
    image: &[f32],
    batch: usize,
    patch_size: usize,
    img_size: usize,
) -> Result<Vec<f32>> {
    let w = pre.width;
    let np = pre.num_patches;
    let pd = pre.patch_dim;
    let n_side = img_size / patch_size;

    ensure!(
        image.len() == batch * 3 * img_size * img_size,
        "image length {} != B·3·H·W ({batch}·3·{img_size}·{img_size})",
        image.len()
    );
    ensure!(np == n_side * n_side, "num_patches / img_size mismatch");

    let mut hidden = vec![0f32; batch * np * w];
    let mut patch_buf = vec![0f32; pd];
    for b in 0..batch {
        let img_off = b * 3 * img_size * img_size;
        let out_off = b * np * w;
        for py in 0..n_side {
            for px in 0..n_side {
                // CHW order to match Conv2d weight layout [width, C, ph, pw].
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
                let row = py * n_side + px;
                let out_slice = &mut hidden[out_off + row * w..out_off + (row + 1) * w];
                out_slice.copy_from_slice(&pre.patch_b);
                for d in 0..pd {
                    let val = patch_buf[d];
                    if val == 0.0 {
                        continue;
                    }
                    let w_row = &pre.patch_w[d * w..(d + 1) * w];
                    for k in 0..w {
                        out_slice[k] += val * w_row[k];
                    }
                }
                // Position embedding (row-indexed, no CLS offset).
                let pos = &pre.pos_embed[row * w..(row + 1) * w];
                for k in 0..w {
                    out_slice[k] += pos[k];
                }
            }
        }
    }
    Ok(hidden)
}

/// Convert an RGB8 (HWC) image of arbitrary size to a SigLIP-normalized
/// NCHW f32 tensor at `img_size×img_size`, matching `SiglipImageProcessor`:
/// **bilinear** resize to exactly `img×img` (no aspect preservation, no
/// crop), rescale `1/255`, normalize with mean = std = 0.5.
pub fn siglip_normalize_nchw(rgb: &[u8], h_in: usize, w_in: usize, img_size: usize) -> Vec<f32> {
    use rlx_core::image_preprocess::{Filter, ImagePreprocessor, ResizeMode};
    ImagePreprocessor {
        size: img_size,
        mean: SIGLIP_MEAN,
        std: SIGLIP_STD,
        filter: Filter::Bilinear,
        resize_mode: ResizeMode::Exact,
        center_crop: false,
    }
    .from_rgb(rgb, w_in, h_in)
}
