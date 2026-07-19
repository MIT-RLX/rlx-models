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

//! Host-side DINOv3 preprocessing: patch projection + token assembly.
//!
//! rlx-ir has no f32 forward Conv2d today, so the stride-`patch_size`
//! Conv2d patch embedding is done on the CPU as one matmul over unfolded
//! patches (identical to DINOv2/UNI2). The output is the `"hidden"`
//! tensor `[B, seq, E]` consumed by the encoder graph.
//!
//! ## Pipeline (differs from DINOv2: **no `pos_embed`**)
//! ```text
//!   image [B, 3, H, W] (ImageNet-normalized, NCHW f32)
//!     → unfold to [B, np, 3·ps·ps]
//!     → matmul proj_w + proj_b            → [B, np, E]
//!     → prepend CLS + register_tokens     → [B, seq, E]
//! ```
//! Spatial position is added later by 2-D axial RoPE *inside attention*,
//! so there is no position tensor to assemble here.

use super::config::DinoV3Config;
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

/// Preprocess weights extracted from the safetensors checkpoint (HF keys).
pub struct DinoV3PreprocessWeights {
    /// Patch projection: Conv2d `[E, 3, ps, ps]` → `[3·ps·ps, E]`
    /// (row-major, sgemm-friendly).
    pub proj_w: Vec<f32>,
    /// Patch projection bias `[E]`.
    pub proj_b: Vec<f32>,
    /// CLS token, flattened `[1,1,E]` → `[E]`.
    pub cls_token: Vec<f32>,
    /// Register tokens, flattened `[1,reg,E]` → `[reg·E]`. Empty if none.
    pub register_tokens: Vec<f32>,
    /// Encoder width `E`.
    pub embed_dim: usize,
    /// Flattened patch length (`channels · patch_size²`).
    pub patch_dim: usize,
    /// Number of patch tokens.
    pub num_patches: usize,
    /// Number of register tokens.
    pub num_register_tokens: usize,
    /// Full sequence length (`1 + reg + num_patches`).
    pub seq: usize,
}

pub(super) fn extract_preprocess_weights(
    weights: &mut WeightMap,
    cfg: &DinoV3Config,
) -> Result<DinoV3PreprocessWeights> {
    let embed_dim = cfg.hidden_size;
    let patch_dim = cfg.patch_dim();
    let num_patches = cfg.num_patches();
    let seq = cfg.seq_len();
    let c = cfg.num_channels;
    let ps = cfg.patch_size;

    // Conv2d [E, C, ps, ps] → flatten to [E, patch_dim] → transpose [patch_dim, E].
    let (proj_raw, proj_shape) = weights.take("embeddings.patch_embeddings.weight")?;
    ensure!(
        proj_shape.len() == 4
            && proj_shape[0] == embed_dim
            && proj_shape[1] == c
            && proj_shape[2] == ps
            && proj_shape[3] == ps,
        "embeddings.patch_embeddings.weight expected [E={embed_dim}, {c}, {ps}, {ps}], got {proj_shape:?}"
    );
    let mut proj_w = vec![0f32; embed_dim * patch_dim];
    for e in 0..embed_dim {
        for d in 0..patch_dim {
            proj_w[d * embed_dim + e] = proj_raw[e * patch_dim + d];
        }
    }

    let (proj_b, _) = weights.take("embeddings.patch_embeddings.bias")?;
    let (cls_token, _) = weights.take("embeddings.cls_token")?;
    ensure!(
        cls_token.len() == embed_dim,
        "embeddings.cls_token expected {embed_dim} elems, got {}",
        cls_token.len()
    );

    let register_tokens = if cfg.num_register_tokens > 0 {
        let (data, shape) = weights.take("embeddings.register_tokens")?;
        ensure!(
            shape.len() == 3 && shape[1] == cfg.num_register_tokens && shape[2] == embed_dim,
            "embeddings.register_tokens expected [1, {n}, {embed_dim}], got {shape:?}",
            n = cfg.num_register_tokens
        );
        data
    } else {
        Vec::new()
    };

    // `mask_token` is a pretraining-only parameter; discard if present so
    // the checkpoint is fully consumed.
    let _ = weights.take("embeddings.mask_token");

    Ok(DinoV3PreprocessWeights {
        proj_w,
        proj_b,
        cls_token,
        register_tokens,
        embed_dim,
        patch_dim,
        num_patches,
        num_register_tokens: cfg.num_register_tokens,
        seq,
    })
}

/// Image → hidden tensor for the encoder graph.
///
/// `image`: NCHW float32, length `batch · C · img_size · img_size`,
///   pre-normalized with ImageNet mean/std.
///
/// Returns `[batch · seq · embed_dim]` flat row-major (`[CLS, reg…, patches]`).
pub fn assemble_hidden(
    pre: &DinoV3PreprocessWeights,
    image: &[f32],
    batch: usize,
    patch_size: usize,
    img_size: usize,
) -> Result<Vec<f32>> {
    let e = pre.embed_dim;
    let np = pre.num_patches;
    let seq = pre.seq;
    let pd = pre.patch_dim;
    let channels = pd / (patch_size * patch_size);
    let n_side = img_size / patch_size;

    ensure!(
        image.len() == batch * channels * img_size * img_size,
        "image length {} != B·C·H·W ({}·{}·{}·{})",
        image.len(),
        batch,
        channels,
        img_size,
        img_size
    );
    ensure!(
        np == n_side * n_side,
        "num_patches mismatch — img_size/patch_size inconsistent"
    );

    let mut hidden = vec![0f32; batch * seq * e];

    for b in 0..batch {
        let img_off = b * channels * img_size * img_size;
        let out_off = b * seq * e;

        // 1) CLS token → row 0.
        hidden[out_off..out_off + e].copy_from_slice(&pre.cls_token);

        // 2) Register tokens → rows 1..1+n_reg.
        if pre.num_register_tokens > 0 {
            let dst = &mut hidden[out_off + e..out_off + e * (1 + pre.num_register_tokens)];
            dst.copy_from_slice(&pre.register_tokens);
        }

        // 3) Patch tokens: unfold (CHW order to match the flattened
        //    Conv2d weight) + project. Patch (py,px) → row 1+n_reg+py*n_side+px.
        let patch_row_base = 1 + pre.num_register_tokens;
        let mut patch_buf = vec![0f32; pd];
        for py in 0..n_side {
            for px in 0..n_side {
                for ch in 0..channels {
                    for ry in 0..patch_size {
                        let src_y = py * patch_size + ry;
                        for rx in 0..patch_size {
                            let src_x = px * patch_size + rx;
                            let src_idx =
                                img_off + ch * img_size * img_size + src_y * img_size + src_x;
                            let dst_idx = ch * patch_size * patch_size + ry * patch_size + rx;
                            patch_buf[dst_idx] = image[src_idx];
                        }
                    }
                }
                let row = patch_row_base + py * n_side + px;
                let out_slice = &mut hidden[out_off + row * e..out_off + (row + 1) * e];
                out_slice.copy_from_slice(&pre.proj_b);
                for d in 0..pd {
                    let v = patch_buf[d];
                    if v == 0.0 {
                        continue;
                    }
                    let w_row = &pre.proj_w[d * e..(d + 1) * e];
                    for k in 0..e {
                        out_slice[k] += v * w_row[k];
                    }
                }
            }
        }
    }

    Ok(hidden)
}

/// RGB u8 (HWC, arbitrary size) → normalized NCHW f32 at
/// `(img_size, img_size)` via bilinear resize + ImageNet stats.
///
/// Note: HF's DINOv3 image processor uses bicubic resize; at the native
/// checkpoint resolution (no upscaling) the two agree closely. For rigorous
/// numeric parity, feed raw `pixel_values` through the runner's
/// `forward_nchw` path instead of an image.
pub fn rgb_u8_to_imagenet_nchw(rgb: &[u8], h_in: usize, w_in: usize, img_size: usize) -> Vec<f32> {
    use super::config::{IMAGENET_MEAN, IMAGENET_STD};
    let mut out = vec![0f32; 3 * img_size * img_size];
    let sx = (w_in as f32 - 1.0) / (img_size.max(1) as f32 - 1.0).max(1.0);
    let sy = (h_in as f32 - 1.0) / (img_size.max(1) as f32 - 1.0).max(1.0);
    for y in 0..img_size {
        let fy = y as f32 * sy;
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(h_in - 1);
        let dy = fy - y0 as f32;
        for x in 0..img_size {
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
                let v = (top * (1.0 - dy) + bot * dy) / 255.0;
                out[c * img_size * img_size + y * img_size + x] =
                    (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
    out
}
