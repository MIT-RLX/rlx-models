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

//! Host-side UNI2-h preprocessing: patch projection + token assembly.
//!
//! rlx-ir has no f32 forward Conv2d today, so the patch embedding (a
//! stride-`patch_size` Conv2d in PyTorch) runs on the CPU as a single
//! matmul over unfolded patches. The output is the `"hidden"` tensor
//! consumed by the IR graph built by [`crate::flow`].
//!
//! ## `no_embed_class` layout
//! UNI2-h is created with `no_embed_class=True`, so the position
//! embedding covers **only the patch tokens** — the `[CLS]` and the 8
//! register tokens receive none. timm's `_pos_embed` is therefore:
//!
//! ```text
//!   patches = patches + pos_embed          # pos on patches only
//!   x       = cat([cls, reg×8, patches])   # prefix tokens have no pos
//! ```
//!
//! ```text
//!   image [B, 3, H, W] (ImageNet-normalized, NCHW f32)
//!     → unfold to [B, np, 3·ps·ps]
//!     → matmul proj_w + proj_b            → [B, np, E]
//!     → add pos_embed (patch rows only)   → [B, np, E]
//!     → prepend [CLS] + register_tokens   → [B, seq, E]
//! ```

use super::config::Uni2Config;
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

/// Preprocess weights extracted from the checkpoint, held alongside the
/// compiled graph so [`assemble_hidden`] can run before each forward.
pub struct Uni2PreprocessWeights {
    /// Patch projection: Conv2d `[E, 3, ps, ps]` reshaped + transposed to
    /// `[3·ps·ps, E]` (row-major sgemm-friendly).
    pub proj_w: Vec<f32>,
    /// Patch projection bias `[E]`.
    pub proj_b: Vec<f32>,
    /// `[CLS]` token, flattened from `[1, 1, E]` to `[E]`.
    pub cls_token: Vec<f32>,
    /// Register tokens, flattened to `[num_register_tokens · E]`.
    pub register_tokens: Vec<f32>,
    /// Patch position embeddings, flattened from `[1, np, E]` to `[np · E]`.
    pub pos_embed: Vec<f32>,
    pub embed_dim: usize,
    pub patch_dim: usize,
    pub num_patches: usize,
    pub num_register_tokens: usize,
    pub seq: usize,
}

pub(super) fn extract_preprocess_weights(
    weights: &mut WeightMap,
    cfg: &Uni2Config,
) -> Result<Uni2PreprocessWeights> {
    let embed_dim = cfg.hidden_size;
    let patch_dim = cfg.patch_dim();
    let num_patches = cfg.num_patches();
    let seq = cfg.seq_len();

    // Conv2d [E, 3, ps, ps] → flatten to [E, patch_dim] → transpose to [patch_dim, E].
    let (proj_raw, proj_shape) = weights.take("patch_embed.proj.weight")?;
    ensure!(
        proj_shape.len() == 4
            && proj_shape[0] == embed_dim
            && proj_shape[1] * proj_shape[2] * proj_shape[3] == patch_dim,
        "patch_embed.proj.weight expected [E={embed_dim}, 3, ps, ps] (patch_dim={patch_dim}), got {proj_shape:?}"
    );
    let mut proj_w = vec![0f32; embed_dim * patch_dim];
    for e in 0..embed_dim {
        for d in 0..patch_dim {
            proj_w[d * embed_dim + e] = proj_raw[e * patch_dim + d];
        }
    }

    let (proj_b, _) = weights.take("patch_embed.proj.bias")?;
    let (cls_token, _) = weights.take("cls_token")?;

    // `no_embed_class=True`: pos_embed spans the patch tokens only.
    let (pos_embed, pos_shape) = weights.take("pos_embed")?;
    ensure!(
        pos_embed.len() == num_patches * embed_dim,
        "pos_embed length {} does not match num_patches*E ({}*{}) — UNI2-h uses \
         no_embed_class (pos on patches only); interpolation to other resolutions \
         is not yet supported, so use cfg.img_size matching the checkpoint (224). \
         shape={pos_shape:?}",
        pos_embed.len(),
        num_patches,
        embed_dim
    );

    // timm names the register parameter `reg_token`; accept the Meta-style
    // `register_tokens` too for converted checkpoints.
    let register_tokens = if cfg.num_register_tokens > 0 {
        let key = if weights.has("reg_token") {
            "reg_token"
        } else {
            "register_tokens"
        };
        let (data, shape) = weights.take(key)?;
        ensure!(
            shape.len() == 3 && shape[1] == cfg.num_register_tokens && shape[2] == embed_dim,
            "{key} expected [1, {n}, {embed_dim}], got {shape:?}",
            n = cfg.num_register_tokens
        );
        data
    } else {
        Vec::new()
    };

    Ok(Uni2PreprocessWeights {
        proj_w,
        proj_b,
        cls_token,
        register_tokens,
        pos_embed,
        embed_dim,
        patch_dim,
        num_patches,
        num_register_tokens: cfg.num_register_tokens,
        seq,
    })
}

/// Image → hidden tensor for the encoder graph.
///
/// `image`: NCHW float32, length `batch · 3 · img_size · img_size`,
///   pre-normalized with ImageNet mean/std (see
///   [`super::config::IMAGENET_MEAN`] / `IMAGENET_STD`).
///
/// Returns `[batch · seq · embed_dim]` flat row-major, laid out as
/// `[CLS, register_tokens…, patches]` per batch.
pub fn assemble_hidden(
    pre: &Uni2PreprocessWeights,
    image: &[f32],
    batch: usize,
    patch_size: usize,
    img_size: usize,
) -> Result<Vec<f32>> {
    let e = pre.embed_dim;
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

    let mut hidden = vec![0f32; batch * seq * e];

    for b in 0..batch {
        let img_off = b * 3 * img_size * img_size;
        let out_off = b * seq * e;

        // 1) [CLS] token — row 0 (no position embedding under no_embed_class)
        hidden[out_off..out_off + e].copy_from_slice(&pre.cls_token);

        // 2) Register tokens — rows 1..1+n_reg (no position embedding)
        if pre.num_register_tokens > 0 {
            let dst = &mut hidden[out_off + e..out_off + e * (1 + pre.num_register_tokens)];
            dst.copy_from_slice(&pre.register_tokens);
        }

        // 3) Patch tokens — unfold + project + add patch position embedding.
        // Patch (py, px) → encoder row 1 + n_reg + py*n_side + px.
        let patch_row_base = 1 + pre.num_register_tokens;
        let mut patch_buf = vec![0f32; pd];
        for py in 0..n_side {
            for px in 0..n_side {
                // CHW order (c, row, col) to match the flattened Conv2d weight
                // layout `[E, C=3, ph, pw]`.
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
                let p_idx = py * n_side + px;
                let row = patch_row_base + p_idx;
                let out_slice = &mut hidden[out_off + row * e..out_off + (row + 1) * e];
                out_slice.copy_from_slice(&pre.proj_b);
                // proj_w layout: [patch_dim, embed_dim] row-major.
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
                // Patch position embedding (pos_embed is patch-only here).
                let pos_row = &pre.pos_embed[p_idx * e..(p_idx + 1) * e];
                for k in 0..e {
                    out_slice[k] += pos_row[k];
                }
            }
        }
    }

    Ok(hidden)
}

/// RGB u8 (HWC) at arbitrary resolution → normalized NCHW f32 at
/// `(img_size, img_size)` via bilinear resize + ImageNet stats.
///
/// `rgb` must be `H_in · W_in · 3` row-major (u8). Note: the reference
/// pipeline resizes with bicubic interpolation; pathology tiles are
/// typically supplied at native 224×224, where the resize is a no-op.
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
                let norm = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                out[c * img_size * img_size + y * img_size + x] = norm;
            }
        }
    }
    out
}
