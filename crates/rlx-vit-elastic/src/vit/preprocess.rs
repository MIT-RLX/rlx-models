// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Host-side patch projection + token assembly, generalized over the two pos
//! embedding layouts:
//!   - `no_embed_class = false` (plain ViT / DINO): `pos_embed` covers
//!     `[CLS] + patches` (`[1, 1+np, E]`), added to CLS row 0 and patch rows.
//!   - `no_embed_class = true` (UNI2): `pos_embed` covers patches only
//!     (`[1, np, E]`), and 8 register tokens sit between CLS and patches with
//!     no position embedding.
//!
//! rlx-ir has no f32 forward Conv2d, so the stride-`patch_size` patch
//! embedding runs on the host as an unfolded matmul (identical to
//! `rlx-uni2`). The resize/normalize step reuses [`rlx_uni2::rgb_u8_to_imagenet_nchw`].

use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

use super::config::VitConfig;

pub use rlx_uni2::rgb_u8_to_imagenet_nchw;

/// Preprocessing weights extracted from a checkpoint, held alongside the
/// compiled graph so [`assemble_hidden`] can run before each forward.
#[derive(Debug, Clone)]
pub struct PreprocessWeights {
    /// Patch projection Conv2d `[E,3,ps,ps]` reshaped+transposed to `[patch_dim, E]`.
    pub proj_w: Vec<f32>,
    /// Patch projection bias `[E]`.
    pub proj_b: Vec<f32>,
    /// `[CLS]` token `[E]`.
    pub cls_token: Vec<f32>,
    /// Register tokens `[n_reg · E]` (empty when `num_register_tokens == 0`).
    pub register_tokens: Vec<f32>,
    /// Position embeddings, flattened. Length `P · E` where `P = num_patches`
    /// (`no_embed_class`) or `1 + num_patches` (CLS + patches).
    pub pos_embed: Vec<f32>,
    pub embed_dim: usize,
    pub patch_dim: usize,
    pub num_patches: usize,
    pub num_register_tokens: usize,
    pub seq: usize,
    pub no_embed_class: bool,
    pub patch_size: usize,
    pub img_size: usize,
}

/// Extract preprocessing weights from a canonicalized checkpoint. Consumes
/// `patch_embed.proj.{weight,bias}`, `cls_token`, `pos_embed`, and (when
/// present) `reg_token` / `register_tokens` from `weights`.
pub fn extract_preprocess_weights(
    weights: &mut WeightMap,
    cfg: &VitConfig,
) -> Result<PreprocessWeights> {
    let embed_dim = cfg.hidden_size;
    let patch_dim = cfg.patch_dim();
    let num_patches = cfg.num_patches();
    let seq = cfg.seq_len();

    // Conv2d [E, 3, ps, ps] → flatten [E, patch_dim] → transpose [patch_dim, E].
    let (proj_raw, proj_shape) = weights.take("patch_embed.proj.weight")?;
    ensure!(
        proj_shape.len() == 4
            && proj_shape[0] == embed_dim
            && proj_shape[1] * proj_shape[2] * proj_shape[3] == patch_dim,
        "patch_embed.proj.weight expected [E={embed_dim},3,ps,ps] (patch_dim={patch_dim}), got {proj_shape:?}"
    );
    let mut proj_w = vec![0f32; embed_dim * patch_dim];
    for e in 0..embed_dim {
        for d in 0..patch_dim {
            proj_w[d * embed_dim + e] = proj_raw[e * patch_dim + d];
        }
    }
    let (proj_b, _) = weights.take("patch_embed.proj.bias")?;
    let (cls_token, _) = weights.take("cls_token")?;

    // Position embedding: [1, P, E] with P = np (no_embed_class) or 1+np.
    let (pos_embed, pos_shape) = weights.take("pos_embed")?;
    let expect_p = if cfg.no_embed_class {
        num_patches
    } else {
        1 + num_patches
    };
    ensure!(
        pos_embed.len() == expect_p * embed_dim,
        "pos_embed length {} != P*E ({expect_p}*{embed_dim}); no_embed_class={}, shape={pos_shape:?}",
        pos_embed.len(),
        cfg.no_embed_class
    );

    let register_tokens = if cfg.num_register_tokens > 0 {
        let key = if weights.has("reg_token") {
            "reg_token"
        } else {
            "register_tokens"
        };
        let (data, shape) = weights.take(key)?;
        ensure!(
            data.len() == cfg.num_register_tokens * embed_dim,
            "{key} expected {n}*{embed_dim} values, got {} (shape {shape:?})",
            data.len(),
            n = cfg.num_register_tokens
        );
        data
    } else {
        Vec::new()
    };

    Ok(PreprocessWeights {
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
        no_embed_class: cfg.no_embed_class,
        patch_size: cfg.patch_size,
        img_size: cfg.img_size,
    })
}

/// Image (ImageNet-normalized NCHW f32, `batch·3·img·img`) → `"hidden"` tensor
/// `[batch·seq·E]` laid out `[CLS, register…, patches]` per batch.
pub fn assemble_hidden(pre: &PreprocessWeights, image: &[f32], batch: usize) -> Result<Vec<f32>> {
    let e = pre.embed_dim;
    let np = pre.num_patches;
    let seq = pre.seq;
    let pd = pre.patch_dim;
    let ps = pre.patch_size;
    let img = pre.img_size;
    let n_side = img / ps;

    ensure!(
        image.len() == batch * 3 * img * img,
        "image length {} != B·3·H·W ({batch}·3·{img}·{img})",
        image.len()
    );
    ensure!(
        np == n_side * n_side,
        "num_patches mismatch — img_size/patch_size inconsistent"
    );

    // Position-embedding row offset (in tokens) of the first patch.
    let pos_patch_base = if pre.no_embed_class { 0 } else { 1 };
    let patch_row_base = 1 + pre.num_register_tokens;

    let mut hidden = vec![0f32; batch * seq * e];
    for b in 0..batch {
        let img_off = b * 3 * img * img;
        let out_off = b * seq * e;

        // [CLS] — row 0. Under plain-ViT layout it also gets pos_embed row 0.
        let cls = &mut hidden[out_off..out_off + e];
        cls.copy_from_slice(&pre.cls_token);
        if !pre.no_embed_class {
            for k in 0..e {
                cls[k] += pre.pos_embed[k];
            }
        }

        // Register tokens — no position embedding (UNI2).
        if pre.num_register_tokens > 0 {
            let dst = &mut hidden[out_off + e..out_off + e * (1 + pre.num_register_tokens)];
            dst.copy_from_slice(&pre.register_tokens);
        }

        // Patch tokens — unfold + project + add patch position embedding.
        let mut patch_buf = vec![0f32; pd];
        for py in 0..n_side {
            for px in 0..n_side {
                for c in 0..3 {
                    for ry in 0..ps {
                        let src_y = py * ps + ry;
                        for rx in 0..ps {
                            let src_x = px * ps + rx;
                            let src_idx = img_off + c * img * img + src_y * img + src_x;
                            let dst_idx = c * ps * ps + ry * ps + rx;
                            patch_buf[dst_idx] = image[src_idx];
                        }
                    }
                }
                let p_idx = py * n_side + px;
                let row = patch_row_base + p_idx;
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
                let pos_row =
                    &pre.pos_embed[(pos_patch_base + p_idx) * e..(pos_patch_base + p_idx + 1) * e];
                for k in 0..e {
                    out_slice[k] += pos_row[k];
                }
            }
        }
    }
    Ok(hidden)
}
