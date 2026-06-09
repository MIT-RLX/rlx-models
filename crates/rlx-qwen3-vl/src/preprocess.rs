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

//! Qwen3-VL image preprocessing.
//!
//! SigLIP normalization: per-channel mean/std = (0.5, 0.5, 0.5).
//! Resize-to-square at `cfg.image_size` (bicubic via the `image` crate)
//! then unfold into patches and run the patch-embed projection on the
//! host, matching the `rlx-dinov2` pattern (rlx-ir has no f32 forward
//! Conv2d yet, so we do the stride-`patch_size` convolution as a
//! matmul over unfolded patches).

use anyhow::{Result, anyhow, ensure};
use image::imageops::FilterType;
use rlx_core::weight_map::WeightMap;
use rlx_vlm_base::{ImagePatches, ImagePreprocessor};
use std::path::Path;

use super::config::Qwen3VlVisionConfig;

const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Host-side patch-embed weights (Conv2d → matmul reshape).
pub struct Qwen3VlPreprocessWeights {
    /// `[patch_dim, embed_dim]` — Conv2d `[E, 3, ps, ps]` flattened and
    /// transposed for row-major sgemm.
    pub proj_w: Vec<f32>,
    /// `[embed_dim]` — patch-embed bias.
    pub proj_b: Vec<f32>,
    /// `[seq, embed_dim]` — absolute positional embedding (flattened).
    pub pos_embed: Vec<f32>,
    pub embed_dim: usize,
    pub patch_dim: usize,
    pub num_patches: usize,
}

pub fn extract_preprocess_weights(
    weights: &mut WeightMap,
    cfg: &Qwen3VlVisionConfig,
) -> Result<Qwen3VlPreprocessWeights> {
    let embed_dim = cfg.hidden_size;
    let patch_dim = cfg.patch_dim();
    let num_patches = cfg.num_patches();

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
    let (pos_embed, pos_shape) = weights.take("pos_embed")?;
    ensure!(
        pos_embed.len() == num_patches * embed_dim,
        "pos_embed length {} does not match num_patches*E ({}*{}); shape={pos_shape:?}",
        pos_embed.len(),
        num_patches,
        embed_dim,
    );

    Ok(Qwen3VlPreprocessWeights {
        proj_w,
        proj_b,
        pos_embed,
        embed_dim,
        patch_dim,
        num_patches,
    })
}

/// Image → `[num_patches, patch_dim]` row-major patches, ready to be
/// matmul'd with `proj_w` and have `pos_embed` added.
pub fn image_to_patch_tensor(
    img: &image::DynamicImage,
    cfg: &Qwen3VlVisionConfig,
) -> Result<ImagePatches> {
    let target = cfg.image_size as u32;
    let resized = img.resize_exact(target, target, FilterType::CatmullRom);
    let rgb = resized.to_rgb8();
    let (w, h) = rgb.dimensions();
    let ps = cfg.patch_size as u32;
    ensure!(
        w % ps == 0 && h % ps == 0,
        "image_size {target} not divisible by patch_size {ps}"
    );

    let grid_h = (h / ps) as usize;
    let grid_w = (w / ps) as usize;
    let num_patches = grid_h * grid_w;
    let patch_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size;

    let mut patches = vec![0f32; num_patches * patch_dim];

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let row = gy * grid_w + gx;
            for py in 0..cfg.patch_size {
                for px in 0..cfg.patch_size {
                    let x = (gx * cfg.patch_size + px) as u32;
                    let y = (gy * cfg.patch_size + py) as u32;
                    let pix = rgb.get_pixel(x, y);
                    for c in 0..cfg.num_channels {
                        let raw = pix.0[c] as f32 / 255.0;
                        let v = (raw - SIGLIP_MEAN[c]) / SIGLIP_STD[c];
                        // CHW within each patch (matches Conv2d weight layout).
                        let inner = c * cfg.patch_size * cfg.patch_size + py * cfg.patch_size + px;
                        patches[row * patch_dim + inner] = v;
                    }
                }
            }
        }
    }

    Ok(ImagePatches {
        patches,
        grid_h,
        grid_w,
        patch_h: cfg.patch_size,
        patch_w: cfg.patch_size,
        channels: cfg.num_channels,
    })
}

/// Run the host-side Conv2d patch-embed + pos-embed add.
/// Output: `[1, num_patches, embed_dim]` row-major.
pub fn assemble_hidden(pp: &Qwen3VlPreprocessWeights, patches: &ImagePatches) -> Result<Vec<f32>> {
    ensure!(
        patches.num_patches() == pp.num_patches,
        "patch count mismatch: {} vs {}",
        patches.num_patches(),
        pp.num_patches
    );
    ensure!(
        patches.patch_dim() == pp.patch_dim,
        "patch_dim mismatch: {} vs {}",
        patches.patch_dim(),
        pp.patch_dim
    );

    let n = pp.num_patches;
    let e = pp.embed_dim;
    let d = pp.patch_dim;
    let mut out = vec![0f32; n * e];
    // out[n,e] = patches[n,d] @ proj_w[d,e] + proj_b[e] + pos_embed[n,e]
    for row in 0..n {
        for col in 0..e {
            let mut acc = pp.proj_b[col];
            for k in 0..d {
                acc += patches.patches[row * d + k] * pp.proj_w[k * e + col];
            }
            acc += pp.pos_embed[row * e + col];
            out[row * e + col] = acc;
        }
    }
    Ok(out)
}

/// `ImagePreprocessor` impl backed by `image::open` + bicubic resize.
pub struct Qwen3VlImagePreprocessor {
    pub cfg: Qwen3VlVisionConfig,
}

impl ImagePreprocessor for Qwen3VlImagePreprocessor {
    fn preprocess_path(&self, path: &Path) -> Result<ImagePatches> {
        let img = image::open(path).map_err(|e| anyhow!("rlx-qwen3-vl: open {path:?}: {e}"))?;
        image_to_patch_tensor(&img, &self.cfg)
    }
    fn preprocess_bytes(&self, bytes: &[u8]) -> Result<ImagePatches> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| anyhow!("rlx-qwen3-vl: decode image bytes: {e}"))?;
        image_to_patch_tensor(&img, &self.cfg)
    }
}
