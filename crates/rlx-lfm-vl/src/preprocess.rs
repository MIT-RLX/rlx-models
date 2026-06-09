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

//! LFM2.5-VL image preprocessing. SigLIP normalization
//! (mean/std = 0.5) + host-side patch embedding + absolute pos-embed.
//!
//! Weight names follow the HuggingFace LFM2-VL layout:
//! `vision_tower.vision_model.embeddings.{patch_embedding,position_embedding}.*`.

use anyhow::{Result, anyhow, ensure};
use image::imageops::FilterType;
use rlx_core::weight_map::WeightMap;
use rlx_vlm_base::{ImagePatches, ImagePreprocessor};
use std::path::Path;

use super::config::LfmVlVisionConfig;

pub const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];

pub struct LfmVlPreprocessWeights {
    pub proj_w: Vec<f32>,
    pub proj_b: Vec<f32>,
    pub pos_embed: Vec<f32>,
    pub embed_dim: usize,
    pub patch_dim: usize,
    pub num_patches: usize,
}

const PATCH_EMBED_W: &str = "vision_tower.vision_model.embeddings.patch_embedding.weight";
const PATCH_EMBED_B: &str = "vision_tower.vision_model.embeddings.patch_embedding.bias";
const POS_EMBED: &str = "vision_tower.vision_model.embeddings.position_embedding.weight";

pub fn extract_preprocess_weights(
    weights: &mut WeightMap,
    cfg: &LfmVlVisionConfig,
) -> Result<LfmVlPreprocessWeights> {
    let embed_dim = cfg.hidden_size;
    let patch_dim = cfg.patch_dim();
    let num_patches = cfg.num_patches();

    let (proj_raw, proj_shape) = weights.take(PATCH_EMBED_W)?;
    ensure!(
        proj_shape.len() == 4
            && proj_shape[0] == embed_dim
            && proj_shape[1] * proj_shape[2] * proj_shape[3] == patch_dim,
        "{PATCH_EMBED_W} expected [E={embed_dim}, 3, ps, ps] (patch_dim={patch_dim}), got {proj_shape:?}"
    );
    let mut proj_w = vec![0f32; embed_dim * patch_dim];
    for e in 0..embed_dim {
        for d in 0..patch_dim {
            proj_w[d * embed_dim + e] = proj_raw[e * patch_dim + d];
        }
    }

    let (proj_b, _) = weights.take(PATCH_EMBED_B)?;
    let (pos_embed, pos_shape) = weights.take(POS_EMBED)?;
    ensure!(
        pos_embed.len() == num_patches * embed_dim,
        "{POS_EMBED} length {} does not match num_patches*E ({}*{}); shape={pos_shape:?}",
        pos_embed.len(),
        num_patches,
        embed_dim
    );

    Ok(LfmVlPreprocessWeights {
        proj_w,
        proj_b,
        pos_embed,
        embed_dim,
        patch_dim,
        num_patches,
    })
}

pub fn image_to_patch_tensor(
    img: &image::DynamicImage,
    cfg: &LfmVlVisionConfig,
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

pub fn assemble_hidden(pp: &LfmVlPreprocessWeights, patches: &ImagePatches) -> Result<Vec<f32>> {
    ensure!(patches.num_patches() == pp.num_patches);
    ensure!(patches.patch_dim() == pp.patch_dim);
    let (n, e, d) = (pp.num_patches, pp.embed_dim, pp.patch_dim);
    let mut out = vec![0f32; n * e];
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

pub struct LfmVlImagePreprocessor {
    pub cfg: LfmVlVisionConfig,
}
impl ImagePreprocessor for LfmVlImagePreprocessor {
    fn preprocess_path(&self, path: &Path) -> Result<ImagePatches> {
        let img = image::open(path).map_err(|e| anyhow!("rlx-lfm-vl: open {path:?}: {e}"))?;
        image_to_patch_tensor(&img, &self.cfg)
    }
    fn preprocess_bytes(&self, bytes: &[u8]) -> Result<ImagePatches> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| anyhow!("rlx-lfm-vl: decode image bytes: {e}"))?;
        image_to_patch_tensor(&img, &self.cfg)
    }
}
