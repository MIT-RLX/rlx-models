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

//! Host image preprocessing for Llama-4 vision (`Llama4UnfoldConvolution` +
//! class token + position embedding). Normalization is IMAGENET_STANDARD
//! (mean/std 0.5 → `pixel/127.5 - 1`), same as mllama.
//!
//! v1 uses a single global tile (resize the whole image to `image_size²`); the
//! processor's aspect-ratio tiling + global thumbnail are a follow-up.

use anyhow::{Result, ensure};
use image::{RgbImage, imageops::FilterType};
use rlx_core::weight_map::WeightMap;

use crate::config::Llama4VisionConfig;

/// Host-side vision stem weights (taken from the checkpoint before graph build).
pub struct Llama4VisionStem {
    pub hidden: usize,
    pub patch_dim: usize,
    pub grid: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub num_patches: usize,
    patch_embed_t: Vec<f32>, // [patch_dim, hidden]
    class_embed: Vec<f32>,   // [hidden]
    pos_embed: Vec<f32>,     // [num_patches, hidden]
}

fn transpose_2d(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

/// Extract and cache the vision stem (consumes its keys from `wm`).
pub fn extract_vision_stem(
    wm: &mut WeightMap,
    cfg: &Llama4VisionConfig,
) -> Result<Llama4VisionStem> {
    let hidden = cfg.hidden_size;
    let ps = cfg.patch_size;
    let patch_dim = cfg.num_channels * ps * ps;
    let grid = cfg.image_size / ps;
    let np = cfg.num_patches();

    let (pe, _) = wm.take("vision_model.patch_embedding.linear.weight")?;
    ensure!(
        pe.len() == hidden * patch_dim,
        "patch_embedding.linear.weight len {}",
        pe.len()
    );
    let patch_embed_t = transpose_2d(&pe, hidden, patch_dim);
    let (class_embed, _) = wm.take("vision_model.class_embedding")?;
    let (pos_embed, _) = wm.take("vision_model.positional_embedding_vlm")?;

    Ok(Llama4VisionStem {
        hidden,
        patch_dim,
        grid,
        patch_size: ps,
        image_size: cfg.image_size,
        num_patches: np,
        patch_embed_t,
        class_embed,
        pos_embed,
    })
}

impl Llama4VisionStem {
    /// Preprocess one RGB image (HWC `u8`) → `hidden [num_patches * hidden]`
    /// (patches + class token at the end + position embedding).
    pub fn preprocess(&self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<f32>> {
        ensure!(rgb.len() == w * h * 3, "rgb len {} != w*h*3", rgb.len());
        let hidden = self.hidden;
        let ps = self.patch_size;
        let grid = self.grid;
        let np = self.num_patches;
        let sz = self.image_size;

        let src = RgbImage::from_raw(w as u32, h as u32, rgb.to_vec())
            .ok_or_else(|| anyhow::anyhow!("bad rgb buffer"))?;
        let img = image::imageops::resize(&src, sz as u32, sz as u32, FilterType::Triangle);

        let mut hidden_seq = vec![0.0f32; np * hidden];
        let mut patch_vec = vec![0.0f32; self.patch_dim];
        for py in 0..grid {
            for px in 0..grid {
                let patch = py * grid + px;
                // channel-major [c, ky, kx] to match nn.Unfold ordering.
                for c in 0..3 {
                    for ky in 0..ps {
                        for kx in 0..ps {
                            let p = img.get_pixel((px * ps + kx) as u32, (py * ps + ky) as u32);
                            patch_vec[c * ps * ps + ky * ps + kx] = p[c] as f32 / 127.5 - 1.0;
                        }
                    }
                }
                let dst = patch * hidden;
                for o in 0..hidden {
                    let mut acc = 0.0f32;
                    for k in 0..self.patch_dim {
                        acc += patch_vec[k] * self.patch_embed_t[k * hidden + o];
                    }
                    hidden_seq[dst + o] = acc;
                }
            }
        }
        // Class token is appended AFTER the patches (position num_patches-1).
        let cls = (np - 1) * hidden;
        hidden_seq[cls..cls + hidden].copy_from_slice(&self.class_embed[..hidden]);
        // Learned position embedding.
        for i in 0..np * hidden {
            hidden_seq[i] += self.pos_embed[i];
        }
        Ok(hidden_seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn preprocess_shapes() {
        let cfg: Llama4VisionConfig = serde_json::from_str(
            r#"{"hidden_size":8,"num_attention_heads":2,"image_size":28,"patch_size":14}"#,
        )
        .unwrap();
        let np = cfg.num_patches(); // 5
        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let pd = 3 * 14 * 14;
        t.insert(
            "vision_model.patch_embedding.linear.weight".into(),
            (vec![0.01; 8 * pd], vec![8, pd]),
        );
        t.insert(
            "vision_model.class_embedding".into(),
            (vec![0.1; 8], vec![8]),
        );
        t.insert(
            "vision_model.positional_embedding_vlm".into(),
            (vec![0.0; np * 8], vec![np, 8]),
        );
        let mut wm = WeightMap::from_tensors(t);
        let stem = extract_vision_stem(&mut wm, &cfg).unwrap();
        let rgb = vec![128u8; 56 * 56 * 3];
        let hidden = stem.preprocess(&rgb, 56, 56).unwrap();
        assert_eq!(hidden.len(), np * 8);
        assert!(hidden.iter().all(|v| v.is_finite()));
    }
}
