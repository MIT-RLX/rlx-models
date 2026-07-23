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

//! DeepEncoder — SAM-ViT-B (local detail) + CLIP-L/14-224 (global semantics)
//! feature concatenation, feeding the `2048 → 1280` [`crate::projector`], and
//! the per-view token assembly (mosaic + newlines + view separator) that
//! `modeling_unlimitedocr.py` uses to lay projected patch tokens into the LM
//! input sequence.

use crate::clip_tower::ClipTower;
use crate::config::UnlimitedOcrConfig;
use crate::preprocess::PreprocessedImage;
use crate::projector::Projector;
use crate::sam_tower::SamTower;
use crate::weights::UnlimitedOcrWeightStore;
use anyhow::{Result, ensure};

/// Combined vision tower: SAM + CLIP branches, concatenated per patch.
pub struct DeepEncoder {
    pub sam: SamTower,
    pub clip: ClipTower,
}

impl DeepEncoder {
    pub fn from_config(config: &UnlimitedOcrConfig) -> Self {
        Self {
            sam: SamTower::from_config(&config.vision_config.sam),
            clip: ClipTower::from_config(&config.vision_config.clip),
        }
    }

    pub fn load(&mut self, store: &UnlimitedOcrWeightStore) -> Result<()> {
        self.sam.load(store)?;
        self.clip.load(store)?;
        Ok(())
    }

    /// Encode one `side x side` view (CHW pixels): SAM → CLIP (fed SAM's
    /// features as patch tokens) → concat `[clip[1:], sam]` → project.
    /// Returns `(projected [q*q, hidden], q)`.
    fn encode_view(
        &self,
        projector: &Projector,
        pixels: &[f32],
        side: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let sam_feat = self.sam.encode(pixels, side)?; // [1024, q, q] NCHW
        let n_patches = sam_feat.len() / 1024;
        let q = (n_patches as f64).sqrt().round() as usize;
        ensure!(
            q * q == n_patches,
            "deep_encoder: SAM output {n_patches} not a perfect square"
        );

        let clip_out = self.clip.encode(pixels, &sam_feat)?; // [(1+q*q)*1024]
        let clip_hidden = self.clip.config.hidden_size;
        let clip_patches = &clip_out[clip_hidden..]; // drop cls token -> [q*q, clip_hidden]

        // sam_feat is NCHW [1024, q, q]; transpose to token-major [q*q, 1024]
        // to match clip_patches' row-major token order.
        let mut sam_tokens = vec![0f32; n_patches * 1024];
        for c in 0..1024 {
            for p in 0..n_patches {
                sam_tokens[p * 1024 + c] = sam_feat[c * n_patches + p];
            }
        }

        let in_features = projector.in_features;
        ensure!(
            in_features == clip_hidden + 1024,
            "deep_encoder: projector in_features {in_features} != clip({clip_hidden}) + sam(1024)"
        );
        let mut fused = vec![0f32; n_patches * in_features];
        for p in 0..n_patches {
            let dst = &mut fused[p * in_features..(p + 1) * in_features];
            dst[..clip_hidden]
                .copy_from_slice(&clip_patches[p * clip_hidden..(p + 1) * clip_hidden]);
            dst[clip_hidden..].copy_from_slice(&sam_tokens[p * 1024..(p + 1) * 1024]);
        }

        let projected = projector.forward(&fused, n_patches)?;
        Ok((projected, q))
    }

    /// Append a learned newline embedding after each `cols`-wide row of a
    /// `[rows*cols, dim]` token grid.
    fn insert_newlines(
        tokens: &[f32],
        rows: usize,
        cols: usize,
        dim: usize,
        newline: &[f32],
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(rows * (cols + 1) * dim);
        for r in 0..rows {
            out.extend_from_slice(&tokens[r * cols * dim..(r + 1) * cols * dim]);
            out.extend_from_slice(newline);
        }
        out
    }

    /// Tile `cw x ch` per-tile token grids (`[q*q, dim]` each, tile order
    /// row-major matching [`PreprocessedImage::spatial_crop`]) into one big
    /// `[(q*ch) * (q*cw), dim]` mosaic.
    fn mosaic(tiles: &[Vec<f32>], q: usize, cw: usize, ch: usize, dim: usize) -> Vec<f32> {
        let big_cols = q * cw;
        let mut out = vec![0f32; (q * ch) * big_cols * dim];
        for tile_row in 0..ch {
            for tile_col in 0..cw {
                let tile = &tiles[tile_row * cw + tile_col];
                for sub_row in 0..q {
                    let big_row = tile_row * q + sub_row;
                    let dst_off = (big_row * big_cols + tile_col * q) * dim;
                    let src_off = (sub_row * q) * dim;
                    out[dst_off..dst_off + q * dim]
                        .copy_from_slice(&tile[src_off..src_off + q * dim]);
                }
            }
        }
        out
    }

    /// Encode one preprocessed image (global view + optional local crop
    /// tiles) into its full LM-input token pack, including newlines and the
    /// trailing view separator.
    fn encode_image_pack(
        &self,
        projector: &Projector,
        image: &PreprocessedImage,
    ) -> Result<Vec<f32>> {
        let dim = projector.out_features;
        let newline = projector.newline()?.to_vec();
        let separator = projector.separator()?.to_vec();

        let (global_tokens, qb) =
            self.encode_view(projector, &image.global, image.global_size as usize)?;
        let global_packed = Self::insert_newlines(&global_tokens, qb, qb, dim, &newline);

        let mut pack = Vec::new();
        if image.has_tiles() {
            let [cw, ch] = image.spatial_crop;
            let (cw, ch) = (cw as usize, ch as usize);
            let n_tiles = image.tiles.len();
            ensure!(
                n_tiles == cw * ch,
                "deep_encoder: tile count {n_tiles} != spatial_crop {cw}x{ch}"
            );
            let mut tile_tokens = Vec::with_capacity(n_tiles);
            let mut q_crop = 0usize;
            for tile in &image.tiles {
                let (tokens, q) = self.encode_view(projector, tile, image.tile_size as usize)?;
                q_crop = q;
                tile_tokens.push(tokens);
            }
            let mosaic_tokens = Self::mosaic(&tile_tokens, q_crop, cw, ch, dim);
            let local_packed =
                Self::insert_newlines(&mosaic_tokens, q_crop * ch, q_crop * cw, dim, &newline);
            pack.extend_from_slice(&local_packed);
        }
        pack.extend_from_slice(&global_packed);
        pack.extend_from_slice(&separator);
        Ok(pack)
    }

    /// Encode + project every view of every image into one packed vision
    /// embedding buffer (`[n_vision_tokens, hidden]`, row-major), in the
    /// order [`crate::embed::fuse_inputs_embeds`] expects to match the
    /// prompt's `<image>` placeholder expansion.
    pub fn encode_and_project(
        &self,
        images: &[PreprocessedImage],
        projector: &Projector,
    ) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        for image in images {
            let pack = self.encode_image_pack(projector, image)?;
            out.extend_from_slice(&pack);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_newlines_shape() {
        let dim = 2;
        let tokens = vec![1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]; // 2x2 grid
        let newline = vec![9.0f32, 9.0];
        let out = DeepEncoder::insert_newlines(&tokens, 2, 2, dim, &newline);
        assert_eq!(out.len(), 2 * 3 * dim);
        assert_eq!(&out[4..6], &[9.0, 9.0]);
        assert_eq!(&out[10..12], &[9.0, 9.0]);
    }

    #[test]
    fn mosaic_places_tiles_in_row_major_order() {
        let dim = 1;
        // 2 tiles, each 2x2 (q=2), spatial_crop = [2,1] (cw=2, ch=1).
        let tile0 = vec![1.0f32, 2.0, 3.0, 4.0];
        let tile1 = vec![5.0f32, 6.0, 7.0, 8.0];
        let out = DeepEncoder::mosaic(&[tile0, tile1], 2, 2, 1, dim);
        // Expected big grid (2 rows x 4 cols):
        // row0: 1 2 5 6
        // row1: 3 4 7 8
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }
}
