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

//! Host-side image preprocessing for mllama (ports `image_processing_mllama.py`).
//!
//! Produces the two graph inputs the vision tower expects:
//! - `hidden`   `[1, num_tiles*num_patches, hidden]` — patch embeddings + the
//!   pre-tile positional embedding + the class token + the gated positional
//!   embedding, in `[tile, position]` order.
//! - `post_tile` `[1, num_tiles*num_patches, hidden]` — the post-tile positional
//!   embedding (added in-graph after `layernorm_post`).
//!
//! We run with the image's *exact* tile count and no /8 patch padding: HF masks
//! both the padded tiles and the alignment-pad patches, so an exact run is
//! numerically equivalent while dropping all masks. mllama normalizes with
//! `IMAGENET_STANDARD` mean/std = 0.5 → `pixel/127.5 - 1` (pad pixels = 0 → -1).

use anyhow::{Result, ensure};
use image::{RgbImage, imageops::FilterType};
use rlx_core::weight_map::WeightMap;

use crate::config::MllamaVisionConfig;

/// Host-side vision-stem weights, taken from the checkpoint before the graph is
/// built (so the graph never sees them). All positional / tile tables are kept
/// raw and indexed by aspect-ratio id at preprocess time.
pub struct VisionEmbedWeights {
    pub hidden: usize,
    pub patch_dim: usize,   // channels * patch^2
    pub num_patches: usize, // incl. class token (grid^2 + 1)
    pub grid: usize,        // patches per tile side
    pub patch_size: usize,
    pub tile_size: usize,
    pub max_num_tiles: usize,
    pub supported: Vec<(usize, usize)>, // (num_tiles_height, num_tiles_width)

    patch_embed_t: Vec<f32>,  // [patch_dim, hidden]
    class_embed: Vec<f32>,    // [hidden]
    pos_embed: Vec<f32>,      // [num_patches, hidden]
    pos_gate: f32,            // tanh applied at use
    tile_pos_table: Vec<f32>, // [rows, max_tiles*num_patches*hidden]
    pre_table: Vec<f32>,      // [rows, max_tiles*hidden]
    pre_gate: f32,
    post_table: Vec<f32>, // [rows, max_tiles*hidden]
    post_gate: f32,
}

/// Vision graph inputs for one image.
pub struct VisionInputs {
    pub hidden: Vec<f32>,    // [num_tiles*num_patches*hidden]
    pub post_tile: Vec<f32>, // [num_tiles*num_patches*hidden]
    pub num_tiles: usize,
    pub aspect_ratio_id: usize,
    pub seq: usize, // num_tiles * num_patches
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

/// `get_all_supported_aspect_ratios(max)` — arrangements `(a, b)` used both as
/// tile grid `(height, width)` and as the aspect-ratio-id ordering.
pub fn supported_aspect_ratios(max: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for a in 1..=max {
        for b in 1..=max {
            if a * b <= max {
                v.push((a, b));
            }
        }
    }
    v
}

/// Port of `get_optimal_tiled_canvas` → `(num_tiles_height, num_tiles_width)`.
pub fn optimal_canvas(img_h: usize, img_w: usize, max_tiles: usize, tile: usize) -> (usize, usize) {
    let arrangements = supported_aspect_ratios(max_tiles);
    let (imgh, imgw) = (img_h as f64, img_w as f64);
    // scale[i] = min( (a*tile)/img_h , (b*tile)/img_w )
    let scales: Vec<f64> = arrangements
        .iter()
        .map(|&(a, b)| {
            let sh = (a * tile) as f64 / imgh;
            let sw = (b * tile) as f64 / imgw;
            if sw > sh { sh } else { sw }
        })
        .collect();
    let up: Vec<f64> = scales.iter().copied().filter(|&s| s >= 1.0).collect();
    let selected = if !up.is_empty() {
        up.iter().copied().fold(f64::INFINITY, f64::min)
    } else {
        scales
            .iter()
            .copied()
            .filter(|&s| s < 1.0)
            .fold(f64::NEG_INFINITY, f64::max)
    };
    // among arrangements with scale == selected, pick minimum canvas area.
    let mut best: Option<(usize, usize)> = None;
    let mut best_area = usize::MAX;
    for (i, &(a, b)) in arrangements.iter().enumerate() {
        if scales[i] == selected {
            let area = (a * tile) * (b * tile);
            if area < best_area {
                best_area = area;
                best = Some((a, b));
            }
        }
    }
    best.unwrap_or((1, 1))
}

/// Port of `get_image_size_fit_to_canvas` → `(new_height, new_width)`.
fn fit_to_canvas(
    img_h: usize,
    img_w: usize,
    canvas_h: usize,
    canvas_w: usize,
    tile: usize,
) -> (usize, usize) {
    let target_w = (img_w.max(tile)).min(canvas_w);
    let target_h = (img_h.max(tile)).min(canvas_h);
    let scale_h = target_h as f64 / img_h as f64;
    let scale_w = target_w as f64 / img_w as f64;
    if scale_w < scale_h {
        let new_w = target_w;
        let nh = ((img_h as f64 * scale_w).floor() as usize)
            .max(1)
            .min(target_h);
        (nh, new_w)
    } else {
        let new_h = target_h;
        let nw = ((img_w as f64 * scale_h).floor() as usize)
            .max(1)
            .min(target_w);
        (new_h, nw)
    }
}

/// Extract and cache the vision stem weights (consumes them from `wm`).
pub fn extract_vision_embed_weights(
    wm: &mut WeightMap,
    cfg: &MllamaVisionConfig,
) -> Result<VisionEmbedWeights> {
    let hidden = cfg.hidden_size;
    let ps = cfg.patch_size;
    let patch_dim = cfg.num_channels * ps * ps;
    let np = cfg.num_patches();
    let grid = cfg.image_size / ps;

    let (pe, pe_shape) = wm.take("vision_model.patch_embedding.weight")?;
    ensure!(
        pe.len() == hidden * patch_dim,
        "patch_embedding.weight len {} != hidden*patch_dim {} (shape {:?})",
        pe.len(),
        hidden * patch_dim,
        pe_shape
    );
    let patch_embed_t = transpose_2d(&pe, hidden, patch_dim);

    let (class_embed, _) = wm.take("vision_model.class_embedding")?;
    let (pos_embed, _) = wm.take("vision_model.gated_positional_embedding.embedding")?;
    let (pos_gate, _) = wm.take("vision_model.gated_positional_embedding.gate")?;
    let (tile_pos_table, _) =
        wm.take("vision_model.gated_positional_embedding.tile_embedding.weight")?;
    let (pre_table, _) = wm.take("vision_model.pre_tile_positional_embedding.embedding.weight")?;
    let (pre_gate, _) = wm.take("vision_model.pre_tile_positional_embedding.gate")?;
    let (post_table, _) =
        wm.take("vision_model.post_tile_positional_embedding.embedding.weight")?;
    let (post_gate, _) = wm.take("vision_model.post_tile_positional_embedding.gate")?;

    Ok(VisionEmbedWeights {
        hidden,
        patch_dim,
        num_patches: np,
        grid,
        patch_size: ps,
        tile_size: cfg.image_size,
        max_num_tiles: cfg.max_num_tiles,
        supported: supported_aspect_ratios(cfg.max_num_tiles),
        patch_embed_t,
        class_embed,
        pos_embed,
        pos_gate: pos_gate.first().copied().unwrap_or(0.0),
        tile_pos_table,
        pre_table,
        pre_gate: pre_gate.first().copied().unwrap_or(0.0),
        post_table,
        post_gate: post_gate.first().copied().unwrap_or(0.0),
    })
}

impl VisionEmbedWeights {
    /// Preprocess one RGB image (`rgb` is HWC `u8`, length `w*h*3`).
    pub fn preprocess(&self, rgb: &[u8], w: usize, h: usize) -> Result<VisionInputs> {
        ensure!(rgb.len() == w * h * 3, "rgb len {} != w*h*3", rgb.len());
        let tile = self.tile_size;
        let ps = self.patch_size;
        let grid = self.grid;
        let hidden = self.hidden;
        let np = self.num_patches;

        let (nth, ntw) = optimal_canvas(h, w, self.max_num_tiles, tile);
        let num_tiles = nth * ntw;
        let canvas_h = nth * tile;
        let canvas_w = ntw * tile;
        let ar_pos = self
            .supported
            .iter()
            .position(|&r| r == (nth, ntw))
            .ok_or_else(|| anyhow::anyhow!("aspect ratio ({nth},{ntw}) unsupported"))?;
        let ar_id = ar_pos + 1;

        let (new_h, new_w) = fit_to_canvas(h, w, canvas_h, canvas_w, tile);

        // Bilinear resize, then zero-pad (bottom/right) into the canvas.
        let src = RgbImage::from_raw(w as u32, h as u32, rgb.to_vec())
            .ok_or_else(|| anyhow::anyhow!("bad rgb buffer"))?;
        let resized =
            image::imageops::resize(&src, new_w as u32, new_h as u32, FilterType::Triangle);

        // Planar normalized canvas [3, canvas_h, canvas_w]; pad pixels (raw 0) → -1.
        let plane = canvas_h * canvas_w;
        let mut canvas = vec![-1.0f32; 3 * plane]; // 0/127.5 - 1 == -1
        for y in 0..new_h {
            for x in 0..new_w {
                let p = resized.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    canvas[c * plane + y * canvas_w + x] = p[c] as f32 / 127.5 - 1.0;
                }
            }
        }

        // Patch-embed each tile, then assemble hidden + post_tile with embeddings.
        let pre_gate_t = self.pre_gate.tanh();
        let pos_gate_t = self.pos_gate.tanh();
        let post_gate_t = self.post_gate.tanh();
        let tile_row = ar_id * self.max_num_tiles * hidden; // into pre_/post_table
        let tilepos_row = ar_id * self.max_num_tiles * np * hidden; // into tile_pos_table

        let mut hidden_seq = vec![0.0f32; num_tiles * np * hidden];
        let mut post_seq = vec![0.0f32; num_tiles * np * hidden];

        let mut patch_vec = vec![0.0f32; self.patch_dim];
        for th in 0..nth {
            for tw in 0..ntw {
                let t = th * ntw + tw; // row-major tile index (matches split_to_tiles)
                let pre_off = tile_row + t * hidden;
                let post_off = tile_row + t * hidden;
                // per-tile position bases
                for py in 0..grid {
                    for px in 0..grid {
                        let patch = py * grid + px; // 0..ppt
                        // channel-major patch vector [c, ky, kx]
                        for c in 0..3 {
                            for ky in 0..ps {
                                let row = (th * tile + py * ps + ky) * canvas_w;
                                let base = c * plane + row + tw * tile + px * ps;
                                let dst = c * ps * ps + ky * ps;
                                patch_vec[dst..dst + ps].copy_from_slice(&canvas[base..base + ps]);
                            }
                        }
                        // patch_embed: [patch_dim] · [patch_dim, hidden] -> [hidden]
                        // position in the tile sequence: class is 0, patches are 1..np
                        let q = patch + 1;
                        let dst = (t * np + q) * hidden;
                        for o in 0..hidden {
                            let mut acc = 0.0f32;
                            for k in 0..self.patch_dim {
                                acc += patch_vec[k] * self.patch_embed_t[k * hidden + o];
                            }
                            hidden_seq[dst + o] = acc + pre_gate_t * self.pre_table[pre_off + o];
                        }
                    }
                }
                // class token at position 0 of this tile
                let cdst = (t * np) * hidden;
                hidden_seq[cdst..cdst + hidden].copy_from_slice(&self.class_embed[..hidden]);

                // gated positional (all np positions) + post-tile additive
                for q in 0..np {
                    let dst = (t * np + q) * hidden;
                    let po = q * hidden;
                    let tpo = tilepos_row + (t * np + q) * hidden;
                    for o in 0..hidden {
                        hidden_seq[dst + o] += (1.0 - pos_gate_t) * self.pos_embed[po + o]
                            + pos_gate_t * self.tile_pos_table[tpo + o];
                        post_seq[dst + o] = post_gate_t * self.post_table[post_off + o];
                    }
                }
            }
        }

        Ok(VisionInputs {
            hidden: hidden_seq,
            post_tile: post_seq,
            num_tiles,
            aspect_ratio_id: ar_id,
            seq: num_tiles * np,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canvas_selection_matches_reference() {
        // 448x448 square → 1x1 tile.
        assert_eq!(optimal_canvas(448, 448, 4, 448), (1, 1));
        // very wide image → prefers more width tiles.
        let (a, b) = optimal_canvas(200, 1600, 4, 448);
        assert!(a * b <= 4);
        assert!(
            b >= a,
            "wide image should use >= width tiles, got ({a},{b})"
        );
    }

    #[test]
    fn supported_ratios_order() {
        assert_eq!(
            supported_aspect_ratios(4),
            vec![
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 1),
                (2, 2),
                (3, 1),
                (4, 1)
            ]
        );
    }
}
