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

//! SAM's point and box prompts as sparse embeddings, and the image's
//! positional encoding: the random-Fourier features candle's
//! `PromptEncoder` computes. Plain host math over the prompt encoder's
//! weights, which the exported package's program builds too.

use super::config::SAM_IMG_SIZE;
use anyhow::{Result, ensure};

/// The prompt-encoder weights the sparse embeddings read.
pub struct PointWeights<'a> {
    pub embed_dim: usize,
    /// `[2, embed_dim/2]`.
    pub pe_gaussian: &'a [f32],
    /// `[embed_dim]`.
    pub not_a_point_embed: &'a [f32],
    /// `[4, embed_dim]`: background, foreground, box corners.
    pub point_embeddings: &'a [f32],
}

/// `[tokens, embed_dim]`: the points (padded with a not-a-point token when
/// there is no box), then each box's two corners. Coordinates are pixels
/// of the `SAM_IMG_SIZE` input.
pub fn sparse_embeddings(
    w: &PointWeights<'_>,
    points: Option<(&[f32], &[f32])>,
    boxes: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let pad_points = boxes.is_none();
    let mut sparse = Vec::new();

    if let Some((coords, labels)) = points {
        let n = labels.len();
        ensure!(
            coords.len() == n * 2,
            "points coords len {} ≠ N·2 ({}·2)",
            coords.len(),
            n
        );
        // Candle adds 0.5 to point coords.
        let mut pts: Vec<f32> = coords.iter().map(|c| c + 0.5).collect();
        let mut lbls = labels.to_vec();
        if pad_points {
            // Pad with a single "not-a-point" sentinel (label -1).
            pts.push(0.0);
            pts.push(0.0);
            lbls.push(-1.0);
        }
        let n_padded = lbls.len();
        let emb = embed_points_and_boxes(w, &pts, n_padded, /*is_box=*/ false, Some(&lbls))?;
        sparse.extend_from_slice(&emb);
    }
    if let Some(box_coords) = boxes {
        let m = box_coords.len() / 4;
        ensure!(box_coords.len() == m * 4, "boxes len must be multiple of 4");
        let coords_with_half: Vec<f32> = box_coords.iter().map(|c| c + 0.5).collect();
        let emb = embed_points_and_boxes(w, &coords_with_half, m * 2, /*is_box=*/ true, None)?;
        sparse.extend_from_slice(&emb);
    }

    Ok(sparse)
}

/// Random-Fourier positional encoding for a `(h, w)` grid.
/// Output shape `[embed_dim, h, w]`.
pub fn image_pe(w: &PointWeights<'_>, h: usize, ww: usize) -> Vec<f32> {
    let e = w.embed_dim;
    let half = e / 2;
    let mut out = vec![0f32; e * h * ww];
    // For each (y, x) cell, normalize to (x+0.5)/w, (y+0.5)/h, then map
    // through the Gaussian + sin/cos pe_encoding.
    for y in 0..h {
        let fy = (y as f32 + 0.5) / h as f32;
        for x in 0..ww {
            let fx = (x as f32 + 0.5) / ww as f32;
            // Candle's pe_encoding: coords = 2*coords - 1, then @ M, then *2π
            let cx = fx * 2.0 - 1.0;
            let cy = fy * 2.0 - 1.0;
            // [cx, cy] @ gaussian [2, half]  → [half]
            for k in 0..half {
                let mut acc = cx * w.pe_gaussian[k] + cy * w.pe_gaussian[half + k];
                acc *= 2.0 * std::f32::consts::PI;
                out[k * h * ww + y * ww + x] = acc.sin();
                out[(half + k) * h * ww + y * ww + x] = acc.cos();
            }
        }
    }
    out
}

/// Apply the Gaussian + sin/cos PE to arbitrary `[N, 2]` coords already
/// in `[0, 1]` (or with extra padding columns that are passed through
/// unchanged via the candle convention). Returns `[N, embed_dim]`.
fn pe_encode_normalized(w: &PointWeights<'_>, coords: &[f32], n: usize) -> Vec<f32> {
    let e = w.embed_dim;
    let half = e / 2;
    let mut out = vec![0f32; n * e];
    for i in 0..n {
        let cx = coords[i * 2] * 2.0 - 1.0;
        let cy = coords[i * 2 + 1] * 2.0 - 1.0;
        for k in 0..half {
            let mut acc = cx * w.pe_gaussian[k] + cy * w.pe_gaussian[half + k];
            acc *= 2.0 * std::f32::consts::PI;
            out[i * e + k] = acc.sin();
            out[i * e + half + k] = acc.cos();
        }
    }
    out
}

/// Embed N points or N boxes (each box becomes 2 corner points).
///
/// For points: applies labels to add per-label embeddings.
/// For boxes: adds `point_embeddings[2]` to first corner, `[3]` to second.
fn embed_points_and_boxes(
    w: &PointWeights<'_>,
    coords_in_pixels: &[f32], // [n*2]
    n: usize,
    is_box: bool,
    labels: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let e = w.embed_dim;
    // Normalize pixel coords to [0,1] using SAM_IMG_SIZE.
    let img = SAM_IMG_SIZE as f32;
    let normed: Vec<f32> = coords_in_pixels.iter().map(|c| c / img).collect();
    let mut emb = pe_encode_normalized(w, &normed, n);

    if is_box {
        // Box: 2 corners per box. Alternate [2], [3] per pair.
        for i in 0..n {
            let pe_idx = if i % 2 == 0 { 2 } else { 3 };
            for k in 0..e {
                emb[i * e + k] += w.point_embeddings[pe_idx * e + k];
            }
        }
    } else if let Some(lbls) = labels {
        ensure!(lbls.len() == n, "labels len {} ≠ n {n}", lbls.len());
        for i in 0..n {
            let label = lbls[i];
            if label < 0.0 {
                // "not-a-point" padding token replaces the PE entirely.
                for k in 0..e {
                    emb[i * e + k] = w.not_a_point_embed[k];
                }
            } else if label == 0.0 {
                for k in 0..e {
                    emb[i * e + k] += w.point_embeddings[k];
                }
            } else {
                // label == 1.0 (foreground)
                for k in 0..e {
                    emb[i * e + k] += w.point_embeddings[e + k];
                }
            }
        }
    }
    Ok(emb)
}
