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

//! Spatial geometry helpers for HOCT attention masks and edge distance bias.

use ndarray::{Array2, Array3, Array4, Axis};

const PARALLEL_EPS: f32 = 1e-4;
const CLAMP_EPS: f32 = 1e-6;

/// Pairwise squared Euclidean distance: `||a||² + ||b||² - 2 a·b`.
///
/// `pos` shape `(B, N, 3)` → `(B, N, N)`.
pub fn pairwise_sqdist(pos: &Array3<f32>) -> Array3<f32> {
    let b = pos.len_of(Axis(0));
    let n = pos.len_of(Axis(1));
    let mut out = Array3::<f32>::zeros((b, n, n));
    for bi in 0..b {
        for i in 0..n {
            let pi = pos.slice(ndarray::s![bi, i, ..]);
            let norm_i: f32 = pi.iter().map(|x| x * x).sum();
            for j in 0..n {
                let pj = pos.slice(ndarray::s![bi, j, ..]);
                let norm_j: f32 = pj.iter().map(|x| x * x).sum();
                let dot: f32 = pi.iter().zip(pj.iter()).map(|(a, b)| a * b).sum();
                let d2 = (norm_i + norm_j - 2.0 * dot).max(1e-30);
                out[[bi, i, j]] = d2;
            }
        }
    }
    out
}

/// Attention additive mask `(B, 1, N, N)`: `0` keep, `-inf` block.
pub fn make_attn_mask(pos: &Array3<f32>, mask: &Array2<bool>, tau_sq: f32) -> Array4<f32> {
    let b = pos.len_of(Axis(0));
    let n = pos.len_of(Axis(1));
    let d2 = pairwise_sqdist(pos);
    let mut out = Array4::<f32>::from_elem((b, 1, n, n), f32::NEG_INFINITY);
    for bi in 0..b {
        for i in 0..n {
            for j in 0..n {
                let keep = mask[[bi, i]] && mask[[bi, j]] && d2[[bi, i, j]] < tau_sq;
                if keep {
                    out[[bi, 0, i, j]] = 0.0;
                }
            }
        }
    }
    out
}

/// Line-to-line distance matrix between edge segments.
///
/// For each edge `e`, source point `P0 = node_pos[i]`, target `P1 = node_pos[j]`.
/// Returns `(B, E, E)` distances between closest points on segment pairs.
pub fn line_to_line_distances(node_pos: &Array3<f32>, edge_indices: &Array3<i64>) -> Array3<f32> {
    let b = node_pos.len_of(Axis(0));
    let e = edge_indices.len_of(Axis(1));
    let mut dist = Array3::<f32>::zeros((b, e, e));

    for bi in 0..b {
        let mut p0 = vec![[0.0f32; 3]; e];
        let mut u = vec![[0.0f32; 3]; e];
        for ei in 0..e {
            let i = edge_indices[[bi, ei, 0]] as usize;
            let j = edge_indices[[bi, ei, 1]] as usize;
            for d in 0..3 {
                p0[ei][d] = node_pos[[bi, i, d]];
                u[ei][d] = node_pos[[bi, j, d]] - node_pos[[bi, i, d]];
            }
        }

        for ei in 0..e {
            for ej in 0..e {
                dist[[bi, ei, ej]] = segment_segment_dist(p0[ei], u[ei], p0[ej], u[ej]);
            }
        }
    }
    dist
}

fn segment_segment_dist(p0_i: [f32; 3], u_i: [f32; 3], p0_j: [f32; 3], u_j: [f32; 3]) -> f32 {
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let a = dot(u_i, u_i);
    let b = dot(u_i, u_j);
    let c = dot(u_j, u_j);
    let v0 = [p0_i[0] - p0_j[0], p0_i[1] - p0_j[1], p0_i[2] - p0_j[2]];
    let d = dot(u_i, v0);
    let e = dot(u_j, v0);
    let big_d = a * c - b * b;

    let (sc, tc) = if big_d < PARALLEL_EPS {
        let tc_p = (e / c.max(CLAMP_EPS)).clamp(0.0, 1.0);
        (0.0, tc_p)
    } else {
        let sc_np = ((b * e - c * d) / big_d.max(CLAMP_EPS)).clamp(0.0, 1.0);
        let tc_np = ((b * sc_np + e) / c.max(CLAMP_EPS)).clamp(0.0, 1.0);
        let sc_np2 = ((b * tc_np - d) / a.max(CLAMP_EPS)).clamp(0.0, 1.0);
        (sc_np2, tc_np)
    };

    let mut closest_i = [0.0f32; 3];
    let mut closest_j = [0.0f32; 3];
    for k in 0..3 {
        closest_i[k] = p0_i[k] + sc * u_i[k];
        closest_j[k] = p0_j[k] + tc * u_j[k];
    }
    let dx = closest_i[0] - closest_j[0];
    let dy = closest_i[1] - closest_j[1];
    let dz = closest_i[2] - closest_j[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn pairwise_sqdist_zero_for_same_point() {
        let pos = Array3::from_shape_vec((1, 2, 3), vec![0.0, 0.0, 0.0, 3.0, 4.0, 0.0]).unwrap();
        let d2 = pairwise_sqdist(&pos);
        assert!((d2[[0, 0, 0]]).abs() < 1e-6);
        assert!((d2[[0, 0, 1]] - 25.0).abs() < 1e-4);
    }

    #[test]
    fn mask_blocks_far_pairs() {
        let pos = Array3::from_shape_vec((1, 2, 3), vec![0.0, 0.0, 0.0, 1000.0, 0.0, 0.0]).unwrap();
        let mask = arr2(&[[true, true]]);
        let m = make_attn_mask(&pos, &mask, 90000.0);
        assert_eq!(m[[0, 0, 0, 0]], 0.0);
        assert!(m[[0, 0, 0, 1]].is_infinite());
    }
}
