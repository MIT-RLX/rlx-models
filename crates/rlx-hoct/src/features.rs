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

//! Region-property features for HOCT nodes (19-d).
//!
//! Layout matches upstream `REGIONPROPS` plus `(t,z,y,x)`: diameter, intensity
//! stats, 3×3 inertia (skimage convention), and centroid `border_dist`.
//! 2D labels are padded to Z=1 before measuring (same as HOCT's convert-to-3D).

use crate::config::{FEATURE_MEAN, FEATURE_STD, HoctConfig};
use ndarray::{Array1, Array3, Array4, Axis};

/// One detected object / node.
#[derive(Debug, Clone)]
pub struct NodeFeatures {
    pub label: u32,
    pub t: f32,
    pub z: f32,
    pub y: f32,
    pub x: f32,
    pub diameter: f32,
    pub intensity_min: f32,
    pub intensity_max: f32,
    pub intensity_mean: f32,
    pub intensity_std: f32,
    pub inertia: [f32; 9],
    pub border_dist: f32,
}

impl NodeFeatures {
    pub fn to_raw(&self) -> [f32; 19] {
        [
            self.t,
            self.z,
            self.y,
            self.x,
            self.diameter,
            self.intensity_min,
            self.intensity_max,
            self.intensity_mean,
            self.intensity_std,
            self.inertia[0],
            self.inertia[1],
            self.inertia[2],
            self.inertia[3],
            self.inertia[4],
            self.inertia[5],
            self.inertia[6],
            self.inertia[7],
            self.inertia[8],
            self.border_dist,
        ]
    }

    pub fn standardized(&self) -> [f32; 19] {
        let raw = self.to_raw();
        let mut out = raw;
        for i in 0..19 {
            out[i] = (raw[i] - FEATURE_MEAN[i]) / FEATURE_STD[i];
        }
        out
    }
}

pub fn standardize_batch(feats: &mut Array3<f32>) {
    for bi in 0..feats.len_of(Axis(0)) {
        for ni in 0..feats.len_of(Axis(1)) {
            for i in 0..19 {
                feats[[bi, ni, i]] = (feats[[bi, ni, i]] - FEATURE_MEAN[i]) / FEATURE_STD[i];
            }
        }
    }
}

/// Border distance matching `hoct.features._border_dist_nd` (centroid-based).
///
/// `1 - min(1, min_dist_to_border / cutoff)` with default `cutoff = 5`.
pub fn border_dist_centroid(coords: &[f32], shape: &[usize], cutoff: f32) -> f32 {
    debug_assert_eq!(coords.len(), shape.len());
    let mut dmin = f32::INFINITY;
    for (c, &s) in coords.iter().zip(shape.iter()) {
        dmin = dmin.min(*c).min(s as f32 - *c);
    }
    1.0 - (dmin / cutoff).min(1.0)
}

/// Extract nodes from a label image `(Y, X)` at frame `t` (2D).
///
/// Matches HOCT's 2D→3D path: singleton Z, skimage-style inertia (`/n`),
/// 3D equivalent diameter, and centroid border_dist.
pub fn regionprops_2d(
    labels: &Array3<u32>,
    t: i32,
    image: Option<ndarray::ArrayView3<f32>>,
    _border_dist_scale: f32,
) -> Vec<NodeFeatures> {
    let (y_max, x_max) = (labels.len_of(Axis(1)), labels.len_of(Axis(2)));
    // Pad to Z=1 volume — same as `hoct.features.graph.convert_to_3d`.
    let mut lab4 = Array4::<u32>::zeros((1, 1, y_max, x_max));
    for y in 0..y_max {
        for x in 0..x_max {
            lab4[[0, 0, y, x]] = labels[[0, y, x]];
        }
    }
    let img4 = image.map(|img| {
        let mut a = Array4::<f32>::zeros((1, 1, y_max, x_max));
        for y in 0..y_max {
            for x in 0..x_max {
                a[[0, 0, y, x]] = img[[0, y, x]];
            }
        }
        a
    });
    regionprops_3d(&lab4, t, img4.as_ref())
}

/// Stack node features to `(N, 19)` array (unstandardized).
pub fn nodes_to_array(nodes: &[NodeFeatures]) -> Array3<f32> {
    let n = nodes.len();
    let mut out = Array3::<f32>::zeros((1, n, HoctConfig::default().feature_dim));
    for (i, node) in nodes.iter().enumerate() {
        let std = node.standardized();
        for d in 0..19 {
            out[[0, i, d]] = std[d];
        }
    }
    out
}

/// Positions `(1, N, 3)` as spatial `[z, y, x]` (RoPE; time lives in features).
pub fn nodes_to_pos(nodes: &[NodeFeatures]) -> Array3<f32> {
    let n = nodes.len();
    let mut pos = Array3::<f32>::zeros((1, n, 3));
    for (i, node) in nodes.iter().enumerate() {
        pos[[0, i, 0]] = node.z;
        pos[[0, i, 1]] = node.y;
        pos[[0, i, 2]] = node.x;
    }
    pos
}

/// Positions from raw feature rows: indices 1,2,3 are `z,y,x`.
pub fn pos_from_raw_features(raw: &Array3<f32>) -> Array3<f32> {
    let n = raw.len_of(Axis(1));
    let mut pos = Array3::<f32>::zeros((1, n, 3));
    for i in 0..n {
        pos[[0, i, 0]] = raw[[0, i, 1]];
        pos[[0, i, 1]] = raw[[0, i, 2]];
        pos[[0, i, 2]] = raw[[0, i, 3]];
    }
    pos
}

/// 3D labels `(1, Z, Y, X)` batch-squeezed as `(Z, Y, X)` at time `t`.
///
/// Inertia matches skimage `regionprops(...).inertia_tensor` (moments / area),
/// flattened row-major into 9 scalars. Border uses centroid formula (cutoff=5).
pub fn regionprops_3d(
    labels: &Array4<u32>,
    t: i32,
    image: Option<&Array4<f32>>,
) -> Vec<NodeFeatures> {
    let z_max = labels.len_of(Axis(1));
    let y_max = labels.len_of(Axis(2));
    let x_max = labels.len_of(Axis(3));
    let mut by_label: std::collections::HashMap<u32, Vec<(usize, usize, usize)>> =
        std::collections::HashMap::new();
    for z in 0..z_max {
        for y in 0..y_max {
            for x in 0..x_max {
                let lb = labels[[0, z, y, x]];
                if lb == 0 {
                    continue;
                }
                by_label.entry(lb).or_default().push((z, y, x));
            }
        }
    }
    let mut nodes = Vec::new();
    for (label, vox) in by_label {
        let n = vox.len() as f32;
        let mut cz = 0.0;
        let mut cy = 0.0;
        let mut cx = 0.0;
        let mut min_i = f32::INFINITY;
        let mut max_i = f32::NEG_INFINITY;
        let mut sum_i = 0.0;
        let mut sum_i2 = 0.0;
        for &(z, y, x) in &vox {
            cz += z as f32;
            cy += y as f32;
            cx += x as f32;
            let inten = image.map(|img| img[[0, z, y, x]]).unwrap_or(0.0);
            min_i = min_i.min(inten);
            max_i = max_i.max(inten);
            sum_i += inten;
            sum_i2 += inten * inten;
        }
        cz /= n;
        cy /= n;
        cx /= n;
        let mean_i = sum_i / n;
        let var_i = (sum_i2 / n - mean_i * mean_i).max(0.0);
        // skimage `equivalent_diameter_area` for 3D: (6V/π)^(1/3)
        let eq_d = (6.0 * n / std::f32::consts::PI).powf(1.0 / 3.0);

        // Central second moments (unnormalized), then /n like skimage.
        let mut mu_zz = 0.0f32;
        let mut mu_yy = 0.0f32;
        let mut mu_xx = 0.0f32;
        let mut mu_zy = 0.0f32;
        let mut mu_zx = 0.0f32;
        let mut mu_yx = 0.0f32;
        for &(z, y, x) in &vox {
            let dz = z as f32 - cz;
            let dy = y as f32 - cy;
            let dx = x as f32 - cx;
            mu_zz += dz * dz;
            mu_yy += dy * dy;
            mu_xx += dx * dx;
            mu_zy += dz * dy;
            mu_zx += dz * dx;
            mu_yx += dy * dx;
        }
        mu_zz /= n;
        mu_yy /= n;
        mu_xx /= n;
        mu_zy /= n;
        mu_zx /= n;
        mu_yx /= n;
        // Inertia tensor I_ij = δ_ij * tr(μ) - μ_ij (skimage convention).
        let tr = mu_xx + mu_yy + mu_zz;
        let i_zz = tr - mu_zz;
        let i_yy = tr - mu_yy;
        let i_xx = tr - mu_xx;
        let i_zy = -mu_zy;
        let i_zx = -mu_zx;
        let i_yx = -mu_yx;
        // Row-major 3×3: [[Izz, Izy, Izx], [Iyz, Iyy, Iyx], [Ixz, Ixy, Ixx]]
        // Matching skimage's (z,y,x) axis order for 3D volumes.
        let inertia = [
            i_zz, i_zy, i_zx, //
            i_zy, i_yy, i_yx, //
            i_zx, i_yx, i_xx,
        ];

        let border = border_dist_centroid(&[cz, cy, cx], &[z_max, y_max, x_max], 5.0);
        nodes.push(NodeFeatures {
            label,
            t: t as f32,
            z: cz,
            y: cy,
            x: cx,
            diameter: eq_d,
            intensity_min: if image.is_some() { min_i } else { 0.0 },
            intensity_max: if image.is_some() { max_i } else { 0.0 },
            intensity_mean: if image.is_some() { mean_i } else { 0.0 },
            intensity_std: if image.is_some() { var_i.sqrt() } else { 0.0 },
            inertia,
            border_dist: border,
        });
    }
    nodes.sort_by_key(|n| n.label);
    nodes
}

/// Helper: centroid vector for quick tests.
pub fn centroid_vec(nodes: &[NodeFeatures]) -> Array1<f32> {
    Array1::from_iter(nodes.iter().flat_map(|n| [n.t, n.y, n.x]))
}
