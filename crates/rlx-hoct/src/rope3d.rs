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

//! 3D rotary position embedding + Householder reflection (HOCT).
//!
//! Positions are scaled by `1/τ`, mixed with learnable `log_freq`, rotate-half,
//! then reflected with `I − 2vvᵀ` per head.

use crate::config::HoctConfig;
use ndarray::{ArrayView4, Axis};

const SQRT3: f32 = 1.732_050_8;

/// Apply RoPE-3D then Householder reflection.
///
/// - `x`: `(B, H, N, Hd)` query or key head tensor (mutated conceptually; returns new)
/// - `pos`: `(B, N, 3)` node or edge positions
/// - `log_freq`: `(1, H, 1, 12, 1)`
/// - `reflect_vec`: `(H, Hd)`
/// - `eye`: `(Hd, Hd)`
pub fn apply_rope3d(
    cfg: &HoctConfig,
    x: &ArrayView4<f32>,
    pos: &ndarray::Array3<f32>,
    log_freq: &[f32],
    reflect_vec: &[f32],
    eye: &[f32],
) -> ndarray::Array4<f32> {
    let b = x.len_of(Axis(0));
    let h = x.len_of(Axis(1));
    let n = x.len_of(Axis(2));
    let hd = x.len_of(Axis(3));
    assert_eq!(hd, cfg.head_dim);

    let tau = cfg.tau;
    let mut rotated = ndarray::Array4::<f32>::zeros((b, h, n, hd));

    for bi in 0..b {
        for hi in 0..h {
            for ni in 0..n {
                let pos_n = [
                    pos[[bi, ni, 0]] / tau,
                    pos[[bi, ni, 1]] / tau,
                    pos[[bi, ni, 2]] / tau,
                ];
                let mut cos_hd = vec![0.0f32; hd];
                let mut sin_hd = vec![0.0f32; hd];
                for fi in 0..12 {
                    let lf = log_freq[hi * 12 + fi];
                    for ci in 0..3 {
                        let freq = pos_n[ci] * lf.exp() / SQRT3;
                        let c = freq.cos();
                        let s = freq.sin();
                        let c = if c.is_nan() { 1.0 } else { c };
                        let s = if s.is_nan() { 0.0 } else { s };
                        cos_hd[fi * 6 + ci * 2] = c;
                        cos_hd[fi * 6 + ci * 2 + 1] = c;
                        sin_hd[fi * 6 + ci * 2] = s;
                        sin_hd[fi * 6 + ci * 2 + 1] = s;
                    }
                }
                for d in 0..hd {
                    let xv = x[[bi, hi, ni, d]];
                    let rot = if d % 2 == 0 {
                        -x[[bi, hi, ni, d + 1]]
                    } else {
                        x[[bi, hi, ni, d - 1]]
                    };
                    rotated[[bi, hi, ni, d]] = xv * cos_hd[d] + rot * sin_hd[d];
                }
            }
        }
    }

    // Householder: refl = eye - 2 outer(v,v), out = einsum bhnd,hde->bhne
    let mut out = rotated.clone();
    for hi in 0..h {
        let rv: Vec<f32> = (0..hd).map(|d| reflect_vec[hi * hd + d]).collect();
        let norm: f32 = rv.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        let v: Vec<f32> = rv.iter().map(|x| x / norm).collect();
        let mut refl = vec![0.0f32; hd * hd];
        for i in 0..hd {
            for j in 0..hd {
                refl[i * hd + j] = eye[i * hd + j] - 2.0 * v[i] * v[j];
            }
        }
        for bi in 0..b {
            for ni in 0..n {
                for e in 0..hd {
                    let mut sum = 0.0f32;
                    for d in 0..hd {
                        sum += rotated[[bi, hi, ni, d]] * refl[d * hd + e];
                    }
                    out[[bi, hi, ni, e]] = sum;
                }
            }
        }
    }
    out
}

/// Rotation part of RoPE-3D (cos/sin), without Householder — for tests.
pub fn apply_rope_rotation(
    cfg: &HoctConfig,
    x: &ArrayView4<f32>,
    pos: &ndarray::Array3<f32>,
    log_freq: &[f32],
) -> ndarray::Array4<f32> {
    let b = x.len_of(Axis(0));
    let h = x.len_of(Axis(1));
    let n = x.len_of(Axis(2));
    let hd = x.len_of(Axis(3));
    let tau = cfg.tau;
    let mut rotated = ndarray::Array4::<f32>::zeros((b, h, n, hd));
    for bi in 0..b {
        for hi in 0..h {
            for ni in 0..n {
                let pos_n = [
                    pos[[bi, ni, 0]] / tau,
                    pos[[bi, ni, 1]] / tau,
                    pos[[bi, ni, 2]] / tau,
                ];
                let mut cos_hd = vec![0.0f32; hd];
                let mut sin_hd = vec![0.0f32; hd];
                for fi in 0..12 {
                    let lf = log_freq[hi * 12 + fi];
                    for ci in 0..3 {
                        let freq = pos_n[ci] * lf.exp() / SQRT3;
                        let c = freq.cos();
                        let s = freq.sin();
                        let c = if c.is_nan() { 1.0 } else { c };
                        let s = if s.is_nan() { 0.0 } else { s };
                        cos_hd[fi * 6 + ci * 2] = c;
                        cos_hd[fi * 6 + ci * 2 + 1] = c;
                        sin_hd[fi * 6 + ci * 2] = s;
                        sin_hd[fi * 6 + ci * 2 + 1] = s;
                    }
                }
                for d in 0..hd {
                    let xv = x[[bi, hi, ni, d]];
                    let rot = if d % 2 == 0 {
                        -x[[bi, hi, ni, d + 1]]
                    } else {
                        x[[bi, hi, ni, d - 1]]
                    };
                    rotated[[bi, hi, ni, d]] = xv * cos_hd[d] + rot * sin_hd[d];
                }
            }
        }
    }
    rotated
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn rope_finite_on_random() {
        let cfg = HoctConfig::default();
        let x = Array4::<f32>::from_elem((1, 4, 3, 72), 0.01);
        let pos = ndarray::Array3::from_shape_vec((1, 3, 3), vec![1.0; 9]).unwrap();
        let log_freq = vec![0.0f32; 4 * 12];
        let reflect_vec = vec![1.0f32; 4 * 72];
        let mut eye = vec![0.0f32; 72 * 72];
        for i in 0..72 {
            eye[i * 72 + i] = 1.0;
        }
        let y = apply_rope3d(&cfg, &x.view(), &pos, &log_freq, &reflect_vec, &eye);
        assert!(y.iter().all(|v| v.is_finite()));
    }
}
