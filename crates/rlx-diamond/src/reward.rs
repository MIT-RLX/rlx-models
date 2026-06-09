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

//! Pluggable reward functions for Diamond guidance.

/// Reward on clean latent / decoded state z (data at t=1).
pub trait LatentReward: Send + Sync {
    fn reward(&self, z: &[f32]) -> f32;
    /// ∂r/∂z (same length as z).
    fn grad_wrt_z(&self, z: &[f32]) -> Vec<f32>;
}

/// Proxy “blueness” on packed latent: maximize channel index 2 in each 3-group.
#[derive(Debug, Clone, Copy, Default)]
pub struct BluenessReward {
    pub scale: f32,
}

impl LatentReward for BluenessReward {
    fn reward(&self, z: &[f32]) -> f32 {
        let s: f32 = z.chunks(3).map(|c| c.get(2).copied().unwrap_or(0.0)).sum();
        self.scale * s
    }

    fn grad_wrt_z(&self, z: &[f32]) -> Vec<f32> {
        let mut g = vec![0.0f32; z.len()];
        for (i, chunk) in z.chunks(3).enumerate() {
            if chunk.len() > 2 {
                g[i * 3 + 2] = self.scale;
            }
        }
        g
    }
}

/// Linear measurement y = A z + noise; reward = -||y - A z||².
#[derive(Debug, Clone)]
pub struct LinearMeasurementReward {
    pub matrix: Vec<f32>,
    pub measurement: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl LinearMeasurementReward {
    pub fn new(matrix: Vec<f32>, measurement: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(matrix.len(), rows * cols);
        assert_eq!(measurement.len(), rows);
        Self {
            matrix,
            measurement,
            rows,
            cols,
        }
    }

    fn matvec(&self, z: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows];
        for r in 0..self.rows {
            let mut acc = 0.0f32;
            for c in 0..self.cols.min(z.len()) {
                acc += self.matrix[r * self.cols + c] * z[c];
            }
            out[r] = acc;
        }
        out
    }
}

impl LatentReward for LinearMeasurementReward {
    fn reward(&self, z: &[f32]) -> f32 {
        let pred = self.matvec(z);
        let err: f32 = pred
            .iter()
            .zip(self.measurement.iter())
            .map(|(p, m)| (p - m).powi(2))
            .sum();
        -err
    }

    fn grad_wrt_z(&self, z: &[f32]) -> Vec<f32> {
        let pred = self.matvec(z);
        let mut g = vec![0.0f32; z.len()];
        for r in 0..self.rows {
            let residual = 2.0 * (pred[r] - self.measurement[r]);
            for c in 0..self.cols.min(z.len()) {
                g[c] -= residual * self.matrix[r * self.cols + c];
            }
        }
        g
    }
}

/// Chain rule proxy: ∂r/∂x_t ≈ ∂r/∂z when z is a posterior sample near x_t.
pub fn grad_xt_via_z(grad_z: &[f32]) -> Vec<f32> {
    grad_z.to_vec()
}

/// SPSA estimate of ∂V/∂x_t using random Rademacher directions.
pub fn spsa_grad(
    x_t: &[f32],
    mut eval_v: impl FnMut(&[f32]) -> f32,
    eps: f32,
    num_dirs: usize,
    seed: u64,
) -> Vec<f32> {
    let dim = x_t.len();
    let mut grad = vec![0.0f32; dim];
    let mut state = seed;
    for _ in 0..num_dirs {
        let mut delta = vec![0.0f32; dim];
        for d in &mut delta {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *d = if state & 1 == 0 { 1.0 } else { -1.0 };
        }
        let mut xp = x_t.to_vec();
        let mut xm = x_t.to_vec();
        for i in 0..dim {
            xp[i] += eps * delta[i];
            xm[i] -= eps * delta[i];
        }
        let vp = eval_v(&xp);
        let vm = eval_v(&xm);
        let scale = (vp - vm) / (2.0 * eps);
        for i in 0..dim {
            grad[i] += scale * delta[i];
        }
    }
    let inv = 1.0 / num_dirs as f32;
    grad.iter_mut().for_each(|g| *g *= inv);
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blueness_grad_sparsity() {
        let r = BluenessReward { scale: 1.0 };
        let z = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let g = r.grad_wrt_z(&z);
        assert!((g[2] - 1.0).abs() < 1e-6);
        assert!((g[5] - 1.0).abs() < 1e-6);
        assert!(g[0].abs() < 1e-6);
    }
}
