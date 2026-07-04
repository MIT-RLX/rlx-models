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

//! Banded wide-sparse correction after pruned butterfly (ternary distill).

use anyhow::{Result, ensure};

/// Mel-path band half-width (fused compile still one `matmul`).
pub const DEFAULT_BAND_RADIUS: usize = 32;
/// Spectrum-path band half-width — wider to absorb pruned-butterfly error.
pub const SPECTRUM_BAND_RADIUS: usize = 96;

fn band_radius(n_bins: usize, half_width: usize) -> usize {
    half_width.min(n_bins.saturating_sub(1) / 2)
}

/// Row-packed banded map: `out[i] = bias[i] + Σ_t W[i,t] · x[i+t-radius]`.
#[derive(Debug, Clone)]
pub struct BandedCorrector {
    pub n_bins: usize,
    pub band_width: usize,
    pub radius: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl BandedCorrector {
    pub fn identity(n_fft: usize) -> Self {
        Self::identity_with_radius(n_fft, DEFAULT_BAND_RADIUS)
    }

    /// Wide band for raw spectrum / denoise / q8 / welch correction.
    pub fn identity_spectrum(n_fft: usize) -> Self {
        Self::identity_with_radius(n_fft, SPECTRUM_BAND_RADIUS)
    }

    /// Full band (= dense n×n matmul when baked).
    pub fn identity_dense(n_fft: usize) -> Self {
        let n_bins = n_fft * 2;
        Self::identity_with_radius(n_fft, n_bins.saturating_sub(1) / 2)
    }

    fn identity_with_radius(n_fft: usize, half_width: usize) -> Self {
        let n_bins = n_fft * 2;
        let radius = band_radius(n_bins, half_width);
        let band_width = radius * 2 + 1;
        let mut weights = vec![0.0; n_bins * band_width];
        for i in 0..n_bins {
            weights[i * band_width + radius] = 1.0;
        }
        Self {
            n_bins,
            band_width,
            radius,
            weights,
            bias: vec![0.0; n_bins],
        }
    }

    #[inline]
    fn in_tap(&self, out_i: usize, t: usize) -> Option<usize> {
        let j = out_i as isize + t as isize - self.radius as isize;
        (j >= 0 && (j as usize) < self.n_bins).then_some(j as usize)
    }

    /// Fused apply — single pass, no input clone.
    pub fn apply_batch(&self, spectrum: &[f32], batch: usize, n_fft: usize) -> Result<Vec<f32>> {
        ensure!(spectrum.len() == batch * n_fft * 2 && self.n_bins == n_fft * 2);
        let n_bins = self.n_bins;
        let bw = self.band_width;
        let mut out = vec![0f32; spectrum.len()];
        for b in 0..batch {
            let base = b * n_bins;
            for i in 0..n_bins {
                let mut acc = self.bias[i];
                let row = i * bw;
                for t in 0..bw {
                    if let Some(j) = self.in_tap(i, t) {
                        acc += self.weights[row + t] * spectrum[base + j];
                    }
                }
                out[base + i] = acc;
            }
        }
        Ok(out)
    }

    /// Dense RHS for fused compile `matmul`: `[batch,n] @ [n,n]`.
    pub fn dense_rhs_matrix(&self) -> Vec<f32> {
        let n = self.n_bins;
        let mut dense = vec![0f32; n * n];
        for i in 0..n {
            let row = i * self.band_width;
            for t in 0..self.band_width {
                if let Some(j) = self.in_tap(i, t) {
                    dense[j * n + i] = self.weights[row + t];
                }
            }
        }
        dense
    }

    pub fn dense_rhs_with_freq_mask(&self, freq_mask: &[f32]) -> Vec<f32> {
        debug_assert_eq!(freq_mask.len(), self.n_bins);
        let n = self.n_bins;
        let mut dense = vec![0f32; n * n];
        for i in 0..n {
            let row = i * self.band_width;
            for t in 0..self.band_width {
                if let Some(j) = self.in_tap(i, t) {
                    dense[j * n + i] = self.weights[row + t] * freq_mask[j];
                }
            }
        }
        dense
    }

    pub fn train_step_mse(
        &mut self,
        input: &[f32],
        target: &[f32],
        batch: usize,
        n_fft: usize,
        lr: f32,
    ) -> Result<f32> {
        ensure!(input.len() == target.len() && input.len() == batch * n_fft * 2);
        ensure!(self.n_bins == n_fft * 2);
        let n_bins = self.n_bins;
        let bw = self.band_width;
        let mut mse = 0f32;
        let n = (batch * n_bins) as f32;
        let mut dw = vec![0f32; self.weights.len()];
        let mut db = vec![0f32; n_bins];
        for b in 0..batch {
            let base = b * n_bins;
            for i in 0..n_bins {
                let mut acc = self.bias[i];
                let row = i * bw;
                for t in 0..bw {
                    if let Some(j) = self.in_tap(i, t) {
                        acc += self.weights[row + t] * input[base + j];
                    }
                }
                let d = acc - target[base + i];
                mse += d * d;
                db[i] += d;
                for t in 0..bw {
                    if let Some(j) = self.in_tap(i, t) {
                        dw[row + t] += d * input[base + j];
                    }
                }
            }
        }
        for i in 0..n_bins {
            self.bias[i] -= lr * 2.0 * db[i] / n;
        }
        for k in 0..self.weights.len() {
            self.weights[k] -= lr * 2.0 * dw[k] / n;
        }
        Ok(mse / n)
    }

    /// Backprop mel loss gradient through the band (updates weights + bias).
    pub fn train_step_spectrum_grad(
        &mut self,
        input: &[f32],
        spec_grad: &[f32],
        batch: usize,
        n_fft: usize,
        lr: f32,
    ) {
        let n_bins = n_fft * 2;
        let bw = self.band_width;
        let n = (batch * n_bins) as f32;
        let mut dw = vec![0f32; self.weights.len()];
        let mut db = vec![0f32; n_bins];
        for b in 0..batch {
            let base = b * n_bins;
            for i in 0..n_bins {
                let g = spec_grad[base + i] / n.max(1.0);
                db[i] += g;
                let row = i * bw;
                for t in 0..bw {
                    if let Some(j) = self.in_tap(i, t) {
                        dw[row + t] += g * input[base + j];
                    }
                }
            }
        }
        for i in 0..n_bins {
            self.bias[i] -= lr * db[i];
        }
        for k in 0..self.weights.len() {
            self.weights[k] -= lr * dw[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::reference::max_abs_error;

    #[test]
    fn banded_identity_passthrough() {
        let c = BandedCorrector::identity(64);
        let x = vec![1.0; 128];
        let y = c.apply_batch(&x, 1, 64).unwrap();
        assert!(max_abs_error(&y, &x) < 1e-5);
    }

    #[test]
    fn dense_rhs_matches_eager() {
        let mut c = BandedCorrector::identity(32);
        c.weights[10 * c.band_width + c.radius] = 0.5;
        let x: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let y = c.apply_batch(&x, 1, 32).unwrap();
        let w = c.dense_rhs_matrix();
        let n = 64usize;
        let mut mm = vec![0f32; n];
        for i in 0..n {
            let mut s = 0f32;
            for j in 0..n {
                s += x[j] * w[j * n + i];
            }
            mm[i] = s + c.bias[i];
        }
        assert!(max_abs_error(&y, &mm) < 1e-4);
    }
}
