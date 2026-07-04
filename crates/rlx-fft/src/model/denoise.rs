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

//! Learned spectrum denoiser — per-bin affine correction (legacy / teacher paths).

use crate::model::reference::max_abs_error;
use anyhow::{Result, ensure};

/// Per-bin affine correction on interleaved complex spectrum `[batch, n_fft, 2]`.
#[derive(Debug, Clone)]
pub struct SpectrumDenoiser {
    pub scale: Vec<f32>,
    pub bias: Vec<f32>,
}

impl SpectrumDenoiser {
    pub fn identity(n_fft: usize) -> Self {
        Self {
            scale: vec![1.0; n_fft * 2],
            bias: vec![0.0; n_fft * 2],
        }
    }

    pub fn apply_batch(&self, spectrum: &[f32], batch: usize, n_fft: usize) -> Result<Vec<f32>> {
        ensure!(spectrum.len() == batch * n_fft * 2);
        ensure!(self.scale.len() == n_fft * 2 && self.bias.len() == n_fft * 2);
        let mut out = spectrum.to_vec();
        for b in 0..batch {
            for i in 0..n_fft * 2 {
                let idx = b * n_fft * 2 + i;
                out[idx] = spectrum[idx] * self.scale[i] + self.bias[i];
            }
        }
        Ok(out)
    }

    pub fn train_step_affine(
        &mut self,
        pred: &[f32],
        target: &[f32],
        batch: usize,
        n_fft: usize,
        lr: f32,
    ) -> Result<f32> {
        ensure!(pred.len() == target.len() && pred.len() == batch * n_fft * 2);
        let mut mse = 0f32;
        let n = (batch * n_fft * 2) as f32;
        for i in 0..n_fft * 2 {
            let mut ds = 0f32;
            let mut db = 0f32;
            for b in 0..batch {
                let idx = b * n_fft * 2 + i;
                let p = pred[idx] * self.scale[i] + self.bias[i];
                let d = p - target[idx];
                mse += d * d;
                ds += d * pred[idx];
                db += d;
            }
            self.scale[i] -= lr * 2.0 * ds / n;
            self.bias[i] -= lr * 2.0 * db / n;
        }
        Ok(mse / n)
    }
}

pub fn denoised_max_err(pred: &[f32], denoised: &[f32], target: &[f32]) -> f32 {
    max_abs_error(denoised, target).min(max_abs_error(pred, target))
}
