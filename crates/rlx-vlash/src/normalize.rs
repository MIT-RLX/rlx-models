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

//! Host-side MEAN_STD normalization for robot state (input) and actions
//! (output), matching `policies/normalize.py`.
//!
//! ```text
//!   normalize:   x' = (x - mean) / (std + eps)
//!   unnormalize: x  = x' * std + mean
//! ```
//! with `eps = 1e-8`. Stats live at the *raw* feature dimension; the model
//! pads state to `max_state_dim` and predicts padded actions, so normalization
//! runs on the raw dims and padding happens separately (see [`pad_to`]).

use anyhow::{Result, anyhow};
use rlx_core::weight_map::WeightMap;

const EPS: f32 = 1e-8;

/// MEAN_STD statistics for one feature (raw, un-padded dimension).
#[derive(Debug, Clone)]
pub struct MeanStd {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl MeanStd {
    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// `(x - mean) / (std + eps)`.
    pub fn normalize(&self, x: &[f32]) -> Vec<f32> {
        x.iter()
            .zip(self.mean.iter().zip(self.std.iter()))
            .map(|(v, (m, s))| (v - m) / (s + EPS))
            .collect()
    }

    /// `x * std + mean` (applied per trailing feature dim; `x` may be a flat
    /// `[n · dim]` chunk of actions).
    pub fn unnormalize(&self, x: &[f32]) -> Vec<f32> {
        let d = self.dim();
        x.iter()
            .enumerate()
            .map(|(i, v)| {
                let j = i % d;
                v * self.std[j] + self.mean[j]
            })
            .collect()
    }
}

/// Robot state (input) + action (output) normalization stats extracted from a
/// checkpoint. Either may be absent (identity) if the checkpoint stores stats
/// out-of-band; callers then pass pre-normalized values.
#[derive(Debug, Clone, Default)]
pub struct Normalization {
    pub state: Option<MeanStd>,
    pub action: Option<MeanStd>,
}

impl Normalization {
    /// Read `norm.state.{mean,std}` / `norm.action.{mean,std}` (canonical names
    /// produced by [`crate::weights::canonical_key`]) from a remapped map.
    pub fn from_weight_map(wm: &WeightMap) -> Self {
        let ms = |m: &str, s: &str| -> Option<MeanStd> {
            let mean = wm.get(m)?.0.to_vec();
            let std = wm.get(s)?.0.to_vec();
            if mean.is_empty() || mean.len() != std.len() {
                return None;
            }
            // Uninitialized lerobot buffers are filled with inf; treat as absent.
            if mean.iter().chain(std.iter()).any(|v| !v.is_finite()) {
                return None;
            }
            Some(MeanStd { mean, std })
        };
        Normalization {
            state: ms("norm.state.mean", "norm.state.std"),
            action: ms("norm.action.mean", "norm.action.std"),
        }
    }

    /// Normalize raw state (no-op if stats absent).
    pub fn normalize_state(&self, state: &[f32]) -> Vec<f32> {
        match &self.state {
            Some(s) => s.normalize(state),
            None => state.to_vec(),
        }
    }

    /// Unnormalize actions `[n · action_dim]` (no-op if stats absent).
    pub fn unnormalize_action(&self, action: &[f32]) -> Vec<f32> {
        match &self.action {
            Some(a) => a.unnormalize(action),
            None => action.to_vec(),
        }
    }

    /// Raw action dimension, if known from stats.
    pub fn action_dim(&self) -> Option<usize> {
        self.action.as_ref().map(|a| a.dim())
    }
}

/// Zero-pad the trailing dimension of `x` (`rows × cur`) to `new` columns.
pub fn pad_to(x: &[f32], cur: usize, new: usize) -> Result<Vec<f32>> {
    if cur > new {
        return Err(anyhow!("cannot pad {cur} → {new} (shrink)"));
    }
    if x.len() % cur != 0 {
        return Err(anyhow!("pad_to: len {} not divisible by cur {cur}", x.len()));
    }
    let rows = x.len() / cur;
    let mut out = vec![0f32; rows * new];
    for r in 0..rows {
        out[r * new..r * new + cur].copy_from_slice(&x[r * cur..r * cur + cur]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_mean_std() {
        let ms = MeanStd {
            mean: vec![1.0, -2.0, 0.5],
            std: vec![2.0, 0.5, 4.0],
        };
        let x = vec![3.0, -1.0, 2.5];
        let n = ms.normalize(&x);
        let r = ms.unnormalize(&n);
        for (a, b) in x.iter().zip(r.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn unnormalize_tiles_over_rows() {
        let ms = MeanStd {
            mean: vec![0.0, 10.0],
            std: vec![1.0, 1.0],
        };
        // two rows of dim 2
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let r = ms.unnormalize(&x);
        assert_eq!(r, vec![1.0, 12.0, 3.0, 14.0]);
    }

    #[test]
    fn pad_expands_trailing_dim() {
        let x = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows × 2
        let p = pad_to(&x, 2, 4).unwrap();
        assert_eq!(p, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
    }
}
