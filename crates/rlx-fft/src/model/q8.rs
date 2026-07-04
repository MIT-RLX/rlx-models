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

//! Q8 quantized twiddles for inference (Tier B).

use crate::model::butterfly::butterfly_forward_real_batch;
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct Q8Twiddles {
    pub q: Vec<i8>,
    pub scale: f32,
}

impl Q8Twiddles {
    pub fn from_f32(tw: &[f32]) -> Self {
        let max_abs = tw.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-8);
        let scale = max_abs / 127.0;
        let q = tw
            .iter()
            .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        Self { q, scale }
    }

    pub fn dequant(&self) -> Vec<f32> {
        self.q.iter().map(|&v| v as f32 * self.scale).collect()
    }

    pub fn forward_real_batch(
        &self,
        signal: &[f32],
        batch: usize,
        n_fft: usize,
    ) -> Result<Vec<f32>> {
        let tw = self.dequant();
        butterfly_forward_real_batch(signal, &tw, batch, n_fft)
    }
}

pub fn q8_max_twiddle_error(tw: &[f32]) -> f32 {
    let q8 = Q8Twiddles::from_f32(tw);
    let dq = q8.dequant();
    tw.iter()
        .zip(dq.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}
