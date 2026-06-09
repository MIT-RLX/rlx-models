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

//! Per-layer decode state for [`crate::Mamba1Block::step`].

/// One layer's rolling state. `conv` is the most-recent `d_conv` activations
/// per channel (oldest at index 0); `ssm` is the hidden SSM state.
#[derive(Debug, Clone)]
pub struct Mamba1Cache {
    /// Flat `[batch, d_inner, d_conv]`, row-major.
    pub conv: Vec<f32>,
    /// Flat `[batch, d_inner, d_state]`, row-major.
    pub ssm: Vec<f32>,
    pub batch: usize,
    pub d_inner: usize,
    pub d_conv: usize,
    pub d_state: usize,
}

impl Mamba1Cache {
    pub fn zeros(batch: usize, d_inner: usize, d_conv: usize, d_state: usize) -> Self {
        Self {
            conv: vec![0.0; batch * d_inner * d_conv],
            ssm: vec![0.0; batch * d_inner * d_state],
            batch,
            d_inner,
            d_conv,
            d_state,
        }
    }
}

/// One cache per layer.
#[derive(Debug, Clone)]
pub struct Mamba1Caches {
    pub caches: Vec<Mamba1Cache>,
}

impl Mamba1Caches {
    pub fn zeros(
        n_layer: usize,
        batch: usize,
        d_inner: usize,
        d_conv: usize,
        d_state: usize,
    ) -> Self {
        Self {
            caches: (0..n_layer)
                .map(|_| Mamba1Cache::zeros(batch, d_inner, d_conv, d_state))
                .collect(),
        }
    }
}
