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

//! Mamba1 hyperparameters. Mirrors `burn_mamba::mamba1::Mamba1Config` /
//! `Mamba1NetworkConfig` so configs can be ported field-for-field.

#[derive(Debug, Clone)]
pub struct Mamba1Config {
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
    /// Optional override; defaults to `d_model.div_ceil(d_state)` to
    /// match burn-mamba.
    pub dt_rank: Option<usize>,
    /// Optional override; defaults to `expand * d_model`.
    pub d_inner: Option<usize>,
    pub conv_bias: bool,
    pub bias: bool,
}

impl Mamba1Config {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            dt_rank: None,
            d_inner: None,
            conv_bias: true,
            bias: false,
        }
    }
    pub fn d_inner(&self) -> usize {
        self.d_inner.unwrap_or(self.expand * self.d_model)
    }
    pub fn dt_rank(&self) -> usize {
        self.dt_rank.unwrap_or(self.d_model.div_ceil(self.d_state))
    }
}

#[derive(Debug, Clone)]
pub struct Mamba1NetworkConfig {
    pub n_layer: usize,
    pub vocab_size: usize,
    pub pad_vocab_size_multiple: usize,
    pub mamba_block: Mamba1Config,
    /// When `true`, tie lm_head to the embedding (transpose) instead of
    /// owning a separate weight.
    pub tied_lm_head: bool,
}

impl Mamba1NetworkConfig {
    pub fn padded_vocab_size(&self) -> usize {
        if self.vocab_size.is_multiple_of(self.pad_vocab_size_multiple) {
            self.vocab_size
        } else {
            ((self.vocab_size / self.pad_vocab_size_multiple) + 1) * self.pad_vocab_size_multiple
        }
    }
}
