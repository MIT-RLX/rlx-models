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

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Flux2VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub block_out_channels: Vec<usize>,
    #[serde(default = "default_act_fn")]
    pub act_fn: String,
    #[serde(default = "default_batch_norm_eps")]
    pub batch_norm_eps: f32,
    #[serde(default = "default_mid_block_add_attention")]
    pub mid_block_add_attention: bool,
    #[serde(default = "default_use_post_quant_conv")]
    pub use_post_quant_conv: bool,
    #[serde(default)]
    pub scaling_factor: f32,
    #[serde(default)]
    pub shift_factor: f32,
}

fn default_act_fn() -> String {
    "silu".into()
}
fn default_batch_norm_eps() -> f32 {
    1e-4
}
fn default_mid_block_add_attention() -> bool {
    true
}
fn default_use_post_quant_conv() -> bool {
    true
}

impl Flux2VaeConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).with_context(|| format!("reading {path:?}"))?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn flux2_klein() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 32,
            layers_per_block: 2,
            norm_num_groups: 32,
            block_out_channels: vec![128, 256, 512, 512],
            act_fn: "silu".into(),
            batch_norm_eps: 1e-4,
            mid_block_add_attention: true,
            use_post_quant_conv: true,
            scaling_factor: 1.0,
            shift_factor: 0.0,
        }
    }

    /// Tiny VAE for unit tests (no mid attention to shrink graph).
    pub fn tiny() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            layers_per_block: 1,
            norm_num_groups: 2,
            block_out_channels: vec![8, 16],
            act_fn: "silu".into(),
            batch_norm_eps: 1e-4,
            mid_block_add_attention: false,
            use_post_quant_conv: true,
            scaling_factor: 1.0,
            shift_factor: 0.0,
        }
    }

    pub fn bn_channels(&self) -> usize {
        4 * self.latent_channels
    }

    /// Spatial downsample factor from RGB to latent (one halving per encoder down block except the last).
    pub fn encode_spatial_stride(&self) -> usize {
        1 << self.block_out_channels.len().saturating_sub(1)
    }
}
