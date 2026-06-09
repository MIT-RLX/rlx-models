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

//! Nemotron-3 Nano Omni vision-tower config.
//!
//! NVIDIA's Nemotron Omni accepts text + vision + audio. The vision
//! tower is a SigLIP-variant (NVIDIA's RADIO-derived encoder uses the
//! same separate-Q/K/V pre-LN ViT block shape). The audio side
//! delegates to a Whisper-shaped mel encoder loaded separately.

use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct NemotronOmniVisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub layer_norm_eps: f64,
    pub projector_output_dim: usize,
}

impl NemotronOmniVisionConfig {
    pub fn num_patches(&self) -> usize {
        let p = self.image_size / self.patch_size;
        p * p
    }
    pub fn patch_dim(&self) -> usize {
        self.num_channels * self.patch_size * self.patch_size
    }
    pub fn seq_len(&self) -> usize {
        self.num_patches()
    }
}

#[derive(Debug, Deserialize)]
struct HfVision {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    image_size: usize,
    patch_size: usize,
    #[serde(default = "default_channels")]
    num_channels: usize,
    #[serde(default = "default_eps")]
    layer_norm_eps: f64,
}
fn default_channels() -> usize {
    3
}
fn default_eps() -> f64 {
    1e-6
}

#[derive(Debug, Deserialize)]
struct HfText {
    hidden_size: usize,
}

#[derive(Debug, Deserialize)]
struct HfTop {
    vision_config: HfVision,
    text_config: HfText,
}

impl NemotronOmniVisionConfig {
    pub fn from_hf_config_json(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("rlx-nemotron-omni: read {path:?}: {e}"))?;
        let cfg: HfTop = serde_json::from_str(&raw)
            .map_err(|e| anyhow!("rlx-nemotron-omni: parse {path:?}: {e}"))?;
        Ok(Self {
            hidden_size: cfg.vision_config.hidden_size,
            num_hidden_layers: cfg.vision_config.num_hidden_layers,
            num_attention_heads: cfg.vision_config.num_attention_heads,
            intermediate_size: cfg.vision_config.intermediate_size,
            image_size: cfg.vision_config.image_size,
            patch_size: cfg.vision_config.patch_size,
            num_channels: cfg.vision_config.num_channels,
            layer_norm_eps: cfg.vision_config.layer_norm_eps,
            projector_output_dim: cfg.text_config.hidden_size,
        })
    }
}
