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

//! SAM 3 configuration.
//!
//! The defaults mirror `facebookresearch/sam3::model_builder` for the
//! base SAM3 release. SAM3.1 multiplex is a distinct architecture and is
//! intentionally not represented by this config.

use serde::Deserialize;

/// SAM3 normalizes RGB values after scaling to `[0, 1]`.
pub const SAM3_PIXEL_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const SAM3_PIXEL_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Base SAM3 image side used by the public model builder.
pub const SAM3_IMG_SIZE: usize = 1008;
pub const SAM3_PATCH_SIZE: usize = 14;
pub const SAM3_PATCH_GRID: usize = SAM3_IMG_SIZE / SAM3_PATCH_SIZE; // 72
pub const SAM3_VISION_DIM: usize = 1024;
pub const SAM3_DET_DIM: usize = 256;

#[derive(Debug, Clone, Deserialize)]
pub struct Sam3VitConfig {
    pub img_size: usize,
    pub pretrain_img_size: usize,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub qkv_bias: bool,
    pub bias_patch_embed: bool,
    pub use_abs_pos: bool,
    pub tile_abs_pos: bool,
    pub use_rope: bool,
    pub use_interp_rope: bool,
    pub window_size: usize,
    pub global_att_blocks: Vec<usize>,
    pub layer_norm_eps: f64,
}

impl Sam3VitConfig {
    pub fn base() -> Self {
        Self {
            img_size: SAM3_IMG_SIZE,
            pretrain_img_size: 336,
            patch_size: SAM3_PATCH_SIZE,
            embed_dim: SAM3_VISION_DIM,
            depth: 32,
            num_heads: 16,
            mlp_ratio: 4.625,
            qkv_bias: true,
            bias_patch_embed: false,
            use_abs_pos: true,
            tile_abs_pos: true,
            use_rope: true,
            use_interp_rope: true,
            window_size: 24,
            global_att_blocks: vec![7, 15, 23, 31],
            layer_norm_eps: 1e-6,
        }
    }

    pub fn patch_grid(&self) -> usize {
        self.img_size / self.patch_size
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sam3TextConfig {
    pub d_model: usize,
    pub width: usize,
    pub heads: usize,
    pub layers: usize,
}

impl Default for Sam3TextConfig {
    fn default() -> Self {
        Self {
            d_model: SAM3_DET_DIM,
            width: 1024,
            heads: 16,
            layers: 24,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sam3DetectorConfig {
    pub d_model: usize,
    pub num_queries: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub transformer_heads: usize,
    pub dim_feedforward: usize,
    pub presence_token: bool,
    pub num_feature_levels: usize,
}

impl Default for Sam3DetectorConfig {
    fn default() -> Self {
        Self {
            d_model: SAM3_DET_DIM,
            num_queries: 200,
            encoder_layers: 6,
            decoder_layers: 6,
            transformer_heads: 8,
            dim_feedforward: 2048,
            presence_token: true,
            num_feature_levels: 1,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sam3TrackerConfig {
    pub image_size: usize,
    pub backbone_stride: usize,
    pub num_maskmem: usize,
    pub max_cond_frames_in_attn: usize,
    pub memory_dim: usize,
    pub transformer_dim: usize,
    pub transformer_layers: usize,
    pub feat_hw: usize,
}

impl Default for Sam3TrackerConfig {
    fn default() -> Self {
        Self {
            image_size: SAM3_IMG_SIZE,
            backbone_stride: SAM3_PATCH_SIZE,
            num_maskmem: 7,
            max_cond_frames_in_attn: 4,
            memory_dim: 64,
            transformer_dim: SAM3_DET_DIM,
            transformer_layers: 4,
            feat_hw: SAM3_PATCH_GRID,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sam3Config {
    pub vit: Sam3VitConfig,
    pub text: Sam3TextConfig,
    pub detector: Sam3DetectorConfig,
    pub tracker: Sam3TrackerConfig,
    pub enable_inst_interactivity: bool,
    pub enable_video: bool,
}

impl Sam3Config {
    pub fn base() -> Self {
        Self {
            vit: Sam3VitConfig::base(),
            text: Sam3TextConfig::default(),
            detector: Sam3DetectorConfig::default(),
            tracker: Sam3TrackerConfig::default(),
            enable_inst_interactivity: false,
            enable_video: true,
        }
    }
}
