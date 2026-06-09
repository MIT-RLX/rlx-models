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

//! SAM v1 model configuration. Mirrors Meta's `segment-anything` Python
//! reference and candle's `segment_anything` module.
//!
//! Three ViT image-encoder variants (B/L/H) and one MobileSAM TinyViT
//! variant. Decoder + prompt-encoder hyperparameters are fixed across
//! all variants.

use serde::Deserialize;

/// ImageNet mean/std applied to raw 0..255 pixel values *before* the
/// /255 scaling — SAM uses unnormalized pixel values directly, unlike
/// most ViTs. Match `sam.rs::preprocess()` in candle exactly.
pub const SAM_PIXEL_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
pub const SAM_PIXEL_STD: [f32; 3] = [58.395, 57.12, 57.375];

/// Target image side after preprocessing. SAM always operates at
/// 1024×1024 internally; smaller inputs are resized + zero-padded.
pub const SAM_IMG_SIZE: usize = 1024;
pub const SAM_PATCH_SIZE: usize = 16;
/// Spatial resolution of image embeddings produced by the encoder.
pub const SAM_EMBED_HW: usize = SAM_IMG_SIZE / SAM_PATCH_SIZE; // 64

/// Channel count of the embeddings emitted by the encoder neck and
/// consumed by the prompt encoder + mask decoder.
pub const SAM_PROMPT_EMBED_DIM: usize = 256;

/// Encoder configuration — ViT-B/L/H or TinyViT variants.
#[derive(Debug, Clone, Deserialize)]
pub struct SamEncoderConfig {
    pub encoder_kind: EncoderKind,
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    /// Per-block flag: blocks listed here use global attention
    /// (no windowing); all others use windowed attention with
    /// `window_size`.
    pub global_attn_indexes: Vec<usize>,
    pub window_size: usize,
    pub use_rel_pos: bool,
    pub use_abs_pos: bool,
    pub qkv_bias: bool,
    /// LayerNorm eps used throughout the encoder.
    pub layer_norm_eps: f64,
    /// Channel count of the final image embeddings (after the neck).
    pub out_chans: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum EncoderKind {
    ViT,
    TinyViT,
}

impl SamEncoderConfig {
    /// SAM ViT-B (default, ~91 M params).
    pub fn vit_b() -> Self {
        Self {
            encoder_kind: EncoderKind::ViT,
            embed_dim: 768,
            depth: 12,
            num_heads: 12,
            global_attn_indexes: vec![2, 5, 8, 11],
            window_size: 14,
            use_rel_pos: true,
            use_abs_pos: true,
            qkv_bias: true,
            layer_norm_eps: 1e-6,
            out_chans: SAM_PROMPT_EMBED_DIM,
        }
    }
    /// SAM ViT-L (~308 M params).
    pub fn vit_l() -> Self {
        Self {
            embed_dim: 1024,
            depth: 24,
            num_heads: 16,
            global_attn_indexes: vec![5, 11, 17, 23],
            ..Self::vit_b()
        }
    }
    /// SAM ViT-H (~632 M params).
    pub fn vit_h() -> Self {
        Self {
            embed_dim: 1280,
            depth: 32,
            num_heads: 16,
            global_attn_indexes: vec![7, 15, 23, 31],
            ..Self::vit_b()
        }
    }

    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }
    pub fn num_patches_per_side(&self) -> usize {
        SAM_EMBED_HW
    }
}

/// Mask decoder configuration. Same across SAM variants.
#[derive(Debug, Clone)]
pub struct SamDecoderConfig {
    pub transformer_dim: usize,
    pub transformer_depth: usize,
    pub transformer_num_heads: usize,
    pub transformer_mlp_dim: usize,
    /// 4 = 1 IoU token + 3 mask tokens; downstream code picks one or
    /// all three depending on `multimask_output`.
    pub num_mask_tokens: usize,
    pub iou_head_depth: usize,
    pub iou_head_hidden_dim: usize,
    pub layer_norm_eps: f64,
}

impl Default for SamDecoderConfig {
    fn default() -> Self {
        Self {
            transformer_dim: SAM_PROMPT_EMBED_DIM,
            transformer_depth: 2,
            transformer_num_heads: 8,
            transformer_mlp_dim: 2048,
            num_mask_tokens: 4,
            iou_head_depth: 3,
            iou_head_hidden_dim: SAM_PROMPT_EMBED_DIM,
            layer_norm_eps: 1e-6,
        }
    }
}

/// Top-level SAM configuration — encoder + decoder + a few constants
/// shared between them.
#[derive(Debug, Clone)]
pub struct SamConfig {
    pub encoder: SamEncoderConfig,
    pub decoder: SamDecoderConfig,
}

impl SamConfig {
    pub fn vit_b() -> Self {
        Self {
            encoder: SamEncoderConfig::vit_b(),
            decoder: SamDecoderConfig::default(),
        }
    }
    pub fn vit_l() -> Self {
        Self {
            encoder: SamEncoderConfig::vit_l(),
            decoder: SamDecoderConfig::default(),
        }
    }
    pub fn vit_h() -> Self {
        Self {
            encoder: SamEncoderConfig::vit_h(),
            decoder: SamDecoderConfig::default(),
        }
    }
}
