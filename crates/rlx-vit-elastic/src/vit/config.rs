// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Generic Vision Transformer configuration covering both backbones used by
//! SnapViT / GLARE here:
//!
//!   - **DINO ViT-B/16** (`facebook/dino-vitb16`) — a plain ViT: learned pos
//!     embedding over `[CLS] + patches`, GELU FFN, no LayerScale, no register
//!     tokens. The SSL backbone both papers target.
//!   - **UNI2-h** (`MahmoodLab/UNI2-h`) — DINOv2-family ViT-H/14: packed
//!     SwiGLU FFN, LayerScale, 8 register tokens, `no_embed_class` pos
//!     embedding. Reuses [`rlx_uni2`] presets/preprocess.
//!
//! Weight keys are the timm-style canonical names shared with `rlx-uni2`
//! (`blocks.{i}.attn.qkv`, `.attn.proj`, `.norm1/2`, `.mlp.fc1/2`,
//! `.ls1/2.gamma`, `cls_token`, `reg_token`, `pos_embed`, `patch_embed.proj`,
//! `norm`); non-timm checkpoints are remapped to these by the loader.

use serde::{Deserialize, Serialize};

/// ImageNet-1k normalization (shared with `rlx-uni2`).
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Feed-forward block variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FfnKind {
    /// Standard ViT MLP: `fc2(gelu(fc1(x)))`. `mlp.fc1 [mlp_hidden, hidden]`,
    /// `mlp.fc2 [hidden, mlp_hidden]`. Inner width = `mlp_hidden_dim`.
    Gelu,
    /// timm `SwiGLUPacked` (UNI2): `fc2(silu(value) * gate)` where
    /// `value,gate = chunk(fc1(x), 2)`. `mlp.fc1 [2*inner, hidden]`,
    /// `mlp.fc2 [hidden, inner]`. Inner width = `mlp_hidden_dim / 2`.
    PackedSwiGLU,
}

/// A Vision Transformer encoder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitConfig {
    /// Transformer width / embedding dimension (`embed_dim`).
    pub hidden_size: usize,
    /// Number of transformer blocks (`depth`).
    pub num_hidden_layers: usize,
    /// Attention heads per block (`head_dim = hidden_size / heads`).
    pub num_attention_heads: usize,
    /// Square input resolution in pixels.
    pub img_size: usize,
    /// Patch side length.
    pub patch_size: usize,
    /// Output width of `mlp.fc1` (see [`FfnKind`] for how it maps to inner width).
    pub mlp_hidden_dim: usize,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f64,
    /// Number of learnable register tokens (`reg_tokens`).
    pub num_register_tokens: usize,
    /// Whether each block applies LayerScale (`ls1`/`ls2.gamma`).
    pub layer_scale: bool,
    /// `no_embed_class`: when true the position embedding covers **patches
    /// only** (UNI2); when false it covers `[CLS] + patches` (plain ViT/DINO).
    pub no_embed_class: bool,
    /// Feed-forward variant.
    pub ffn_kind: FfnKind,
    /// Whether `attn.qkv` carries a bias (both backbones: true).
    pub qkv_bias: bool,
}

impl VitConfig {
    /// `facebook/dino-vitb16` — DINO ViT-B/16 (the paper's SSL backbone).
    pub fn dino_vitb16() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            img_size: 224,
            patch_size: 16,
            mlp_hidden_dim: 3072,
            // HF `ViTModel` uses 1e-12; pinned against a reference dump in Phase 1.
            layer_norm_eps: 1e-12,
            num_register_tokens: 0,
            layer_scale: false,
            no_embed_class: false,
            ffn_kind: FfnKind::Gelu,
            qkv_bias: true,
        }
    }

    /// `MahmoodLab/UNI2-h` — DINOv2-family ViT-H/14 (mirrors [`rlx_uni2`]).
    pub fn uni2_h(img_size: usize) -> Self {
        Self {
            hidden_size: 1536,
            num_hidden_layers: 24,
            num_attention_heads: 24,
            img_size,
            patch_size: 14,
            mlp_hidden_dim: 8192,
            layer_norm_eps: 1e-6,
            num_register_tokens: 8,
            layer_scale: true,
            no_embed_class: true,
            ffn_kind: FfnKind::PackedSwiGLU,
            qkv_bias: true,
        }
    }

    /// A tiny real-topology ViT for fast cross-backend tests (plain-ViT shape).
    pub fn synthetic() -> Self {
        Self {
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            img_size: 32,
            patch_size: 16,
            mlp_hidden_dim: 64,
            layer_norm_eps: 1e-6,
            num_register_tokens: 0,
            layer_scale: false,
            no_embed_class: false,
            ffn_kind: FfnKind::Gelu,
            qkv_bias: true,
        }
    }

    /// A tiny UNI2-shaped config (LayerScale + packed SwiGLU + registers).
    pub fn synthetic_uni2() -> Self {
        Self {
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            img_size: 28,
            patch_size: 14,
            mlp_hidden_dim: 64, // inner = 32
            layer_norm_eps: 1e-6,
            num_register_tokens: 8,
            layer_scale: true,
            no_embed_class: true,
            ffn_kind: FfnKind::PackedSwiGLU,
            qkv_bias: true,
        }
    }

    /// SwiGLU / MLP inner width (the `mlp.fc2` input dimension).
    pub fn ffn_inner(&self) -> usize {
        match self.ffn_kind {
            FfnKind::Gelu => self.mlp_hidden_dim,
            FfnKind::PackedSwiGLU => self.mlp_hidden_dim / 2,
        }
    }
    /// Per-head attention dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    /// Number of patch tokens (`(img_size / patch_size)²`).
    pub fn num_patches(&self) -> usize {
        let n = self.img_size / self.patch_size;
        n * n
    }
    /// Encoder sequence length: `[CLS] + register_tokens + patches`.
    pub fn seq_len(&self) -> usize {
        1 + self.num_register_tokens + self.num_patches()
    }
    /// Flattened patch dimension (`3 · patch_size²`).
    pub fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }
    /// Row index (in the assembled sequence) of the first patch token.
    pub fn patch_row_base(&self) -> usize {
        1 + self.num_register_tokens
    }
}
