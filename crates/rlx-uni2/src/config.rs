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

//! UNI2-h configuration.
//!
//! Mirrors the exact `timm.create_model("hf-hub:MahmoodLab/UNI2-h", ...)`
//! keyword arguments published on the model card:
//!
//! ```text
//!   img_size=224, patch_size=14, depth=24, num_heads=24,
//!   embed_dim=1536, init_values=1e-5, mlp_ratio=2.66667*2 (=16/3),
//!   num_classes=0, no_embed_class=True, reg_tokens=8,
//!   mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
//!   dynamic_img_size=True
//! ```

use serde::Deserialize;
use std::path::Path;

/// ImageNet-1k mean/std applied to RGB pixels in `[0, 1]` (same recipe
/// as the model card's `transforms.Normalize`).
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// UNI2-h ViT-H/14 configuration.
///
/// This is a DINOv2-family ViT with three deviations from a plain
/// DINOv2 encoder, all captured here:
///   1. **Packed SwiGLU MLP** (`timm.layers.SwiGLUPacked`, SiLU gate) in
///      place of the GELU FFN. The single `mlp.fc1` projection emits
///      [`Self::mlp_hidden_dim`] channels which are split in half; the
///      first half is the SiLU-activated value, the second the gate.
///   2. **8 register tokens** (`reg_tokens=8`), stored under the timm
///      `reg_token` parameter.
///   3. **`no_embed_class`** — the position embedding covers only the
///      patch tokens; the `[CLS]` and register tokens receive none.
#[derive(Debug, Clone, Deserialize)]
pub struct Uni2Config {
    /// Transformer width / embedding dimension (`embed_dim`); 1536 for UNI2-h.
    pub hidden_size: usize,
    /// Number of transformer blocks (`depth`); 24 for UNI2-h.
    pub num_hidden_layers: usize,
    /// Attention heads per block; 24 for UNI2-h (head_dim 64).
    pub num_attention_heads: usize,
    /// Square input resolution in pixels (224 = the checkpoint's native size).
    pub img_size: usize,
    /// Patch side length (14 → 16×16 = 256 patches at 224).
    pub patch_size: usize,
    /// Output width of the packed `mlp.fc1` (= `int(hidden * mlp_ratio)`).
    /// The SwiGLU inner width is half of this (see [`Self::swiglu_inner`]).
    #[serde(default = "default_uni2_mlp_hidden")]
    pub mlp_hidden_dim: usize,
    /// LayerNorm epsilon (timm ViT default `1e-6`).
    #[serde(default = "default_uni2_ln_eps")]
    pub layer_norm_eps: f64,
    /// Number of learnable register tokens (`reg_tokens`); 8 for UNI2-h.
    #[serde(default = "default_uni2_reg_tokens")]
    pub num_register_tokens: usize,
}

fn default_uni2_mlp_hidden() -> usize {
    8192
}
fn default_uni2_ln_eps() -> f64 {
    // timm's VisionTransformer defaults `norm_layer` to
    // `partial(nn.LayerNorm, eps=1e-6)`.
    1e-6
}
fn default_uni2_reg_tokens() -> usize {
    8
}

impl Uni2Config {
    /// Load a config from a JSON file whose keys match this struct's fields.
    ///
    /// Note: the timm checkpoint's own `config.json` is a nested
    /// `pretrained_cfg` and is **not** directly deserializable here — use
    /// [`Self::uni2_h`] for the published model.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    /// The published `MahmoodLab/UNI2-h` preset (ViT-H/14, 681M params).
    ///
    /// `img_size` must be a multiple of the patch size (14). The
    /// checkpoint is trained at 224; other resolutions require position
    /// embedding interpolation, which is not yet implemented.
    pub fn uni2_h(img_size: usize) -> Self {
        Self {
            hidden_size: 1536,
            num_hidden_layers: 24,
            num_attention_heads: 24,
            img_size,
            patch_size: 14,
            mlp_hidden_dim: 8192, // int(1536 * 16/3)
            layer_norm_eps: 1e-6,
            num_register_tokens: 8,
        }
    }

    /// SwiGLU inner width — the chunked half of `mlp.fc1`, i.e. the
    /// `mlp.fc2` input dimension.
    pub fn swiglu_inner(&self) -> usize {
        self.mlp_hidden_dim / 2
    }
    /// Per-head attention dimension (`hidden_size / num_attention_heads`).
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
    /// Flattened patch dimension (`3 · patch_size²`) fed to the patch projection.
    pub fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }
}
