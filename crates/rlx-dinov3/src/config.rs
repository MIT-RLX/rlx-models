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

//! DINOv3 configuration. Field names + defaults mirror HF
//! `transformers.models.dinov3_vit.DINOv3ViTConfig` so a checkpoint's
//! `config.json` deserializes directly via [`DinoV3Config::from_file`].

use serde::Deserialize;
use std::path::Path;

/// ImageNet-1k per-channel mean applied to RGB pixels in `[0, 1]`. DINOv3's
/// fast image processor uses these stats (same as DINOv2 / timm).
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet-1k per-channel standard deviation (see [`IMAGENET_MEAN`]).
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// DINOv3 ViT configuration.
///
/// Unlike DINOv2, DINOv3 carries **no learned `pos_embed`**: spatial
/// position is injected entirely by 2-D axial RoPE inside attention. It
/// also uses **separate q/k/v/o projections** (asymmetric bias:
/// `key_bias` defaults to false), LayerScale, and an optional gated
/// (GeGLU) MLP (`use_gated_mlp`).
#[derive(Debug, Clone, Deserialize)]
pub struct DinoV3Config {
    /// Encoder width (`E`). ViT-S/16 = 384, ViT-B/16 = 768, ViT-L/16 = 1024.
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    /// FFN inner width. For the gated (GeGLU) MLP this is the width of each
    /// of the gate/up projections.
    #[serde(default = "d_intermediate")]
    pub intermediate_size: usize,
    /// Number of transformer blocks.
    #[serde(default = "d_layers")]
    pub num_hidden_layers: usize,
    /// Attention heads per block (`head_dim = hidden_size / num_attention_heads`,
    /// and must be a multiple of 4 for the axial RoPE split).
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    /// Square input resolution the runner assembles patches / RoPE for. Must
    /// be a multiple of `patch_size`.
    #[serde(default = "d_image_size")]
    pub image_size: usize,
    /// Conv2d patch size / stride (16 for all published DINOv3 ViTs).
    #[serde(default = "d_patch_size")]
    pub patch_size: usize,
    /// Input channels (3 = RGB).
    #[serde(default = "d_num_channels")]
    pub num_channels: usize,
    /// FFN activation. `"gelu"` = exact erf GELU (the DINOv3 default);
    /// `"gelu_pytorch_tanh"` / `"gelu_new"` select the tanh approximation.
    #[serde(default = "d_hidden_act")]
    pub hidden_act: String,
    /// LayerNorm epsilon.
    #[serde(default = "d_ln_eps")]
    pub layer_norm_eps: f64,
    /// Base period `theta` for the 2-D axial RoPE frequencies.
    #[serde(default = "d_rope_theta")]
    pub rope_theta: f64,
    /// Whether the query projection has a bias (DINOv3: true).
    #[serde(default = "d_true")]
    pub query_bias: bool,
    /// Whether the key projection has a bias (DINOv3: **false**).
    #[serde(default)]
    pub key_bias: bool,
    /// Whether the value projection has a bias (DINOv3: true).
    #[serde(default = "d_true")]
    pub value_bias: bool,
    /// Whether the attention output projection has a bias (DINOv3: true).
    #[serde(default = "d_true")]
    pub proj_bias: bool,
    /// Whether the MLP projections have biases (DINOv3: true).
    #[serde(default = "d_true")]
    pub mlp_bias: bool,
    /// Initial LayerScale value (`lambda1 = layerscale_value · 1`).
    #[serde(default = "d_layerscale")]
    pub layerscale_value: f64,
    /// Use the gated (GeGLU) MLP (`down(act(gate(x)) · up(x))`) instead of
    /// the plain `down(act(up(x)))` MLP. Larger variants (ViT-H+/7B) set this.
    #[serde(default)]
    pub use_gated_mlp: bool,
    /// Number of register tokens prepended after the CLS token (DINOv3: 4).
    #[serde(default)]
    pub num_register_tokens: usize,
    /// When `true` (HF default), apply the learned final `norm.{weight,bias}`.
    /// Trellis's `DinoV3FeatureExtractor` instead uses non-affine
    /// `F.layer_norm` on the pre-norm activations — set this to `false` there.
    #[serde(default = "d_true")]
    pub final_layer_norm_affine: bool,
}

fn d_hidden() -> usize {
    384
}
fn d_intermediate() -> usize {
    1536
}
fn d_layers() -> usize {
    12
}
fn d_heads() -> usize {
    6
}
fn d_image_size() -> usize {
    224
}
fn d_patch_size() -> usize {
    16
}
fn d_num_channels() -> usize {
    3
}
fn d_hidden_act() -> String {
    "gelu".to_string()
}
fn d_ln_eps() -> f64 {
    1e-5
}
fn d_rope_theta() -> f64 {
    100.0
}
fn d_true() -> bool {
    true
}
fn d_layerscale() -> f64 {
    1.0
}

impl DinoV3Config {
    /// Deserialize a checkpoint's `config.json` (HF `DINOv3ViTConfig` fields).
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    /// Per-head width (`hidden_size / num_attention_heads`).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    /// Patches along one side (image is square in these presets).
    pub fn num_patches_side(&self) -> usize {
        self.image_size / self.patch_size
    }
    /// Total patch tokens (`num_patches_side²`).
    pub fn num_patches(&self) -> usize {
        let n = self.num_patches_side();
        n * n
    }
    /// Prefix tokens = CLS + register tokens (RoPE is *not* applied to these).
    pub fn num_prefix_tokens(&self) -> usize {
        1 + self.num_register_tokens
    }
    /// Full token sequence length (`prefix + patches`).
    pub fn seq_len(&self) -> usize {
        self.num_prefix_tokens() + self.num_patches()
    }
    /// Flattened Conv2d patch length (`channels · patch_size²`).
    pub fn patch_dim(&self) -> usize {
        self.num_channels * self.patch_size * self.patch_size
    }
    /// Whether the FFN activation is tanh-approx GELU (`gelu_pytorch_tanh`
    /// / `gelu_new`) rather than the exact erf GELU (`gelu`).
    pub fn gelu_is_tanh(&self) -> bool {
        matches!(self.hidden_act.as_str(), "gelu_pytorch_tanh" | "gelu_new")
    }

    fn preset(
        image_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        use_gated_mlp: bool,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            image_size,
            patch_size: 16,
            num_channels: 3,
            hidden_act: "gelu".to_string(),
            layer_norm_eps: 1e-5,
            rope_theta: 100.0,
            query_bias: true,
            key_bias: false,
            value_bias: true,
            proj_bias: true,
            mlp_bias: true,
            layerscale_value: 1.0,
            use_gated_mlp,
            num_register_tokens: 4,
            final_layer_norm_affine: true,
        }
    }

    /// `facebook/dinov3-vits16-pretrain-lvd1689m` (ViT-S/16, standard MLP).
    pub fn vit_s16(image_size: usize) -> Self {
        Self::preset(image_size, 384, 1536, 12, 6, false)
    }
    /// `facebook/dinov3-vitb16-pretrain-lvd1689m` (ViT-B/16, standard MLP) —
    /// the default embedder model.
    pub fn vit_b16(image_size: usize) -> Self {
        Self::preset(image_size, 768, 3072, 12, 12, false)
    }
    /// `facebook/dinov3-vitl16-pretrain-lvd1689m` (ViT-L/16, standard MLP).
    pub fn vit_l16(image_size: usize) -> Self {
        Self::preset(image_size, 1024, 4096, 24, 16, false)
    }
}
