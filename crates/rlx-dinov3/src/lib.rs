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

//! DINOv3 — Meta's self-supervised ViT with 2-D axial RoPE.
//!
//! A native RLX port that runs on every RLX backend (CPU / Metal / MLX /
//! wgpu / CoreML / CUDA / Vulkan / ROCm). Architecture matches HF
//! `transformers.models.dinov3_vit` exactly:
//!   - CLS + register tokens + Conv2d patch embed (**no learned pos_embed**);
//!   - per-block: pre-norm → RoPE self-attention (separate q/k/v/o, no key
//!     bias) → LayerScale → residual → pre-norm → MLP (or gated GeGLU) →
//!     LayerScale → residual;
//!   - final LayerNorm; pooled output = CLS token.
//!
//! Public entry points:
//!   - [`DinoV3Config`] — dimensions + `vit_s16` / `vit_b16` / `vit_l16`
//!     factories (or [`DinoV3Config::from_file`] for a checkpoint's config).
//!   - [`build_dinov3_built`] / [`DinoV3Flow`] — emit the encoder graph.
//!   - [`DinoV3Runner`] — image → embedding, with [`DinoV3Runner::forward_nchw`]
//!     for rigorous `pixel_values` parity checks.
//!   - [`assemble_hidden`] / [`rgb_u8_to_imagenet_nchw`] — host plumbing.
//!
//! Weight keys match HF safetensors verbatim
//! (`embeddings.{cls_token,register_tokens,patch_embeddings.*}`,
//! `layer.{i}.{norm1,norm2,attention.{q,k,v,o}_proj,layer_scale{1,2}.lambda1,mlp.*}`,
//! `norm.*`) — no remapping. The pretrained weights are gated on the HF Hub.

/// Command-line entry point (`rlx-dinov3` / `rlx-run dinov3`).
pub mod cli;
/// Model configuration + variant presets.
pub mod config;
/// Native encoder graph assembly (RoPE attention + MLP plugins).
pub mod flow;
/// Host-side patch embedding + token assembly.
pub mod preprocess;
/// 2-D axial RoPE cos/sin table construction.
pub mod rope;
/// Image → embedding runner.
pub mod runner;

pub use config::{DinoV3Config, IMAGENET_MEAN, IMAGENET_STD};
pub use flow::{DinoV3Built, DinoV3Flow, build_dinov3_built};
pub use preprocess::{DinoV3PreprocessWeights, assemble_hidden, rgb_u8_to_imagenet_nchw};
pub use rope::rope_tables;
pub use runner::{DinoV3Output, DinoV3Runner, DinoV3RunnerBuilder};
