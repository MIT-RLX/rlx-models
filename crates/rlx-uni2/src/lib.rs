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

//! UNI2-h — MahmoodLab's pathology foundation model (ViT-H/14).
//!
//! `UNI2-h` is a DINOv2-family Vision Transformer with three deviations
//! from a plain DINOv2 encoder, all handled here:
//!   - **packed SwiGLU MLP** (`timm.layers.SwiGLUPacked`, SiLU gate),
//!   - **8 register tokens** (`reg_token`),
//!   - **`no_embed_class`** position embeddings (patch tokens only).
//!
//! Public entry points:
//!   - [`Uni2Config`] — model dimensions + the `uni2_h` preset,
//!   - [`Uni2Runner`] — load a checkpoint and embed images,
//!   - [`build_uni2_built`] — emit the IR graph + host preprocessing,
//!   - [`assemble_hidden`] / [`rgb_u8_to_imagenet_nchw`] — image → encoder
//!     input plumbing.
//!
//! ```no_run
//! use rlx_uni2::Uni2Runner;
//! use rlx_runtime::Device;
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut runner = Uni2Runner::builder()
//!     .weights("uni2h.safetensors")
//!     .device(Device::Cpu)
//!     .build()?;
//!
//! // `rgb`: HWC u8 (any size; resized + ImageNet-normalized to 224).
//! let rgb = vec![0u8; 224 * 224 * 3];
//! let embedding: Vec<f32> = runner.embed_image(&rgb, 224, 224)?; // [1536]
//! assert_eq!(embedding.len(), 1536);
//! # Ok(())
//! # }
//! ```
//!
//! Parity is verified **bit-exact vs the reference timm forward** (cosine 1.0)
//! on CPU/Metal/MLX/wgpu — see the crate README.
//!
//! Weight keys match the published timm checkpoint verbatim
//! (`blocks.{i}.attn.qkv`, `mlp.fc1`/`fc2`, `ls1`/`ls2.gamma`,
//! `reg_token`, `cls_token`, `pos_embed`, `patch_embed.proj`), so no key
//! remapping is required — only a pickle → safetensors conversion of the
//! gated `pytorch_model.bin` (see the crate README).
//!
//! License note: the UNI2-h weights are **CC-BY-NC-ND 4.0** (gated,
//! non-commercial academic use only). This crate is the RLX runtime; the
//! weights are the user's responsibility to obtain under those terms.

pub mod cli;
pub mod config;
pub mod flow;
pub mod preprocess;
pub mod runner;

pub use config::{IMAGENET_MEAN, IMAGENET_STD, Uni2Config};
pub use flow::{Uni2Built, Uni2Flow, build_uni2_built};
pub use preprocess::{Uni2PreprocessWeights, assemble_hidden, rgb_u8_to_imagenet_nchw};
pub use runner::{Uni2Output, Uni2Runner, Uni2RunnerBuilder};
