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

//! Grounding DINO (`IDEA-Research/grounding-dino-base`) on RLX.
//!
//! Open-vocabulary object detection: an RGB image plus a free-text prompt
//! (e.g. `"a cat. a remote control."`) produces bounding boxes, grounded
//! labels, and scores. Runs on every RLX backend (CPU, Metal, MLX, CUDA,
//! WGPU, Vulkan) — the heavy math is composed from existing IR ops so GPU
//! backends execute natively.
//!
//! Pipeline: Swin-B vision backbone + BERT text backbone → feature enhancer
//! (multi-scale deformable + bidirectional cross-attention) → language-guided
//! query selection → cross-modality decoder → contrastive class + box heads.

pub mod cli;
pub mod config;
pub mod decoder;
pub mod decoder_ir;
pub mod deform_attn;
pub mod deform_attn_ir;
pub mod deform_op;
pub mod download;
pub mod enhancer;
pub mod enhancer_ir;
pub mod grounding_dino;
pub mod ir;
pub mod mlp;
pub mod neck;
pub(crate) mod nn;
pub mod postprocess;
pub mod preprocess;
pub mod query_select;
pub mod swin;
pub mod swin_ir;
pub mod text_encoder;
pub mod text_encoder_ir;
pub mod tokenizer;
pub(crate) mod weights;

pub use config::{GroundingDinoConfig, SwinConfig, TextConfig};
pub use grounding_dino::GroundingDino;
pub use postprocess::Detection;
pub use rlx_runtime::Device;
pub use text_encoder::{TextEncoder, TextFeatures};
pub use tokenizer::{TextTokens, text_tokens_from_ids};
