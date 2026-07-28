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

//! Llama-3.2-Vision (mllama) for RLX.
//!
//! Unlike the embed-splice VLMs (Pixtral, Qwen2.5-VL), mllama fuses vision via
//! **cross-attention**: a ViT vision tower produces per-tile features that a
//! subset of the Llama-3.2 text-decoder layers attend to as K/V (with tanh
//! output gates), while the `<|image|>` token stays a normal text token id.
//!
//! Modules:
//! - [`config`] — nested vision + text configuration (HF `config.json`).
//! - [`vision`] — the ViT tower + multi-modal projector as a native flow.

pub mod cli;
pub mod config;
pub mod cross_attn;
pub mod preprocess;
pub mod runner;
pub mod vision;

pub use runner::MllamaRunner;

pub use config::{MllamaConfig, MllamaTextConfig, MllamaVisionConfig};
pub use cross_attn::{CROSS_STATES_INPUT, CrossAttnDims, cross_attn_stage};
pub use preprocess::{VisionEmbedWeights, VisionInputs, extract_vision_embed_weights};
