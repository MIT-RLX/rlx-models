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

//! Native Rust port of [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts).
//!
//! Composition:
//! - [`flow_lm`] — 6-layer transformer + per-step `SimpleMLPAdaLN` flow head
//! - [`mimi`] — decoder path of the Mimi codec (`output_proj` → upsample →
//!   2-layer projected transformer → SEANet decoder)
//! - [`tokenizer`] — SentencePiece bridge + sentence splitter
//! - [`voice`] — loads `audio_prompt [1, 125, 1024]` voice conditioning
//! - [`model::TtsModel`] — top-level entry point
//!
//! See the crate-level README for the weights layout.

#![allow(clippy::too_many_arguments)]

pub mod audio;
pub mod config;
pub mod flow_lm;
pub mod mimi;
pub mod model;
pub mod ops;
pub mod tokenizer;
pub mod voice;
pub mod weights;

#[cfg(feature = "hf-download")]
pub mod download;

pub use config::PocketTtsConfig;
pub use model::{Audio, GenerationOptions, TtsModel};
pub use tokenizer::{PocketTokenizer, split_into_chunks};
pub use voice::Voice;

/// Output audio sample rate (24 kHz).
pub const SAMPLE_RATE: u32 = 24_000;

/// Latent frame rate emitted by the FlowLM (12.5 Hz; 80 ms per frame).
pub const FRAME_RATE: f32 = 12.5;

/// Audio samples per FlowLM latent frame (`SAMPLE_RATE / FRAME_RATE`).
pub const SAMPLES_PER_FRAME: usize = 1_920;
