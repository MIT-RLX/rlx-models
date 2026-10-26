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

//! Native Rust port of [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
//! ([model card](https://huggingface.co/kyutai/pocket-tts)).
//!
//! [`PocketTts`] runs every released model — English (100M, 24-layer, drifting,
//! and the 2M/7M/18M small models) and French, German, Italian, Portuguese,
//! Spanish, Dutch — as rlx graphs on any RLX device:
//! - [`graph::flow_lm`] — one FlowLM step (backbone + sampler head + EOS head)
//!   with a host-held, bucketed KV cache;
//! - [`graph::mimi`] — the Mimi decoder as a streaming block graph;
//! - [`spec`] — the upstream YAML configs (bundled);
//! - [`text`] — upstream text preparation and sentence chunking;
//! - [`voice_state`] — the released voices (KV-cache states) and
//!   `audio_prompt` voices.
//!
//! The older eager ndarray path ([`TtsModel`]) is kept for the January
//! checkpoint; see the crate README for the weights layout.

#![allow(clippy::too_many_arguments)]

pub mod audio;
pub mod checkpoint;
pub mod config;
pub mod engine;
#[cfg(feature = "export")]
pub mod export;
pub mod flow_lm;
pub mod graph;
pub mod mimi;
pub mod model;
pub mod ops;
pub mod spec;
pub mod text;
pub mod tokenizer;
pub mod voice;
pub mod voice_state;
pub mod weights;

#[cfg(feature = "hf-download")]
pub mod download;

pub use config::PocketTtsConfig;
pub use engine::{ChunkOutput, ModelFiles, PocketTts, SynthOptions};
pub use model::{Audio, GenerationOptions, TtsModel};
pub use spec::{FlowType, ModelSpec};
pub use tokenizer::{PocketTokenizer, split_into_chunks};
pub use voice::Voice;
pub use voice_state::{VoiceFile, VoiceState};

/// Output audio sample rate (24 kHz).
pub const SAMPLE_RATE: u32 = 24_000;

/// Latent frame rate emitted by the FlowLM (12.5 Hz; 80 ms per frame).
pub const FRAME_RATE: f32 = 12.5;

/// Audio samples per FlowLM latent frame (`SAMPLE_RATE / FRAME_RATE`).
pub const SAMPLES_PER_FRAME: usize = 1_920;
