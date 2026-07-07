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

//! F5-TTS — flow-matching DiT voice-cloning TTS for RLX.
//!
//! Runs the community DakeQQ 3-file ONNX export (`F5_Preprocess`,
//! `F5_Transformer`, `F5_Decode`, all f16) with a thin Rust orchestrator: text
//! tokenization (char-level over `vocab.txt`), F5's duration estimate, and the
//! NFE denoising loop. The DiT does classifier-free guidance + the ODE step
//! internally; the decoder folds in the Vocos vocoder. Weights are CC-BY-NC.

pub mod config;
pub mod model;
pub mod tokenize;

pub use config::{DEFAULT_HF_REPO, DEFAULT_LOCAL_DIR, Layout, SAMPLE_RATE, Vocab};
pub use model::{F5Tts, InferOpts, peak_amplitude};
pub use rlx_runtime::{Device, parse_device};
