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

//! ChatterBox — Resemble AI's zero-shot voice-cloning TTS for RLX (MIT, 24 kHz).
//!
//! A 0.5B-Llama **T3** (text→speech-token) backbone + **S3Gen** flow vocoder.
//! The runtime is **native RLX** (ONNX graphs imported to rlx-ir, compiled
//! per backend — no ONNX Runtime at inference).
//!
//! Pipeline: reference audio → `speech_encoder` → conditioning; T3 AR samples
//! speech tokens; `conditional_decoder` / HiFT → 24 kHz PCM.

pub mod common;

/// Native ort-free runtime — imports the ONNX graphs → rlx-ir → compile → run
/// on any RLX backend (cpu/metal/mlx/wgpu/coreml).
#[cfg(feature = "native")]
pub mod native;

pub use common::{SAMPLE_RATE, SynthOpts, peak_amplitude, polish_onset};

#[cfg(feature = "native")]
pub use native::NativeChatterBox;

pub use rlx_runtime::{Device, parse_device};

/// Default weights dir (`synath/chatterbox-ONNX` layout).
pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/chatterbox";
