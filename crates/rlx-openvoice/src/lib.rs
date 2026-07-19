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

//! OpenVoice v2 — MyShell's ~100M zero-shot voice-cloning TTS for RLX (MIT).
//!
//! Pipeline: a MeloTTS base voice (via the native [`rlx_tiny_tts`] engine)
//! generates the utterance, then an ONNX **tone-color converter** transfers the
//! timbre of a reference clip onto it. The reference speaker embedding is pulled
//! by an ONNX **tone-color extractor**. Both ONNX graphs run on ONNX Runtime
//! (`tone_extract.onnx`, `tone_color.onnx`); the base TTS runs on native rlx.
//!
//! Weights: a tiny-tts MeloTTS bundle + the OpenVoice-ONNX-v2 export
//! (`Hinotsuba/OpenVoice-ONNX-v2`).

pub mod dsp;
pub mod model;

pub use model::{OpenVoice, peak_amplitude};
pub use rlx_runtime::{Device, parse_device};

/// Default MeloTTS base bundle (tiny-tts engine).
pub const DEFAULT_MELO_DIR: &str = "weights/tts/melotts";
/// Default OpenVoice ONNX dir (`tone_extract.onnx`, `tone_color.onnx`).
pub const DEFAULT_OPENVOICE_DIR: &str = "weights/tts/openvoice";
/// OpenVoice flow sampling temperature.
pub const DEFAULT_TAU: f32 = 0.3;
