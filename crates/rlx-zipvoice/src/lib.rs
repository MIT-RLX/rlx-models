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

//! ZipVoice — k2-fsa flow-matching voice-cloning TTS for RLX (Apache-2.0).
//!
//! ZipVoice and LuxTTS are the same architecture — LuxTTS is ZipVoice-distill
//! fine-tuned — with byte-identical ONNX interfaces (`text_encoder`,
//! `fm_decoder`) and the same Vocos vocoder. This crate reuses the
//! [`rlx_luxtts`] runner + DSP wholesale; the only difference is the inference
//! defaults ([`zipvoice_opts`]): the 4-step anchor-ODE sampler with **no** LuxTTS
//! `×1.3` speed bump (`speed_mult = 1.0`).
//!
//! Setup mirrors LuxTTS: `F5`-style dir with `text_encoder.onnx`,
//! `fm_decoder.onnx`, `tokens.txt`, and an exported `onnx/vocoder_spec.onnx`
//! (from `charactr/vocos-mel-24khz`; see `scripts/export_vocoder.py`).

pub use rlx_luxtts::dsp;
pub use rlx_luxtts::{InferOpts, LuxTts as ZipVoice, peak_amplitude};
pub use rlx_runtime::{Device, parse_device};

/// Weights repo (Apache-2.0); use the `zipvoice_distill/` subdir for the 4-step
/// distilled model this crate's sampler targets.
pub const DEFAULT_HF_REPO: &str = "k2-fsa/ZipVoice";
pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/zipvoice-distill";

/// ZipVoice-distill inference defaults (4 steps; no LuxTTS `×1.3` speed bump).
pub fn zipvoice_opts() -> InferOpts {
    InferOpts { num_step: 4, speed_mult: 1.0, ..InferOpts::default() }
}
