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

//! LuxTTS — ZipVoice-distill flow-matching voice-cloning TTS for RLX.
//!
//! Three ONNX subgraphs (`text_encoder`, `fm_decoder`, and a `vocoder_spec`
//! head we export from the Vocos vocoder to sidestep ONNX's missing ISTFT) are
//! chained with Rust glue mirroring the reference `zipvoice` ONNX inference:
//! a prompt wav + its transcript condition a 4-step anchor-ODE flow-matching
//! sampler; the vocoder head yields STFT coefficients and a Rust ISTFT produces
//! 24 kHz audio. The DSP (VocosFbank log-mel + ISTFT) is validated bit-close to
//! torch/torchaudio in `tests/dsp_parity.rs`.

pub mod config;
pub mod dsp;
pub mod model;
pub mod tokenize;

pub use config::{DEFAULT_HF_REPO, DEFAULT_LOCAL_DIR, Layout, Tokens};
pub use model::{InferOpts, LuxTts, peak_amplitude};
pub use rlx_runtime::{Device, parse_device};
