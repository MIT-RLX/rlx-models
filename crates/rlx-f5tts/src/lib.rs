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
//! The runtime is native RLX ([`F5Native`]: ONNX graphs → rlx-ir → compile →
//! run; no ONNX Runtime). Weights are CC-BY-NC.

pub mod config;
pub mod dsp;
pub mod model;
/// Native RLX path (no ONNX Runtime): the 3 F5 graphs imported + compiled + run.
pub mod native;
pub mod tokenize;

pub use config::{DEFAULT_HF_REPO, DEFAULT_LOCAL_DIR, Layout, SAMPLE_RATE, Vocab};
pub use dsp::{preprocess_ref_audio, soft_peak_limit};
pub use model::{InferOpts, peak_amplitude, write_wav};
pub use native::F5Native;
pub use rlx_runtime::{Device, parse_device};
pub use tokenize::normalize_ref_text;
