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

//! Piper — VITS text-to-speech for RLX.
//!
//! Piper voices (`rhasspy/piper-voices`, MIT) are single VITS ONNX graphs with
//! an espeak-ng phoneme frontend. This crate reuses the bundled espeak-ng
//! phonemizer + ONNX Runtime EP selector from [`rlx_kittentts`] and adds the
//! Piper `phoneme_id_map` tokenizer and the `input`/`input_lengths`/`scales`
//! runner.

pub mod config;
pub mod tokenize;

#[cfg(feature = "native")]
pub mod native;
#[cfg(feature = "native")]
pub mod sdp;

pub use config::{DEFAULT_HF_REPO, DEFAULT_LOCAL_DIR, PiperConfig, find_voice};
#[cfg(feature = "native")]
pub use native::NativeVits;
pub use rlx_runtime::{Device, parse_device};

/// Peak absolute amplitude of a signal, ignoring non-finite samples.
pub fn peak_amplitude(a: &[f32]) -> f32 {
    a.iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0, f32::max)
}
