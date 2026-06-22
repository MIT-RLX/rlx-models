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

//! Acoustic echo cancellation at 16 kHz — FDAF-NLMS + RLX residual suppression.
//!
//! Pre-ASR front-end: feed cleaned mono f32 PCM to Whisper / Voxtral unchanged.

pub mod audio;
pub mod cli;
pub mod delay;
pub mod dtd;
pub mod fdaf;
pub mod metrics;
pub mod residual;
pub mod session;
pub mod synth;

pub use audio::{
    SAMPLE_RATE, load_wav_16k, load_wav_mono_f32, parse_wav_mono_f32, resample_linear,
    write_wav_mono_16,
};
pub use delay::{ReferenceRing, estimate_delay_samples};
pub use dtd::DoubleTalkDetector;
pub use fdaf::{FdafConfig, FdafNlms};
pub use metrics::{correlation, erle_db, max_abs_error, mse, mse_improvement_db, signal_power};
pub use residual::{ResidualWeights, embedded_residual_weights};
pub use session::{AecConfig, AecSession};
pub use synth::{apply_echo, simple_delayed_echo};
