// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Parler-TTS Mini v1 — voice-description TTS for RLX (878M).
//!
//! Native (ort-free) path: T5 text encoder + 9-codebook delay-pattern decoder
//! imported from ONNX via `rlx-onnx-import`, then Descript DAC decode via
//! [`rlx_dac`]. Runs on any RLX backend (cpu / metal / mlx / …).
//!
//! ## Quick start
//!
//! ```no_run
//! use rlx_parlertts::{InferOpts, NativeParler, DEFAULT_DAC_DIR, DEFAULT_LOCAL_DIR};
//! use rlx_runtime::Device;
//!
//! # fn main() -> anyhow::Result<()> {
//! let tts = NativeParler::open(DEFAULT_LOCAL_DIR, DEFAULT_DAC_DIR, Device::Cpu)?;
//! let pcm = tts.synthesize(
//!     "Hello from Parler.",
//!     "A clear female voice speaks slowly.",
//!     &InferOpts::default(),
//! )?;
//! tts.write_wav(&pcm, std::path::Path::new("out.wav"))?;
//! # Ok(()) }
//! ```
//!
//! Weights: `weights/tts/parlertts/` (ONNX + tokenizer) and
//! `weights/tts/parler-dac/` (Descript DAC). See crate `README.md`.

pub mod config;
pub mod native;

pub use config::{
    DEFAULT_DAC_DIR, DEFAULT_HF_DAC_REPO, DEFAULT_HF_REPO, DEFAULT_LOCAL_DIR, ParlerTTSConfig,
    SAMPLE_RATE,
};
pub use native::{DEFAULT_DESCRIPTION, InferOpts, NativeParler, write_wav};
pub use rlx_runtime::{Device, parse_device};

/// Peak absolute amplitude (audibility check).
pub fn peak_amplitude(audio: &[f32]) -> f32 {
    audio.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}
