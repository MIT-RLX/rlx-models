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

//! Supertonic-3 — multilingual flow-matching latent TTS for RLX.
//!
//! Four ONNX subgraphs (`duration_predictor`, `text_encoder`, `vector_estimator`,
//! `vocoder`) are chained with a small Rust glue that mirrors the reference
//! `supertonic-py` pipeline: a scalar total-duration prediction sets the latent
//! length, a flow-matching ODE loop (default 8 steps, the estimator integrates
//! internally) denoises the latent, and the vocoder emits 44.1 kHz audio. The
//! tokenizer is pure char/unicode (`unicode_indexer`), so no phonemizer is
//! needed — it covers 31 languages via `<lang>…</lang>` wrapping.
//!
//! ```no_run
//! use rlx_supertonic::{Supertonic, Voice, InferOpts};
//! # fn main() -> anyhow::Result<()> {
//! let dir = std::path::Path::new("weights/tts/supertonic-3");
//! let tts = Supertonic::load_from_dir(dir)?;
//! let voice = Voice::load(&dir.join("voice_styles/F1.json"))?;
//! let audio = tts.synthesize("Hello from Supertonic.", "en", &voice, &InferOpts::default())?;
//! tts.write_wav(&audio, std::path::Path::new("out.wav"))?;
//! # Ok(()) }
//! ```

pub mod config;
pub mod download;
pub mod model;
pub mod tokenize;
pub mod voices;

pub use config::{AVAILABLE_LANGS, DEFAULT_HF_REPO, DEFAULT_LOCAL_DIR, StConfig};
pub use model::{
    DEFAULT_SPEED, DEFAULT_TOTAL_STEP, InferOpts, MIN_AUDIBLE_PEAK, Rng, Supertonic, peak_amplitude,
};
pub use rlx_runtime::{Device, parse_device};
pub use tokenize::{UnicodeIndexer, preprocess};
pub use voices::{StyleTensor, Voice, list_voices};
