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

//! Kokoro-82M — StyleTTS2 + ISTFTNet text-to-speech for RLX.
//!
//! Kokoro is an 82M-parameter StyleTTS2 model with an ISTFTNet vocoder head. It
//! shares the espeak-ng → IPA frontend and the 3-input ONNX interface
//! (`input_ids` / `style` / `speed`) with [`rlx_kittentts`], from which the
//! phonemizer, text preprocessor and ONNX Runtime execution-provider selector
//! are reused.
//!
//! ## Quick start
//!
//! ```no_run
//! use rlx_kokoro::Kokoro;
//! # fn main() -> anyhow::Result<()> {
//! let tts = Kokoro::load_from_dir(std::path::Path::new(".cache/kokoro-82m"))?;
//! let audio = tts.generate_from_text("Hello from Kokoro.", "af_heart", 1.0)?;
//! tts.write_wav(&audio, std::path::Path::new("out.wav"))?;
//! # Ok(()) }
//! ```
//!
//! ## Backends
//!
//! The default path runs the exported ONNX graph on ONNX Runtime (CPU, plus
//! CoreML / CUDA / DirectML execution providers via the `metal` / `mlx` /
//! `cuda` / `gpu` features).
//!
//! The **`native`** feature adds an ort-free RLX path: the monolithic graph is
//! split into `encoder.onnx` + `decoder_raw.onnx` (via `scripts/split_kokoro.py`)
//! with the data-dependent length regulator and ISTFT overlap-add done in Rust;
//! both subgraphs import through `rlx-onnx-import` → rlx-ir and run on any RLX
//! backend (cpu / metal / mlx / wgpu). See [`native::NativeKokoro`]. The split
//! decomposition is verified bit-exact against the monolithic model.

pub mod config;
pub mod download;
pub mod model;
#[cfg(feature = "native")]
pub mod native;
pub mod tokenize;
pub mod voices;

pub use config::{
    DEFAULT_HF_REPO, DEFAULT_LOCAL_DIR, ModelLayout, SAMPLE_RATE, is_english_voice, voice_lang,
};
pub use download::{ENGLISH_VOICES, fetch_default};
pub use model::{Kokoro, MIN_AUDIBLE_PEAK, peak_amplitude, write_wav};
#[cfg(feature = "native")]
pub use native::{NativeKokoro, resolve_native_device};
pub use rlx_runtime::{Device, is_available, parse_device};
pub use tokenize::{MAX_PHONEME_LEN, PAD_ID, Vocab};
pub use voices::{STYLE_DIM, Voice, VoiceBank};
