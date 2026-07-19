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

//! MeloTTS — MyShell's ~52M multilingual VITS2 TTS for RLX (MIT).
//!
//! Real inference is provided by the shared [`rlx_tiny_tts`] engine: the MeloTTS
//! ONNX graphs (`text_encoder`, `duration_predictor`, `flow`, `decoder`) are
//! imported to native rlx-ir and run on **native rlx backends**
//! (CPU/Metal/MLX/CUDA/wgpu — all whisper-validated), not ONNX Runtime. This
//! crate is a thin MeloTTS-named surface over that engine.
//!
//! ```no_run
//! use rlx_melotts::{MeloTts, InferOpts};
//! let tts = MeloTts::load("weights/tiny-tts-rlx")?;
//! let wav = tts.synthesize("The quick brown fox.", &InferOpts::default())?;
//! # Ok::<(), anyhow::Error>(())
//! ```

pub use rlx_runtime::{Device, parse_device};
pub use rlx_tiny_tts::{
    AssetSource, InferOpts, KernelVariant, MIN_AUDIBLE_PEAK, MIN_AUDIBLE_SAMPLES,
    TinyTts as MeloTts, Wav, ensure_audible, normalize_audio, peak_amplitude, write_wav,
};

/// Default bundle directory for the MeloTTS/VITS2 assets.
pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/melotts";
