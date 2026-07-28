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
//! Real inference is provided by the shared [`rlx_tiny_tts`] engine. Prefer the
//! Hub pack [`eugenehp/tiny-tts-rlx`](https://huggingface.co/eugenehp/tiny-tts-rlx)
//! (`tiny-tts.rlxp` with nested `graphs/*.rlxp`). Local ONNX trees remain a
//! pack-time source only. This crate is a thin MeloTTS-named surface over that
//! engine.
//!
//! Bundle layout matches TinyTTS. Prefer `weights/tts/melotts` (symlink) or
//! `weights/tts/tiny-tts-rlx`.
//!
//! ```no_run
//! use rlx_melotts::{MeloTts, InferOpts};
//! let tts = MeloTts::load(rlx_melotts::resolve_bundle_dir()?)?;
//! let wav = tts.synthesize("The quick brown fox.", &InferOpts::default())?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

pub use rlx_runtime::{Device, parse_device};
pub use rlx_tiny_tts::{
    AssetSource, InferOpts, KernelVariant, MIN_AUDIBLE_PEAK, MIN_AUDIBLE_SAMPLES,
    TinyTts as MeloTts, Wav, ensure_audible, normalize_audio, peak_amplitude, write_wav,
};

/// Preferred MeloTTS/VITS2 bundle directory (usually a symlink to TinyTTS).
pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/melotts";

/// Canonical TinyTTS bundle (same ONNX + frontend assets MeloTTS uses).
pub const TINY_TTS_LOCAL_DIR: &str = "weights/tts/tiny-tts-rlx";

/// Resolve a loadable MeloTTS/TinyTTS bundle directory.
///
/// Order: `RLX_MELOTTS_DIR` → `RLX_TINY_TTS_DIR` → `weights/tts/melotts` →
/// `weights/tts/tiny-tts-rlx` → `weights/tiny-tts-rlx` (compat symlink).
pub fn resolve_bundle_dir() -> Result<PathBuf> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("RLX_MELOTTS_DIR") {
        cands.push(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("RLX_TINY_TTS_DIR") {
        cands.push(PathBuf::from(p));
    }
    cands.push(PathBuf::from(DEFAULT_LOCAL_DIR));
    cands.push(PathBuf::from(TINY_TTS_LOCAL_DIR));
    cands.push(PathBuf::from("weights/tiny-tts-rlx"));
    for p in cands {
        if is_bundle_dir(&p) {
            return Ok(p);
        }
    }
    bail!(
        "MeloTTS/TinyTTS bundle not found (tried {DEFAULT_LOCAL_DIR}, {TINY_TTS_LOCAL_DIR}, \
         env RLX_MELOTTS_DIR / RLX_TINY_TTS_DIR). Fetch with `just fetch-tiny-tts` \
         then `ln -sfn tiny-tts-rlx weights/tts/melotts`."
    );
}

/// True when `dir` looks like a TinyTTS/MeloTTS ONNX bundle.
pub fn is_bundle_dir(dir: &Path) -> bool {
    dir.join("config.json").is_file()
        && dir.join("onnx/text_encoder.onnx").is_file()
        && dir.join("onnx/decoder.onnx").is_file()
}

/// Load MeloTTS from [`resolve_bundle_dir`], or from an explicit path if it is a
/// valid bundle / `.rlxp`.
pub fn load_default() -> Result<MeloTts> {
    MeloTts::load(resolve_bundle_dir()?)
}
