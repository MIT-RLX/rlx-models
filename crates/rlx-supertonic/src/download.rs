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

//! Hugging Face download for the Supertonic-3 bundle.

use std::path::PathBuf;

use anyhow::Result;

use crate::config::{DEFAULT_HF_REPO, DEFAULT_LOCAL_DIR};

/// Voice-style presets shipped with the repo.
pub const VOICES: &[&str] = &["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"];

/// ONNX subgraphs + pipeline config/tokenizer files.
pub const ONNX_FILES: &[&str] = &[
    "onnx/duration_predictor.onnx",
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
    "onnx/tts.json",
    "onnx/unicode_indexer.json",
];

/// Fetch the default bundle into [`DEFAULT_LOCAL_DIR`] if not present.
#[cfg(feature = "hf-download")]
pub fn fetch_default() -> Result<PathBuf> {
    use anyhow::Context;
    let dest = PathBuf::from(DEFAULT_LOCAL_DIR);
    if dest.join("onnx/tts.json").is_file() && dest.join("onnx/vocoder.onnx").is_file() {
        return Ok(dest);
    }
    let api = hf_hub::api::sync::ApiBuilder::new().build().context("hf_hub ApiBuilder")?;
    let repo = api.model(DEFAULT_HF_REPO.to_string());
    std::fs::create_dir_all(dest.join("onnx"))?;
    std::fs::create_dir_all(dest.join("voice_styles"))?;
    std::fs::copy(repo.get("config.json")?, dest.join("config.json")).ok();
    for f in ONNX_FILES {
        let src = repo.get(f).with_context(|| format!("download {f}"))?;
        std::fs::copy(&src, dest.join(f))?;
    }
    for v in VOICES {
        let rel = format!("voice_styles/{v}.json");
        if let Ok(src) = repo.get(&rel) {
            std::fs::copy(&src, dest.join(&rel))?;
        }
    }
    eprintln!("[supertonic] wrote bundle to {}", dest.display());
    Ok(dest)
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_default() -> Result<PathBuf> {
    let dest = PathBuf::from(DEFAULT_LOCAL_DIR);
    if dest.join("onnx/tts.json").is_file() {
        return Ok(dest);
    }
    anyhow::bail!(
        "Supertonic-3 weights not found at {}; rebuild with `--features hf-download` or download {} manually",
        dest.display(),
        DEFAULT_HF_REPO
    )
}
