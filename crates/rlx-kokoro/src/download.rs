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

//! Hugging Face download for Kokoro ONNX bundles.

use std::path::PathBuf;

#[cfg(feature = "hf-download")]
use anyhow::{Context, Result};
#[cfg(not(feature = "hf-download"))]
use anyhow::Result;

use crate::config::{DEFAULT_HF_REPO, DEFAULT_LOCAL_DIR};

/// English voice packs shipped with the v1.0 ONNX repo (American + British).
pub const ENGLISH_VOICES: &[&str] = &[
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma",
    "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
];

/// HF cache root (respects `HF_HOME` / `HUGGINGFACE_HUB_CACHE`).
pub fn hf_hub_root() -> PathBuf {
    if let Ok(h) = std::env::var("HF_HOME") {
        return PathBuf::from(h).join("hub");
    }
    if let Ok(h) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(h);
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

/// Download the default Kokoro bundle (`model_file` + English voices) into
/// `dest`, matching the on-disk layout the loader expects.
#[cfg(feature = "hf-download")]
pub fn fetch_to_local_dir(repo_id: &str, model_file: &str, dest: &std::path::Path) -> Result<PathBuf> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(hf_hub_root())
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.model(repo_id.to_string());

    std::fs::create_dir_all(dest.join("onnx"))?;
    std::fs::create_dir_all(dest.join("voices"))?;

    // Small metadata + the requested ONNX variant.
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
        let src = repo.get(name).with_context(|| format!("download {name}"))?;
        std::fs::copy(&src, dest.join(name))?;
    }
    let onnx_rel = format!("onnx/{model_file}");
    let src = repo.get(&onnx_rel).with_context(|| format!("download {onnx_rel}"))?;
    std::fs::copy(&src, dest.join(&onnx_rel))?;

    // Voices (English set; others need non-English espeak data).
    for v in ENGLISH_VOICES {
        let rel = format!("voices/{v}.bin");
        match repo.get(&rel) {
            Ok(src) => {
                std::fs::copy(&src, dest.join(&rel))?;
            }
            Err(e) => eprintln!("[kokoro] skip {rel}: {e}"),
        }
    }
    eprintln!("[kokoro] wrote bundle to {}", dest.display());
    Ok(dest.to_path_buf())
}

/// Fetch the default bundle into [`DEFAULT_LOCAL_DIR`] if not already present.
#[cfg(feature = "hf-download")]
pub fn fetch_default() -> Result<PathBuf> {
    let dest = PathBuf::from(DEFAULT_LOCAL_DIR);
    if dest.join("tokenizer.json").is_file() && dest.join("onnx/model.onnx").is_file() {
        return Ok(dest);
    }
    fetch_to_local_dir(DEFAULT_HF_REPO, "model.onnx", &dest)
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_default() -> Result<PathBuf> {
    let dest = PathBuf::from(DEFAULT_LOCAL_DIR);
    if dest.join("tokenizer.json").is_file() {
        return Ok(dest);
    }
    anyhow::bail!(
        "Kokoro weights not found at {} and HF download is disabled.\n\
         Rebuild with `--features hf-download`, or download manually from {}",
        dest.display(),
        DEFAULT_HF_REPO
    )
}
