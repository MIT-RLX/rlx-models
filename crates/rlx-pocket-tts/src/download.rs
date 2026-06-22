// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Hugging Face download helpers (gated behind the `hf-download` feature).

use std::path::PathBuf;

use anyhow::{Context, Result};
use hf_hub::api::sync::Api;

/// Ungated mirror used by default — see crate README.
pub const POCKET_TTS_REPO: &str = "Verylicious/pocket-tts-ungated";
pub const WEIGHTS_FILE: &str = "tts_b6369a24.safetensors";
pub const TOKENIZER_FILE: &str = "tokenizer.model";

#[derive(Debug, Clone)]
pub struct PocketTtsAssets {
    pub weights: PathBuf,
    pub tokenizer: PathBuf,
}

/// Download (or look up cached) weights and tokenizer.
pub fn fetch_default_assets() -> Result<PocketTtsAssets> {
    let api = Api::new().context("hf-hub Api::new")?;
    let repo = api.model(POCKET_TTS_REPO.to_string());
    let weights = repo.get(WEIGHTS_FILE).context("download weights")?;
    let tokenizer = repo.get(TOKENIZER_FILE).context("download tokenizer")?;
    Ok(PocketTtsAssets { weights, tokenizer })
}

/// Download a voice embedding by name (`alba`, `cosette`, etc.) — see the
/// crate README for the list. Returns the local path to the safetensors file.
pub fn fetch_voice(name: &str) -> Result<PathBuf> {
    let api = Api::new().context("hf-hub Api::new")?;
    let repo = api.model(POCKET_TTS_REPO.to_string());
    let relpath = format!("embeddings/{name}.safetensors");
    repo.get(&relpath)
        .with_context(|| format!("download voice {name}"))
}
