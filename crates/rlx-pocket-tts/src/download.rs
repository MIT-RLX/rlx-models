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

/// Kyutai's ungated repository: every language, the released voice states,
/// no Mimi encoder weights (so no voice cloning).
pub const KYUTAI_REPO: &str = "kyutai/pocket-tts-without-voice-cloning";
/// Kyutai's gated repository (accept the terms on the model card, then log in):
/// the same models with the Mimi encoder weights that voice cloning needs.
pub const KYUTAI_GATED_REPO: &str = "kyutai/pocket-tts";

/// Download `language`'s model, tokenizer and `voices` from `repo` (one of
/// [`KYUTAI_REPO`] / [`KYUTAI_GATED_REPO`]). Returns the local snapshot root,
/// laid out like the repository, ready for [`crate::PocketTts::open`].
pub fn fetch_language(repo: &str, language: &str, voices: &[&str]) -> Result<PathBuf> {
    let spec = crate::ModelSpec::builtin(language)?;
    let api = Api::new().context("hf-hub Api::new")?;
    let r = api.model(repo.to_string());
    let lang = format!("languages/{language}");
    let weights = r
        .get(&format!("{lang}/model.safetensors"))
        .with_context(|| format!("download {repo}/{lang}/model.safetensors"))?;
    // The tokenizer may live in another language's directory (drifting reuses
    // english_2026-09's); prefer the SentencePiece model next to the JSON.
    if let Some(tok) = spec.tokenizer_repo_path() {
        let model = std::path::Path::new(&tok).with_extension("model");
        let model = model.to_string_lossy().replace('\\', "/");
        if r.get(&model).is_err() {
            r.get(&tok)
                .with_context(|| format!("download tokenizer {tok}"))?;
        }
    }
    for v in voices {
        r.get(&format!("{lang}/embeddings/{v}.safetensors"))
            .with_context(|| format!("download voice {v}"))?;
    }
    // `<snapshot>/languages/<language>/model.safetensors` → `<snapshot>`.
    weights
        .ancestors()
        .nth(3)
        .map(PathBuf::from)
        .context("unexpected hf-hub cache layout")
}
