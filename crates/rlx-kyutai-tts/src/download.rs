//! Weight download helpers and default cache directory resolution.
//!
//! The repo [`kyutai/tts-1.6b-en_fr`](https://huggingface.co/kyutai/tts-1.6b-en_fr)
//! ships three files:
//!
//! - `dsm_tts_1e68beda@240.safetensors` — backbone + DepFormer weights
//! - `tokenizer-e351c8d8-checkpoint125.safetensors` — Mimi codec sidecar
//! - `tokenizer_spm_8k_en_fr_audio.model` — SentencePiece text tokenizer
//!
//! The Mimi sidecar matches the one shipped with `kyutai/moshiko-*`, so
//! [`rlx_mimi::resolve_candle_weights`] can pick it up out of the TTS dir.

use crate::checkpoint::KyutaiTtsCheckpoint;
use anyhow::Result;
use std::path::{Path, PathBuf};

#[cfg(feature = "hf-download")]
use anyhow::Context;

/// HuggingFace repo for the 1.6B en/fr Kyutai TTS model.
pub const HF_KYUTAI_TTS_REPO: &str = "kyutai/tts-1.6b-en_fr";

/// HuggingFace repo for pre-computed speaker embeddings (`speaker_wavs` tensors).
pub const HF_KYUTAI_TTS_VOICES_REPO: &str = "kyutai/tts-voices";

/// Default voice used in tests / examples when none is specified.
pub const DEFAULT_VOICE_NAME: &str = "alba-mackenna/casual.wav";

/// Primary backbone + DepFormer weights.
pub const TTS_WEIGHTS_FILE: &str = "dsm_tts_1e68beda@240.safetensors";

/// Mimi codec sidecar (Candle layout, ~385 MB) — same file as Moshi ships.
pub const MIMI_SIDECAR_FILE: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";

/// SentencePiece text tokenizer (8k vocab, en/fr + audio control tokens).
pub const SPM_TOKENIZER_FILE: &str = "tokenizer_spm_8k_en_fr_audio.model";

/// Static config filename.
pub const CONFIG_FILE: &str = "config.json";

/// Default `.cache/…` path for the env-default checkpoint.
pub fn default_kyutai_tts_dir() -> PathBuf {
    std::env::var("RLX_KYUTAI_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| KyutaiTtsCheckpoint::from_env_or_default().default_cache_dir())
}

/// Default Mimi codec cache (delegates to `rlx-mimi`).
pub fn default_mimi_dir() -> PathBuf {
    rlx_mimi::default_mimi_dir()
}

/// Default cache for `kyutai/tts-voices` embeddings.
pub fn default_voices_dir() -> PathBuf {
    std::env::var("RLX_KYUTAI_TTS_VOICES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/kyutai-tts-voices"))
}

/// Local path for a voice embedding file (may not exist until fetched).
pub fn voice_embedding_path(
    voices_dir: &Path,
    checkpoint: KyutaiTtsCheckpoint,
    voice_name: &str,
) -> PathBuf {
    voices_dir.join(checkpoint.voice_hf_filename(voice_name))
}

/// Ensure a voice embedding exists locally, fetching from HF when `hf-download` is enabled.
pub fn ensure_voice_embedding(
    voices_dir: &Path,
    checkpoint: KyutaiTtsCheckpoint,
    voice_name: &str,
) -> Result<PathBuf> {
    let dest = voice_embedding_path(voices_dir, checkpoint, voice_name);
    if dest.is_file() {
        return Ok(dest);
    }
    #[cfg(feature = "hf-download")]
    {
        fetch_voice_embedding(voices_dir, checkpoint, voice_name)?;
        Ok(dest)
    }
    #[cfg(not(feature = "hf-download"))]
    {
        anyhow::bail!(
            "missing voice embedding {} — set RLX_KYUTAI_TTS_VOICES_DIR or rebuild with --features hf-download",
            dest.display()
        )
    }
}

/// Download one voice embedding from [`HF_KYUTAI_TTS_VOICES_REPO`].
#[cfg(feature = "hf-download")]
pub fn fetch_voice_embedding(
    voices_dir: &Path,
    checkpoint: KyutaiTtsCheckpoint,
    voice_name: &str,
) -> Result<PathBuf> {
    use hf_hub::api::sync::Api;
    let filename = checkpoint.voice_hf_filename(voice_name);
    std::fs::create_dir_all(voices_dir)
        .with_context(|| format!("create {}", voices_dir.display()))?;
    let api = Api::new()?;
    let repo = api.model(checkpoint.voice_repo().to_string());
    let path = repo
        .get(&filename)
        .with_context(|| format!("download voice {filename}"))?;
    let dest = voices_dir.join(&filename);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if path != dest {
        std::fs::copy(&path, &dest)
            .with_context(|| format!("copy {} -> {}", path.display(), dest.display()))?;
    }
    eprintln!("fetched voice {}", dest.display());
    Ok(dest)
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_voice_embedding(
    _voices_dir: &Path,
    _checkpoint: KyutaiTtsCheckpoint,
    _voice_name: &str,
) -> Result<PathBuf> {
    anyhow::bail!("rebuild with feature `hf-download` to fetch voices")
}

/// Use explicit path when provided, otherwise [`default_kyutai_tts_dir`].
pub fn resolve_kyutai_tts_dir(explicit: Option<&Path>) -> PathBuf {
    explicit
        .map(Path::to_path_buf)
        .unwrap_or_else(default_kyutai_tts_dir)
}

/// Absolute path to LM weights using the env-default checkpoint.
pub fn tts_weights_path(model_dir: &Path) -> PathBuf {
    KyutaiTtsCheckpoint::from_env_or_default().lm_weights_path(model_dir)
}

/// Absolute path to SentencePiece tokenizer using the env-default checkpoint.
pub fn tokenizer_path(model_dir: &Path) -> PathBuf {
    KyutaiTtsCheckpoint::from_env_or_default().tokenizer_path(model_dir)
}

/// Download Kyutai TTS weights (default 1.6B en/fr preset) into `out_dir`.
#[cfg(feature = "hf-download")]
pub fn fetch_kyutai_tts(out_dir: &Path) -> Result<PathBuf> {
    fetch_kyutai_tts_checkpoint(KyutaiTtsCheckpoint::from_env_or_default(), out_dir)
}

/// Download Kyutai TTS weights for a specific checkpoint preset.
#[cfg(feature = "hf-download")]
pub fn fetch_kyutai_tts_checkpoint(
    checkpoint: KyutaiTtsCheckpoint,
    out_dir: &Path,
) -> Result<PathBuf> {
    use hf_hub::api::sync::Api;
    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let api = Api::new()?;
    let repo = api.model(checkpoint.hf_repo().to_string());
    let files = [
        CONFIG_FILE,
        checkpoint.lm_filename(),
        checkpoint.tokenizer_filename(),
        checkpoint.mimi_sidecar_filename(),
    ];
    for file in files {
        let path = repo.get(file).with_context(|| format!("download {file}"))?;
        let dest = out_dir.join(file);
        if path != dest {
            std::fs::copy(&path, &dest)
                .with_context(|| format!("copy {} -> {}", path.display(), dest.display()))?;
        }
        eprintln!("fetched {}", dest.display());
    }
    Ok(out_dir.to_path_buf())
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_kyutai_tts(_out_dir: &Path) -> Result<PathBuf> {
    anyhow::bail!("rebuild with feature `hf-download` to fetch weights")
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_kyutai_tts_checkpoint(
    _checkpoint: KyutaiTtsCheckpoint,
    _out_dir: &Path,
) -> Result<PathBuf> {
    anyhow::bail!("rebuild with feature `hf-download` to fetch weights")
}

/// Ensure LM weights + tokenizer + Mimi sidecar exist under `model_dir`.
pub fn ensure_weights(model_dir: &Path) -> Result<()> {
    ensure_weights_checkpoint(model_dir, KyutaiTtsCheckpoint::from_env_or_default())
}

/// Ensure LM + tokenizer + Mimi sidecar exist, auto-fetching when `hf-download` is enabled.
pub fn ensure_weights_checkpoint(model_dir: &Path, checkpoint: KyutaiTtsCheckpoint) -> Result<()> {
    let lm = checkpoint.lm_weights_path(model_dir);
    let tok = checkpoint.tokenizer_path(model_dir);
    let mimi = model_dir.join(checkpoint.mimi_sidecar_filename());
    if lm.is_file() && tok.is_file() && mimi.is_file() {
        return Ok(());
    }
    #[cfg(feature = "hf-download")]
    {
        fetch_kyutai_tts_checkpoint(checkpoint, model_dir)?;
        Ok(())
    }
    #[cfg(not(feature = "hf-download"))]
    {
        anyhow::bail!(
            "missing weights under {} (expected {}) — set RLX_KYUTAI_TTS_DIR or rebuild with --features hf-download",
            model_dir.display(),
            lm.display()
        )
    }
}
