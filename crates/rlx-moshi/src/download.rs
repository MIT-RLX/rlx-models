//! Weight download helpers and default cache directory resolution.

use crate::checkpoint::MoshiCheckpoint;
use crate::config::{MoshiVariant, MoshiVoice};
use anyhow::Result;
use std::path::{Path, PathBuf};

#[cfg(feature = "hf-download")]
use anyhow::Context;

/// Default Moshiko bf16 HuggingFace repo (legacy constant).
pub const HF_MOSHI_MOSHIKO_REPO: &str = "kyutai/moshiko-candle-bf16";

/// Default `.cache/…` path for a variant + checkpoint pair.
pub fn default_moshi_dir_for(variant: MoshiVariant, checkpoint: MoshiCheckpoint) -> PathBuf {
    checkpoint.default_cache_dir(variant.voice())
}

/// Resolve Moshi LM directory: `RLX_MOSHI_DIR` or voice/checkpoint-scoped cache.
pub fn default_moshi_dir() -> PathBuf {
    std::env::var("RLX_MOSHI_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            default_moshi_dir_for(
                MoshiVariant::MoshikoOneWay,
                MoshiCheckpoint::from_env_or_default(),
            )
        })
}

/// Default Mimi codec cache (delegates to `rlx-mimi`).
pub fn default_mimi_dir() -> PathBuf {
    rlx_mimi::default_mimi_dir()
}

/// Use explicit path when provided, otherwise [`default_moshi_dir`].
pub fn resolve_moshi_dir(explicit: Option<&Path>) -> PathBuf {
    explicit
        .map(Path::to_path_buf)
        .unwrap_or_else(default_moshi_dir)
}

/// Download default Moshiko bf16 weights into `out_dir`.
#[cfg(feature = "hf-download")]
pub fn fetch_moshi(out_dir: &Path) -> Result<PathBuf> {
    fetch_moshi_checkpoint(
        MoshiVariant::MoshikoOneWay,
        MoshiCheckpoint::Bf16Safetensors,
        out_dir,
    )
}

/// Download LM weights for a variant + checkpoint preset (voice inferred from variant).
#[cfg(feature = "hf-download")]
pub fn fetch_moshi_checkpoint(
    variant: MoshiVariant,
    checkpoint: MoshiCheckpoint,
    out_dir: &Path,
) -> Result<PathBuf> {
    fetch_moshi_voice_checkpoint(variant.voice(), checkpoint, out_dir)
}

/// Download LM weights for a voice + checkpoint preset.
#[cfg(feature = "hf-download")]
pub fn fetch_moshi_voice_checkpoint(
    voice: MoshiVoice,
    checkpoint: MoshiCheckpoint,
    out_dir: &Path,
) -> Result<PathBuf> {
    use hf_hub::api::sync::Api;
    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let api = Api::new()?;
    let repo = api.model(checkpoint.hf_repo(voice).to_string());
    let mut files = vec![checkpoint.lm_filename()];
    files.extend_from_slice(MoshiCheckpoint::SHARED_FILES);
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
pub fn fetch_moshi(_out_dir: &Path) -> Result<PathBuf> {
    anyhow::bail!("rebuild with feature `hf-download` to fetch weights")
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_moshi_checkpoint(
    _variant: MoshiVariant,
    _checkpoint: MoshiCheckpoint,
    _out_dir: &Path,
) -> Result<PathBuf> {
    anyhow::bail!("rebuild with feature `hf-download` to fetch weights")
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_moshi_voice_checkpoint(
    _voice: MoshiVoice,
    _checkpoint: MoshiCheckpoint,
    _out_dir: &Path,
) -> Result<PathBuf> {
    anyhow::bail!("rebuild with feature `hf-download` to fetch weights")
}

/// Ensure default env checkpoint exists under `model_dir` (Moshiko one-way).
pub fn ensure_weights(model_dir: &Path) -> Result<()> {
    ensure_weights_checkpoint(
        model_dir,
        MoshiVariant::MoshikoOneWay,
        MoshiCheckpoint::from_env_or_default(),
    )
}

/// Ensure LM + tokenizer exist, auto-fetching from HuggingFace when `hf-download` is enabled.
pub fn ensure_weights_checkpoint(
    model_dir: &Path,
    // `variant` is only consumed by the `hf-download` fetch path.
    #[cfg_attr(not(feature = "hf-download"), allow(unused_variables))] variant: MoshiVariant,
    checkpoint: MoshiCheckpoint,
) -> Result<()> {
    let lm = checkpoint.lm_weights_path(model_dir);
    let tok = checkpoint.tokenizer_path(model_dir);
    if lm.is_file() && tok.is_file() {
        return Ok(());
    }
    #[cfg(feature = "hf-download")]
    {
        fetch_moshi_checkpoint(variant, checkpoint, model_dir)?;
        Ok(())
    }
    #[cfg(not(feature = "hf-download"))]
    {
        anyhow::bail!(
            "missing weights under {} (expected {}) — set RLX_MOSHI_DIR or run with --fetch",
            model_dir.display(),
            lm.display()
        )
    }
}

/// LM weights path using env default checkpoint preset.
pub fn lm_weights_path(model_dir: &Path) -> PathBuf {
    MoshiCheckpoint::from_env_or_default().lm_weights_path(model_dir)
}

/// Tokenizer path using env default checkpoint preset.
pub fn tokenizer_path(model_dir: &Path) -> PathBuf {
    MoshiCheckpoint::from_env_or_default().tokenizer_path(model_dir)
}
