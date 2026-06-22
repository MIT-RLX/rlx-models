use anyhow::Result;

#[cfg(feature = "hf-download")]
use anyhow::Context;
use std::path::{Path, PathBuf};

pub const HF_MIMI_REPO: &str = "kyutai/mimi";

/// Kyutai Moshi bundle sidecar — Candle/moshi weight layout (~385 MB).
pub const MIMI_CANDLE_SIDECAR: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";

pub fn default_mimi_dir() -> PathBuf {
    std::env::var("RLX_MIMI_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/mimi"))
}

pub fn resolve_model_dir(explicit: Option<&Path>) -> PathBuf {
    explicit
        .map(Path::to_path_buf)
        .unwrap_or_else(default_mimi_dir)
}

#[cfg(feature = "hf-download")]
pub fn fetch_mimi(out_dir: &Path) -> Result<PathBuf> {
    use hf_hub::api::sync::Api;
    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let api = Api::new()?;
    let repo = api.model(HF_MIMI_REPO.to_string());
    for file in [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    ] {
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
pub fn fetch_mimi(_out_dir: &Path) -> Result<PathBuf> {
    anyhow::bail!("rebuild with feature `hf-download` to fetch weights")
}

pub fn ensure_weights(model_dir: &Path) -> Result<()> {
    let st = model_dir.join("model.safetensors");
    let cfg = model_dir.join("config.json");
    if st.is_file() && cfg.is_file() {
        return Ok(());
    }
    #[cfg(feature = "hf-download")]
    {
        fetch_mimi(model_dir)?;
        Ok(())
    }
    #[cfg(not(feature = "hf-download"))]
    {
        anyhow::bail!(
            "missing weights under {} — set RLX_MIMI_DIR or run with --fetch (needs hf-download feature)",
            model_dir.display()
        )
    }
}

/// Candle/moshi-format Mimi weights (sidecar from `kyutai/moshiko-*` repos).
pub fn resolve_candle_weights(model_dir: &Path, moshi_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = moshi_dir {
        let p = dir.join(MIMI_CANDLE_SIDECAR);
        if p.is_file() {
            return Some(p);
        }
    }
    let p = model_dir.join(MIMI_CANDLE_SIDECAR);
    if p.is_file() {
        return Some(p);
    }
    None
}
