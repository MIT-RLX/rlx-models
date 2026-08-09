#[cfg(feature = "hf-download")]
use crate::config::DacConfig;
use anyhow::Result;
use std::path::{Path, PathBuf};

pub const HF_DAC_24KHZ: &str = "hance-ai/descript-audio-codec-24khz";
pub const HF_DAC_16KHZ: &str = "hance-ai/descript-audio-codec-16khz";
pub const HF_DAC_44KHZ: &str = "hance-ai/descript-audio-codec-44khz";

pub fn default_dac_dir() -> PathBuf {
    std::env::var("RLX_DAC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/dac/24khz"))
}

pub fn resolve_model_dir(explicit: Option<&Path>) -> PathBuf {
    explicit
        .map(Path::to_path_buf)
        .unwrap_or_else(default_dac_dir)
}

#[cfg(feature = "hf-download")]
fn hf_repo(model_type: &str) -> &'static str {
    match model_type {
        "16khz" => HF_DAC_16KHZ,
        "44khz" => HF_DAC_44KHZ,
        _ => HF_DAC_24KHZ,
    }
}

#[cfg(feature = "hf-download")]
fn config_for(model_type: &str) -> DacConfig {
    match model_type {
        "16khz" => DacConfig::config_16khz(),
        "44khz" => DacConfig::config_44khz(),
        _ => DacConfig::config_24khz(),
    }
}

#[cfg(feature = "hf-download")]
pub fn fetch_dac(out_dir: &Path, model_type: &str) -> Result<PathBuf> {
    use anyhow::Context;
    use hf_hub::api::sync::Api;

    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let api = Api::new()?;
    let repo = api.model(hf_repo(model_type).to_string());
    let path = repo
        .get("model.safetensors")
        .with_context(|| "download model.safetensors")?;
    let dest = out_dir.join("model.safetensors");
    if path != dest {
        std::fs::copy(&path, &dest)
            .with_context(|| format!("copy {} -> {}", path.display(), dest.display()))?;
    }
    let cfg = config_for(model_type);
    let cfg_path = out_dir.join("config.json");
    std::fs::write(&cfg_path, serde_json::to_string_pretty(&cfg)?)?;
    eprintln!("fetched {} ({})", dest.display(), model_type);
    Ok(out_dir.to_path_buf())
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_dac(_out_dir: &Path, _model_type: &str) -> Result<PathBuf> {
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
        let model_type = model_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("24khz");
        fetch_dac(model_dir, model_type)?;
        Ok(())
    }
    #[cfg(not(feature = "hf-download"))]
    {
        anyhow::bail!(
            "missing weights under {} — set RLX_DAC_DIR or run with --fetch (needs hf-download feature)",
            model_dir.display()
        )
    }
}
