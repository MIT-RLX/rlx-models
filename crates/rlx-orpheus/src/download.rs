// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Hugging Face Hub download for Orpheus GGUF + SNAC raw weights.
//!
//! Enable with the `hf-download` feature (`hf-hub`).

#[cfg(feature = "hf-download")]
use anyhow::Context;
use anyhow::Result;
#[cfg(not(feature = "hf-download"))]
use anyhow::bail;
use std::path::{Path, PathBuf};

/// Finetune-prod Orpheus GGUF (named voices).
pub const HF_ORPHEUS_FT_GGUF_REPO: &str = "unsloth/orpheus-3b-0.1-ft-GGUF";

/// SNAC 24 kHz codec (PyTorch checkpoint — export to safetensors separately).
pub const HF_SNAC_REPO: &str = "hubertsiuzdak/snac_24khz";

/// Default GGUF quant for Orpheus TTS. Q8_0 is the best size/quality tradeoff on
/// the CPU GGUF reference path; F16 is also supported via `ORPHEUS_QUANT`.
pub const DEFAULT_ORPHEUS_QUANT: &str = "Q8_0";

/// Quant tag for Hub download — `ORPHEUS_QUANT` overrides [`DEFAULT_ORPHEUS_QUANT`].
pub fn resolve_orpheus_quant() -> String {
    std::env::var("ORPHEUS_QUANT")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_ORPHEUS_QUANT.to_string())
}

pub const DEFAULT_ORPHEUS_DIR: &str = "/tmp/rlx-weights/orpheus";
pub const DEFAULT_SNAC_DIR: &str = "/tmp/rlx-weights/snac";

pub const SNAC_DECODER_SAFETENSORS: &str = "snac_24khz_decoder.safetensors";

#[cfg(feature = "hf-download")]
const SNAC_RAW_FILES: &[&str] = &["config.json", "pytorch_model.bin"];

/// Default Hugging Face cache root (`$HF_HOME` or `~/.cache/huggingface`).
pub fn default_hf_cache_dir() -> PathBuf {
    std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs_or_home().join(".cache").join("huggingface"))
}

fn dirs_or_home() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

/// GGUF filename on [`HF_ORPHEUS_FT_GGUF_REPO`].
pub fn orpheus_gguf_filename(quant: &str) -> String {
    format!("orpheus-3b-0.1-ft-{quant}.gguf")
}

/// Resolve default on-disk Orpheus weights directory (`ORPHEUS_WEIGHTS_DIR` or
/// [`DEFAULT_ORPHEUS_DIR`]).
pub fn default_orpheus_dir() -> PathBuf {
    std::env::var("ORPHEUS_WEIGHTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_ORPHEUS_DIR))
}

/// Resolve default SNAC working directory (`ORPHEUS_SNAC_DIR` or [`DEFAULT_SNAC_DIR`]).
pub fn default_snac_dir() -> PathBuf {
    std::env::var("ORPHEUS_SNAC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_SNAC_DIR))
}

/// Expected SNAC decoder export path under `snac_dir`.
pub fn snac_decoder_path(snac_dir: &Path) -> PathBuf {
    snac_dir.join(SNAC_DECODER_SAFETENSORS)
}

/// Download one Orpheus finetune GGUF quant into the HF cache.
#[cfg(feature = "hf-download")]
pub fn download_orpheus_gguf(cache_dir: &Path, quant: &str) -> Result<PathBuf> {
    let filename = orpheus_gguf_filename(quant);
    eprintln!(
        "Fetching {filename} from {HF_ORPHEUS_FT_GGUF_REPO} into {}",
        cache_dir.display()
    );
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.model(HF_ORPHEUS_FT_GGUF_REPO.to_string());
    repo.get(&filename)
        .with_context(|| format!("download {filename} from {HF_ORPHEUS_FT_GGUF_REPO}"))
}

/// Copy/symlink the GGUF from the HF cache into `dest_dir`.
#[cfg(feature = "hf-download")]
pub fn materialize_orpheus_gguf(src_gguf: &Path, dest_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dest_dir).with_context(|| format!("create {dest_dir:?}"))?;
    let name = src_gguf
        .file_name()
        .context("gguf path has no filename")?
        .to_owned();
    let dst = dest_dir.join(name);
    link_or_copy(src_gguf, &dst)?;
    eprintln!("wrote {}", dst.display());
    Ok(dst)
}

/// Download a quant and materialize under `dest_dir`.
#[cfg(feature = "hf-download")]
pub fn fetch_orpheus_gguf(cache_dir: &Path, dest_dir: &Path, quant: &str) -> Result<PathBuf> {
    let src = download_orpheus_gguf(cache_dir, quant)?;
    materialize_orpheus_gguf(&src, dest_dir)
}

/// Download SNAC PyTorch weights into the HF cache; returns the snapshot directory.
#[cfg(feature = "hf-download")]
pub fn download_snac_raw(cache_dir: &Path) -> Result<PathBuf> {
    eprintln!("Fetching {HF_SNAC_REPO} into {}", cache_dir.display());
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.model(HF_SNAC_REPO.to_string());
    let config = repo.get("config.json").context("download config.json")?;
    let snapshot = config
        .parent()
        .context("config.json has no parent dir")?
        .to_path_buf();
    for name in SNAC_RAW_FILES {
        if name == &"config.json" {
            continue;
        }
        repo.get(name)
            .with_context(|| format!("download {name} from {HF_SNAC_REPO}"))?;
    }
    Ok(snapshot)
}

/// Materialize SNAC raw files under `dest` for `scripts/export_snac_decoder.py`.
#[cfg(feature = "hf-download")]
pub fn materialize_snac_raw(snapshot: &Path, dest: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dest).with_context(|| format!("create {dest:?}"))?;
    for name in SNAC_RAW_FILES {
        let src = snapshot.join(name);
        if src.is_file() {
            link_or_copy(&src, &dest.join(name))?;
        }
    }
    eprintln!("wrote SNAC raw weights to {}", dest.display());
    Ok(dest.to_path_buf())
}

/// Download SNAC into the HF cache, then copy into `dest`.
#[cfg(feature = "hf-download")]
pub fn fetch_snac_raw(cache_dir: &Path, dest: &Path) -> Result<PathBuf> {
    let snapshot = download_snac_raw(cache_dir)?;
    materialize_snac_raw(&snapshot, dest)
}

/// Download finetune GGUF + SNAC raw weights into default directories.
#[cfg(feature = "hf-download")]
pub fn fetch_default() -> Result<(PathBuf, PathBuf)> {
    let cache = default_hf_cache_dir();
    let orpheus = fetch_orpheus_gguf(&cache, &default_orpheus_dir(), &resolve_orpheus_quant())?;
    let snac = fetch_snac_raw(&cache, &default_snac_dir())?;
    print_snac_export_hint(&snac);
    Ok((orpheus, snac))
}

#[cfg(feature = "hf-download")]
pub fn print_snac_export_hint(snac_dir: &Path) {
    let out = snac_decoder_path(snac_dir);
    eprintln!(
        "\nSNAC decoder safetensors are not on Hugging Face — export once:\n\
         \n\
           python3 scripts/export_snac_decoder.py --repo {} --out {}\n\
         \n\
         then:\n\
           export ORPHEUS_SNAC_PATH={}\n",
        snac_dir.display(),
        snac_dir.display(),
        out.display()
    );
}

#[cfg(feature = "hf-download")]
fn link_or_copy(src: &Path, dst: &Path) -> Result<()> {
    if dst.exists() {
        return Ok(());
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(src, dst)
            .or_else(|_| std::fs::copy(src, dst).map(|_| ()))
            .with_context(|| format!("link {src:?} -> {dst:?}"))?;
    }
    #[cfg(not(unix))]
    {
        std::fs::copy(src, dst).with_context(|| format!("copy {src:?} -> {dst:?}"))?;
    }
    Ok(())
}

#[cfg(not(feature = "hf-download"))]
pub fn download_orpheus_gguf(_cache_dir: &Path, _quant: &str) -> Result<PathBuf> {
    hf_download_disabled()
}

#[cfg(not(feature = "hf-download"))]
pub fn materialize_orpheus_gguf(_src_gguf: &Path, _dest_dir: &Path) -> Result<PathBuf> {
    hf_download_disabled()
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_orpheus_gguf(_cache_dir: &Path, _dest_dir: &Path, _quant: &str) -> Result<PathBuf> {
    hf_download_disabled()
}

#[cfg(not(feature = "hf-download"))]
pub fn download_snac_raw(_cache_dir: &Path) -> Result<PathBuf> {
    hf_download_disabled()
}

#[cfg(not(feature = "hf-download"))]
pub fn materialize_snac_raw(_snapshot: &Path, _dest: &Path) -> Result<PathBuf> {
    hf_download_disabled()
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_snac_raw(_cache_dir: &Path, _dest: &Path) -> Result<PathBuf> {
    hf_download_disabled()
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_default() -> Result<(PathBuf, PathBuf)> {
    hf_download_disabled_pair()
}

#[cfg(not(feature = "hf-download"))]
pub fn print_snac_export_hint(_snac_dir: &Path) {}

#[cfg(not(feature = "hf-download"))]
fn hf_download_disabled() -> Result<PathBuf> {
    bail!(
        "HF download requires the `hf-download` feature — rebuild with \
         `cargo build -p rlx-orpheus --features hf-download` or run `just fetch-orpheus`"
    )
}

#[cfg(not(feature = "hf-download"))]
fn hf_download_disabled_pair() -> Result<(PathBuf, PathBuf)> {
    hf_download_disabled()?;
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orpheus_gguf_filename_matches_unsloth() {
        assert_eq!(
            orpheus_gguf_filename("Q4_K_M"),
            "orpheus-3b-0.1-ft-Q4_K_M.gguf"
        );
    }
}
