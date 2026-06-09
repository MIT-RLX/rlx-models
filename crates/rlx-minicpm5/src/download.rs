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

//! Hugging Face Hub download for [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B).
//!
//! Enable with the `hf-download` feature (`hf-hub`).

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

use crate::{HF_MODEL_ID_1B, HF_MODEL_ID_GGUF, MINICPM5_GGUF_FILES};

const CONFIG_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
];

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

/// Download MiniCPM5-1B into the HF cache; returns the snapshot directory.
#[cfg(feature = "hf-download")]
pub fn download_minicpm5_1b(cache_dir: &Path) -> Result<PathBuf> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.model(HF_MODEL_ID_1B.to_string());

    let config = repo.get("config.json").context("download config.json")?;
    let snapshot = config
        .parent()
        .context("config.json has no parent dir")?
        .to_path_buf();

    for name in CONFIG_FILES {
        if *name == "config.json" {
            continue;
        }
        let _ = repo.get(name);
    }

    let weight_files = weight_shard_names(&repo)?;
    for name in &weight_files {
        repo.get(name)
            .with_context(|| format!("download weight shard {name}"))?;
    }

    Ok(snapshot)
}

#[cfg(feature = "hf-download")]
fn weight_shard_names(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<String>> {
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let text = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value =
            serde_json::from_str(&text).context("parse model.safetensors.index.json")?;
        if let Some(map) = index.get("weight_map").and_then(|m| m.as_object()) {
            let mut files: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            files.sort();
            files.dedup();
            if !files.is_empty() {
                return Ok(files);
            }
        }
    }
    if repo.get("model-00000-of-00001.safetensors").is_ok() {
        return Ok(vec!["model-00000-of-00001.safetensors".into()]);
    }
    if repo.get("model.safetensors").is_ok() {
        return Ok(vec!["model.safetensors".into()]);
    }
    bail!("no safetensors shards found for {HF_MODEL_ID_1B}")
}

/// Copy or symlink checkpoint files from the HF cache snapshot into `dest`.
#[cfg(feature = "hf-download")]
pub fn materialize_minicpm5_1b(snapshot: &Path, dest: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dest).with_context(|| format!("create {dest:?}"))?;

    for name in CONFIG_FILES {
        let src = snapshot.join(name);
        if src.is_file() {
            link_or_copy(&src, &dest.join(name))?;
        }
    }

    let weight_files = list_weight_files(snapshot)?;
    for name in &weight_files {
        link_or_copy(&snapshot.join(name), &dest.join(name))?;
    }

    let canonical = dest.join("model.safetensors");
    if !canonical.exists() {
        let shard = weight_files
            .iter()
            .find(|n| n.ends_with(".safetensors"))
            .context("no .safetensors shard in snapshot")?;
        link_or_copy(&dest.join(shard), &canonical)?;
    }

    let profile_src =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../rlx-llama32/src/llama32.rlx.toml");
    let profile_dst = dest.join("llama32.rlx.toml");
    if profile_src.is_file() && !profile_dst.exists() {
        std::fs::copy(&profile_src, &profile_dst)
            .with_context(|| format!("install {}", profile_dst.display()))?;
    }

    Ok(dest.to_path_buf())
}

#[cfg(feature = "hf-download")]
fn list_weight_files(snapshot: &Path) -> Result<Vec<String>> {
    let index_path = snapshot.join("model.safetensors.index.json");
    if index_path.is_file() {
        let text = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&text)?;
        if let Some(map) = index.get("weight_map").and_then(|m| m.as_object()) {
            let mut files: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            files.sort();
            files.dedup();
            if !files.is_empty() {
                return Ok(files);
            }
        }
    }
    for candidate in ["model-00000-of-00001.safetensors", "model.safetensors"] {
        if snapshot.join(candidate).is_file() {
            return Ok(vec![candidate.into()]);
        }
    }
    bail!("no weight files under {snapshot:?}")
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

/// Download one GGUF quant into the HF cache; returns the local file path.
#[cfg(feature = "hf-download")]
pub fn download_minicpm5_gguf(cache_dir: &Path, quant_label: &str) -> Result<PathBuf> {
    let filename = MINICPM5_GGUF_FILES
        .iter()
        .find(|(label, _)| *label == quant_label)
        .map(|(_, f)| *f)
        .ok_or_else(|| {
            let names: Vec<_> = MINICPM5_GGUF_FILES.iter().map(|(l, _)| *l).collect();
            anyhow::anyhow!("unknown quant {quant_label:?}; expected one of {names:?}")
        })?;
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.model(HF_MODEL_ID_GGUF.to_string());
    repo.get(filename)
        .with_context(|| format!("download {filename} from {HF_MODEL_ID_GGUF}"))
}

/// Copy/symlink a GGUF from the HF cache snapshot into `dest_dir`.
#[cfg(feature = "hf-download")]
pub fn materialize_minicpm5_gguf(src_gguf: &Path, dest_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dest_dir).with_context(|| format!("create {dest_dir:?}"))?;
    let name = src_gguf
        .file_name()
        .context("gguf path has no filename")?
        .to_owned();
    let dst = dest_dir.join(name);
    link_or_copy(src_gguf, &dst)?;
    Ok(dst)
}

/// Download a quant and materialize under `dest_dir`.
#[cfg(feature = "hf-download")]
pub fn fetch_minicpm5_gguf(
    cache_dir: &Path,
    dest_dir: &Path,
    quant_label: &str,
) -> Result<PathBuf> {
    let src = download_minicpm5_gguf(cache_dir, quant_label)?;
    materialize_minicpm5_gguf(&src, dest_dir)
}

/// Download into the HF cache, then materialize under `dest` (flat layout for RLX).
#[cfg(feature = "hf-download")]
pub fn fetch_minicpm5_1b(cache_dir: &Path, dest: &Path) -> Result<PathBuf> {
    let snapshot = download_minicpm5_1b(cache_dir)?;
    materialize_minicpm5_1b(&snapshot, dest)
}

#[cfg(not(feature = "hf-download"))]
pub fn download_minicpm5_1b(_cache_dir: &Path) -> Result<PathBuf> {
    bail!(
        "HF download requires the `hf-download` feature — rebuild with \
         `cargo build -p rlx-minicpm5 --features hf-download`"
    )
}

#[cfg(not(feature = "hf-download"))]
pub fn materialize_minicpm5_1b(_snapshot: &Path, _dest: &Path) -> Result<PathBuf> {
    download_minicpm5_1b(Path::new("."))
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_minicpm5_1b(_cache_dir: &Path, _dest: &Path) -> Result<PathBuf> {
    download_minicpm5_1b(_cache_dir)
}

#[cfg(not(feature = "hf-download"))]
pub fn download_minicpm5_gguf(_cache_dir: &Path, _quant_label: &str) -> Result<PathBuf> {
    bail!("HF download requires the `hf-download` feature")
}

#[cfg(not(feature = "hf-download"))]
pub fn materialize_minicpm5_gguf(_src: &Path, _dest_dir: &Path) -> Result<PathBuf> {
    download_minicpm5_gguf(Path::new("."), "Q4_K_M")
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_minicpm5_gguf(
    _cache_dir: &Path,
    _dest_dir: &Path,
    _quant_label: &str,
) -> Result<PathBuf> {
    download_minicpm5_gguf(Path::new("."), "Q4_K_M")
}
