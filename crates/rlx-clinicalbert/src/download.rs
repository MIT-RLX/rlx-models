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

//! Hugging Face Hub download for ClinicalBERT variants.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

use crate::config::ClinicalBertVariant;

const CONFIG_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
];

/// Default Hugging Face cache root (`$HF_HOME` or `~/.cache/huggingface`).
pub fn default_hf_cache_dir() -> PathBuf {
    std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."));
            home.join(".cache").join("huggingface")
        })
}

/// Download a ClinicalBERT variant into `cache_dir`; returns the snapshot dir.
pub fn download_clinicalbert(cache_dir: &Path, variant: ClinicalBertVariant) -> Result<PathBuf> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.model(variant.hf_repo().to_string());

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

    let shards = weight_shard_names(&repo, variant)?;
    for name in &shards {
        repo.get(name)
            .with_context(|| format!("download weight shard {name}"))?;
    }

    Ok(snapshot)
}

fn weight_shard_names(
    repo: &hf_hub::api::sync::ApiRepo,
    variant: ClinicalBertVariant,
) -> Result<Vec<String>> {
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
    if repo.get("model.safetensors").is_ok() {
        return Ok(vec!["model.safetensors".into()]);
    }
    if repo.get("pytorch_model.bin").is_ok() {
        return Ok(vec!["pytorch_model.bin".into()]);
    }
    bail!("no weight shards found for {}", variant.hf_repo())
}

/// Download into the HF cache, then materialize under `dest` (flat layout).
pub fn fetch_clinicalbert(
    cache_dir: &Path,
    dest: &Path,
    variant: ClinicalBertVariant,
) -> Result<PathBuf> {
    let snapshot = download_clinicalbert(cache_dir, variant)?;
    materialize(&snapshot, dest)
}

fn materialize(snapshot: &Path, dest: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dest).with_context(|| format!("create {dest:?}"))?;
    for name in CONFIG_FILES {
        let src = snapshot.join(name);
        if src.is_file() {
            link_or_copy(&src, &dest.join(name))?;
        }
    }
    for entry in std::fs::read_dir(snapshot)? {
        let entry = entry?;
        let name = entry.file_name();
        let ns = name.to_string_lossy();
        if ns.ends_with(".safetensors") || ns == "pytorch_model.bin" {
            link_or_copy(&entry.path(), &dest.join(&*ns))?;
        }
    }
    #[cfg(feature = "prepare")]
    {
        crate::prepare::prepare_clinicalbert_dir(dest)?;
    }
    Ok(dest.to_path_buf())
}

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
