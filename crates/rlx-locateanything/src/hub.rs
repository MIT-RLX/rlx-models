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

//! Hugging Face Hub cache (`HF_HOME`, `~/.cache/huggingface/hub/…`).

use crate::config::LocateAnythingConfig;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// Default cache root (`HF_HOME` / `HUGGINGFACE_HUB_CACHE` / `~/.cache/huggingface`).
pub fn default_hf_cache_dir() -> PathBuf {
    if let Ok(h) = std::env::var("HF_HOME") {
        return PathBuf::from(h);
    }
    if let Ok(h) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(h);
    }
    home_dir().join(".cache").join("huggingface")
}

fn home_dir() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

/// `org/model` style Hub id (not an existing filesystem path).
pub fn is_hub_model_id(s: &str) -> bool {
    let s = s.trim();
    s.contains('/') && !s.starts_with('.') && !s.starts_with('/') && !Path::new(s).exists()
}

/// Snapshot directory for `repo_id` already in the HF cache.
#[cfg(feature = "hf-cache")]
pub fn hf_snapshot_dir(repo_id: &str) -> Result<PathBuf> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(default_hf_cache_dir())
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.model(repo_id.to_string());
    let config = repo.get("config.json").with_context(|| {
        format!(
            "locate {repo_id} in Hugging Face cache under {}\n\
             Download once: `huggingface-cli download {repo_id}` or `just fetch-locateanything`",
            default_hf_cache_dir().display()
        )
    })?;
    config
        .parent()
        .map(Path::to_path_buf)
        .context("config.json has no parent (snapshot dir)")
}

#[cfg(not(feature = "hf-cache"))]
pub fn hf_snapshot_dir(_repo_id: &str) -> Result<PathBuf> {
    bail!(
        "Hugging Face cache support disabled; rebuild with `--features hf-cache` on rlx-locateanything"
    )
}

/// Best-effort checkpoint directory: `RLX_LOCATEANYTHING_DIR` → HF cache → `just fetch` layout.
pub fn default_model_dir() -> Result<PathBuf> {
    if let Ok(raw) = std::env::var("RLX_LOCATEANYTHING_DIR") {
        if let Some(p) = crate::fixtures::resolve_model_dir_path(&raw) {
            return Ok(p);
        }
    }

    if let Ok(p) = hf_snapshot_dir(LocateAnythingConfig::HF_MODEL_ID) {
        return Ok(p);
    }

    let local = PathBuf::from(".cache/locateanything/LocateAnything-3B");
    if local.join("config.json").is_file() {
        return Ok(local);
    }

    bail!(
        "LocateAnything weights not found.\n\
         • Hugging Face cache: `huggingface-cli download {}` (or `just fetch-locateanything`)\n\
         • Or: `export RLX_LOCATEANYTHING_DIR=/path/to/LocateAnything-3B`\n\
         Cache root: {}",
        LocateAnythingConfig::HF_MODEL_ID,
        default_hf_cache_dir().display()
    )
}

/// Resolve CLI/API path: directory, index file, `hf`/`hub`, or `org/model` Hub id.
pub fn resolve_weights_path(path: &Path) -> Result<PathBuf> {
    let lossy = path.to_string_lossy();
    let token = lossy.trim();
    if token.eq_ignore_ascii_case("hf")
        || token.eq_ignore_ascii_case("hub")
        || token.eq_ignore_ascii_case("cache")
    {
        return default_model_dir();
    }
    if path.join("config.json").is_file() {
        return Ok(path.to_path_buf());
    }
    if path.is_file() {
        return path
            .parent()
            .map(Path::to_path_buf)
            .context("weights file has no parent directory");
    }
    if is_hub_model_id(token) {
        return hf_snapshot_dir(token);
    }
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    bail!("weights path not found: {}", path.display())
}
