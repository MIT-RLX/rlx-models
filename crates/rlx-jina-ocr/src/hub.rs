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

//! Hugging Face Hub cache resolution for `jinaai/jina-ocr-v1`.

use crate::config::JinaOcrConfig;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

pub use rlx_unlimited_ocr::hub::{default_hf_cache_dir, is_hub_model_id};

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
             Download once: `huggingface-cli download {repo_id}` or `rlx-jina-ocr --download`",
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
    bail!("Hugging Face cache support disabled; rebuild with `--features hf-cache`")
}

/// Best-effort checkpoint directory: `RLX_JINA_OCR_DIR` → HF cache → local.
pub fn default_model_dir() -> Result<PathBuf> {
    if let Ok(raw) = rlx_ir::env::var_result("RLX_JINA_OCR_DIR")
        && let Some(p) = crate::fixtures::resolve_model_dir_path(&raw)
    {
        return Ok(p);
    }

    #[cfg(feature = "hf-download")]
    if let Some(p) = crate::download::read_snapshot_pointer(&default_hf_cache_dir()) {
        return Ok(p);
    }

    if let Ok(p) = hf_snapshot_dir(JinaOcrConfig::HF_MODEL_ID) {
        return Ok(p);
    }

    let local = PathBuf::from(".cache/jina-ocr/jina-ocr-v1");
    if local.join("config.json").is_file() {
        return Ok(local);
    }

    bail!(
        "jina-ocr-v1 weights not found.\n\
         • Hugging Face cache: `huggingface-cli download {}` (or `rlx-jina-ocr --download`)\n\
         • Or: `export RLX_JINA_OCR_DIR=/path/to/jina-ocr-v1`\n\
         Cache root: {}",
        JinaOcrConfig::HF_MODEL_ID,
        default_hf_cache_dir().display()
    )
}

/// Resolve a CLI/API path: directory, index file, `hf`/`hub`, or Hub id.
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
