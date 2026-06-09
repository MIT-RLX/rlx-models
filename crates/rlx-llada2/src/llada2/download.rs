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

//! HuggingFace download helper for LLaDA2 checkpoints (`hf-download` feature).

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub const DEFAULT_HF_REPO: &str = "inclusionAI/LLaDA2.0-mini";

/// Download `inclusionAI/LLaDA2.0-mini` (smallest public LLaDA2 MoE) into `cache_dir`.
#[cfg(feature = "hf-download")]
pub fn download_llada2_mini(cache_dir: &Path) -> Result<PathBuf> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.model(DEFAULT_HF_REPO.to_string());
    let config = repo.get("config.json").context("download config.json")?;
    let index = repo
        .get("model.safetensors.index.json")
        .ok()
        .or_else(|| repo.get("pytorch_model.bin.index.json").ok());
    if let Some(index_path) = index {
        let text = std::fs::read_to_string(&index_path)?;
        let index_json: serde_json::Value =
            serde_json::from_str(&text).context("parse weight index json")?;
        if let Some(weight_map) = index_json.get("weight_map").and_then(|m| m.as_object()) {
            let mut files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            files.sort();
            files.dedup();
            for f in files {
                repo.get(&f)
                    .with_context(|| format!("download shard {f}"))?;
            }
        }
    } else {
        let _ = repo.get("model.safetensors").ok();
    }
    Ok(config.parent().unwrap_or(cache_dir).to_path_buf())
}

#[cfg(not(feature = "hf-download"))]
pub fn download_llada2_mini(_cache_dir: &Path) -> Result<PathBuf> {
    anyhow::bail!(
        "HF download requires `hf-download` feature — rebuild with \
         `cargo build -p rlx-models --features hf-download`"
    )
}
