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

//! Hugging Face Hub download helpers for Nanbeige4.2-3B.

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use std::path::{Path, PathBuf};

use crate::HF_MODEL_ID_3B;

pub fn default_hf_cache_dir() -> String {
    std::env::var("HF_HOME")
        .or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE"))
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            format!("{home}/.cache/huggingface")
        })
}

/// Download Nanbeige4.2-3B safetensors + config/tokenizer into `dest`.
pub fn download_nanbeige42_3b(cache: &str, dest: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dest).with_context(|| format!("mkdir {dest:?}"))?;
    let api = ApiBuilder::new()
        .with_cache_dir(cache.into())
        .build()
        .context("hf-hub ApiBuilder")?;
    let repo = api.model(HF_MODEL_ID_3B.to_string());

    let required = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ];
    let optional = [
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "added_tokens.json",
        "configuration_nanbeige.py",
        "modeling_nanbeige.py",
    ];
    for name in required {
        let src = repo.get(name).with_context(|| format!("download {name}"))?;
        let dst = dest.join(name);
        if !dst.exists() {
            std::fs::copy(&src, &dst).with_context(|| format!("copy {src:?} → {dst:?}"))?;
        }
    }
    for name in optional {
        if let Ok(src) = repo.get(name) {
            let dst = dest.join(name);
            if !dst.exists() {
                let _ = std::fs::copy(&src, &dst);
            }
        }
    }
    Ok(dest.to_path_buf())
}

pub fn fetch_nanbeige42_3b(cache: &str, dest: &Path) -> Result<PathBuf> {
    download_nanbeige42_3b(cache, dest)
}

pub fn materialize_nanbeige42_3b(dest: &Path) -> Result<PathBuf> {
    let cache = default_hf_cache_dir();
    fetch_nanbeige42_3b(&cache, dest)
}
