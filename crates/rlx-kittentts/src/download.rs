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

//! Hugging Face Hub download for KittenTTS checkpoints.

#[cfg(feature = "hf-download")]
use std::path::{Path, PathBuf};

#[cfg(feature = "hf-download")]
use anyhow::{Context, Result};

#[cfg(feature = "hf-download")]
use crate::assets::{self, ModelLayout, normalize_repo_id};
#[cfg(feature = "hf-download")]
use crate::config::DEFAULT_HF_REPO;

/// Download (or refresh) a KittenTTS repo into the HF cache; returns the snapshot dir.
#[cfg(feature = "hf-download")]
pub fn fetch_repo(repo_id: &str, cache_dir: &Path) -> Result<PathBuf> {
    use crate::config::ModelConfig;

    let repo_id = normalize_repo_id(repo_id);
    eprintln!("Fetching {repo_id} into {}", cache_dir.display());

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.model(repo_id.clone());

    let config_path = repo.get("config.json").context("download config.json")?;
    let snapshot = config_path
        .parent()
        .context("config.json has no parent")?
        .to_path_buf();
    eprintln!("snapshot: {}", snapshot.display());

    let config = ModelConfig::load_from_dir(&snapshot)?;
    for name in [&config.model_file, &config.voices] {
        eprintln!("  {name}");
        repo.get(name).with_context(|| format!("download {name}"))?;
    }

    Ok(snapshot)
}

/// Download repo files into `dest` (flat layout matching `huggingface-cli download --local-dir`).
#[cfg(feature = "hf-download")]
pub fn fetch_to_local_dir(repo_id: &str, dest: &Path) -> Result<PathBuf> {
    let snapshot = fetch_repo(repo_id, &assets::hf_hub_root())?;
    let layout = ModelLayout::resolve(&snapshot)?;
    std::fs::create_dir_all(dest)?;
    for src in [
        snapshot.join("config.json"),
        layout.onnx.clone(),
        layout.voices.clone(),
    ] {
        let name = src.file_name().context("path file_name")?;
        let dst = dest.join(name);
        if src != dst {
            std::fs::copy(&src, &dst)
                .with_context(|| format!("copy {} → {}", src.display(), dst.display()))?;
        }
    }
    eprintln!("wrote {}", dest.display());
    Ok(dest.to_path_buf())
}

/// Default fetch into `.cache/kittentts-mini-0.8`.
#[cfg(feature = "hf-download")]
pub fn fetch_default() -> Result<PathBuf> {
    let local = PathBuf::from(assets::DEFAULT_LOCAL_DIR);
    if assets::layout_exists(&local) {
        if let Ok(layout) = ModelLayout::resolve(&local) {
            eprintln!("using existing weights at {}", layout.dir.display());
            return Ok(layout.dir);
        }
    }
    fetch_to_local_dir(DEFAULT_HF_REPO, &local)
}

#[cfg(not(feature = "hf-download"))]
use std::path::{Path, PathBuf};

#[cfg(not(feature = "hf-download"))]
use anyhow::Result;

#[cfg(not(feature = "hf-download"))]
use crate::assets;
#[cfg(not(feature = "hf-download"))]
pub fn fetch_to_local_dir(_repo_id: &str, _dest: &Path) -> Result<PathBuf> {
    anyhow::bail!(
        "HF download disabled; rebuild with `--features hf-download` or run:\n\
         just fetch-kittentts"
    )
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_default() -> Result<PathBuf> {
    assets::default_model_dir()
}
