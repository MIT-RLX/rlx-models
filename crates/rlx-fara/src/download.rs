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

//! HuggingFace download helpers for Fara1.5.

use crate::config::{FaraSize, default_cache_root, default_model_dir, is_model_dir};
#[cfg(feature = "hf-download")]
use anyhow::Context;
use anyhow::Result;
use std::path::{Path, PathBuf};

fn snapshot_pointer_path(cache_root: &Path, size: FaraSize) -> PathBuf {
    cache_root.join(format!(".rlx_fara_{}_snapshot", size.cache_subdir()))
}

#[cfg(feature = "hf-download")]
fn write_snapshot_pointer(cache_root: &Path, size: FaraSize, snapshot: &Path) -> Result<()> {
    std::fs::create_dir_all(cache_root)
        .with_context(|| format!("mkdir {}", cache_root.display()))?;
    std::fs::write(
        snapshot_pointer_path(cache_root, size),
        snapshot.display().to_string(),
    )?;
    Ok(())
}

/// Read the last downloaded snapshot path for `size`, if still valid.
pub fn read_snapshot_pointer(cache_root: &Path, size: FaraSize) -> Option<PathBuf> {
    let text = std::fs::read_to_string(snapshot_pointer_path(cache_root, size)).ok()?;
    let path = PathBuf::from(text.trim());
    is_model_dir(&path).then_some(path)
}

/// Download `microsoft/Fara1.5-{4B,9B}` via the HF hub cache; returns the
/// snapshot directory that contains `config.json` + safetensors shards.
#[cfg(feature = "hf-download")]
pub fn download_fara(size: FaraSize, cache_root: &Path) -> Result<PathBuf> {
    eprintln!(
        "Downloading {} into HF cache under {}",
        size.hf_model_id(),
        cache_root.display()
    );
    std::fs::create_dir_all(cache_root)
        .with_context(|| format!("mkdir {}", cache_root.display()))?;
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_root.to_path_buf())
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.model(size.hf_model_id().to_string());

    let config = repo.get("config.json").context("download config.json")?;
    let snapshot = config
        .parent()
        .context("config.json has no parent")?
        .to_path_buf();
    eprintln!("snapshot: {}", snapshot.display());

    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ] {
        eprintln!("  {name}");
        let _ = repo.get(name);
    }

    eprintln!("  model.safetensors.index.json");
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let raw = std::fs::read_to_string(&index_path).context("read index")?;
        let v: serde_json::Value = serde_json::from_str(&raw).context("parse index")?;
        if let Some(map) = v.get("weight_map").and_then(|m| m.as_object()) {
            let mut shards: Vec<String> = map
                .values()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect();
            shards.sort();
            shards.dedup();
            for shard in shards {
                eprintln!("  {shard}");
                repo.get(&shard)
                    .with_context(|| format!("download {shard}"))?;
            }
        }
    } else {
        eprintln!("  model.safetensors");
        repo.get("model.safetensors")
            .context("download model.safetensors")?;
    }

    write_snapshot_pointer(cache_root, size, &snapshot)?;
    // Also write a convenience pointer under the size subdir for just recipes.
    let size_dir = cache_root.join(size.cache_subdir());
    std::fs::create_dir_all(&size_dir)?;
    std::fs::write(size_dir.join(".snapshot"), snapshot.display().to_string())?;
    Ok(snapshot)
}

#[cfg(not(feature = "hf-download"))]
pub fn download_fara(size: FaraSize, _cache_root: &Path) -> Result<PathBuf> {
    let _ = size;
    anyhow::bail!("rlx-fara: rebuild with `--features hf-download` to fetch from HuggingFace")
}

/// Resolve a local model dir, optionally downloading when missing.
pub fn resolve_or_download(size: FaraSize, model_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(d) = model_dir {
        if is_model_dir(d) {
            return Ok(d.to_path_buf());
        }
        // Convenience: `.cache/fara/4b/.snapshot` written by download_fara.
        let snap = d.join(".snapshot");
        if let Ok(text) = std::fs::read_to_string(&snap) {
            let path = PathBuf::from(text.trim());
            if is_model_dir(&path) {
                return Ok(path);
            }
        }
        anyhow::bail!("not a Fara model dir: {d:?}");
    }
    let cache = default_cache_root();
    if let Some(p) = read_snapshot_pointer(&cache, size) {
        return Ok(p);
    }
    let def = default_model_dir(size);
    if is_model_dir(&def) {
        return Ok(def);
    }
    let snap = def.join(".snapshot");
    if let Ok(text) = std::fs::read_to_string(&snap) {
        let path = PathBuf::from(text.trim());
        if is_model_dir(&path) {
            return Ok(path);
        }
    }
    download_fara(size, &cache)
}
