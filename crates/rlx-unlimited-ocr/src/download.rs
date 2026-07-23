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

//! Download `baidu/Unlimited-OCR` into the HF cache.

use crate::config::UnlimitedOcrConfig;
use crate::hub::default_hf_cache_dir;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const CONFIG_AND_TOKENIZER: &[&str] = &[
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.json",
    // Custom modeling code required by `trust_remote_code` HF loads / parity helper.
    "modeling_unlimitedocr.py",
    "modeling_deepseekv2.py",
    "configuration_deepseek_v2.py",
    "deepencoder.py",
    "conversation.py",
];

/// Download weights into the HF cache; returns the snapshot directory.
#[cfg(feature = "hf-download")]
pub fn download_unlimited_ocr(cache_dir: &Path) -> Result<PathBuf> {
    eprintln!(
        "Downloading {} into {}",
        UnlimitedOcrConfig::HF_MODEL_ID,
        cache_dir.display()
    );
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.model(UnlimitedOcrConfig::HF_MODEL_ID.to_string());

    let config = repo.get("config.json").context("download config.json")?;
    let snapshot = config
        .parent()
        .context("config.json has no parent")?
        .to_path_buf();
    eprintln!("snapshot: {}", snapshot.display());

    for name in CONFIG_AND_TOKENIZER {
        if *name == "config.json" {
            continue;
        }
        eprintln!("  {name}");
        match repo.get(name) {
            Ok(_) => {}
            Err(e) => eprintln!("    (skip {name}: {e})"),
        }
    }
    // Prefer a single `tokenizer.json`; fall back to vocab+merges if absent.
    let has_tok = snapshot.join("tokenizer.json").is_file()
        || (snapshot.join("vocab.json").is_file() && snapshot.join("merges.txt").is_file());
    if !has_tok {
        anyhow::bail!(
            "downloaded snapshot missing tokenizer.json (and vocab.json+merges.txt) at {}",
            snapshot.display()
        );
    }

    eprintln!("  model.safetensors.index.json");
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let index_text = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_text)?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .context("weight_map in index")?;
        let mut shards: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        shards.sort();
        shards.dedup();
        for shard in &shards {
            eprintln!("  {shard}");
            repo.get(shard)
                .with_context(|| format!("download shard {shard}"))?;
        }
    } else {
        eprintln!("  model.safetensors");
        repo.get("model.safetensors")
            .context("download model.safetensors")?;
    }

    write_snapshot_pointer(cache_dir, &snapshot)?;
    Ok(snapshot)
}

/// Path written for `just fetch-unlimited-ocr` (under HF cache root).
pub fn snapshot_pointer_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join(".rlx_unlimited_ocr_snapshot")
}

#[cfg(feature = "hf-download")]
fn write_snapshot_pointer(cache_dir: &Path, snapshot: &Path) -> Result<()> {
    std::fs::write(
        snapshot_pointer_path(cache_dir),
        snapshot.display().to_string(),
    )?;
    Ok(())
}

#[cfg(feature = "hf-download")]
pub fn read_snapshot_pointer(cache_dir: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(snapshot_pointer_path(cache_dir)).ok()?;
    let path = PathBuf::from(text.trim());
    path.join("config.json").is_file().then_some(path)
}

#[cfg(feature = "hf-download")]
pub fn fetch_unlimited_ocr(cache_dir: &Path) -> Result<PathBuf> {
    download_unlimited_ocr(cache_dir)
}

#[cfg(not(feature = "hf-download"))]
pub fn fetch_unlimited_ocr(_cache_dir: &Path) -> Result<PathBuf> {
    anyhow::bail!("enable feature `hf-download` on rlx-unlimited-ocr")
}

/// Default cache + download (for `just fetch-unlimited-ocr`).
pub fn fetch_default() -> Result<PathBuf> {
    fetch_unlimited_ocr(&default_hf_cache_dir())
}
