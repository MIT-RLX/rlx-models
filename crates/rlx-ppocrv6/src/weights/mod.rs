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

//! Resolve PP-OCRv6 model directories (safetensors for native HIR).

use crate::config::Tier;
use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};

/// `(det_weights_dir, rec_weights_dir, dict_path)`.
pub fn resolve_model_dir(dir: &Path, tier: Tier) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let tier_dir = if dir.join("det").is_dir() {
        dir.to_path_buf()
    } else if dir.join(tier.as_str()).join("det").is_dir() {
        dir.join(tier.as_str())
    } else {
        dir.to_path_buf()
    };
    let det = prefer_weights_dir(&tier_dir.join("det"), tier, "det")?;
    let rec = prefer_weights_dir(&tier_dir.join("rec"), tier, "rec")?;
    let dict = prefer_dict(&tier_dir, tier)?;
    Ok((det, rec, dict))
}

fn prefer_weights_dir(task_dir: &Path, tier: Tier, task: &str) -> Result<PathBuf> {
    let model = task_dir.join("model.safetensors");
    if model.is_file() {
        return Ok(task_dir.to_path_buf());
    }
    let named = task_dir.join(format!("ppocrv6_{}_{}.safetensors", tier.as_str(), task));
    if named.is_file() {
        // Native loaders expect `model.safetensors` in the directory.
        std::fs::copy(&named, &model)
            .map_err(|e| anyhow!("copy {} → {}: {e}", named.display(), model.display()))?;
        return Ok(task_dir.to_path_buf());
    }
    Err(anyhow!(
        "no safetensors in {} (expected model.safetensors or ppocrv6_{}_{}.safetensors)",
        task_dir.display(),
        tier.as_str(),
        task
    ))
}

fn prefer_dict(tier_dir: &Path, tier: Tier) -> Result<PathBuf> {
    let candidates = [
        tier_dir.join("rec").join("keys.txt"),
        tier_dir.join(format!("{}_keys.txt", tier.as_str())),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("assets/dicts")
            .join(format!("{}_keys.txt", tier.as_str())),
    ];
    for p in candidates {
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(anyhow!("no character dict for tier {}", tier.as_str()))
}
