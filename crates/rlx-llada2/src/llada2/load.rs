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

// RLX — load LLaDA2 config + safetensors weights from a model directory.

use crate::config::LLaDA2MoeConfig;
use crate::weights::{LLaDA2Weights, tensor_keys_for_config};
use anyhow::{Context, Result, anyhow};
use rlx_core::weight_loader::WeightLoader;
use rlx_core::weight_map::WeightMap;
use std::path::{Path, PathBuf};

/// Load config + only tensors needed for the first `max_layers` blocks (e2e / dev).
pub fn load_llada2_partial(
    dir: &Path,
    max_layers: usize,
) -> Result<(LLaDA2MoeConfig, LLaDA2Weights)> {
    let dir = normalize_dir(dir)?;
    let mut cfg = LLaDA2MoeConfig::from_file(&dir.join("config.json"))
        .with_context(|| format!("read {}", dir.join("config.json").display()))?;
    cfg.num_hidden_layers = max_layers.min(cfg.num_hidden_layers);
    let keys = tensor_keys_for_config(&cfg);
    let mut loader = WeightMap::from_safetensors_dir_selected(&dir, &keys)?;
    let weights = LLaDA2Weights::load(&cfg, &mut loader)?;
    Ok((cfg, weights))
}

fn normalize_dir(dir: &Path) -> Result<PathBuf> {
    Ok(if dir.is_file() {
        dir.parent()
            .ok_or_else(|| anyhow!("weights path has no parent directory"))?
            .to_path_buf()
    } else {
        dir.to_path_buf()
    })
}

/// Resolve `config.json` and the primary weights file under `dir`.
pub fn load_llada2_from_dir(dir: &Path) -> Result<(LLaDA2MoeConfig, LLaDA2Weights)> {
    let dir = normalize_dir(dir)?;
    let cfg_path = dir.join("config.json");
    let cfg = LLaDA2MoeConfig::from_file(&cfg_path)
        .with_context(|| format!("read {}", cfg_path.display()))?;
    let mut loader = load_weights_loader(dir.as_path())?;
    let weights = LLaDA2Weights::load(&cfg, &mut *loader)?;
    Ok((cfg, weights))
}

fn load_weights_loader(dir: &Path) -> Result<Box<dyn WeightLoader>> {
    for name in ["model.safetensors", "pytorch_model.bin", "model.bin"] {
        let p = dir.join(name);
        if p.is_file() {
            let s = p
                .to_str()
                .ok_or_else(|| anyhow!("non-UTF-8 path {}", p.display()))?;
            return Ok(Box::new(WeightMap::from_file(s)?));
        }
    }
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    shards.sort();
    match shards.len() {
        0 => Err(anyhow!(
            "no model.safetensors or *.safetensors shards under {}",
            dir.display()
        )),
        1 => {
            let s = shards[0]
                .to_str()
                .ok_or_else(|| anyhow!("non-UTF-8 path {}", shards[0].display()))?;
            Ok(Box::new(WeightMap::from_file(s)?))
        }
        _ => Ok(Box::new(WeightMap::from_safetensors_dir(dir)?)),
    }
}
