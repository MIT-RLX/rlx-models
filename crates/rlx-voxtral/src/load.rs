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

//! Safetensors loading helpers for sharded HF checkpoints.

use anyhow::Result;
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_core::weight_map::WeightMap;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub type WeightSnapshot = HashMap<String, (Vec<f32>, Vec<usize>)>;

pub const PREFIX_AUDIO_TOWER: &str = "audio_tower.";
pub const PREFIX_PROJECTOR: &str = "multi_modal_projector.";
pub const PREFIX_LANGUAGE_MODEL: &str = "language_model.";

/// Mmap-cached Voxtral checkpoint — open once, load selective prefixes cheaply.
#[derive(Clone)]
pub struct VoxtralWeightStore {
    dir: PathBuf,
    checkpoint: Arc<SafetensorsCheckpoint>,
    all_keys: Arc<HashSet<String>>,
}

impl VoxtralWeightStore {
    pub fn open(weights_path: &Path) -> Result<Self> {
        let dir = resolve_model_dir(weights_path)?;
        let checkpoint = Arc::new(SafetensorsCheckpoint::open(&dir)?);
        let all_keys = Arc::new(checkpoint.keys().map(str::to_string).collect());
        Ok(Self {
            dir,
            checkpoint,
            all_keys,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.dir
    }

    pub fn load_prefixes(&self, prefixes: &[&str]) -> Result<WeightMap> {
        let want = keys_matching_prefixes(self.all_keys.as_ref(), prefixes);
        if want.is_empty() {
            anyhow::bail!(
                "no checkpoint keys match prefixes {:?} under {:?}",
                prefixes,
                self.dir
            );
        }
        self.checkpoint.load_selected(&want)
    }

    pub fn load_keys(&self, keys: &[&str]) -> Result<WeightMap> {
        let want: HashSet<String> = keys.iter().map(|k| (*k).to_string()).collect();
        self.checkpoint.load_selected(&want)
    }

    pub fn load_audio_weights(&self) -> Result<WeightMap> {
        self.load_prefixes(&[PREFIX_AUDIO_TOWER])
    }

    pub fn load_projector_weights(&self) -> Result<WeightMap> {
        self.load_prefixes(&[PREFIX_PROJECTOR])
    }

    pub fn load_language_model_weights(&self) -> Result<WeightMap> {
        self.load_prefixes(&[PREFIX_LANGUAGE_MODEL])
    }
}

pub fn resolve_model_dir(weights_path: &Path) -> Result<PathBuf> {
    if weights_path.is_dir() {
        return Ok(weights_path.to_path_buf());
    }
    weights_path
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow::anyhow!("weights path has no parent: {weights_path:?}"))
}

pub fn list_checkpoint_keys(dir: &Path) -> Result<HashSet<String>> {
    let dir = resolve_model_dir(dir)?;
    Ok(SafetensorsCheckpoint::open(&dir)?
        .keys()
        .map(str::to_string)
        .collect())
}

pub fn keys_matching_prefixes(all: &HashSet<String>, prefixes: &[&str]) -> HashSet<String> {
    all.iter()
        .filter(|key| prefixes.iter().any(|prefix| key.starts_with(prefix)))
        .cloned()
        .collect()
}

pub fn load_weight_map_with_prefixes(dir: &Path, prefixes: &[&str]) -> Result<WeightMap> {
    VoxtralWeightStore::open(dir)?.load_prefixes(prefixes)
}

pub fn load_weight_map_keys(dir: &Path, keys: &[&str]) -> Result<WeightMap> {
    VoxtralWeightStore::open(dir)?.load_keys(keys)
}

pub fn load_audio_weights(dir: &Path) -> Result<WeightMap> {
    load_weight_map_with_prefixes(dir, &[PREFIX_AUDIO_TOWER])
}

pub fn load_projector_weights(dir: &Path) -> Result<WeightMap> {
    load_weight_map_with_prefixes(dir, &[PREFIX_PROJECTOR])
}

pub fn load_language_model_weights(dir: &Path) -> Result<WeightMap> {
    load_weight_map_with_prefixes(dir, &[PREFIX_LANGUAGE_MODEL])
}

pub fn load_weight_snapshot(weights_path: &Path) -> Result<WeightSnapshot> {
    let store = VoxtralWeightStore::open(weights_path)?;
    snapshot_from_weight_map(store.load_prefixes(&[
        PREFIX_AUDIO_TOWER,
        PREFIX_PROJECTOR,
        PREFIX_LANGUAGE_MODEL,
    ])?)
}

fn snapshot_from_weight_map(mut wm: WeightMap) -> Result<WeightSnapshot> {
    let keys: Vec<String> = wm.keys().map(str::to_string).collect();
    let mut out = HashMap::with_capacity(keys.len());
    for key in keys {
        out.insert(key.clone(), wm.take(&key)?);
    }
    Ok(out)
}
