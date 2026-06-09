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

//! Safetensors loading for sharded `nvidia/LocateAnything-3B` checkpoints.

use anyhow::Result;
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_core::weight_map::WeightMap;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub type WeightSnapshot = HashMap<String, (Vec<f32>, Vec<usize>)>;

pub const PREFIX_VISION: &str = "vision_model.";
pub const PREFIX_PROJECTOR: &str = "mlp1.";
pub const PREFIX_LANGUAGE_MODEL: &str = "language_model.";

pub const EXPECTED_TENSOR_COUNT: usize = 770;
pub const EXPECTED_VISION_TENSORS: usize = 329;
pub const EXPECTED_PROJECTOR_TENSORS: usize = 6;
pub const EXPECTED_LANGUAGE_MODEL_TENSORS: usize = 435;

/// Mmap-cached LocateAnything checkpoint.
#[derive(Clone)]
pub struct LocateAnythingWeightStore {
    dir: PathBuf,
    checkpoint: Arc<SafetensorsCheckpoint>,
    all_keys: Arc<HashSet<String>>,
}

impl LocateAnythingWeightStore {
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

    pub fn keys(&self) -> &HashSet<String> {
        self.all_keys.as_ref()
    }

    pub fn count_keys_with_prefix(&self, prefix: &str) -> usize {
        self.all_keys
            .iter()
            .filter(|k| k.starts_with(prefix))
            .count()
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

    pub fn load_vision_weights(&self) -> Result<WeightMap> {
        self.load_prefixes(&[PREFIX_VISION])
    }

    pub fn load_projector_weights(&self) -> Result<WeightMap> {
        self.load_prefixes(&[PREFIX_PROJECTOR])
    }

    pub fn load_language_model_weights(&self) -> Result<WeightMap> {
        self.load_prefixes(&[PREFIX_LANGUAGE_MODEL])
    }

    /// F32 snapshot of all LM tensors (for compile caches across decode/MTP steps).
    pub fn load_language_model_snapshot(&self) -> Result<WeightSnapshot> {
        let mut wm = self.load_language_model_weights()?;
        let keys: Vec<String> = wm.keys().map(str::to_string).collect();
        let mut out = HashMap::with_capacity(keys.len());
        for k in keys {
            out.insert(k.clone(), wm.take(&k)?);
        }
        Ok(out)
    }

    pub fn load_keys(&self, keys: &[&str]) -> Result<WeightMap> {
        let want: HashSet<String> = keys.iter().map(|k| (*k).to_string()).collect();
        self.checkpoint.load_selected(&want)
    }

    /// Embedding rows for `token_ids` only (mmap slice; avoids ~1.2GB full vocab table).
    pub fn load_lm_embed_rows_for_tokens(
        &self,
        token_ids: &[u32],
        vocab: usize,
        hidden: usize,
    ) -> Result<HashMap<u32, Vec<f32>>> {
        use crate::weights::LocateAnythingWeightPrefix;
        let key = LocateAnythingWeightPrefix::lm_embed_tokens();
        let mut unique = Vec::new();
        let mut seen = HashSet::new();
        for &t in token_ids {
            let ti = t as usize;
            if ti < vocab && seen.insert(t) {
                unique.push(t);
            }
        }
        if unique.is_empty() {
            return Ok(HashMap::new());
        }
        let rows = self.checkpoint.load_tensor_rows_f32(key, &unique, hidden)?;
        Ok(unique.into_iter().zip(rows).collect())
    }

    pub fn validate_tensor_layout(&self) -> Result<()> {
        let total = self.all_keys.len();
        anyhow::ensure!(
            total == EXPECTED_TENSOR_COUNT,
            "expected {EXPECTED_TENSOR_COUNT} tensors, got {total}"
        );
        anyhow::ensure!(
            self.count_keys_with_prefix(PREFIX_VISION) == EXPECTED_VISION_TENSORS,
            "vision tensor count"
        );
        anyhow::ensure!(
            self.count_keys_with_prefix(PREFIX_PROJECTOR) == EXPECTED_PROJECTOR_TENSORS,
            "projector tensor count"
        );
        anyhow::ensure!(
            self.count_keys_with_prefix(PREFIX_LANGUAGE_MODEL) == EXPECTED_LANGUAGE_MODEL_TENSORS,
            "language_model tensor count"
        );
        Ok(())
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
    LocateAnythingWeightStore::open(dir)?.load_prefixes(prefixes)
}

pub fn load_vision_weights(dir: &Path) -> Result<WeightMap> {
    load_weight_map_with_prefixes(dir, &[PREFIX_VISION])
}

pub fn load_projector_weights(dir: &Path) -> Result<WeightMap> {
    load_weight_map_with_prefixes(dir, &[PREFIX_PROJECTOR])
}

pub fn load_language_model_weights(dir: &Path) -> Result<WeightMap> {
    load_weight_map_with_prefixes(dir, &[PREFIX_LANGUAGE_MODEL])
}
