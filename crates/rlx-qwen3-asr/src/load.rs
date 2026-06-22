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

//! Mmap-cached safetensors loader for `Qwen/Qwen3-ASR-*` checkpoints.

use crate::weights::{PREFIX_AUDIO_TOWER, PREFIX_LANGUAGE_MODEL};
use anyhow::Result;
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_core::weight_map::WeightMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
pub struct AsrWeightStore {
    dir: PathBuf,
    checkpoint: Arc<SafetensorsCheckpoint>,
    all_keys: Arc<HashSet<String>>,
}

impl AsrWeightStore {
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
        let want: HashSet<String> = self
            .all_keys
            .iter()
            .filter(|k| prefixes.iter().any(|p| k.starts_with(p)))
            .cloned()
            .collect();
        if want.is_empty() {
            anyhow::bail!("no checkpoint keys match {prefixes:?} under {:?}", self.dir);
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

    /// Decoder + embeddings (`thinker.model.*`). `thinker.lm_head.weight` is
    /// intentionally excluded — the Qwen3 head is tied to `embed_tokens`.
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
