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

//! Mmap-cached safetensors loader for FireRedAudio checkpoints.

use crate::weights::{
    PREFIX_AUDIO, PREFIX_BACKBONE, PREFIX_DIT, PREFIX_PATCH_ENCODER, PREFIX_RED_VAE,
};
use anyhow::Result;
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_core::weight_map::WeightMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
pub struct WeightStore {
    dir: PathBuf,
    checkpoint: Arc<SafetensorsCheckpoint>,
    all_keys: Arc<HashSet<String>>,
}

impl WeightStore {
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
        self.load_prefixes(&[PREFIX_AUDIO])
    }

    pub fn load_backbone_weights(&self) -> Result<WeightMap> {
        self.load_prefixes(&[PREFIX_BACKBONE])
    }

    pub fn load_red_vae_weights(&self) -> Result<WeightMap> {
        self.load_prefixes(&[PREFIX_RED_VAE])
    }

    pub fn load_patch_encoder_weights(&self) -> Result<WeightMap> {
        self.load_prefixes(&[PREFIX_PATCH_ENCODER])
    }

    pub fn load_dit_weights(&self) -> Result<WeightMap> {
        self.load_prefixes(&[PREFIX_DIT])
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
