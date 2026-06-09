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

//! Mmap-backed weight access for `consolidated.safetensors`.

use anyhow::{Result, bail};
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_core::weight_map::WeightMap;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub type WeightSnapshot = HashMap<String, (Vec<f32>, Vec<usize>)>;

pub const PREFIX_CODEC: &str = "audio_tokenizer.";
pub const PREFIX_ACOUSTIC: &str = "acoustic_transformer.";
pub const PREFIX_BACKBONE: &str = "layers.";
pub const PREFIX_MM_EMBED: &str = "mm_audio_embeddings.";

#[derive(Clone)]
pub struct VoxtralTtsWeightStore {
    dir: PathBuf,
    checkpoint: Arc<SafetensorsCheckpoint>,
    keys: Arc<HashSet<String>>,
}

impl VoxtralTtsWeightStore {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let dir = crate::config::resolve_model_dir(model_dir)?;
        let consolidated = dir.join(crate::config::CONSOLIDATED_WEIGHTS);
        if !consolidated.is_file() {
            bail!(
                "missing {} — run `just fetch-voxtral-tts`",
                consolidated.display()
            );
        }
        let checkpoint = Arc::new(SafetensorsCheckpoint::open(&dir)?);
        let keys = Arc::new(checkpoint.keys().map(str::to_string).collect());
        Ok(Self {
            dir,
            checkpoint,
            keys,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.dir
    }

    pub fn keys(&self) -> &std::collections::HashSet<String> {
        &self.keys
    }

    pub fn load_prefix(&self, prefix: &str) -> Result<WeightMap> {
        let want: HashSet<String> = self
            .keys
            .iter()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        if want.is_empty() {
            bail!("no tensors with prefix {prefix:?} under {:?}", self.dir);
        }
        self.checkpoint.load_selected(&want)
    }

    pub fn load_codec(&self) -> Result<WeightMap> {
        self.load_prefix(PREFIX_CODEC)
    }

    pub fn load_acoustic(&self) -> Result<WeightMap> {
        self.load_prefix(PREFIX_ACOUSTIC)
    }

    pub fn load_backbone(&self) -> Result<WeightMap> {
        let mut want: HashSet<String> = self
            .keys
            .iter()
            .filter(|k| k.starts_with(PREFIX_BACKBONE) || *k == "norm.weight")
            .cloned()
            .collect();
        for k in self.keys.iter() {
            if k.starts_with(PREFIX_MM_EMBED) || k.starts_with("tok_embeddings.") {
                want.insert(k.clone());
            }
        }
        if want.is_empty() {
            bail!("no backbone tensors found under {:?}", self.dir);
        }
        self.checkpoint.load_selected(&want)
    }

    pub fn tensor_snapshot_for_embed(&self) -> Result<WeightSnapshot> {
        let want: HashSet<String> = self
            .keys
            .iter()
            .filter(|k| {
                k.starts_with(PREFIX_MM_EMBED)
                    || k.starts_with("tok_embeddings.")
                    || k.as_str() == "norm.weight"
            })
            .cloned()
            .collect();
        if want.is_empty() {
            bail!("no embedding tensors found under {:?}", self.dir);
        }
        let mut wm = self.checkpoint.load_selected(&want)?;
        let keys: Vec<String> = wm.keys().map(str::to_string).collect();
        let mut out = HashMap::with_capacity(keys.len());
        for key in keys {
            out.insert(key.clone(), wm.take(&key)?);
        }
        Ok(out)
    }

    pub fn tensor_snapshot_for_backbone(&self) -> Result<WeightSnapshot> {
        let mut wm = self.load_backbone()?;
        let keys: Vec<String> = wm.keys().map(str::to_string).collect();
        let mut out = HashMap::with_capacity(keys.len());
        for key in keys {
            out.insert(key.clone(), wm.take(&key)?);
        }
        Ok(out)
    }

    pub fn tensor_snapshot(&self, prefix: &str) -> Result<WeightSnapshot> {
        let mut wm = self.load_prefix(prefix)?;
        let keys: Vec<String> = wm.keys().map(str::to_string).collect();
        let mut out = HashMap::with_capacity(keys.len());
        for key in keys {
            out.insert(key.clone(), wm.take(&key)?);
        }
        Ok(out)
    }
}
