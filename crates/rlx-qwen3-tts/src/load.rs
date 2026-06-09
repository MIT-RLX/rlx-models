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

//! Weight loading from `model.safetensors` (HuggingFace layout).

use crate::config::{WEIGHTS_FILE, resolve_model_dir};
use anyhow::{Result, bail};
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_core::weight_map::WeightMap;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub type TensorSnapshot = HashMap<String, (Vec<f32>, Vec<usize>)>;

pub const PREFIX_TALKER: &str = "talker.";

#[derive(Clone)]
pub struct Qwen3TtsWeightStore {
    dir: PathBuf,
    checkpoint: Arc<SafetensorsCheckpoint>,
    keys: Arc<HashSet<String>>,
    /// Memoize `tensor_snapshot` / prompt loads — talker, CP, and prompt share codec tables.
    tensor_cache: Arc<Mutex<TensorSnapshot>>,
}

impl Qwen3TtsWeightStore {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let dir = resolve_model_dir(model_dir)?;
        let weights = dir.join(WEIGHTS_FILE);
        if !weights.is_file() {
            bail!("missing {} — run `just fetch-qwen3-tts`", weights.display());
        }
        let checkpoint = Arc::new(SafetensorsCheckpoint::open(&dir)?);
        let keys = Arc::new(checkpoint.keys().map(str::to_string).collect());
        Ok(Self {
            dir,
            checkpoint,
            keys,
            tensor_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.dir
    }

    pub fn keys(&self) -> &HashSet<String> {
        &self.keys
    }

    pub fn load_selected(&self, want: &HashSet<String>) -> Result<WeightMap> {
        self.checkpoint.load_selected(want)
    }

    pub fn load_talker_backbone(&self) -> Result<WeightMap> {
        let want: HashSet<String> = self
            .keys
            .iter()
            .filter(|k| {
                k.starts_with("talker.model.")
                    && !k.contains("code_predictor")
                    && k.as_str() != "talker.codec_head.weight"
                    && k.as_str() != "talker.text_projection.weight"
            })
            .cloned()
            .collect();
        if want.is_empty() {
            bail!("no talker backbone tensors under {:?}", self.dir);
        }
        self.load_selected(&want)
    }

    pub fn load_code_predictor_backbone(&self) -> Result<WeightMap> {
        let want: HashSet<String> = self
            .keys
            .iter()
            .filter(|k| k.starts_with("talker.code_predictor.model."))
            .cloned()
            .collect();
        if want.is_empty() {
            bail!("no code_predictor tensors under {:?}", self.dir);
        }
        self.load_selected(&want)
    }

    pub fn take_codec_head(&self) -> Result<(Vec<f32>, Vec<usize>)> {
        let mut wm = self.load_selected(&HashSet::from(["talker.codec_head.weight".into()]))?;
        wm.take("talker.codec_head.weight")
    }

    pub fn tensor_snapshot(&self, keys: &[&str]) -> Result<TensorSnapshot> {
        let mut out = HashMap::with_capacity(keys.len());
        let mut missing = Vec::new();
        {
            let cache = self.tensor_cache.lock().expect("tensor_cache");
            for k in keys {
                let key = (*k).to_string();
                if let Some(v) = cache.get(&key) {
                    out.insert(key, v.clone());
                } else {
                    missing.push(key);
                }
            }
        }
        if missing.is_empty() {
            return Ok(out);
        }
        let want: HashSet<String> = missing.iter().cloned().collect();
        let mut wm = self.load_selected(&want)?;
        let mut cache = self.tensor_cache.lock().expect("tensor_cache");
        for k in missing {
            let v = wm.take(&k)?;
            cache.insert(k.clone(), v.clone());
            out.insert(k, v);
        }
        Ok(out)
    }

    /// Share talker codec embedding table with CP megakernel (one mmap-backed copy).
    pub fn talker_codec_embedding_flat(&self) -> Result<Vec<f32>> {
        let snap = self.tensor_snapshot(&["talker.model.codec_embedding.weight"])?;
        Ok(snap["talker.model.codec_embedding.weight"].0.clone())
    }
}

fn map_talker_key(hf: &str) -> Option<String> {
    let rest = hf.strip_prefix("talker.")?;
    match rest {
        "model.codec_embedding.weight" => Some("model.embed_tokens.weight".into()),
        s if s.starts_with("model.") => Some(rest.to_string()),
        _ => None,
    }
}

fn map_code_predictor_key(hf: &str) -> Option<String> {
    let rest = hf.strip_prefix("talker.code_predictor.")?;
    match rest {
        s if s.starts_with("model.") => Some(rest.to_string()),
        _ => None,
    }
}

/// Populate Qwen3-canonical names from code-predictor tensors.
pub fn remap_code_predictor_weights(wm: &mut WeightMap) -> Result<TensorSnapshot> {
    let keys: Vec<String> = wm.keys().map(str::to_string).collect();
    let mut out = HashMap::new();
    for k in keys {
        if let Some(mapped) = map_code_predictor_key(&k) {
            let v = wm.take(&k)?;
            out.insert(mapped, v);
        }
    }
    Ok(out)
}

/// Populate Qwen3-canonical names from talker tensors.
pub fn remap_talker_weights(talker_wm: &mut WeightMap) -> Result<TensorSnapshot> {
    let keys: Vec<String> = talker_wm.keys().map(str::to_string).collect();
    let mut out = HashMap::new();
    for k in keys {
        if let Some(mapped) = map_talker_key(&k) {
            let v = talker_wm.take(&k)?;
            out.insert(mapped, v);
        }
    }
    Ok(out)
}
