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

//! Model directory discovery and path layout.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::config::{DEFAULT_HF_REPO, ModelConfig};

/// Default local checkout from `just fetch-kittentts`.
pub const DEFAULT_LOCAL_DIR: &str = ".cache/kittentts-mini-0.8";

/// Resolved ONNX + voices (+ optional native weights) for one checkpoint directory.
#[derive(Debug, Clone)]
pub struct ModelLayout {
    pub dir: PathBuf,
    pub config: ModelConfig,
    pub onnx: PathBuf,
    pub voices: PathBuf,
    pub native_weights: Option<PathBuf>,
}

impl ModelLayout {
    pub fn resolve(model_dir: &Path) -> Result<Self> {
        let dir = model_dir
            .canonicalize()
            .unwrap_or_else(|_| model_dir.to_path_buf());
        let config = ModelConfig::load_from_dir(&dir)?;
        let onnx = dir.join(&config.model_file);
        let voices = dir.join(&config.voices);
        let native_weights = find_native_weights(&dir);
        if !onnx.is_file() && native_weights.is_none() {
            bail!(
                "ONNX model missing: {}\n\
                 Fetch weights: `just fetch-kittentts` or set RLX_KITTENTTS_DIR",
                onnx.display()
            );
        }
        if !voices.is_file() {
            bail!("voices NPZ missing: {}", voices.display());
        }
        Ok(Self {
            native_weights,
            dir,
            config,
            onnx,
            voices,
        })
    }

    /// Voice keys from the NPZ plus friendly alias names from config.
    pub fn voice_names(&self) -> Result<Vec<String>> {
        let raw = crate::load_npz(&self.voices)
            .with_context(|| format!("load voices {}", self.voices.display()))?;
        let mut names: Vec<String> = raw.into_keys().collect();
        for alias in self.config.voice_aliases.keys() {
            if !names.iter().any(|n| n == alias) {
                names.push(alias.clone());
            }
        }
        names.sort();
        Ok(names)
    }
}

/// Best-effort model directory for CLI and tests.
pub fn default_model_dir() -> Result<PathBuf> {
    for key in ["RLX_KITTENTTS_DIR", "KITTENTTS_MODEL_DIR"] {
        if let Ok(raw) = std::env::var(key) {
            let p = PathBuf::from(&raw);
            if layout_exists(&p) {
                return Ok(p);
            }
        }
    }

    let local = PathBuf::from(DEFAULT_LOCAL_DIR);
    if layout_exists(&local) {
        return Ok(local);
    }

    #[cfg(feature = "hf-download")]
    if let Ok(p) = hf_snapshot_dir(DEFAULT_HF_REPO) {
        return Ok(p);
    }

    if let Some(p) = model_dir_from_bundle_manifest() {
        return Ok(p);
    }

    if let Some(p) = hf_hub_cache_snapshot(DEFAULT_HF_REPO) {
        return Ok(p);
    }

    bail!(
        "KittenTTS weights not found.\n\
         Quick start:\n\
           just fetch-kittentts\n\
           just kittentts-demo\n\
         Or set RLX_KITTENTTS_DIR to a directory containing config.json, \
         the ONNX file, and voices.npz."
    )
}

pub fn layout_exists(dir: &Path) -> bool {
    dir.join("config.json").is_file()
}

fn home_dir() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

pub fn hf_hub_root() -> PathBuf {
    if let Ok(h) = std::env::var("HF_HOME") {
        return PathBuf::from(h).join("hub");
    }
    if let Ok(h) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(h);
    }
    home_dir().join(".cache").join("huggingface").join("hub")
}

/// Locate a cached HF snapshot without downloading (no `hf-hub` dependency).
fn hf_hub_cache_snapshot(repo_id: &str) -> Option<PathBuf> {
    let cache_name = format!("models--{}", repo_id.replace('/', "--"));
    let snapshots = hf_hub_root().join(cache_name).join("snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|snap| layout_exists(snap))
        .collect();
    candidates.sort();
    candidates.into_iter().last()
}

fn model_dir_from_bundle_manifest() -> Option<PathBuf> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../kitten_tts_mini_rlx/weights/rlx_bundle/manifest.json");
    let data = std::fs::read_to_string(&manifest).ok()?;
    let v: serde_json::Value = serde_json::from_str(&data).ok()?;
    let onnx = v.get("source_onnx")?.as_str()?;
    let dir = PathBuf::from(onnx).parent()?.to_path_buf();
    layout_exists(&dir).then_some(dir)
}

/// RLX ONNX bundle (`graph.json` + weights) under a native weights directory.
pub fn find_rlx_bundle(weights_dir: &Path) -> Option<PathBuf> {
    if let Ok(raw) =
        std::env::var("RLX_ONNX_BUNDLE").or_else(|_| std::env::var("KITTEN_RLX_BUNDLE"))
    {
        let p = PathBuf::from(raw);
        if p.join("graph.json").is_file() {
            return Some(p);
        }
    }
    let in_dir = weights_dir.join("rlx_bundle");
    if in_dir.join("graph.json").is_file() {
        return Some(in_dir);
    }
    None
}

/// Workspace decomposed weights (`crates/kitten_tts_mini_rlx/weights`).
pub fn default_native_weights_dir() -> Option<PathBuf> {
    if let Ok(raw) = std::env::var("KITTEN_RLX_WEIGHTS") {
        let p = PathBuf::from(raw);
        if p.join("model.safetensors").is_file() || p.join("rlx_bundle/graph.json").is_file() {
            return Some(p);
        }
    }
    let sibling = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kitten_tts_mini_rlx/weights");
    if sibling.join("model.safetensors").is_file() && find_rlx_bundle(&sibling).is_some() {
        return Some(sibling);
    }
    if find_rlx_bundle(&sibling).is_some() {
        return Some(sibling);
    }
    None
}

/// Decomposed RLX weights (`model.safetensors`), if present.
pub fn find_native_weights(model_dir: &Path) -> Option<PathBuf> {
    if let Ok(raw) = std::env::var("KITTEN_RLX_WEIGHTS") {
        let p = PathBuf::from(raw);
        if p.join("model.safetensors").is_file() || p.join("rlx_bundle/graph.json").is_file() {
            return Some(p);
        }
    }
    if model_dir.join("model.safetensors").is_file() {
        return Some(model_dir.to_path_buf());
    }
    if model_dir.join("rlx_bundle/graph.json").is_file() {
        return Some(model_dir.to_path_buf());
    }
    default_native_weights_dir()
}

#[cfg(feature = "hf-download")]
pub fn hf_snapshot_dir(repo_id: &str) -> Result<PathBuf> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(hf_hub_root())
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.model(normalize_repo_id(repo_id));
    let config = repo.get("config.json").with_context(|| {
        format!(
            "locate {repo_id} in Hugging Face cache under {}\n\
             Download once: `just fetch-kittentts`",
            hf_hub_root().display()
        )
    })?;
    config
        .parent()
        .map(Path::to_path_buf)
        .context("config.json has no parent (snapshot dir)")
}

#[cfg(not(feature = "hf-download"))]
pub fn hf_snapshot_dir(_repo_id: &str) -> Result<PathBuf> {
    bail!("rebuild with `--features hf-download` on rlx-kittentts")
}

pub fn normalize_repo_id(repo_id: &str) -> String {
    if repo_id.contains('/') {
        repo_id.to_string()
    } else {
        format!("KittenML/{repo_id}")
    }
}
