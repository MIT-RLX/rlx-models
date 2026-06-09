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

//! `config.json` schema from KittenTTS Hugging Face repos.

use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;

/// Default mini model on Hugging Face.
pub const DEFAULT_HF_REPO: &str = "KittenML/kitten-tts-mini-0.8";

/// Deserialised `config.json` from a KittenTTS repository.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[serde(rename = "type")]
    pub model_type: String,
    pub model_file: String,
    pub voices: String,
    #[serde(default)]
    pub speed_priors: HashMap<String, f32>,
    #[serde(default)]
    pub voice_aliases: HashMap<String, String>,
}

impl ModelConfig {
    pub fn load_from_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read config.json in {}", model_dir.display()))?;
        let config: Self =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        if !matches!(config.model_type.as_str(), "ONNX1" | "ONNX2") {
            anyhow::bail!(
                "unsupported model type '{}' in {} (expected ONNX1 or ONNX2)",
                config.model_type,
                path.display()
            );
        }
        Ok(config)
    }
}
