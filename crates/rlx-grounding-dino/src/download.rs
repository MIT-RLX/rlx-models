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

//! Hugging Face hub integration for `IDEA-Research/grounding-dino-base`.

#[cfg(feature = "hf-cache")]
use anyhow::Result;
#[cfg(feature = "hf-cache")]
use std::path::PathBuf;

/// Default HF repo id for the base checkpoint.
pub const DEFAULT_REPO: &str = "IDEA-Research/grounding-dino-base";

/// Files we need from the repo.
#[cfg(feature = "hf-cache")]
const FILES: &[&str] = &[
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
];

/// Resolved local paths for a downloaded checkpoint.
#[cfg(feature = "hf-cache")]
#[derive(Debug, Clone)]
pub struct Resolved {
    pub config: PathBuf,
    pub weights: PathBuf,
    pub tokenizer: PathBuf,
}

/// Resolve (downloading if necessary) the checkpoint files from the HF cache.
#[cfg(feature = "hf-cache")]
pub fn resolve(repo: &str) -> Result<Resolved> {
    use hf_hub::api::sync::Api;
    let api = Api::new()?;
    let repo = api.model(repo.to_string());
    let mut paths: std::collections::HashMap<&str, PathBuf> = std::collections::HashMap::new();
    for f in FILES {
        match repo.get(f) {
            Ok(p) => {
                paths.insert(f, p);
            }
            Err(e) => {
                // tokenizer.json may be absent on some mirrors; tolerate optionals.
                if *f == "model.safetensors" || *f == "config.json" {
                    return Err(anyhow::anyhow!("failed to fetch {f}: {e}"));
                }
            }
        }
    }
    let config = paths
        .get("config.json")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("config.json missing"))?;
    let weights = paths
        .get("model.safetensors")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("model.safetensors missing"))?;
    let tokenizer = paths
        .get("tokenizer.json")
        .cloned()
        .unwrap_or_else(|| config.with_file_name("tokenizer.json"));
    Ok(Resolved {
        config,
        weights,
        tokenizer,
    })
}
