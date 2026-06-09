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

//! HuggingFace checkpoint download for FLUX.2 (`hf-download` feature).

#[cfg(feature = "hf-download")]
use anyhow::Context;
use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

/// Resolved on-disk layout for a FLUX.2 HF repo (diffusers-style tree).
#[derive(Debug, Clone)]
pub struct Flux2Checkpoint {
    pub root: PathBuf,
    pub transformer_weights: PathBuf,
    pub transformer_config: PathBuf,
    pub text_encoder_dir: PathBuf,
    pub vae_dir: PathBuf,
    pub tokenizer_path: PathBuf,
}

impl Flux2Checkpoint {
    pub fn from_root(root: PathBuf) -> Result<Self> {
        let transformer_weights = root.join("transformer/diffusion_pytorch_model.safetensors");
        let transformer_config = root.join("transformer/config.json");
        let text_encoder_dir = root.join("text_encoder");
        let vae_dir = root.join("vae");
        let tokenizer_path = root.join("tokenizer/tokenizer.json");
        if !transformer_weights.is_file() {
            bail!(
                "missing transformer weights at {:?} — expected HF layout transformer/diffusion_pytorch_model.safetensors",
                transformer_weights
            );
        }
        Ok(Self {
            root,
            transformer_weights,
            transformer_config,
            text_encoder_dir,
            vae_dir,
            tokenizer_path,
        })
    }
}

/// Download a FLUX.2 repo from HuggingFace Hub into the local cache and return paths.
#[cfg(feature = "hf-download")]
pub fn download_flux2_repo(repo_id: &str) -> Result<Flux2Checkpoint> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()?;
    let repo = api.model(repo_id.to_string());

    let files = [
        "transformer/diffusion_pytorch_model.safetensors",
        "transformer/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "text_encoder/model.safetensors",
        "text_encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ];
    for f in files {
        let _ = repo.get(f)?;
    }

    let root = repo
        .get("transformer/config.json")?
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .context("inferring repo root from cache")?;
    Flux2Checkpoint::from_root(root)
}

#[cfg(not(feature = "hf-download"))]
pub fn download_flux2_repo(_repo_id: &str) -> Result<Flux2Checkpoint> {
    bail!(
        "HF download requires `hf-download` feature — rebuild with \
         `rlx-models` feature `hf-download`"
    );
}

/// Download when `repo_id` is set, otherwise treat `local` as an existing tree.
pub fn resolve_flux2_checkpoint(
    repo_id: Option<&str>,
    local: Option<&Path>,
) -> Result<Flux2Checkpoint> {
    match (repo_id, local) {
        (Some(id), _) => download_flux2_repo(id),
        (None, Some(p)) => Flux2Checkpoint::from_root(p.to_path_buf()),
        (None, None) => bail!("pass --hf-repo or --weights / checkpoint directory"),
    }
}
