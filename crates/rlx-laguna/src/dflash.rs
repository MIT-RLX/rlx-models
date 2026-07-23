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

//! DFlash speculative decode scaffold for Laguna.
//!
//! Poolside ships separate **draft** checkpoints (block-diffusion drafting),
//! not an in-target MTP/EAGLE3 head:
//!
//! - [`poolside/Laguna-XS-2.1-DFlash`](https://huggingface.co/poolside/Laguna-XS-2.1-DFlash)
//! - [`poolside/Laguna-S-2.1-DFlash`](https://huggingface.co/poolside/Laguna-S-2.1-DFlash)
//!
//! Arch tag: `DFlashLagunaForCausalLM`. Typical `dflash_config` fields include
//! `block_size`, `target_layer_ids`, and a mask token. Distinct from
//! `rlx-eagle3` / Qwen3.5 MTP.
//!
//! **Status:** config parse only — propose/verify is not implemented.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

/// HF-style DFlash draft config (subset).
#[derive(Debug, Clone, Deserialize)]
pub struct DFlashConfig {
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    #[serde(default)]
    pub target_layer_ids: Vec<usize>,
    #[serde(default)]
    pub mask_token_id: Option<u32>,
}

fn default_block_size() -> usize {
    16
}

impl DFlashConfig {
    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read DFlash config {}", path.display()))?;
        serde_json::from_str(&raw).context("parse DFlash config JSON")
    }
}

/// Placeholder propose/verify entry — returns a hard error until draft weights
/// and block-diffusion verify are wired.
pub fn propose_and_verify(
    _cfg: &DFlashConfig,
    _prompt_ids: &[u32],
    _max_draft: usize,
) -> Result<Vec<u32>> {
    bail!(
        "rlx-laguna: DFlash not implemented — load draft from \
         poolside/Laguna-*-DFlash and wire block-diffusion propose/verify \
         (see module docs)"
    )
}

pub fn status() -> &'static str {
    "scaffolded — parse DFlashConfig only; no draft runtime"
}
