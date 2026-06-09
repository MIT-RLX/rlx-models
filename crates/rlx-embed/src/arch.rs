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

//! Architecture detection for embedding checkpoints.

use anyhow::Result;
use rlx_core::gguf_config::{EmbedGgufKind, embed_gguf_kind};
use rlx_gguf::GgufFile;
use std::path::Path;

/// Detected embedding architecture from `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Bert,
    NomicBert,
    NomicVision,
}

/// Detect architecture from GGUF `general.architecture`.
pub fn detect_arch_from_gguf(weights_path: &Path) -> Result<Arch> {
    let raw = GgufFile::from_path(weights_path)?;
    Ok(match embed_gguf_kind(&raw)? {
        EmbedGgufKind::Bert => Arch::Bert,
        EmbedGgufKind::NomicBert => Arch::NomicBert,
    })
}

/// Detect architecture from config.json fields.
pub fn detect_arch(config_path: &Path) -> Result<Arch> {
    let data = std::fs::read_to_string(config_path)?;
    let json: serde_json::Value = serde_json::from_str(&data)?;

    if json.get("img_size").is_some() && json.get("patch_size").is_some() {
        return Ok(Arch::NomicVision);
    }
    if json.get("rotary_emb_base").is_some() || json.get("rotary_emb_fraction").is_some() {
        return Ok(Arch::NomicBert);
    }
    Ok(Arch::Bert)
}

/// Default pooling heuristic from HuggingFace repo id.
pub fn default_pooling(repo_id: &str) -> super::Pooling {
    let lower = repo_id.to_lowercase();
    if lower.contains("bge") || lower.contains("nomic") {
        super::Pooling::Cls
    } else {
        super::Pooling::Mean
    }
}
