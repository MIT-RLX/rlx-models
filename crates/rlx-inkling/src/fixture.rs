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

//! Load `tests/fixtures/hf_tiny_parity` dumps from
//! `scripts/dump_hf_tiny_parity.py`.

use crate::eager::TextWeights;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct IndexEntry {
    offset: usize,
    len: usize,
}

#[derive(Debug, Deserialize)]
pub struct FixtureMeta {
    pub input_ids: Vec<u32>,
    pub vocab_size: usize,
}

pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hf_tiny_parity")
}

pub fn load_text_weights(dir: impl AsRef<Path>) -> Result<TextWeights> {
    let dir = dir.as_ref();
    let index_text =
        std::fs::read_to_string(dir.join("weights_index.json")).context("weights_index.json")?;
    let index: HashMap<String, IndexEntry> =
        serde_json::from_str(&index_text).context("parse weights_index.json")?;
    let bytes = std::fs::read(dir.join("weights.bin")).context("weights.bin")?;
    if bytes.len() % 4 != 0 {
        bail!("weights.bin length {} not multiple of 4", bytes.len());
    }
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut tensors = HashMap::new();
    for (name, ent) in index {
        let end = ent.offset.checked_add(ent.len).context("offset overflow")?;
        if end > floats.len() {
            bail!("tensor {name} out of range ({end} > {})", floats.len());
        }
        tensors.insert(name, floats[ent.offset..end].to_vec());
    }
    Ok(TextWeights { tensors })
}

pub fn load_logits(dir: impl AsRef<Path>) -> Result<Vec<f32>> {
    let bytes = std::fs::read(dir.as_ref().join("logits.bin")).context("logits.bin")?;
    if bytes.len() % 4 != 0 {
        bail!("logits.bin bad len");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn load_meta(dir: impl AsRef<Path>) -> Result<FixtureMeta> {
    let text = std::fs::read_to_string(dir.as_ref().join("meta.json")).context("meta.json")?;
    Ok(serde_json::from_str(&text)?)
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}
