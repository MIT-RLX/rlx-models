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

//! Where a stage's parameters come from — a `.safetensors` file, or tensors already in
//! memory (unpacked from an `.rlxp`).

use anyhow::{Context, Result};
use rlx_core::weight_map::WeightMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Tensor payloads keyed by parameter name: `(values, shape)`.
pub type Tensors = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// A stage's parameter source.
///
/// Graph building drains a [`WeightMap`] (`take` removes each entry), and the recognizer
/// builds one graph per line width, so a source has to be able to hand out a *fresh* map
/// per build rather than a shared one.
#[derive(Clone)]
pub enum WeightSource {
    /// A `.safetensors` file, re-read per build.
    File(PathBuf),
    /// Tensors held in memory, cloned per build.
    Memory(Arc<Tensors>),
}

impl WeightSource {
    pub fn memory(tensors: Tensors) -> Self {
        Self::Memory(Arc::new(tensors))
    }

    /// A fresh [`WeightMap`] for one graph build.
    pub fn weight_map(&self) -> Result<WeightMap> {
        match self {
            Self::File(p) => {
                let s = p
                    .to_str()
                    .with_context(|| format!("weights path is not UTF-8: {}", p.display()))?;
                WeightMap::from_file(s)
            }
            Self::Memory(t) => Ok(WeightMap::from_tensors((**t).clone())),
        }
    }
}

impl From<&Path> for WeightSource {
    fn from(p: &Path) -> Self {
        Self::File(p.to_path_buf())
    }
}

impl From<PathBuf> for WeightSource {
    fn from(p: PathBuf) -> Self {
        Self::File(p)
    }
}

/// Parse a `codemap.txt` (whitespace-separated decimal codepoints, one per class).
pub fn parse_codemap(text: &str) -> Result<Vec<u32>> {
    text.split_whitespace()
        .map(|s| {
            s.parse::<u32>()
                .with_context(|| format!("bad codemap entry {s:?}"))
        })
        .collect()
}
