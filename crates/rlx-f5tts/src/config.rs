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

//! F5-TTS vocab + directory layout.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub const DEFAULT_HF_REPO: &str = "huggingfacess/F5-TTS-ONNX";
pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/f5tts";

/// Audio sample rate + vocoder hop.
pub const SAMPLE_RATE: u32 = 24000;
pub const HOP_LENGTH: usize = 256;
/// Default number of function evaluations (denoising steps).
pub const DEFAULT_NFE: usize = 32;

/// `vocab.txt` char → id table (line index = id, matching the F5 reference).
#[derive(Debug, Clone)]
pub struct Vocab {
    map: HashMap<String, i32>,
}

impl Vocab {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("vocab.txt");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read vocab.txt: {}", path.display()))?;
        // Reference: `for i, char in enumerate(f): vocab[char[:-1]] = i`.
        let map: HashMap<String, i32> =
            text.lines().enumerate().map(|(i, l)| (l.to_string(), i as i32)).collect();
        anyhow::ensure!(!map.is_empty(), "empty vocab.txt");
        Ok(Self { map })
    }

    /// Look up a token string; unknown → 0 (matches `vocab_char_map.get(c, 0)`).
    pub fn id_of(&self, tok: &str) -> i32 {
        self.map.get(tok).copied().unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Resolved ONNX paths.
#[derive(Debug, Clone)]
pub struct Layout {
    pub dir: PathBuf,
    pub preprocess: PathBuf,
    pub transformer: PathBuf,
    pub decode: PathBuf,
}

impl Layout {
    pub fn resolve(dir: &Path) -> Result<Self> {
        let dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        let need = |n: &str| -> Result<PathBuf> {
            let p = dir.join(n);
            p.is_file().then_some(p).with_context(|| format!("missing {n} in {}", dir.display()))
        };
        Ok(Self {
            preprocess: need("F5_Preprocess.onnx")?,
            transformer: need("F5_Transformer.onnx")?,
            decode: need("F5_Decode.onnx")?,
            dir,
        })
    }
}
