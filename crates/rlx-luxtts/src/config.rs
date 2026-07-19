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

//! LuxTTS token table + directory layout.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// HF repo shipping the ONNX subgraphs + tokens + vocoder.
pub const DEFAULT_HF_REPO: &str = "YatharthS/LuxTTS";
/// Default local checkout (centralized, gitignored TTS weights).
pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/luxtts";

/// `tokens.txt` phoneme → id table (`"<token> <id>"` per line, `_`=pad=0).
#[derive(Debug, Clone)]
pub struct Tokens {
    map: HashMap<String, i64>,
}

impl Tokens {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("tokens.txt");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read tokens.txt: {}", path.display()))?;
        let mut map = HashMap::new();
        for line in text.lines() {
            let mut it = line.rsplitn(2, char::is_whitespace);
            let (Some(id), Some(tok)) = (it.next(), it.next()) else {
                continue;
            };
            if let Ok(id) = id.trim().parse::<i64>() {
                map.insert(tok.to_string(), id);
            }
        }
        anyhow::ensure!(!map.is_empty(), "empty tokens.txt");
        Ok(Self { map })
    }

    /// Look up a single-character phoneme token.
    pub fn id_of(&self, c: char) -> Option<i64> {
        let mut buf = [0u8; 4];
        self.map.get(c.encode_utf8(&mut buf) as &str).copied()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Resolve the ONNX/token/vocoder paths under a model directory.
#[derive(Debug, Clone)]
pub struct Layout {
    pub dir: PathBuf,
    pub text_encoder: PathBuf,
    /// Single-input encoder BODY (native path): `[/Pad_output_0 [1,S]] →
    /// encoder output [1,S,100]`. Produced by `scripts/export_encoder_body.py`
    /// from `text_encoder.onnx` (splits off the derived-length token concat+pad
    /// and the scalar length regulator so the graph runs natively; both are
    /// re-done in Rust). Optional — absent when only the ort path is used.
    pub encoder_body: Option<PathBuf>,
    pub fm_decoder: PathBuf,
    pub vocoder_spec: PathBuf,
}

impl Layout {
    pub fn resolve(dir: &Path) -> Result<Self> {
        let dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        let find = |names: &[&str]| -> Result<PathBuf> {
            names
                .iter()
                .map(|n| dir.join(n))
                .find(|p| p.is_file())
                .with_context(|| format!("missing any of {names:?} in {}", dir.display()))
        };
        let opt = |names: &[&str]| -> Option<PathBuf> {
            names.iter().map(|n| dir.join(n)).find(|p| p.is_file())
        };
        Ok(Self {
            text_encoder: find(&["text_encoder.onnx"])?,
            encoder_body: opt(&["encoder_body.onnx", "onnx/encoder_body.onnx"]),
            fm_decoder: find(&["fm_decoder.onnx"])?,
            vocoder_spec: find(&["onnx/vocoder_spec.onnx", "vocoder_spec.onnx"])?,
            dir,
        })
    }
}
