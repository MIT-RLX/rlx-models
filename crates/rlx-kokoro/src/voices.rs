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

//! Kokoro voice packs.
//!
//! Each `voices/<name>.bin` file is a raw little-endian `float32` array of shape
//! `[510, 256]` — one 256-d reference style vector per possible phoneme length.
//! At inference the row is selected by the number of content phonemes:
//! `style = voice[len(tokens)]`.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// Style-vector dimensionality (`ref_s`).
pub const STYLE_DIM: usize = 256;

/// A single voice: `nrows` reference style vectors of length [`STYLE_DIM`].
#[derive(Debug, Clone)]
pub struct Voice {
    nrows: usize,
    ncols: usize,
    data: Vec<f32>,
}

impl Voice {
    /// Parse a raw voice `.bin` (row-major `float32`, `ncols` = [`STYLE_DIM`]).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() % 4 != 0 {
            bail!("voice byte length {} is not a multiple of 4", bytes.len());
        }
        let n = bytes.len() / 4;
        if n % STYLE_DIM != 0 {
            bail!("voice element count {n} not divisible by style dim {STYLE_DIM}");
        }
        let ncols = STYLE_DIM;
        let nrows = n / ncols;
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok(Self { nrows, ncols, data })
    }

    /// Load a voice from a `.bin` file on disk.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("read voice: {}", path.display()))?;
        Self::from_bytes(&bytes).with_context(|| format!("parse voice: {}", path.display()))
    }

    /// Number of style rows.
    pub fn rows(&self) -> usize {
        self.nrows
    }

    /// The 256-d reference style vector for a phoneme content length.
    ///
    /// Clamped to the last row for over-long inputs.
    pub fn style_row(&self, content_len: usize) -> &[f32] {
        let i = content_len.min(self.nrows.saturating_sub(1));
        &self.data[i * self.ncols..(i + 1) * self.ncols]
    }
}

/// All voices discovered under a `voices/` directory, keyed by name.
#[derive(Debug, Clone, Default)]
pub struct VoiceBank {
    voices: BTreeMap<String, Voice>,
}

impl VoiceBank {
    /// Load every `*.bin` under `dir` into the bank.
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let mut voices = BTreeMap::new();
        let rd = std::fs::read_dir(dir)
            .with_context(|| format!("read voices dir: {}", dir.display()))?;
        for entry in rd {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                continue;
            }
            let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            match Voice::load(&path) {
                Ok(v) => {
                    voices.insert(name.to_string(), v);
                }
                Err(e) => eprintln!("[kokoro] skipping voice {}: {e:#}", path.display()),
            }
        }
        if voices.is_empty() {
            bail!("no voice .bin files found in {}", dir.display());
        }
        Ok(Self { voices })
    }

    /// Look up a voice by exact name.
    pub fn get(&self, name: &str) -> Option<&Voice> {
        self.voices.get(name)
    }

    /// Sorted list of available voice names.
    pub fn names(&self) -> Vec<String> {
        self.voices.keys().cloned().collect()
    }

    /// Number of loaded voices.
    pub fn len(&self) -> usize {
        self.voices.len()
    }

    /// Whether the bank is empty.
    pub fn is_empty(&self) -> bool {
        self.voices.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_row_major() {
        // 2 rows x STYLE_DIM cols
        let mut bytes = Vec::new();
        for r in 0..2u32 {
            for c in 0..STYLE_DIM as u32 {
                bytes.extend_from_slice(&((r * 1000 + c) as f32).to_le_bytes());
            }
        }
        let v = Voice::from_bytes(&bytes).unwrap();
        assert_eq!(v.rows(), 2);
        assert_eq!(v.style_row(0)[0], 0.0);
        assert_eq!(v.style_row(1)[0], 1000.0);
        assert_eq!(v.style_row(1)[5], 1005.0);
    }

    #[test]
    fn clamps_over_long() {
        let bytes = vec![0u8; STYLE_DIM * 4]; // 1 row
        let v = Voice::from_bytes(&bytes).unwrap();
        assert_eq!(v.style_row(999).len(), STYLE_DIM); // clamped, no panic
    }

    #[test]
    fn rejects_misshaped() {
        assert!(Voice::from_bytes(&[0u8; 10]).is_err());
    }
}
