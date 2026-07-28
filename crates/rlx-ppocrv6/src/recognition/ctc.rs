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

//! CTC greedy decode + character dictionary.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

/// Character dictionary: index 0 is CTC blank; chars[i] is class i+1 in logits
/// when dict was loaded without an explicit blank (Paddle `CTCLabelDecode`).
#[derive(Debug, Clone)]
pub struct CharDict {
    /// Characters for class indices 1..N (blank = 0).
    pub chars: Vec<String>,
}

impl CharDict {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            fs::read_to_string(path).with_context(|| format!("read dict {}", path.display()))?;
        Ok(Self::from_lines(text.lines()))
    }

    pub fn from_embedded(text: &str) -> Self {
        Self::from_lines(text.lines())
    }

    fn from_lines<'a>(lines: impl Iterator<Item = &'a str>) -> Self {
        let mut chars: Vec<String> = lines.map(|l| l.to_string()).collect();
        // Paddle `use_space_char=True`: append space as the last class before blank offset.
        if chars.last().map(|c| c.as_str()) != Some(" ") {
            chars.push(" ".into());
        }
        Self { chars }
    }

    /// Number of CTC classes including blank (index 0).
    pub fn num_classes(&self) -> usize {
        self.chars.len() + 1
    }

    pub fn decode_greedy(&self, logits: &[f32], seq_len: usize, num_classes: usize) -> String {
        // logits layout: [seq, classes] row-major
        let mut prev = 0usize;
        let mut out = String::new();
        for t in 0..seq_len {
            let start = t * num_classes;
            let end = start + num_classes;
            if end > logits.len() {
                break;
            }
            let row = &logits[start..end];
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            if best_i != 0 && best_i != prev {
                let idx = best_i - 1;
                if idx < self.chars.len() {
                    out.push_str(&self.chars[idx]);
                }
            }
            prev = best_i;
        }
        out
    }
}

pub const TINY_DICT: &str = include_str!("../../assets/dicts/tiny_keys.txt");
pub const SMALL_DICT: &str = include_str!("../../assets/dicts/small_keys.txt");
