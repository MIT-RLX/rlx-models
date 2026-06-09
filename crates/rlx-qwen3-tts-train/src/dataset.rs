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

//! `train_with_codes.jsonl` from HF `prepare_data.py`.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct CodesRecord {
    pub audio: String,
    #[serde(default)]
    pub text: String,
    pub ref_audio: String,
    #[serde(default)]
    pub audio_codes: Vec<Vec<u32>>,
}

pub struct CodesDataset {
    pub records: Vec<CodesRecord>,
}

impl CodesDataset {
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| path.display().to_string())?;
        let mut records = Vec::new();
        for line in BufReader::new(f).lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            records.push(serde_json::from_str(line)?);
        }
        Ok(Self { records })
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}
