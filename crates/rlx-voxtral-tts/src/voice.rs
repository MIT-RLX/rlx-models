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

//! Preset voice embeddings (`voice_embedding/*.pt` or converted `.f32`).

use anyhow::{Context, Result, bail, ensure};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct VoiceEmbedding {
    pub name: String,
    /// Flattened `[n_tokens * hidden]`.
    pub data: Vec<f32>,
    pub n_tokens: usize,
    pub hidden: usize,
}

impl VoiceEmbedding {
    pub fn load(model_dir: &Path, voice: &str, hidden: usize) -> Result<Self> {
        let converted = model_dir
            .join("voice_embedding")
            .join(format!("{voice}.f32"));
        if converted.is_file() {
            return Self::load_f32(&converted, voice, hidden);
        }
        let pt = model_dir
            .join("voice_embedding")
            .join(format!("{voice}.pt"));
        if pt.is_file() {
            bail!(
                "found {} but no converted .f32 embedding.\n\
                 Run: just voxtral-tts-prepare-voices",
                pt.display(),
            );
        }
        bail!(
            "voice {voice:?} not found under {}/voice_embedding/",
            model_dir.display()
        );
    }

    pub fn load_f32(path: &Path, name: &str, hidden: usize) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        ensure!(
            bytes.len().is_multiple_of(4),
            "voice embedding file size not multiple of 4"
        );
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        ensure!(
            !data.is_empty() && data.len().is_multiple_of(hidden),
            "invalid voice embedding length {} for hidden={hidden}",
            data.len()
        );
        Ok(Self {
            name: name.to_string(),
            n_tokens: data.len() / hidden,
            hidden,
            data,
        })
    }

    pub fn row(&self, idx: usize) -> &[f32] {
        let start = idx * self.hidden;
        &self.data[start..start + self.hidden]
    }

    pub fn rows(&self) -> impl Iterator<Item = &[f32]> {
        (0..self.n_tokens).map(move |i| self.row(i))
    }

    pub fn save_f32(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("create {}", parent.display()))?;
            }
        }
        let bytes: Vec<u8> = self.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }
}

pub fn voice_dir(model_dir: &Path) -> PathBuf {
    model_dir.join("voice_embedding")
}
