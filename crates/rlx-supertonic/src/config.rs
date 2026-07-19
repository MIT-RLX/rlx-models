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

//! Supertonic-3 pipeline config (subset of `onnx/tts.json`).

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// HF repo shipping the ONNX subgraphs + configs + voices.
pub const DEFAULT_HF_REPO: &str = "Supertone/supertonic-3";

/// Default local checkout (centralized, gitignored TTS weights).
pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/supertonic-3";

/// Languages the model was trained on (used for the `<lang>` text wrapper).
pub const AVAILABLE_LANGS: &[&str] = &[
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
    "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
];

#[derive(Deserialize)]
struct RawConfig {
    ae: AeConfig,
    ttl: TtlConfig,
}

#[derive(Deserialize)]
struct AeConfig {
    sample_rate: u32,
    base_chunk_size: usize,
}

#[derive(Deserialize)]
struct TtlConfig {
    latent_dim: usize,
    chunk_compress_factor: usize,
}

/// The handful of values the ONNX inference glue needs.
#[derive(Debug, Clone, Copy)]
pub struct StConfig {
    pub sample_rate: u32,
    pub base_chunk_size: usize,
    pub chunk_compress_factor: usize,
    pub latent_dim: usize,
}

impl StConfig {
    pub fn load(onnx_dir: &Path) -> Result<Self> {
        let path = onnx_dir.join("tts.json");
        let bytes =
            std::fs::read(&path).with_context(|| format!("read tts.json: {}", path.display()))?;
        let raw: RawConfig =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        Ok(Self {
            sample_rate: raw.ae.sample_rate,
            base_chunk_size: raw.ae.base_chunk_size,
            chunk_compress_factor: raw.ttl.chunk_compress_factor,
            latent_dim: raw.ttl.latent_dim,
        })
    }

    /// Latent feature channels fed to the vector estimator / vocoder.
    pub fn latent_channels(&self) -> usize {
        self.latent_dim * self.chunk_compress_factor
    }

    /// Samples per latent frame (`base_chunk_size * chunk_compress_factor`).
    pub fn chunk_size(&self) -> usize {
        self.base_chunk_size * self.chunk_compress_factor
    }

    /// Latent length for a duration in seconds: `ceil(dur*sr / chunk_size)`.
    pub fn latent_len(&self, duration_secs: f32) -> usize {
        let wav_len = (duration_secs * self.sample_rate as f32) as usize;
        wav_len.div_ceil(self.chunk_size()).max(1)
    }
}
