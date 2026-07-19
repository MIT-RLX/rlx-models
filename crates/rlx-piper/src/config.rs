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

//! Piper `<voice>.onnx.json` config.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

/// A Piper voices repo (MIT). Individual voices live under lang/region paths.
pub const DEFAULT_HF_REPO: &str = "rhasspy/piper-voices";
pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/piper";

#[derive(Deserialize)]
struct RawConfig {
    audio: Audio,
    espeak: Espeak,
    #[serde(default)]
    inference: Inference,
    phoneme_id_map: HashMap<String, Vec<i64>>,
}

#[derive(Deserialize)]
struct Audio {
    sample_rate: u32,
}
#[derive(Deserialize)]
struct Espeak {
    voice: String,
}
#[derive(Deserialize)]
struct Inference {
    #[serde(default = "def_noise")]
    noise_scale: f32,
    #[serde(default = "def_length")]
    length_scale: f32,
    #[serde(default = "def_noisew")]
    noise_w: f32,
}
impl Default for Inference {
    fn default() -> Self {
        Self {
            noise_scale: def_noise(),
            length_scale: def_length(),
            noise_w: def_noisew(),
        }
    }
}
fn def_noise() -> f32 {
    0.667
}
fn def_length() -> f32 {
    1.0
}
fn def_noisew() -> f32 {
    0.8
}

/// Parsed Piper voice config.
#[derive(Debug, Clone)]
pub struct PiperConfig {
    pub sample_rate: u32,
    pub espeak_voice: String,
    pub noise_scale: f32,
    pub length_scale: f32,
    pub noise_w: f32,
    pub phoneme_id_map: HashMap<String, Vec<i64>>,
}

impl PiperConfig {
    pub fn load(json_path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(json_path).with_context(|| format!("read {}", json_path.display()))?;
        let raw: RawConfig = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse {}", json_path.display()))?;
        Ok(Self {
            sample_rate: raw.audio.sample_rate,
            espeak_voice: raw.espeak.voice,
            noise_scale: raw.inference.noise_scale,
            length_scale: raw.inference.length_scale,
            noise_w: raw.inference.noise_w,
            phoneme_id_map: raw.phoneme_id_map,
        })
    }

    /// The single id for a special/phoneme symbol, if mapped.
    pub fn id_of(&self, sym: &str) -> Option<i64> {
        self.phoneme_id_map
            .get(sym)
            .and_then(|v| v.first().copied())
    }
}

/// Locate `<voice>.onnx` + `<voice>.onnx.json` in a directory.
pub fn find_voice(dir: &Path) -> Result<(PathBuf, PathBuf)> {
    let onnx = std::fs::read_dir(dir)
        .with_context(|| format!("read dir {}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.extension().and_then(|x| x.to_str()) == Some("onnx")
                && p.to_str()
                    .map(|s| !s.ends_with(".onnx.json"))
                    .unwrap_or(false)
        })
        .with_context(|| format!("no .onnx voice in {}", dir.display()))?;
    let json = PathBuf::from(format!("{}.json", onnx.display()));
    anyhow::ensure!(json.is_file(), "missing {}", json.display());
    Ok((onnx, json))
}
