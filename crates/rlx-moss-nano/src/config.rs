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

//! MOSS-TTS-Nano config, prompt templates, and builtin voices — parsed from the
//! `browser_poc_manifest.json` that ships with the ONNX export.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct TtsConfig {
    pub n_vq: usize,
    pub audio_pad_token_id: i32,
    pub pad_token_id: i32,
    pub im_start_token_id: i32,
    pub im_end_token_id: i32,
    pub audio_start_token_id: i32,
    pub audio_end_token_id: i32,
    pub audio_user_slot_token_id: i32,
    pub audio_assistant_slot_token_id: i32,
    pub vocab_size: usize,
}

impl TtsConfig {
    /// Row width = 1 text/slot column + `n_vq` audio-code columns.
    pub fn row_width(&self) -> usize {
        self.n_vq + 1
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationDefaults {
    pub max_new_frames: usize,
    #[serde(default = "default_true")]
    pub do_sample: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct PromptTemplates {
    pub user_prompt_prefix_token_ids: Vec<i32>,
    pub user_prompt_after_reference_token_ids: Vec<i32>,
    pub assistant_prompt_prefix_token_ids: Vec<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuiltinVoice {
    pub voice: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub group: String,
    /// `[n_frames][n_vq]` pre-computed reference-audio codes.
    pub prompt_audio_codes: Vec<Vec<i32>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub tts_config: TtsConfig,
    pub generation_defaults: GenerationDefaults,
    pub prompt_templates: PromptTemplates,
    pub builtin_voices: Vec<BuiltinVoice>,
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
    }

    pub fn voice(&self, name: &str) -> Option<&BuiltinVoice> {
        self.builtin_voices.iter().find(|v| v.voice.eq_ignore_ascii_case(name))
    }

    pub fn voice_names(&self) -> Vec<String> {
        self.builtin_voices.iter().map(|v| v.voice.clone()).collect()
    }
}

/// Codec output format (from `codec_browser_onnx_meta.json`).
#[derive(Debug, Clone, Copy)]
pub struct CodecInfo {
    pub sample_rate: u32,
    pub channels: u16,
}

impl Default for CodecInfo {
    fn default() -> Self {
        Self { sample_rate: 48000, channels: 2 }
    }
}
