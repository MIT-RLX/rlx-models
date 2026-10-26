// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! The repository's `config.json` (`"type": "KITTEN2"`): token layout, decode
//! presets, generation defaults and where every artifact lives.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// Where text, audio and marker ids sit in the shared 158 241-row vocabulary.
///
/// Audio codec tokens occupy `[audio_id_base, audio_id_base + num_audio_tokens)`;
/// the prompt builder writes them as raw integers, the text tokenizer never
/// produces them.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenMap {
    pub audio_id_base: u32,
    pub num_audio_tokens: u32,
    pub speech_start_id: u32,
    pub speech_end_id: u32,
    pub text_start_id: u32,
    pub text_end_id: u32,
    pub start_id: u32,
    pub stop_id: u32,
    #[serde(default)]
    pub final_seg_id: u32,
    #[serde(default)]
    pub reference_speech_start_id: u32,
    #[serde(default)]
    pub reference_speech_end_id: u32,
    #[serde(default)]
    pub reference_text_start_id: u32,
    #[serde(default)]
    pub reference_text_end_id: u32,
}

impl TokenMap {
    /// One past the last audio codec id.
    pub fn audio_id_end(&self) -> u32 {
        self.audio_id_base + self.num_audio_tokens
    }

    /// Checkpoints trained with reference-role wrappers have nonzero
    /// `reference_*` ids; the rest reuse the target segment's own wrappers.
    pub fn uses_reference_role_tokens(&self) -> bool {
        self.reference_text_start_id != 0
    }

    pub fn is_audio(&self, id: u32) -> bool {
        (self.audio_id_base..self.audio_id_end()).contains(&id)
    }
}

/// One named sampling preset (`stable`, `expressive`).
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct DecodePreset {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    #[serde(default)]
    pub min_p: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    #[serde(default = "default_true")]
    pub use_reference_prompt: bool,
    #[serde(default)]
    pub repetition_window: Option<usize>,
    /// Text spliced before the target when the input carries expression tags
    /// (`"{emo: 1}"`).
    #[serde(default)]
    pub emotion_control: Option<String>,
    #[serde(default = "default_chunk_chars")]
    pub chunk_chars: usize,
    #[serde(default = "default_chunk_min_chars")]
    pub chunk_min_chars: usize,
}

fn default_true() -> bool {
    true
}
fn default_chunk_chars() -> usize {
    380
}
fn default_chunk_min_chars() -> usize {
    130
}

/// `config.json` of a KittenTTS 2 repository.
#[derive(Debug, Clone, Deserialize)]
pub struct Kitten2Config {
    #[serde(rename = "type")]
    pub kind: String,
    pub lm_dir: String,
    pub voices: String,
    #[serde(default)]
    pub speaker_embedding: Option<String>,
    pub speaker_embedding_dim: usize,
    pub sample_rate: u32,
    pub default_voice: String,
    pub default_preset: String,
    #[serde(default)]
    pub decoders: BTreeMap<String, String>,
    pub decode_presets: BTreeMap<String, DecodePreset>,
    pub generation: GenerationConfig,
    pub token_map: TokenMap,
    /// Packed 1.58-bit LM checkpoint (`lm/model-ternary.safetensors`).
    #[serde(default)]
    pub lm_packed: Option<String>,
    /// Per-voice S3Gen conditioning shipped for the llama.cpp fork; the same
    /// values `s3gen.embed_ref` derives, so built-in voices need neither the
    /// S3 tokenizer nor the CAM++ speaker encoder.
    #[serde(default)]
    pub cpp: Option<serde_json::Value>,
}

impl Kitten2Config {
    pub fn from_dir(repo: &Path) -> Result<Self> {
        let path = repo.join("config.json");
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        Self::from_json_str(&text)
    }

    pub fn from_json_str(text: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(text).context("parse KittenTTS 2 config.json")?;
        if cfg.kind != "KITTEN2" {
            bail!("config type {:?} is not KITTEN2", cfg.kind);
        }
        Ok(cfg)
    }

    pub fn preset(&self, name: &str) -> Result<DecodePreset> {
        self.decode_presets.get(name).copied().with_context(|| {
            format!(
                "unknown preset {name:?}; choose from {:?}",
                self.decode_presets.keys().collect::<Vec<_>>()
            )
        })
    }

    /// Path of the precomputed per-voice S3Gen conditioning
    /// (`cpp/default/voices.json`), if the repository declares one.
    pub fn s3gen_voices_path(&self) -> Option<String> {
        self.cpp
            .as_ref()?
            .pointer("/decoders/default/voices/file")?
            .as_str()
            .map(str::to_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = include_str!("../tests/fixtures/config.json");

    #[test]
    fn published_config_parses() {
        let cfg = Kitten2Config::from_json_str(CONFIG).unwrap();
        let tm = &cfg.token_map;
        assert_eq!(tm.audio_id_base, 151_675);
        assert_eq!(tm.audio_id_end(), 151_675 + 6561);
        assert!(tm.uses_reference_role_tokens());
        assert_eq!(cfg.sample_rate, 24_000);
        assert_eq!(cfg.preset("stable").unwrap().top_k, 50);
        assert_eq!(
            cfg.s3gen_voices_path().as_deref(),
            Some("cpp/default/voices.json")
        );
        assert_eq!(cfg.generation.repetition_window, Some(50));
    }
}
