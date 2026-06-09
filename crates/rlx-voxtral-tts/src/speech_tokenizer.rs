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

//! Native Tekken speech prompt tokenization (replaces Docker `mistral_common`).

use crate::config::TEKKEN_FILE;
use crate::tokens::PRESET_VOICES;
use anyhow::{Context, Result, bail, ensure};
use kitoken::Kitoken;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const BOS_TOKEN_ID: u32 = 1;
const AUDIO_TOKEN_ID: u32 = 24;
const BEGIN_AUDIO_TOKEN_ID: u32 = 25;
const REPEAT_AUDIO_TEXT_TOKEN_ID: u32 = 35;
const NEXT_AUDIO_TEXT_TOKEN_ID: u32 = 36;

#[derive(Debug, Deserialize)]
struct TekkenRoot {
    #[serde(default)]
    audio: Option<TekkenAudio>,
}

#[derive(Debug, Deserialize)]
struct TekkenAudio {
    voice_num_audio_tokens: HashMap<String, u32>,
}

pub struct SpeechTokenizer {
    tok: Kitoken,
    voice_audio_counts: HashMap<String, u32>,
}

impl SpeechTokenizer {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let tekken_path = model_dir.join(TEKKEN_FILE);
        ensure!(
            tekken_path.is_file(),
            "missing {} under {}",
            TEKKEN_FILE,
            model_dir.display()
        );
        Self::from_tekken_file(&tekken_path)
    }

    pub fn from_tekken_file(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let meta: TekkenRoot =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
        let tok = Kitoken::from_file(path)
            .map_err(|e| anyhow::anyhow!("load tekken tokenizer {path:?}: {e}"))?;
        let voice_audio_counts = meta
            .audio
            .map(|a| a.voice_num_audio_tokens)
            .unwrap_or_default();
        Ok(Self {
            tok,
            voice_audio_counts,
        })
    }

    pub fn encode_speech(&self, text: &str, voice: &str) -> Result<Vec<u32>> {
        ensure!(!text.is_empty(), "speech text must be non-empty");
        let n_audio = self
            .voice_audio_counts
            .get(voice)
            .copied()
            .with_context(|| {
                format!(
                    "unknown voice {voice:?}; expected one of {}",
                    self.voice_list()
                )
            })?;
        self.encode_speech_with_n_audio(text, n_audio)
    }

    /// Build a speech prompt with a custom audio-slot count (cloned / reference embeddings).
    pub fn encode_speech_with_n_audio(&self, text: &str, n_audio: u32) -> Result<Vec<u32>> {
        ensure!(!text.is_empty(), "speech text must be non-empty");
        ensure!(
            n_audio > 0,
            "voice embedding must contain at least one audio frame"
        );
        let text_ids = self.encode_text(text)?;
        Ok(self.build_prompt(n_audio, &text_ids))
    }

    /// Count audio placeholder tokens (`24`) in a pre-built prompt.
    pub fn count_audio_slots(token_ids: &[u32]) -> usize {
        token_ids.iter().filter(|&&id| id == AUDIO_TOKEN_ID).count()
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.tok
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode text: {e}"))
    }

    fn build_prompt(&self, n_audio: u32, text_ids: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(6 + n_audio as usize + text_ids.len());
        out.push(BOS_TOKEN_ID);
        out.push(BEGIN_AUDIO_TOKEN_ID);
        out.extend(std::iter::repeat_n(AUDIO_TOKEN_ID, n_audio as usize));
        out.push(NEXT_AUDIO_TEXT_TOKEN_ID);
        out.extend_from_slice(text_ids);
        out.push(REPEAT_AUDIO_TEXT_TOKEN_ID);
        out.push(BEGIN_AUDIO_TOKEN_ID);
        out
    }

    pub fn voice_list(&self) -> String {
        if self.voice_audio_counts.is_empty() {
            PRESET_VOICES.join(", ")
        } else {
            let mut names: Vec<_> = self.voice_audio_counts.keys().cloned().collect();
            names.sort();
            names.join(", ")
        }
    }

    pub fn write_prompt_tokens(path: &Path, tokens: &[u32]) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("create {}", parent.display()))?;
            }
        }
        let body: String = tokens
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        std::fs::write(path, format!("{body}\n"))
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }
}

pub fn default_prompt_tokens_path() -> PathBuf {
    PathBuf::from(".cache/voxtral/tts/prompt_tokens.txt")
}

pub fn resolve_voice(_model_dir: &Path, voice: &str) -> Result<String> {
    if PRESET_VOICES.contains(&voice) {
        return Ok(voice.to_string());
    }
    bail!(
        "unknown voice {voice:?}; use --list-voices or one of: {}",
        PRESET_VOICES.join(", ")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> Option<PathBuf> {
        std::env::var("RLX_VOXTRAL_TTS_DIR")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.join(TEKKEN_FILE).is_file())
    }

    #[test]
    fn hello_world_matches_mistral_common_layout() {
        let Some(dir) = model_dir() else {
            eprintln!("skip: set RLX_VOXTRAL_TTS_DIR with tekken.json");
            return;
        };
        let tok = SpeechTokenizer::from_model_dir(&dir).expect("tokenizer");
        let ids = tok
            .encode_speech("Hello world", "neutral_female")
            .expect("encode");
        assert_eq!(ids.len(), 225);
        assert_eq!(ids[0], 1);
        assert_eq!(ids[1], 25);
        assert!(ids[2..220].iter().all(|&x| x == 24));
        assert_eq!(&ids[220..], &[36, 22177, 4304, 35, 25]);
    }

    #[test]
    fn parity_phrase_tail_tokens() {
        let Some(dir) = model_dir() else {
            eprintln!("skip: set RLX_VOXTRAL_TTS_DIR with tekken.json");
            return;
        };
        let tok = SpeechTokenizer::from_model_dir(&dir).expect("tokenizer");
        let ids = tok
            .encode_speech("Hello from RLX native parity.", "neutral_female")
            .expect("encode");
        assert_eq!(ids.len(), 230);
        assert_eq!(
            &ids[ids.len() - 10..],
            &[36, 22177, 1562, 105863, 1088, 15191, 73085, 1046, 35, 25]
        );
    }

    #[test]
    fn custom_n_audio_slot_count() {
        let Some(dir) = model_dir() else {
            eprintln!("skip: set RLX_VOXTRAL_TTS_DIR with tekken.json");
            return;
        };
        let tok = SpeechTokenizer::from_model_dir(&dir).expect("tokenizer");
        let ids = tok
            .encode_speech_with_n_audio("Hello from a cloned voice.", 50)
            .expect("encode");
        assert_eq!(SpeechTokenizer::count_audio_slots(&ids), 50);
        assert_eq!(ids.len(), 229 - 218 + 50);
    }
}
