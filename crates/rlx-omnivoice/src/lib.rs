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

//! # rlx-omnivoice
//!
//! **OmniVoice** — a massively-multilingual (**646+ languages**) voice-design TTS
//! on RLX. An LM-style backbone emits neural-codec tokens conditioned on text, a
//! language id, and a voice-design (natural-language description → style)
//! embedding; a neural codec renders the waveform.
//!
//! Native Rust, composing rlx pieces:
//!
//! - **Backbone** → Llama-shaped LM (`rlx-llama32`).
//! - **Codec** → a neural audio codec (`rlx-dac` / `rlx-snac`).
//!
//! Supporting 646+ languages means language ids are **ISO 639-3** (three-letter),
//! so the checkpoint-free core here is the config plus correct 639-3 code handling
//! ([`normalize_language`]). LM + codec + voice-design conditioning wiring is next.

use anyhow::{Result, ensure};

/// OmniVoice model config. Dimensional fields carry plausible values; exact widths
/// come from the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct OmniVoiceConfig {
    pub sample_rate: usize,
    pub hop_length: usize,
    // LM backbone.
    pub backbone_hidden: usize,
    pub backbone_layers: usize,
    pub backbone_heads: usize,
    // Neural codec.
    pub num_codebooks: usize,
    pub codebook_size: usize,
    /// Number of supported languages (639-3 ids).
    pub num_languages: usize,
    /// Language-id embedding width.
    pub language_embed_dim: usize,
    /// Voice-design (description → style) conditioning width.
    pub voice_design_dim: usize,
}

impl Default for OmniVoiceConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24_000,
            hop_length: 320,
            backbone_hidden: 1536,
            backbone_layers: 28,
            backbone_heads: 16,
            num_codebooks: 4,
            codebook_size: 1024,
            num_languages: 646,
            language_embed_dim: 256,
            voice_design_dim: 512,
        }
    }
}

impl OmniVoiceConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.num_codebooks > 0, "num_codebooks must be > 0");
        ensure!(self.codebook_size > 0, "codebook_size must be > 0");
        ensure!(self.num_languages > 0, "num_languages must be > 0");
        Ok(())
    }

    pub fn frames_per_second(&self) -> f32 {
        self.sample_rate as f32 / self.hop_length as f32
    }
}

/// Normalise a language tag to a canonical **ISO 639-3** code: trimmed, lowercased,
/// and required to be exactly three ASCII letters (e.g. `"ENG "` → `"eng"`).
/// Returns `None` for anything that isn't a well-formed 639-3 code.
pub fn normalize_language(code: &str) -> Option<String> {
    let t = code.trim();
    if t.len() == 3 && t.chars().all(|c| c.is_ascii_alphabetic()) {
        Some(t.to_ascii_lowercase())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_validate() {
        let c = OmniVoiceConfig::default();
        assert_eq!(c.num_languages, 646);
        assert_eq!(c.sample_rate, 24_000);
        c.validate().unwrap();
    }

    #[test]
    fn language_normalization_accepts_iso639_3() {
        assert_eq!(normalize_language("eng").as_deref(), Some("eng"));
        assert_eq!(normalize_language("ENG").as_deref(), Some("eng"));
        assert_eq!(normalize_language("  jpn ").as_deref(), Some("jpn"));
        assert_eq!(normalize_language("cmn").as_deref(), Some("cmn"));
    }

    #[test]
    fn language_normalization_rejects_malformed() {
        assert_eq!(normalize_language("en"), None); // 2-letter (639-1)
        assert_eq!(normalize_language("engg"), None); // too long
        assert_eq!(normalize_language("e1g"), None); // non-alpha
        assert_eq!(normalize_language(""), None);
    }
}
