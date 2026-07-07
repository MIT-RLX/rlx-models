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

//! Kokoro phoneme tokenizer.
//!
//! Kokoro uses the **misaki** phoneme vocabulary — similar in spirit to the
//! StyleTTS2 `TextCleaner` table but *not* identical (115 symbols with gaps,
//! e.g. affricates `ʣ ʥ ʦ ʨ ʧ ʤ` as single symbols and `( )` added). We load
//! the exact mapping from the model's `tokenizer.json` rather than hardcoding,
//! then map each phoneme character to its id (dropping unknowns) and wrap the
//! sequence with the pad token `0` at both ends:
//!
//! ```text
//! input_ids = [0, *map(vocab, phonemes), 0]
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Pad / unknown token id (the `$` symbol).
pub const PAD_ID: i64 = 0;

/// Longest phoneme content the model accepts (`model_max_length` 512 − 2 pads).
pub const MAX_PHONEME_LEN: usize = 510;

#[derive(Deserialize)]
struct TokenizerJson {
    model: TokenizerModel,
}

#[derive(Deserialize)]
struct TokenizerModel {
    vocab: HashMap<String, i64>,
}

/// Character → token-id map loaded from a Kokoro `tokenizer.json`.
#[derive(Debug, Clone)]
pub struct Vocab {
    map: HashMap<char, i64>,
}

impl Vocab {
    /// Load the vocabulary from a `tokenizer.json` file.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("read tokenizer.json: {}", path.display()))?;
        Self::from_json_bytes(&bytes)
    }

    /// Parse the vocabulary from raw `tokenizer.json` bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let parsed: TokenizerJson =
            serde_json::from_slice(bytes).context("parse tokenizer.json")?;
        let map = parsed
            .model
            .vocab
            .into_iter()
            .filter_map(|(k, v)| {
                let mut chars = k.chars();
                let c = chars.next()?;
                // Vocabulary keys are single characters.
                chars.next().is_none().then_some((c, v))
            })
            .collect::<HashMap<char, i64>>();
        anyhow::ensure!(!map.is_empty(), "tokenizer.json vocab is empty");
        Ok(Self { map })
    }

    /// Number of symbols in the vocabulary.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Map one phoneme character to its id, `None` when absent.
    pub fn id(&self, c: char) -> Option<i64> {
        self.map.get(&c).copied()
    }

    /// Map a phoneme string to content token ids (no pads), dropping unknowns.
    ///
    /// Whitespace runs are collapsed to a single space (id for `' '`) so word
    /// boundaries survive but stray formatting does not perturb the sequence.
    pub fn phonemes_to_ids(&self, phonemes: &str) -> Vec<i64> {
        let mut ids = Vec::with_capacity(phonemes.len());
        let mut prev_space = false;
        for c in phonemes.trim().chars() {
            if c.is_whitespace() {
                if prev_space {
                    continue;
                }
                prev_space = true;
                if let Some(id) = self.map.get(&' ') {
                    ids.push(*id);
                }
                continue;
            }
            prev_space = false;
            if let Some(&id) = self.map.get(&c) {
                ids.push(id);
            }
            // unknown phoneme chars silently dropped (matches reference)
        }
        ids
    }

    /// Full pipeline: phoneme string → padded `input_ids` (`[0, …, 0]`).
    pub fn to_input_ids(&self, phonemes: &str) -> Vec<i64> {
        let mut content = self.phonemes_to_ids(phonemes);
        if content.len() > MAX_PHONEME_LEN {
            eprintln!(
                "[kokoro] warning: {} phonemes exceeds max {}, truncating",
                content.len(),
                MAX_PHONEME_LEN
            );
            content.truncate(MAX_PHONEME_LEN);
        }
        let mut ids = Vec::with_capacity(content.len() + 2);
        ids.push(PAD_ID);
        ids.extend_from_slice(&content);
        ids.push(PAD_ID);
        ids
    }

    /// Number of content phonemes (used as the voice-style row index).
    pub fn content_len(&self, phonemes: &str) -> usize {
        self.phonemes_to_ids(phonemes).len().min(MAX_PHONEME_LEN)
    }

    /// Characters in `phonemes` that are absent from the vocabulary.
    pub fn unknown_chars(&self, phonemes: &str) -> Vec<char> {
        phonemes
            .chars()
            .filter(|c| !c.is_whitespace() && !self.map.contains_key(c))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_vocab() -> Vocab {
        let json = r#"{"model":{"vocab":{"$":0," ":16,"h":50,"ɛ":86,"l":54,"o":57,"ʊ":135}}}"#;
        Vocab::from_json_bytes(json.as_bytes()).unwrap()
    }

    #[test]
    fn pad_is_zero() {
        let v = tiny_vocab();
        assert_eq!(v.id('$'), Some(0));
    }

    #[test]
    fn wraps_with_pads() {
        let v = tiny_vocab();
        let ids = v.to_input_ids("hɛloʊ");
        assert_eq!(ids.first(), Some(&0));
        assert_eq!(ids.last(), Some(&0));
        assert_eq!(ids.len(), 7); // 5 phonemes + 2 pads
    }

    #[test]
    fn drops_unknown() {
        let v = tiny_vocab();
        // '中' is not in the vocab and must be dropped.
        assert_eq!(v.content_len("h中o"), 2);
    }

    #[test]
    fn collapses_whitespace() {
        let v = tiny_vocab();
        // "h   o" -> h, space, o
        assert_eq!(v.phonemes_to_ids("h   o"), vec![50, 16, 57]);
    }
}
