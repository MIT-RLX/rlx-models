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

//! Prompt tokenization for the Qwen3-VL conditioner.
//!
//! The checkpoint ships a `Qwen2TokenizerFast` (byte-level BPE) in two places —
//! `tokenizer/` and `text_encoder/` — and both carry the same vocabulary.
//!
//! Two details matter:
//!
//! - **The chat markers are added tokens, not text.** `<|im_start|>` is id
//!   151644 and `<|im_end|>` is 151645. [`crate::text_encoder::assemble_prompt`]
//!   writes them literally and the tokenizer maps each to its single id, so the
//!   turn structure costs 5 tokens rather than a dozen byte-level pieces.
//! - **The tokenizer is loaded once.** `Tokenizer::from_file` parses an 11 MB
//!   JSON; doing that per prompt is a few hundred milliseconds of pure waste, so
//!   [`H3Tokenizer`] owns it for its lifetime.
//!
//! This module is behind the `tokenizer` feature, because the rest of the crate
//! is useful without pulling in `tokenizers`.

use crate::text_encoder::assemble_prompt;
use anyhow::{Context, Result, anyhow, bail, ensure};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// `<|im_start|>` in the released vocabulary.
pub const IM_START: u32 = 151_644;
/// `<|im_end|>` in the released vocabulary.
pub const IM_END: u32 = 151_645;
/// `<|endoftext|>` in the released vocabulary.
pub const ENDOFTEXT: u32 = 151_643;

/// The conditioner's tokenizer, loaded once.
pub struct H3Tokenizer {
    inner: Tokenizer,
    source: PathBuf,
}

impl H3Tokenizer {
    /// Load from a checkpoint root, preferring `tokenizer/` over
    /// `text_encoder/`.
    pub fn from_checkpoint(root: &Path) -> Result<Self> {
        for sub in ["tokenizer", "text_encoder"] {
            let p = root.join(sub).join("tokenizer.json");
            if p.is_file() {
                return Self::from_file(&p);
            }
        }
        bail!(
            "MiniMax-H3: no tokenizer.json under {}/{{tokenizer,text_encoder}}",
            root.display()
        )
    }

    /// Load a specific `tokenizer.json`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow!("MiniMax-H3: load tokenizer {}: {e}", path.display()))?;
        Ok(Self {
            inner,
            source: path.to_path_buf(),
        })
    }

    #[must_use]
    pub fn source(&self) -> &Path {
        &self.source
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Encode raw text with no chat framing and no special tokens added.
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("MiniMax-H3: encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Encode a prompt in the chat turn the conditioner expects.
    ///
    /// The markers are already in the assembled string and are added tokens, so
    /// `add_special_tokens` stays off — letting the tokenizer add its own would
    /// double them.
    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let ids = self.encode_raw(&assemble_prompt(prompt))?;
        ensure!(!ids.is_empty(), "the prompt encoded to zero tokens");
        Ok(ids)
    }

    /// Encode a prompt and pad or truncate it to exactly `len` tokens.
    ///
    /// The text encoder compiles for a fixed prompt length, so a request has to
    /// land on it. Padding uses `<|endoftext|>`; truncation keeps the **head**
    /// of the prompt and re-appends the assistant marker, so the turn stays
    /// well-formed rather than ending mid-sentence with no marker at all.
    pub fn encode_prompt_padded(&self, prompt: &str, len: usize) -> Result<Vec<u32>> {
        ensure!(len > 0, "the prompt length must be positive");
        let mut ids = self.encode_prompt(prompt)?;
        match ids.len().cmp(&len) {
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Less => ids.resize(len, ENDOFTEXT),
            std::cmp::Ordering::Greater => {
                ids.truncate(len);
                // Keep the assistant turn open even after a cut.
                if len >= 2 {
                    ids[len - 2] = IM_START;
                    ids[len - 1] = self
                        .encode_raw("assistant\n")?
                        .first()
                        .copied()
                        .unwrap_or(ENDOFTEXT);
                }
            }
        }
        ensure!(ids.len() == len, "padding produced {} tokens", ids.len());
        Ok(ids)
    }

    /// Decode ids back to text, for checks.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| anyhow!("MiniMax-H3: decode: {e}"))
            .context("decode token ids")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint() -> Option<PathBuf> {
        let p = PathBuf::from(std::env::var("RLX_MINIMAX_H3").ok()?);
        p.is_dir().then_some(p)
    }

    macro_rules! tok {
        () => {
            match checkpoint().map(|r| H3Tokenizer::from_checkpoint(&r)) {
                Some(Ok(t)) => t,
                Some(Err(e)) => panic!("load tokenizer: {e}"),
                None => {
                    eprintln!("skipping: set RLX_MINIMAX_H3 to a checkpoint root");
                    return;
                }
            }
        };
    }

    #[test]
    fn special_token_ids_match_the_released_vocabulary() {
        let t = tok!();
        assert_eq!(t.encode_raw("<|im_start|>").unwrap(), vec![IM_START]);
        assert_eq!(t.encode_raw("<|im_end|>").unwrap(), vec![IM_END]);
        assert_eq!(t.encode_raw("<|endoftext|>").unwrap(), vec![ENDOFTEXT]);

        // The tokenizer's vocabulary (151643 base + 26 added = 151669) is
        // *smaller* than the model's `vocab_size` of 151936: the embedding
        // table is padded up. Every id it can emit must still index the table.
        let model_vocab = crate::config::H3TextEncoderConfig::default_vocab_size();
        assert!(
            t.vocab_size() <= model_vocab,
            "tokenizer vocab {} exceeds the embedding table of {model_vocab}",
            t.vocab_size()
        );
        assert!(
            t.vocab_size() > 151_000,
            "vocab looks truncated: {}",
            t.vocab_size()
        );
        for id in [IM_START, IM_END, ENDOFTEXT] {
            assert!((id as usize) < model_vocab);
        }
    }

    #[test]
    fn chat_markers_cost_one_token_each() {
        // If the markers were tokenized as plain text they would cost many
        // byte-level pieces and shift every position in the prompt.
        let t = tok!();
        let ids = t.encode_prompt("a red balloon").unwrap();
        assert_eq!(ids[0], IM_START, "the turn must open with <|im_start|>");
        assert_eq!(
            ids.iter().filter(|&&i| i == IM_START).count(),
            2,
            "one marker for the user turn and one for the assistant turn"
        );
        assert_eq!(
            ids.iter().filter(|&&i| i == IM_END).count(),
            1,
            "only the user turn is closed"
        );
    }

    #[test]
    fn prompt_round_trips_through_decode() {
        let t = tok!();
        let prompt = "a cat riding a skateboard through neon rain";
        let ids = t.encode_prompt(prompt).unwrap();
        let back = t.decode(&ids, true).unwrap();
        assert!(
            back.contains("a cat riding a skateboard"),
            "decoded text lost the prompt: {back:?}"
        );
    }

    #[test]
    fn padding_reaches_the_compiled_length() {
        let t = tok!();
        for len in [8usize, 32, 64, 128] {
            let ids = t.encode_prompt_padded("a short prompt", len).unwrap();
            assert_eq!(ids.len(), len);
            assert!(ids.iter().all(|&i| (i as usize) < t.vocab_size()));
        }
    }

    #[test]
    fn truncation_keeps_the_turn_open() {
        let t = tok!();
        let long = "a ".repeat(400);
        let ids = t.encode_prompt_padded(&long, 16).unwrap();
        assert_eq!(ids.len(), 16);
        assert_eq!(
            ids[14], IM_START,
            "a truncated prompt must still open the assistant turn"
        );
    }

    #[test]
    fn empty_prompt_is_still_a_well_formed_turn() {
        let t = tok!();
        let ids = t.encode_prompt("").unwrap();
        assert!(!ids.is_empty());
        assert_eq!(ids[0], IM_START);
    }

    #[test]
    fn distinct_prompts_encode_differently() {
        let t = tok!();
        let a = t.encode_prompt("a dog").unwrap();
        let b = t.encode_prompt("a cat").unwrap();
        assert_ne!(a, b);
    }
}
