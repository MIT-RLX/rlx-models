// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! SentencePiece tokenizer bridge + simple sentence-aware chunker.
//!
//! Pocket TTS ships a SentencePiece model with vocab=4000. Long inputs must be
//! split into chunks of ≤ 50 tokens at sentence boundaries (`.`, `!`, `?`,
//! `...`) — falling back to `,;:` for sentences that are too long — before
//! handing them to the FlowLM. See `pocket_tts/main.py::split_text_into_chunks`.

use std::path::Path;

use anyhow::{Context, Result};
use sentencepiece::SentencePieceProcessor;

/// Maximum tokens per generation chunk (matches pocket_tts default).
pub const MAX_TOKENS_PER_CHUNK: usize = 50;

/// Normalize a text prompt the same way `pocket_tts.models.tts_model.prepare_text_prompt`
/// does, and return the `frames_after_eos` "guess" the upstream model uses.
///
/// The guess is `3` if the input has ≤ 4 words, else `1`. The downstream
/// caller adds `2` to it (see `generate_audio_stream` in tts_model.py).
///
/// `pad_with_spaces_for_short_inputs` prepends 8 spaces when the input has
/// < 5 words; this avoids the model collapsing on tiny prompts.
pub fn prepare_text_prompt(text: &str, pad_with_spaces_for_short_inputs: bool) -> (String, usize) {
    let mut text = text.trim().to_string();
    if text.is_empty() {
        return (String::new(), 1);
    }
    // Normalize newlines + double-spaces.
    text = text.replace(['\n', '\r'], " ");
    while text.contains("  ") {
        text = text.replace("  ", " ");
    }
    let n_words = text.split_whitespace().count();
    let frames_after_eos_guess = if n_words <= 4 { 3 } else { 1 };

    // Capitalize first char if it's lowercase ASCII.
    if let Some(first) = text.chars().next() {
        if first.is_ascii_lowercase() {
            let mut chars = text.chars();
            let upper = chars.next().unwrap().to_ascii_uppercase();
            text = format!("{upper}{}", chars.as_str());
        }
    }

    // Ensure trailing punctuation.
    if let Some(last) = text.chars().last() {
        if last.is_alphanumeric() {
            text.push('.');
        }
    }

    if pad_with_spaces_for_short_inputs && text.split_whitespace().count() < 5 {
        text = format!("{}{text}", " ".repeat(8));
    }

    (text, frames_after_eos_guess)
}

pub struct PocketTokenizer {
    sp: SentencePieceProcessor,
}

impl PocketTokenizer {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let sp = SentencePieceProcessor::open(path.as_ref())
            .with_context(|| format!("open tokenizer {}", path.as_ref().display()))?;
        Ok(Self { sp })
    }

    /// Encode `text` to token IDs (no BOS/EOS).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let ids = self
            .sp
            .encode(text)
            .with_context(|| format!("encode {text:?}"))?;
        Ok(ids.into_iter().map(|p| p.id).collect())
    }

    pub fn vocab_size(&self) -> usize {
        self.sp.len()
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(self.sp.decode_piece_ids(ids)?)
    }
}

/// Split a text into chunks suitable for one FlowLM call (≤ `max_tokens` IDs
/// each). The split prefers sentence boundaries (`. ! ? ...`); falls back to
/// `, ; :` and finally a hard cut at `max_tokens`.
///
/// Returns the chunks as raw strings (let the caller re-tokenize them, which
/// is what pocket_tts does — it stitches by decoding the surface tokens). For
/// the common case where you just want the token IDs per chunk, use
/// [`PocketTokenizer::encode`] on each returned chunk.
pub fn split_into_chunks(
    tok: &PocketTokenizer,
    text: &str,
    max_tokens: usize,
) -> Result<Vec<String>> {
    let mut chunks = Vec::new();
    for sentence in split_sentences(text) {
        let s = sentence.trim();
        if s.is_empty() {
            continue;
        }
        let ids = tok.encode(s)?;
        if ids.len() <= max_tokens {
            chunks.push(s.to_string());
            continue;
        }
        // Re-split on soft punctuation.
        for soft in split_soft(s) {
            let s2 = soft.trim();
            if s2.is_empty() {
                continue;
            }
            let ids2 = tok.encode(s2)?;
            if ids2.len() <= max_tokens {
                chunks.push(s2.to_string());
            } else {
                // Hard cut: walk word by word.
                for piece in hard_chunk(tok, s2, max_tokens)? {
                    if !piece.trim().is_empty() {
                        chunks.push(piece);
                    }
                }
            }
        }
    }
    Ok(chunks)
}

/// Split on `.!?` (keeping the punctuation attached to the preceding sentence).
/// `...` is collapsed to a single boundary.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let bytes: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        cur.push(c);
        if c == '.' || c == '!' || c == '?' {
            // Greedy-consume runs of dots (e.g. "...").
            while i + 1 < bytes.len() && bytes[i + 1] == '.' {
                i += 1;
                cur.push('.');
            }
            // Flush.
            out.push(std::mem::take(&mut cur));
        }
        i += 1;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn split_soft(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        cur.push(c);
        if c == ',' || c == ';' || c == ':' {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn hard_chunk(tok: &PocketTokenizer, text: &str, max_tokens: usize) -> Result<Vec<String>> {
    // Walk word by word; keep appending as long as the running encoding fits.
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        let candidate = if cur.is_empty() {
            word.to_string()
        } else {
            format!("{cur} {word}")
        };
        let ids = tok.encode(&candidate)?;
        if ids.len() > max_tokens {
            if !cur.is_empty() {
                chunks.push(std::mem::take(&mut cur));
            }
            cur = word.to_string();
        } else {
            cur = candidate;
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    Ok(chunks)
}
