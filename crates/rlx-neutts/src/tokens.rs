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

//! Speech token helpers.
//!
//! NeuTTS uses a vocabulary of 1 024 speech tokens in the form
//! `<|speech_0|>` … `<|speech_1023|>`.  The LLM backbone generates
//! these tokens as plain text; this module extracts the integer IDs
//! and converts between the two representations.

use once_cell::sync::Lazy;
use regex::Regex;

/// Number of distinct NeuCodec token values.
pub const NUM_SPEECH_TOKENS: usize = 1024;

/// Regex that matches `<|speech_N|>` and captures the digit(s).
static RE_SPEECH_TOKEN: Lazy<Regex> = Lazy::new(|| Regex::new(r"<\|speech_(\d+)\|>").unwrap());

// ─── Encoding ─────────────────────────────────────────────────────────────────

/// Convert a slice of codec token IDs to the text representation used in
/// the model prompt, e.g. `[0, 5, 42]` → `"<|speech_0|><|speech_5|><|speech_42|>"`.
pub fn ids_to_token_str(ids: &[i32]) -> String {
    // "<|speech_1023|>" = 15 chars (the longest possible token string).
    // Using 14 previously caused one reallocation for every utterance that
    // contained any 4-digit ID (≥ 1000).
    let mut s = String::with_capacity(ids.len() * 15);
    for &id in ids {
        s.push_str(&format!("<|speech_{id}|>"));
    }
    s
}

// ─── Decoding ─────────────────────────────────────────────────────────────────

/// Extract all speech token IDs from a generated string.
///
/// Mirrors the Python `re.findall(r"<\|speech_(\d+)\|>", output)` step.
/// Returns an empty `Vec` if no tokens are found.
pub fn extract_ids(s: &str) -> Vec<i32> {
    RE_SPEECH_TOKEN
        .captures_iter(s)
        .filter_map(|cap| cap[1].parse::<i32>().ok())
        .collect()
}

// ─── Prompt builder ───────────────────────────────────────────────────────────

/// Stop sequence emitted by the model when generation is complete.
pub const STOP_TOKEN: &str = "<|SPEECH_GENERATION_END|>";

/// Build the GGUF prompt string from phonemized text + reference codec tokens.
///
/// Format (mirrors Python `_infer_ggml`):
/// ```text
/// user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_phones} {input_phones}<|TEXT_PROMPT_END|>
/// assistant:<|SPEECH_GENERATION_START|><|speech_0|><|speech_5|>…
/// ```
pub fn build_prompt(ref_phones: &str, input_phones: &str, ref_codes: &[i32]) -> String {
    let codes_str = ids_to_token_str(ref_codes);
    format!(
        "user: Convert the text to speech:\
         <|TEXT_PROMPT_START|>{ref_phones} {input_phones}<|TEXT_PROMPT_END|>\n\
         assistant:<|SPEECH_GENERATION_START|>{codes_str}"
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ids_to_token_str() {
        assert_eq!(
            ids_to_token_str(&[0, 5, 42]),
            "<|speech_0|><|speech_5|><|speech_42|>"
        );
        assert_eq!(ids_to_token_str(&[]), "");
    }

    #[test]
    fn test_extract_ids() {
        let s = "<|speech_0|><|speech_5|><|speech_42|><|SPEECH_GENERATION_END|>";
        assert_eq!(extract_ids(s), vec![0, 5, 42]);
    }

    #[test]
    fn test_extract_ids_empty() {
        assert_eq!(extract_ids("no tokens here"), Vec::<i32>::new());
    }

    #[test]
    fn test_extract_ids_ignores_non_speech() {
        let s = "<|SPEECH_GENERATION_START|><|speech_10|><|SPEECH_GENERATION_END|>";
        assert_eq!(extract_ids(s), vec![10]);
    }

    #[test]
    fn test_build_prompt_contains_key_parts() {
        let prompt = build_prompt("hɛloʊ", "wɜːld", &[0, 1, 2]);
        assert!(prompt.contains("<|TEXT_PROMPT_START|>hɛloʊ wɜːld<|TEXT_PROMPT_END|>"));
        assert!(prompt.contains("<|SPEECH_GENERATION_START|>"));
        assert!(prompt.contains("<|speech_0|><|speech_1|><|speech_2|>"));
    }
}
