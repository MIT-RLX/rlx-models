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

//! Character-level tokeniser — mirrors Python's `TextCleaner`.
//!
//! Maps each Unicode character in a phoneme string to its integer ID in a
//! fixed vocabulary, then wraps with KittenTTS framing:
//! `[pad=0] + ids + [ellipsis=10] + [pad=0]`.
//!
//! The vocabulary is exactly the Python list:
//!   `[_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)`
//!
//! Unknown characters are silently skipped (same as Python's `except KeyError: pass`).

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Vocabulary definition — must match Python exactly (char ordering matters)
// ─────────────────────────────────────────────────────────────────────────────

const PAD: char = '$';

/// Punctuation string (matches Python `_punctuation`).
/// Characters: ; : , . ! ? ¡ ¿ — … " « » " "  (space at end)
const PUNCTUATION: &str = ";:,.!?¡¿—…\u{201C}«»\u{201D}\" ";

/// ASCII letters A–Z a–z.
const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// IPA characters (matches Python `_letters_ipa`).
/// The combining character ̩ (U+0329) and curly quotes are individual vocab entries.
const IPA_LETTERS: &str = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\u{2019}\u{0329}\u{2018}ᵻ";

/// Build the character → index mapping at first use.
static VOCAB: Lazy<HashMap<char, i64>> = Lazy::new(|| {
    let symbols: Vec<char> = std::iter::once(PAD)
        .chain(PUNCTUATION.chars())
        .chain(LETTERS.chars())
        .chain(IPA_LETTERS.chars())
        .collect();

    symbols
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c, i as i64))
        .collect()
});

// ─────────────────────────────────────────────────────────────────────────────
// Tokenisation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Regex that splits an IPA string into word tokens and individual punctuation,
/// mirroring Python `re.findall(r"\w+|[^\w\s]", text)`.
static RE_TOKENIZE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\w+|[^\w\s]").unwrap());

/// Split an IPA string into tokens (words and punctuation marks), then
/// re-join them with spaces — identical to the Python pipeline:
///   `phonemes = ' '.join(basic_english_tokenize(ipa_string))`
pub fn basic_english_tokenize(text: &str) -> String {
    RE_TOKENIZE
        .find_iter(text)
        .map(|m| m.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Map a character to its vocabulary index, returning `None` for unknowns.
pub fn char_to_id(c: char) -> Option<i64> {
    VOCAB.get(&c).copied()
}

/// Vocab index of `…` — KittenTTS always appends this before the final pad.
pub const ELLIPSIS_TOKEN_ID: i64 = 10;

/// Convert a tokenised IPA string to a sequence of vocabulary indices.
///
/// Unknown characters are silently dropped (matches Python behaviour).
/// Framing matches KittenTTS / ONNX:
/// ```python
/// tokens.insert(0, 0)
/// tokens.append(10)  # …
/// tokens.append(0)
/// ```
pub fn text_to_ids(tokenized: &str) -> Vec<i64> {
    let mut ids = vec![0i64]; // start pad
    for ch in tokenized.chars() {
        if let Some(id) = char_to_id(ch) {
            ids.push(id);
        }
        // unknown chars silently skipped
    }
    ids.push(ELLIPSIS_TOKEN_ID);
    ids.push(0i64); // end pad
    ids
}

/// Full pipeline: IPA string → padded token ID vector.
///
/// Mirrors the Python sequence:
/// ```python
/// phonemes = basic_english_tokenize(ipa)
/// phonemes = ' '.join(phonemes)
/// tokens   = text_cleaner(phonemes)
/// tokens.insert(0, 0); tokens.append(10); tokens.append(0)
/// ```
pub fn ipa_to_ids(ipa: &str) -> Vec<i64> {
    let tokenized = basic_english_tokenize(ipa);
    text_to_ids(&tokenized)
}

/// Number of vocabulary IDs between the start pad and the trailing `…` + pad.
pub fn ipa_content_len(ipa: &str) -> usize {
    ipa_to_ids(ipa).len().saturating_sub(3)
}

/// Style row index for IPA-only synthesis (phoneme content token count, min 1).
pub fn ipa_style_index(ipa: &str) -> usize {
    ipa_content_len(ipa).max(1)
}

/// Style row index from Unicode scalar count — matches Python `len(text)`.
pub fn ipa_text_style_index(text: &str) -> usize {
    text.chars().count().max(1)
}

/// Log characters dropped because they are absent from the IPA vocabulary.
pub fn warn_unknown_ipa_chars(ipa: &str) {
    let tokenized = basic_english_tokenize(ipa);
    let unknown: Vec<char> = tokenized
        .chars()
        .filter(|c| !c.is_whitespace() && char_to_id(*c).is_none())
        .collect();
    if !unknown.is_empty() {
        eprintln!(
            "[kittentts] warning: dropped {} unknown IPA/token chars: {:?}",
            unknown.len(),
            unknown
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_not_empty() {
        assert!(!VOCAB.is_empty());
    }

    #[test]
    fn test_pad_is_zero() {
        assert_eq!(char_to_id('$'), Some(0));
    }

    #[test]
    fn test_known_chars() {
        // Punctuation and letters should all be found
        for ch in ";:,.!?".chars() {
            assert!(char_to_id(ch).is_some(), "char {} not in vocab", ch);
        }
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz".chars() {
            assert!(char_to_id(ch).is_some(), "char {} not in vocab", ch);
        }
    }

    #[test]
    fn test_unknown_char_returns_none() {
        assert_eq!(char_to_id('\u{0000}'), None);
        assert_eq!(char_to_id('中'), None);
    }

    #[test]
    fn test_ids_have_pads_and_ellipsis() {
        let ids = ipa_to_ids("hɛloʊ");
        assert_eq!(ids[0], 0, "should start with pad token 0");
        assert_eq!(*ids.last().unwrap(), 0, "should end with pad token 0");
        assert_eq!(
            ids[ids.len() - 2],
            ELLIPSIS_TOKEN_ID,
            "should append … (10) before final pad"
        );
        assert_eq!(char_to_id('…'), Some(ELLIPSIS_TOKEN_ID));
        assert!(ids.len() > 3, "should have content between pads");
    }

    #[test]
    fn test_ipa_style_index() {
        // "həloʊ" → tokens with spaces from tokenize; content excludes 0/10/0.
        assert_eq!(ipa_content_len("həloʊ"), 5);
        assert_eq!(ipa_style_index("həloʊ"), 5);
        assert_eq!(ipa_content_len(""), 0);
        assert_eq!(ipa_content_len("你好"), 0);
        assert_eq!(ipa_text_style_index("hi"), 2);
        assert_eq!(ipa_text_style_index("café"), 4);
    }

    #[test]
    fn test_basic_english_tokenize() {
        let out = basic_english_tokenize("hɛloʊ wɜːld!");
        // Words and punctuation separated by spaces
        assert!(out.contains("hɛloʊ"), "got: {}", out);
        assert!(out.contains("wɜːld"), "got: {}", out);
        assert!(out.contains('!'), "got: {}", out);
    }

    #[test]
    fn test_vocab_uniqueness() {
        // Every (char → index) mapping should be 1-to-1
        let mut seen_indices = std::collections::HashSet::new();
        for &idx in VOCAB.values() {
            assert!(seen_indices.insert(idx), "duplicate index {}", idx);
        }
    }
}
