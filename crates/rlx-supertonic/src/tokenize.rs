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

//! Supertonic-3 char/unicode tokenizer — a faithful port of the reference
//! `UnicodeProcessor` (`py/helper.py`). Text is NFKD-normalized, cleaned, wrapped
//! in `<lang>…</lang>`, then each character maps to `unicode_indexer[ord(c)]`
//! (a flat 65536-entry codepoint→id table; `-1` = out-of-vocabulary, passed
//! through verbatim to match the reference embedding gather).

use std::path::Path;

use anyhow::{Context, Result};
use unicode_normalization::UnicodeNormalization;

use crate::config::AVAILABLE_LANGS;

/// Codepoint → token-id table (index = `ord(char) as u16`).
#[derive(Debug, Clone)]
pub struct UnicodeIndexer {
    table: Vec<i64>,
}

impl UnicodeIndexer {
    pub fn load(onnx_dir: &Path) -> Result<Self> {
        let path = onnx_dir.join("unicode_indexer.json");
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read unicode_indexer.json: {}", path.display()))?;
        let table: Vec<i64> =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        anyhow::ensure!(!table.is_empty(), "empty unicode_indexer");
        Ok(Self { table })
    }

    fn id(&self, c: char) -> i64 {
        // Reference casts ord(char) to uint16 before indexing.
        let idx = (c as u32 & 0xFFFF) as usize;
        self.table.get(idx).copied().unwrap_or(-1)
    }

    /// Preprocess + wrap + tokenize `text` for `lang`. Returns `input_ids`.
    pub fn encode(&self, text: &str, lang: &str) -> Result<Vec<i64>> {
        let wrapped = preprocess(text, lang)?;
        Ok(wrapped.chars().map(|c| self.id(c)).collect())
    }
}

/// Ordered single-char replacements (dashes, curly quotes, separators → ASCII).
const CHAR_REPLACE: &[(char, &str)] = &[
    ('\u{2013}', "-"), // – en dash
    ('\u{2011}', "-"), // ‑ non-breaking hyphen
    ('\u{2014}', "-"), // — em dash
    ('_', " "),
    ('\u{201C}', "\""),
    ('\u{201D}', "\""),
    ('\u{2018}', "'"),
    ('\u{2019}', "'"),
    ('\u{00B4}', "'"), // ´
    ('`', "'"),
    ('[', " "),
    (']', " "),
    ('|', " "),
    ('/', " "),
    ('#', " "),
    ('\u{2192}', " "), // →
    ('\u{2190}', " "), // ←
];

/// Characters dropped outright (mirrors the reference `[♥☆♡©\\]` strip).
const DROP_CHARS: &[char] = &['\u{2665}', '\u{2606}', '\u{2661}', '\u{00A9}', '\\'];

fn is_emoji(c: char) -> bool {
    let u = c as u32;
    matches!(u,
        0x1F600..=0x1F64F | 0x1F300..=0x1F5FF | 0x1F680..=0x1F6FF | 0x1F700..=0x1F77F
        | 0x1F780..=0x1F7FF | 0x1F800..=0x1F8FF | 0x1F900..=0x1F9FF | 0x1FA00..=0x1FA6F
        | 0x1FA70..=0x1FAFF | 0x2600..=0x26FF | 0x2700..=0x27BF | 0x1F1E6..=0x1F1FF)
}

/// Trailing chars that don't require an appended period (reference regex class).
const END_OK: &[char] = &[
    '.', '!', '?', ';', ':', ',', '\'', '"', ')', ']', '}', '\u{2026}', '\u{3002}', '\u{300D}',
    '\u{300F}', '\u{3011}', '\u{3009}', '\u{300B}', '\u{203A}', '\u{00BB}',
];

/// Full text preprocessing + `<lang>…</lang>` wrap (mirrors `_preprocess_text`).
pub fn preprocess(text: &str, lang: &str) -> Result<String> {
    anyhow::ensure!(
        AVAILABLE_LANGS.contains(&lang),
        "invalid language: {lang} (supported: {AVAILABLE_LANGS:?})"
    );

    // NFKD normalize, drop emojis, apply single-char replacements / drops.
    let mut s = String::with_capacity(text.len());
    for c in text.nfkd() {
        if is_emoji(c) || DROP_CHARS.contains(&c) {
            continue;
        }
        if let Some((_, rep)) = CHAR_REPLACE.iter().find(|(k, _)| *k == c) {
            s.push_str(rep);
        } else {
            s.push(c);
        }
    }

    // Multi-char expression replacements.
    s = s
        .replace('@', " at ")
        .replace("e.g.,", "for example, ")
        .replace("i.e.,", "that is, ");

    // Fix spacing before punctuation.
    for (from, to) in [
        (" ,", ","),
        (" .", "."),
        (" !", "!"),
        (" ?", "?"),
        (" ;", ";"),
        (" :", ":"),
        (" '", "'"),
    ] {
        s = s.replace(from, to);
    }

    // Collapse duplicate quotes.
    while s.contains("\"\"") {
        s = s.replace("\"\"", "\"");
    }
    while s.contains("''") {
        s = s.replace("''", "'");
    }

    // Collapse whitespace runs → single space, trim.
    let mut collapsed = String::with_capacity(s.len());
    let mut prev_ws = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_ws {
                collapsed.push(' ');
            }
            prev_ws = true;
        } else {
            collapsed.push(c);
            prev_ws = false;
        }
    }
    let mut out = collapsed.trim().to_string();

    // Ensure a sentence-final punctuation.
    if !out.chars().last().is_some_and(|c| END_OK.contains(&c)) {
        out.push('.');
    }

    Ok(format!("<{lang}>{out}</{lang}>"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_and_adds_period() {
        assert_eq!(
            preprocess("Hello world", "en").unwrap(),
            "<en>Hello world.</en>"
        );
    }

    #[test]
    fn keeps_terminal_punctuation() {
        assert_eq!(preprocess("Hi!", "en").unwrap(), "<en>Hi!</en>");
    }

    #[test]
    fn collapses_whitespace_and_fixes_spacing() {
        assert_eq!(preprocess("a   b ,c", "en").unwrap(), "<en>a b,c.</en>");
    }

    #[test]
    fn rejects_unknown_lang() {
        assert!(preprocess("hi", "zz").is_err());
    }

    #[test]
    fn indexer_oov_is_negative_one() {
        let idx = UnicodeIndexer {
            table: vec![0; 65536],
        };
        // codepoint 0 maps to table[0]=0 here; a value we didn't set stays 0,
        // but an explicit -1 slot returns -1.
        let mut t = vec![0i64; 65536];
        t['x' as usize] = 42;
        t['q' as usize] = -1;
        let idx2 = UnicodeIndexer { table: t };
        assert_eq!(idx2.id('x'), 42);
        assert_eq!(idx2.id('q'), -1);
        let _ = idx;
    }
}
