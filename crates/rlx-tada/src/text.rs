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

//! Text normalization (`tada.utils.text.normalize_text`).
//!
//! TADA was trained on text pushed through this exact pipeline, and the
//! aligner tokenizes the *normalized* string — so prompt text and target text
//! must both go through it or the 1:1 token↔frame alignment is off by however
//! many tokens the punctuation rewrite adds or removes.
//!
//! Hand-rolled rather than regex-backed: the two patterns upstream uses are
//! simple enough to scan directly, and this keeps the crate free of a regex
//! dependency. The substitution *order* is load-bearing and preserved verbatim
//! (notably `--` collapses to `-` before `-` expands to `, `).

/// Unicode punctuation folded to ASCII before anything else.
const SUBSTITUTIONS: &[(char, &str)] = &[
    // Quotes
    ('\u{201C}', "\""),
    ('\u{201D}', "\""),
    ('\u{201E}', "\""),
    ('\u{201F}', "\""),
    ('\u{2018}', "'"),
    ('\u{2019}', "'"),
    ('\u{201A}', "'"),
    ('\u{201B}', "'"),
    // Dashes
    ('\u{2013}', "-"),
    ('\u{2014}', "-"),
    ('\u{2015}', "-"),
    ('\u{2010}', "-"),
    ('\u{2011}', "-"),
    // Ellipsis
    ('\u{2026}', "..."),
    // Misc
    ('\u{2039}', "<"),
    ('\u{203A}', ">"),
    ('\u{00AB}', "<<"),
    ('\u{00BB}', ">>"),
];

fn fold_punctuation(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match SUBSTITUTIONS.iter().find(|(k, _)| *k == c) {
            Some((_, v)) => out.push_str(v),
            None => out.push(c),
        }
    }
    out
}

/// `re.sub(r"\s+([.,?!])", r"\1", text)` — drop whitespace runs that sit
/// directly in front of sentence punctuation.
fn strip_space_before_punct(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && matches!(chars[j], '.' | ',' | '?' | '!') {
                // Swallow the whitespace run; the punctuation is emitted next.
                i = j;
                continue;
            }
            for &c in &chars[i..j] {
                out.push(c);
            }
            i = j;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `re.sub(r"([.!?]\s*)(\w)", upper, text)` — capitalize the first word
/// character that follows sentence-ending punctuation. Matches are
/// non-overlapping, so scanning resumes *after* the capitalized character.
fn capitalize_after_sentence_end(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if matches!(chars[i], '.' | '!' | '?') {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && is_word_char(chars[j]) {
                for &c in &chars[i..j] {
                    out.push(c);
                }
                out.extend(chars[j].to_uppercase());
                i = j + 1;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Port of `tada.utils.text.normalize_text`.
pub fn normalize_text(text: &str) -> String {
    let text = fold_punctuation(text);
    // Ordered exactly as upstream: each `replace` is a full non-overlapping
    // left-to-right pass, and later rules see the output of earlier ones.
    // Not collapsible — `--` must reach `-` before `-` becomes `, `, and the
    // `,,` pass exists to clean up what that expansion produces.
    #[allow(clippy::collapsible_str_replace)]
    let text = text
        .replace("; ", ". ")
        .replace('"', "")
        .replace(':', ",")
        .replace('(', "")
        .replace(')', "")
        .replace("--", "-")
        .replace('-', ", ")
        .replace(",,", ",")
        .replace(" '", " ")
        .replace("' ", " ")
        .replace("  ", " ");
    let text = strip_space_before_punct(&text);
    let text = capitalize_after_sentence_end(&text.to_lowercase());

    let mut chars = text.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().chain(chars).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_smart_punctuation() {
        assert_eq!(normalize_text("\u{201C}hi\u{201D}"), "Hi");
        assert_eq!(normalize_text("wait\u{2026}"), "Wait...");
    }

    #[test]
    fn em_dash_becomes_a_comma_pause() {
        // `—` → `-` → `, `.
        assert_eq!(normalize_text("yes\u{2014}no"), "Yes, no");
    }

    #[test]
    fn double_hyphen_collapses_before_expanding() {
        assert_eq!(normalize_text("a--b"), "A, b");
    }

    #[test]
    fn semicolon_starts_a_new_sentence_and_recapitalizes() {
        assert_eq!(normalize_text("one; two"), "One. Two");
    }

    #[test]
    fn colon_becomes_a_comma() {
        assert_eq!(normalize_text("note: this"), "Note, this");
    }

    #[test]
    fn whitespace_before_punctuation_is_dropped() {
        assert_eq!(normalize_text("hello , world !"), "Hello, world!");
    }

    #[test]
    fn capitalizes_after_every_sentence_end() {
        assert_eq!(
            normalize_text("one. two! three? four"),
            "One. Two! Three? Four"
        );
    }

    #[test]
    fn lowercases_shouting_but_keeps_sentence_starts() {
        assert_eq!(
            normalize_text("THIS IS LOUD. SO IS THIS"),
            "This is loud. So is this"
        );
    }

    #[test]
    fn parentheses_are_removed() {
        assert_eq!(normalize_text("a (b) c"), "A b c");
    }

    #[test]
    fn empty_input_stays_empty() {
        assert_eq!(normalize_text(""), "");
    }

    #[test]
    fn double_space_collapse_is_a_single_non_overlapping_pass() {
        // Python's str.replace("  ", " ") on four spaces yields two, not one.
        assert_eq!(normalize_text("a    b"), "A  b");
    }
}
