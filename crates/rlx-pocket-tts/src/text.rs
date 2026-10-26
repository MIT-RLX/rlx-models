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

//! Text preparation and chunking, ported from
//! `pocket_tts/models/text_chunking.py`.
//!
//! The model is trained on single sentences, so long inputs are split at
//! sentence-final tokens and regrouped into chunks of at most
//! [`MAX_TOKENS_PER_CHUNK`] tokens; each chunk is generated on its own.

use anyhow::{Result, bail};

use crate::spec::ModelSpec;
use crate::tokenizer::PocketTokenizer;

/// Upstream `MAX_TOKEN_PER_CHUNK`.
pub const MAX_TOKENS_PER_CHUNK: usize = 50;

const TERMINAL: &str = ".!?\u{2026}";
const WEAK: &str = ",;:-\u{2013}\u{2014}";
const CLOSERS: &str = "\"'\u{201d}\u{2019})]\u{00bb}";

/// The text-normalization settings of a model config.
#[derive(Debug, Clone, Default)]
pub struct TextOptions {
    pub pad_with_spaces_for_short_inputs: bool,
    pub remove_semicolons: bool,
    pub append_terminal_punctuation: bool,
    pub capitalize_first_letter: bool,
    pub replace_characters: Vec<(char, String)>,
}

impl TextOptions {
    pub fn from_spec(spec: &ModelSpec) -> Self {
        Self {
            pad_with_spaces_for_short_inputs: spec.pad_with_spaces_for_short_inputs,
            remove_semicolons: spec.remove_semicolons,
            append_terminal_punctuation: spec.append_terminal_punctuation,
            capitalize_first_letter: spec.capitalize_first_letter,
            replace_characters: spec
                .replace_characters
                .iter()
                .filter_map(|(k, v)| {
                    let mut it = k.chars();
                    match (it.next(), it.next()) {
                        (Some(c), None) => Some((c, v.clone())),
                        _ => None,
                    }
                })
                .collect(),
        }
    }
}

/// `prepare_text_prompt`: normalize one prompt; returns the text and the
/// `frames_after_eos` guess (3 for at most four words, else 1).
pub fn prepare_text_prompt(text: &str, opts: &TextOptions) -> Result<(String, usize)> {
    let mut text = text.trim().to_string();
    if !opts.replace_characters.is_empty() {
        let mut replaced = String::with_capacity(text.len());
        for c in text.chars() {
            match opts.replace_characters.iter().find(|(k, _)| *k == c) {
                Some((_, v)) => replaced.push_str(v),
                None => replaced.push(c),
            }
        }
        text = replaced.split_whitespace().collect::<Vec<_>>().join(" ");
        text = drop_comma_after_sentence_mark(&text);
    }
    if text.is_empty() {
        bail!("text prompt cannot be empty");
    }
    text = text.replace(['\n', '\r'], " ").replace("  ", " ");
    if opts.remove_semicolons {
        text = text.replace(';', ",");
    }
    let guess = if text.split_whitespace().count() <= 4 {
        3
    } else {
        1
    };
    if opts.capitalize_first_letter {
        let mut chars = text.chars();
        if let Some(first) = chars.next()
            && !first.is_uppercase()
        {
            text = first.to_uppercase().chain(chars).collect();
        }
    }
    if opts.append_terminal_punctuation {
        text = ensure_terminal_punctuation(&text);
    }
    if opts.pad_with_spaces_for_short_inputs && text.split_whitespace().count() < 5 {
        text = format!("{}{text}", " ".repeat(8));
    }
    Ok((text, guess))
}

/// `re.sub(r"([.!?…])\s*[,;:]", r"\1", text)`.
fn drop_comma_after_sentence_mark(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        out.push(c);
        i += 1;
        if TERMINAL.contains(c) {
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && ",;:".contains(chars[j]) {
                i = j + 1;
            }
        }
    }
    out
}

/// `_ensure_terminal_punctuation`.
fn ensure_terminal_punctuation(text: &str) -> String {
    let core = text.trim_end_matches(|c: char| CLOSERS.contains(c) || c == ' ');
    let closers = text[core.len()..].trim();
    match core.chars().last() {
        None => text.to_string(),
        Some(c) if TERMINAL.contains(c) => text.to_string(),
        Some(c) if WEAK.contains(c) => {
            let base = core.trim_end_matches(|c: char| WEAK.contains(c) || c == ' ');
            format!("{base}.{closers}")
        }
        Some(_) => format!("{text}."),
    }
}

/// `_is_decimal_period_boundary`: `idx` starts right after a decimal period ("3.14").
fn is_decimal_period_boundary(tokens: &[u32], idx: usize, tok: &PocketTokenizer) -> Result<bool> {
    let before: Vec<char> = tok
        .decode(&tokens[idx.saturating_sub(2)..idx])?
        .chars()
        .collect();
    let after: Vec<char> = tok
        .decode(&tokens[idx - 1..(idx + 1).min(tokens.len())])?
        .chars()
        .collect();
    Ok(before.len() >= 2
        && before[before.len() - 1] == '.'
        && before[before.len() - 2].is_numeric()
        && after.len() >= 2
        && after[1].is_numeric())
}

/// `_find_boundary_indices`.
fn find_boundaries(
    tokens: &[u32],
    boundary: &[u32],
    tok: Option<&PocketTokenizer>,
) -> Result<Vec<usize>> {
    let mut indices = vec![0];
    let mut previous_was_boundary = false;
    for (idx, t) in tokens.iter().enumerate() {
        if boundary.contains(t) {
            previous_was_boundary = true;
        } else {
            if previous_was_boundary {
                if let Some(tok) = tok
                    && is_decimal_period_boundary(tokens, idx, tok)?
                {
                    previous_was_boundary = false;
                    continue;
                }
                indices.push(idx);
            }
            previous_was_boundary = false;
        }
    }
    indices.push(tokens.len());
    Ok(indices)
}

fn segments(
    tokens: &[u32],
    bounds: &[usize],
    tok: &PocketTokenizer,
) -> Result<Vec<(usize, String)>> {
    bounds
        .windows(2)
        .map(|w| Ok((w[1] - w[0], tok.decode(&tokens[w[0]..w[1]])?)))
        .collect()
}

/// `split_into_best_sentences`: the chunks a prompt is generated in.
pub fn split_into_chunks(
    tok: &PocketTokenizer,
    text: &str,
    max_tokens: usize,
    opts: &TextOptions,
) -> Result<Vec<String>> {
    let (text, _) = prepare_text_prompt(text, opts)?;
    let tokens = tok.encode(text.trim())?;
    let end_of_sentence: Vec<u32> = tok.encode(".!...?")?.into_iter().skip(1).collect();
    let bounds = find_boundaries(&tokens, &end_of_sentence, Some(tok))?;
    let sentences = segments(&tokens, &bounds, tok)?;

    // Sub-split oversized sentences on commas, semicolons and colons.
    let fallback: Vec<u32> = tok.encode(",;:")?.into_iter().skip(1).collect();
    let mut refined = Vec::new();
    for (n, sentence) in sentences {
        if n <= max_tokens {
            refined.push((n, sentence));
            continue;
        }
        let sub_tokens = tok.encode(sentence.trim())?;
        let sub_bounds = find_boundaries(&sub_tokens, &fallback, None)?;
        let subs = segments(&sub_tokens, &sub_bounds, tok)?;
        if subs.len() > 1 {
            refined.extend(subs);
        } else {
            refined.push((n, sentence));
        }
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_n = 0;
    for (n, sentence) in refined {
        if current.is_empty() {
            current = sentence;
            current_n = n;
        } else if current_n + n > max_tokens {
            chunks.push(current.trim().to_string());
            current = sentence;
            current_n = n;
        } else {
            current.push(' ');
            current.push_str(&sentence);
            current_n += n;
        }
    }
    if !current.is_empty() {
        chunks.push(current.trim().to_string());
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> TextOptions {
        TextOptions {
            append_terminal_punctuation: true,
            capitalize_first_letter: true,
            ..Default::default()
        }
    }

    #[test]
    fn prepare_matches_upstream_rules() {
        let o = opts();
        assert_eq!(
            prepare_text_prompt("hello world", &o).unwrap(),
            ("Hello world.".into(), 3)
        );
        assert_eq!(prepare_text_prompt("Wait,", &o).unwrap().0, "Wait.");
        assert_eq!(
            prepare_text_prompt("He said \"go\"", &o).unwrap().0,
            "He said \"go\"."
        );
        assert_eq!(prepare_text_prompt("Done!\"", &o).unwrap().0, "Done!\"");
        assert_eq!(
            prepare_text_prompt("one two three four five six", &o)
                .unwrap()
                .1,
            1
        );
        let padded = TextOptions {
            pad_with_spaces_for_short_inputs: true,
            ..opts()
        };
        assert_eq!(prepare_text_prompt("Hi", &padded).unwrap().0, "        Hi.");
        assert!(prepare_text_prompt("   ", &o).is_err());
    }

    #[test]
    fn replacement_drops_comma_after_sentence_mark() {
        let o = TextOptions {
            replace_characters: vec![('"', String::new())],
            ..opts()
        };
        assert_eq!(
            prepare_text_prompt("\"Hi?\", she said", &o).unwrap().0,
            "Hi? she said."
        );
    }
}
