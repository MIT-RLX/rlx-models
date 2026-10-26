// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Written → spoken English normalization ("$12.50" → "twelve dollars fifty
//! cents"), the step `kitten_text_processing` performs for KittenTTS 2.
//!
//! The grammars are NVIDIA NeMo Text Processing 1.2.0's cased English tagger,
//! verbalizer and post-processor as exported by `kitten_text_processing`
//! 0.1.0 (Apache-2.0; `data/LICENSE`, `data/NOTICE`), embedded here (~0.8 MB).
//! Pipeline, as upstream:
//!
//! 1. tagger rewrite → `tokens { … }` ([`fst`]);
//! 2. parse, then for each split of tokens try field permutations until the
//!    verbalizer accepts one ([`tokens`]);
//! 3. post-processor rewrite, Moses detokenization and input-matched
//!    punctuation spacing ([`punct`]).
//!
//! Failure modes follow upstream: a malformed or unverbalizable token returns
//! the (bracket-escaped) input; a tagger with no path is an error, which the
//! caller treats as "speak the raw text".

pub mod fst;
pub mod punct;
pub mod tokens;
mod unicode_data;

use std::sync::OnceLock;

use anyhow::{Context, Result};

use fst::Graph;

const TAGGER: &[u8] = include_bytes!("data/en/tagger.fst.gz");
const VERBALIZER: &[u8] = include_bytes!("data/en/verbalizer.fst.gz");
const POST: &[u8] = include_bytes!("data/en/post.fst.gz");
const MAX_PERMUTATIONS: usize = 729;

/// The English normalizer (grammars parsed once per process).
pub struct Normalizer {
    tagger: Graph,
    verbalizer: Graph,
    post: Graph,
}

// The graphs' arc caches are per-thread `RefCell`s; build one per thread.
thread_local! {
    static EN: OnceLock<std::result::Result<Normalizer, String>> = const { OnceLock::new() };
}

impl Normalizer {
    pub fn english() -> Result<Self> {
        Ok(Self {
            tagger: Graph::from_gz(TAGGER, false).context("tagger grammar")?,
            verbalizer: Graph::from_gz(VERBALIZER, false).context("verbalizer grammar")?,
            post: Graph::from_gz(POST, true).context("post-processing grammar")?,
        })
    }

    /// `Normalizer.normalize(text, punct_post_process=True)`.
    pub fn normalize(&self, original: &str) -> Result<String> {
        let text = original.trim();
        if text.is_empty() {
            return Ok(text.to_string());
        }
        let tagged = self
            .tagger
            .rewrite(text)
            .map_err(|_| anyhow::anyhow!("tagger has no path for {:?}", truncate(text)))?;
        let parsed = tokens::Parser::new(&tagged).parse()?;
        let groups = tokens::split_groups(parsed, MAX_PERMUTATIONS)?;
        let mut chunks = Vec::new();
        for group in &groups {
            let Ok(cands) = tokens::candidates(group) else {
                return Ok(escape(text));
            };
            match cands.iter().find_map(|c| self.verbalizer.rewrite(c).ok()) {
                Some(v) => chunks.push(v),
                None => return Ok(escape(text)),
            }
        }
        let mut output = collapse(&chunks.join(" "));
        if !output.trim().is_empty() {
            output = self
                .post
                .rewrite(output.trim())
                .map_err(|_| anyhow::anyhow!("post-processor has no path"))?;
        }
        Ok(punct::post_process_punct(
            original,
            &punct::detokenize_en(&output),
        ))
    }
}

fn truncate(s: &str) -> String {
    s.chars().take(80).collect()
}

/// `re.sub(' {2,}', ' ', s)`.
fn collapse(s: &str) -> String {
    let mut res = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.chars() {
        if c == ' ' && prev_space {
            continue;
        }
        prev_space = c == ' ';
        res.push(c);
    }
    res
}

fn escape(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('[', "\\[")
        .replace(']', "\\]")
}

/// `kitten_text_processing.normalize_text(text)` (English): normalize, then
/// strip. Errors mean "no usable normalization"; callers speak the input.
pub fn normalize_english(text: &str) -> Result<String> {
    if text.trim().is_empty() {
        return Ok(text.to_string());
    }
    EN.with(|cell| {
        let n = cell.get_or_init(|| Normalizer::english().map_err(|e| format!("{e:#}")));
        match n {
            Ok(n) => n.normalize(text).map(|s| s.trim().to_string()),
            Err(e) => Err(anyhow::anyhow!("{e}")),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Outputs of `kitten_text_processing.normalize_text` 0.1.0 on the same
    /// inputs (more in `tests/normalize_parity.rs`).
    #[test]
    fn matches_upstream_on_core_cases() {
        for (input, want) in [
            (
                "I paid $12.50 for 3 books.",
                "I paid twelve dollars fifty cents for three books.",
            ),
            ("Hello world", "Hello world"),
        ] {
            assert_eq!(normalize_english(input).unwrap(), want, "{input}");
        }
    }
}
