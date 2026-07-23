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

//! Native candidate rescorer: N-best CTC candidates scored by
//! `rec + w_ngram·ngram + w_word·word_ngram + w_lex·lexicon` (en-US weights
//! ngram=0.225, lexicon=0.015, word=0).

use crate::ngram::NgramModel;
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

/// Lexicon as a character trie the candidate words are intersected against. Each word
/// path ends in a `word` node. During rescoring a candidate word is classified as:
///   • valid word   → 0 (no penalty; between two real words the recogniser decides)
///   • valid prefix → mild penalty (could be a truncated / hyphen-split word)
///   • off-trie     → full penalty (a genuine non-word, e.g. an OCR slip like "worlc")
/// Scaled by `w_lex`, the valid/off-trie gap acts as a soft lexicon constraint strong enough
/// to overturn a 1–2 nat recogniser character confusion, without touching OOV tokens (names,
/// version numbers) that have no valid alternative in the beam.
#[derive(Default)]
struct TrieNode {
    children: HashMap<char, u32>,
    is_word: bool,
}

pub struct Lexicon {
    nodes: Vec<TrieNode>,
    pub oov_penalty: f32,
    pub prefix_penalty: f32,
}

impl Lexicon {
    pub fn load(path: &Path) -> Result<Self> {
        let mut nodes: Vec<TrieNode> = vec![TrieNode::default()]; // 0 = root
        for line in std::fs::read_to_string(path)?.lines() {
            let w = line.split_once('\t').map(|(w, _)| w).unwrap_or(line);
            if w.is_empty() {
                continue;
            }
            let mut cur = 0u32;
            for ch in w.chars().flat_map(char::to_lowercase) {
                cur = match nodes[cur as usize].children.get(&ch) {
                    Some(&nx) => nx,
                    None => {
                        let nx = nodes.len() as u32;
                        nodes.push(TrieNode::default());
                        nodes[cur as usize].children.insert(ch, nx);
                        nx
                    }
                };
            }
            nodes[cur as usize].is_word = true;
        }
        Ok(Self { nodes, oov_penalty: -1.0, prefix_penalty: -0.35 })
    }

    /// Classify one lowercased word against the trie: 0 (word) / prefix / off-trie.
    fn word_score(&self, w: &str) -> f32 {
        let mut cur = 0u32;
        for ch in w.chars() {
            match self.nodes[cur as usize].children.get(&ch) {
                Some(&nx) => cur = nx,
                None => return self.oov_penalty, // diverges from every word path
            }
        }
        if self.nodes[cur as usize].is_word { 0.0 } else { self.prefix_penalty }
    }

    /// Sum of per-word trie scores. Only multi-letter alphabetic words are judged (digits,
    /// punctuation, single letters and mixed alphanumerics are exempt so codes/URLs survive).
    pub fn score(&self, text: &str) -> f32 {
        let mut s = 0.0;
        for w in text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
            if w.chars().all(|c| c.is_alphabetic()) && w.chars().count() > 1 {
                s += self.word_score(&w.to_lowercase());
            }
        }
        s
    }
}

pub struct Rescorer {
    pub ngram: Option<NgramModel>,
    pub word_ngram: Option<NgramModel>, // same n-gram format, word vocab
    pub lexicon: Option<Lexicon>,
    pub w_ngram: f32,
    pub w_word: f32,
    pub w_lex: f32,
    unk_tok: u32,
    num_tok: u32,
}

impl Rescorer {
    /// Load the English scoring stack from its artifacts (n-gram model `.bin`, lexicon `.tsv`).
    pub fn load_en(ngram: Option<&Path>, lexicon: Option<&Path>) -> Result<Self> {
        let c = match ngram {
            Some(p) => Some(NgramModel::load(p)?),
            None => None,
        };
        let l = match lexicon {
            Some(p) => Some(Lexicon::load(p)?),
            None => None,
        };
        Ok(Self::new(c, None, l))
    }

    /// English OCR weights (`w_ngram` 0.225, `w_word` 0). `w_lex` is tuned for the
    /// trie-validity score (0/prefix/off-trie per word) so the lexicon acts as a soft
    /// constraint (~3 nats valid-vs-off-trie), strong enough to correct a confident
    /// recogniser's single-character slips. Override: OCR2_LEX_W.
    pub fn new(ngram: Option<NgramModel>, word_ngram: Option<NgramModel>, lexicon: Option<Lexicon>) -> Self {
        let unk_tok = 0;
        let num_tok = ngram.as_ref().and_then(|c| c.token_for("xNUMBx")).unwrap_or(3);
        let w_lex = crate::env::lex_weight(3.0);
        Self { ngram, word_ngram, lexicon, w_ngram: 0.225, w_word: 0.0, w_lex, unk_tok, num_tok }
    }

    /// Map a string to n-gram tokens: punctuation/class tokens map directly; digits →
    /// `xNUMBx`; letters/unknowns → `<unk>` (the model is a punctuation/class n-gram).
    fn ngram_tokens(&self, text: &str, model: &NgramModel) -> Vec<u32> {
        text.chars()
            .map(|ch| {
                let s = ch.to_string();
                if let Some(t) = model.token_for(&s) {
                    t
                } else if ch.is_numeric() {
                    self.num_tok
                } else {
                    self.unk_tok
                }
            })
            .collect()
    }

    fn word_tokens(text: &str, model: &NgramModel) -> Vec<u32> {
        text.split_whitespace()
            .map(|w| model.token_for(w).or_else(|| model.token_for(&w.to_lowercase())).unwrap_or(0))
            .collect()
    }

    /// Combined score for a candidate string (higher = better).
    pub fn score(&self, text: &str) -> f32 {
        let mut s = 0.0;
        if let Some(c) = &self.ngram {
            s += self.w_ngram * c.joint(&self.ngram_tokens(text, c));
        }
        if self.w_word != 0.0 {
            if let Some(w) = &self.word_ngram {
                s += self.w_word * w.joint(&Self::word_tokens(text, w));
            }
        }
        if let Some(l) = &self.lexicon {
            s += self.w_lex * l.score(text);
        }
        s
    }
}
