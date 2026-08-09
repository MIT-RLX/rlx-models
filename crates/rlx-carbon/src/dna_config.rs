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

//! DNA 6-mer vocabulary + segmentation — the algorithmic half of Carbon's
//! `HybridDNATokenizer` (`tokenizer.py` + `dna_config.json`).
//!
//! Carbon's tokenizer combines a Qwen3 byte-level BPE (ids `0..dna_start_id`,
//! handled by the base `tokenizer.json`) with an **algorithmic** DNA vocabulary
//! appended above it:
//!
//! ```text
//!   dna_start_id + 0            "<dna>"    (begin marker)
//!   dna_start_id + 1            "</dna>"   (end marker)
//!   dna_start_id + 2            "<oov>"    (out-of-vocab k-mer)
//!   dna_start_id + 3 ..         ATCG^k  6-mers, base-4 over ['A','T','C','G']
//!   … padding                   "<unused_i>" (128-alignment filler)
//! ```
//!
//! For Carbon-500M: `k=6`, `dna_start_id=151669`, `dna_vocab_size=4107`
//! (3 special + 4096 6-mers + 8 padding), so total vocab = 155776.
//!
//! Everything in this module is pure (no base tokenizer, no I/O beyond
//! [`DnaConfig::from_file`]), so the id math is unit-testable on its own.

use serde::Deserialize;
use std::path::Path;

/// Base ordering used to number k-mers: `A=0, T=1, C=2, G=3`
/// (Python `itertools.product(['A','T','C','G'], repeat=k)`).
const BASES: [u8; 4] = *b"ATCG";

fn base_digit(b: u8) -> Option<u32> {
    match b {
        b'A' => Some(0),
        b'T' => Some(1),
        b'C' => Some(2),
        b'G' => Some(3),
        _ => None,
    }
}

fn default_special_tokens() -> Vec<String> {
    vec!["<dna>".into(), "</dna>".into(), "<oov>".into()]
}

/// Parsed `dna_config.json` describing the DNA half of the hybrid vocabulary.
#[derive(Debug, Clone, Deserialize)]
pub struct DnaConfig {
    /// k-mer width (Carbon: 6).
    pub k: usize,
    /// First DNA token id — equals the base (BPE) vocabulary size.
    pub dna_start_id: u32,
    /// Total DNA vocabulary size (special + k-mers + padding).
    pub dna_vocab_size: usize,
    /// DNA special tokens, in id order: `["<dna>", "</dna>", "<oov>"]`.
    #[serde(default = "default_special_tokens")]
    pub dna_special_tokens: Vec<String>,
    /// When true, `HybridDnaTokenizer::encode` wraps raw input
    /// in `<dna>…</dna>` if no `<dna>` tag is present. Carbon-500M ships `false`.
    #[serde(default)]
    pub auto_dna_tags: bool,
}

impl Default for DnaConfig {
    /// Carbon-500M defaults (also correct for the 3B / 8B Carbon variants,
    /// which share the tokenizer).
    fn default() -> Self {
        Self {
            k: 6,
            dna_start_id: 151669,
            dna_vocab_size: 4107,
            dna_special_tokens: default_special_tokens(),
            auto_dna_tags: false,
        }
    }
}

/// One DNA-region segment after tag stripping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DnaRegion {
    pub content: String,
    pub has_start: bool,
    pub has_end: bool,
}

impl DnaConfig {
    /// Load from a `dna_config.json` file.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
        Ok(serde_json::from_str(&data)?)
    }

    #[inline]
    pub fn begin_id(&self) -> u32 {
        self.dna_start_id
    }
    #[inline]
    pub fn end_id(&self) -> u32 {
        self.dna_start_id + 1
    }
    #[inline]
    pub fn oov_id(&self) -> u32 {
        self.dna_start_id + 2
    }
    /// Id of the first k-mer (`"AAAA…"`).
    #[inline]
    pub fn kmer_base_id(&self) -> u32 {
        self.dna_start_id + self.dna_special_tokens.len() as u32
    }
    /// Number of real k-mers = `4^k`.
    #[inline]
    pub fn num_kmers(&self) -> u32 {
        4u32.pow(self.k as u32)
    }
    /// Total model vocabulary (base BPE + DNA).
    #[inline]
    pub fn total_vocab_size(&self) -> usize {
        self.dna_start_id as usize + self.dna_vocab_size
    }
    /// Whether `id` belongs to the DNA range (special / k-mer / padding).
    #[inline]
    pub fn is_dna_id(&self, id: u32) -> bool {
        id >= self.dna_start_id && (id as usize) < self.total_vocab_size()
    }

    /// Encode a k-length nucleotide slice to its k-mer id, or `None` if it is
    /// not exactly `k` valid `ATCG` bases (caller maps `None` → `<oov>`).
    pub fn kmer_bytes_to_id(&self, kmer: &[u8]) -> Option<u32> {
        if kmer.len() != self.k {
            return None;
        }
        let mut idx: u32 = 0;
        for &b in kmer {
            idx = idx * 4 + base_digit(b)?;
        }
        Some(self.kmer_base_id() + idx)
    }

    /// Decode a k-mer table index (`0..4^k`) back to its nucleotide string.
    pub fn kmer_index_to_string(&self, idx: u32) -> String {
        let mut out = vec![0u8; self.k];
        let mut v = idx;
        for p in (0..self.k).rev() {
            out[p] = BASES[(v % 4) as usize];
            v /= 4;
        }
        // Safe: BASES are ASCII.
        String::from_utf8(out).expect("ascii bases")
    }

    /// Reverse of the DNA id space: `id` → token string (special marker, k-mer,
    /// or `<unused_i>` padding). Returns `None` for base (BPE) ids.
    pub fn id_to_dna_token(&self, id: u32) -> Option<String> {
        if !self.is_dna_id(id) {
            return None;
        }
        let off = (id - self.dna_start_id) as usize;
        let ns = self.dna_special_tokens.len();
        if off < ns {
            return Some(self.dna_special_tokens[off].clone());
        }
        let ki = (off - ns) as u32;
        let num_k = self.num_kmers();
        if ki < num_k {
            Some(self.kmer_index_to_string(ki))
        } else {
            Some(format!("<unused_{}>", ki - num_k))
        }
    }

    /// Tokenize a raw DNA string to DNA ids, mirroring
    /// `HybridDNATokenizer._process_dna_sequence`:
    ///
    /// * non-overlapping k-mers left-to-right; any chunk with a non-`ATCG` base
    ///   → `<oov>`;
    /// * a trailing partial chunk is right-padded with `A` to width `k` (so the
    ///   real bases keep positions `0..valid_len`), then encoded (or `<oov>` if
    ///   it contains a non-`ATCG` base).
    ///
    /// The input is upper-cased first (`acgt` == `ACGT`).
    pub fn process_dna_sequence_ids(&self, seq: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        if self.k == 0 {
            return ids;
        }
        let up = seq.to_ascii_uppercase();
        let bytes = up.as_bytes();
        let len = bytes.len();
        let k = self.k;
        let oov = self.oov_id();

        let mut i = 0;
        while i + k <= len {
            ids.push(self.kmer_bytes_to_id(&bytes[i..i + k]).unwrap_or(oov));
            i += k;
        }
        let processed = ids.len() * k;
        if processed < len {
            let mut padded = bytes[processed..].to_vec();
            padded.resize(k, b'A');
            ids.push(self.kmer_bytes_to_id(&padded).unwrap_or(oov));
        }
        ids
    }
}

/// Split `text` into `(segment, is_dna)` runs on `<dna>…</dna>` boundaries.
///
/// Faithful port of `HybridDNATokenizer._split_by_dna_tags`, including its
/// handling of unbalanced / stray tags. DNA regions keep their surrounding
/// tags (stripped later by [`parse_dna_region`]).
pub fn split_by_dna_tags(text: &str) -> Vec<(String, bool)> {
    const START: &str = "<dna>";
    const END: &str = "</dna>";
    let find_from = |from: usize, pat: &str| text[from..].find(pat).map(|p| p + from);

    let mut segments = Vec::new();
    let n = text.len();
    let mut i = 0;
    while i < n {
        let start_pos = find_from(i, START);
        let end_pos = find_from(i, END);
        match (start_pos, end_pos) {
            (None, None) => {
                let remaining = &text[i..];
                if !remaining.is_empty() {
                    segments.push((remaining.to_string(), false));
                }
                break;
            }
            (None, Some(e)) => {
                let region = &text[i..e + END.len()];
                if !region.is_empty() {
                    segments.push((region.to_string(), true));
                }
                i = e + END.len();
            }
            (Some(s), None) => {
                if i < s {
                    let normal = &text[i..s];
                    if !normal.is_empty() {
                        segments.push((normal.to_string(), false));
                    }
                }
                let region = &text[s..];
                if !region.is_empty() {
                    segments.push((region.to_string(), true));
                }
                break;
            }
            (Some(s), Some(e)) => {
                if s < e {
                    if i < s {
                        let normal = &text[i..s];
                        if !normal.is_empty() {
                            segments.push((normal.to_string(), false));
                        }
                    }
                    let region = &text[s..e + END.len()];
                    if !region.is_empty() {
                        segments.push((region.to_string(), true));
                    }
                    i = e + END.len();
                } else {
                    let region = &text[i..e + END.len()];
                    if !region.is_empty() {
                        segments.push((region.to_string(), true));
                    }
                    i = e + END.len();
                }
            }
        }
    }
    segments
}

/// Strip `<dna>` / `</dna>` markers from a DNA segment, mirroring
/// `HybridDNATokenizer._parse_dna_region`. Returns the inner (trimmed) content
/// plus whether the region carried a start / end marker.
pub fn parse_dna_region(region: &str) -> DnaRegion {
    if region == "<dna>" {
        return DnaRegion {
            content: String::new(),
            has_start: true,
            has_end: false,
        };
    }
    if region == "</dna>" {
        return DnaRegion {
            content: String::new(),
            has_start: false,
            has_end: true,
        };
    }
    let has_start = region.starts_with("<dna>");
    let has_end = region.ends_with("</dna>");
    let mut content = region;
    if has_start {
        content = &content[5..];
    }
    if has_end && content.ends_with("</dna>") {
        content = &content[..content.len() - 6];
    }
    DnaRegion {
        content: content.trim().to_string(),
        has_start,
        has_end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DnaConfig {
        DnaConfig::default()
    }

    #[test]
    fn special_ids_and_vocab_match_carbon_500m() {
        let c = cfg();
        assert_eq!(c.begin_id(), 151669);
        assert_eq!(c.end_id(), 151670);
        assert_eq!(c.oov_id(), 151671);
        assert_eq!(c.kmer_base_id(), 151672);
        assert_eq!(c.num_kmers(), 4096);
        // 3 special + 4096 k-mers + 8 padding = 4107 → total 155776.
        assert_eq!(c.total_vocab_size(), 155776);
        assert_eq!(
            c.dna_vocab_size,
            3 + 4096 + (c.total_vocab_size() - 151669 - 3 - 4096)
        );
    }

    #[test]
    fn kmer_id_math_matches_itertools_product_order() {
        let c = cfg();
        // "AAAAAA" is index 0; "GGGGGG" is 4^6-1.
        assert_eq!(c.kmer_bytes_to_id(b"AAAAAA"), Some(151672));
        assert_eq!(c.kmer_bytes_to_id(b"GGGGGG"), Some(151672 + 4095));
        // Hand-computed base-4 (A=0,T=1,C=2,G=3):
        assert_eq!(c.kmer_bytes_to_id(b"ATCGAT"), Some(151672 + 433));
        assert_eq!(c.kmer_bytes_to_id(b"CGATCG"), Some(151672 + 2843));
        // Round-trips.
        for &s in &["AAAAAA", "GGGGGG", "ATCGAT", "CGATCG", "TTTTTT"] {
            let id = c.kmer_bytes_to_id(s.as_bytes()).unwrap();
            let ki = id - c.kmer_base_id();
            assert_eq!(c.kmer_index_to_string(ki), s);
            assert_eq!(c.id_to_dna_token(id).as_deref(), Some(s));
        }
    }

    #[test]
    fn invalid_kmer_is_none() {
        let c = cfg();
        assert_eq!(c.kmer_bytes_to_id(b"ATCGN"), None); // wrong length
        assert_eq!(c.kmer_bytes_to_id(b"ATCGAN"), None); // N not a base
    }

    #[test]
    fn lowercase_is_normalized() {
        let c = cfg();
        assert_eq!(
            c.process_dna_sequence_ids("atcgat"),
            vec![c.kmer_bytes_to_id(b"ATCGAT").unwrap()]
        );
    }

    #[test]
    fn process_full_and_partial_and_oov() {
        let c = cfg();
        let atcgat = c.kmer_bytes_to_id(b"ATCGAT").unwrap();
        let cgatcg = c.kmer_bytes_to_id(b"CGATCG").unwrap();
        // Two clean 6-mers.
        assert_eq!(
            c.process_dna_sequence_ids("ATCGATCGATCG"),
            vec![atcgat, cgatcg]
        );
        // 9 bases: one full 6-mer + "ATC" right-padded to "ATCAAA".
        let atcaaa = c.kmer_bytes_to_id(b"ATCAAA").unwrap();
        assert_eq!(
            c.process_dna_sequence_ids("ATCGATATC"),
            vec![atcgat, atcaaa]
        );
        // Trailing non-base → the padded partial chunk is <oov>.
        assert_eq!(
            c.process_dna_sequence_ids("ATCGATN"),
            vec![atcgat, c.oov_id()]
        );
        // A whole chunk of non-bases → <oov>.
        assert_eq!(c.process_dna_sequence_ids("NNNNNN"), vec![c.oov_id()]);
        // Shorter than k → single right-padded chunk.
        assert_eq!(
            c.process_dna_sequence_ids("AT"),
            vec![c.kmer_bytes_to_id(b"ATAAAA").unwrap()]
        );
    }

    #[test]
    fn split_and_parse_tags() {
        let segs = split_by_dna_tags("hello<dna>ATCG</dna>world");
        assert_eq!(
            segs,
            vec![
                ("hello".to_string(), false),
                ("<dna>ATCG</dna>".to_string(), true),
                ("world".to_string(), false),
            ]
        );
        let r = parse_dna_region("<dna>ATCG</dna>");
        assert_eq!(r.content, "ATCG");
        assert!(r.has_start && r.has_end);

        // Bare start marker → begin-only region.
        let r = parse_dna_region("<dna>");
        assert!(r.has_start && !r.has_end && r.content.is_empty());
        // Open region (start, no end): content keeps everything after <dna>.
        let segs = split_by_dna_tags("<dna>ATCGATCG");
        assert_eq!(segs, vec![("<dna>ATCGATCG".to_string(), true)]);
        let r = parse_dna_region("<dna>ATCGATCG");
        assert_eq!(r.content, "ATCGATCG");
        assert!(r.has_start && !r.has_end);
    }
}
