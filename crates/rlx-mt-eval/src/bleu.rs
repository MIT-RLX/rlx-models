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

//! Corpus / sentence BLEU-4 with modified n-gram precision and brevity penalty.
//!
//! Tokenization is whitespace + punctuation split (case-folded). Good enough for
//! host-side bake-offs; not a drop-in for sacrebleu tokenization.

use std::collections::HashMap;

fn tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_alphanumeric() || c == '\'' || c == '’' {
            cur.extend(c.to_lowercase());
        } else {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            if !c.is_whitespace() {
                out.push(c.to_string());
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn ngram_counts(toks: &[String], n: usize) -> HashMap<Vec<String>, usize> {
    let mut m = HashMap::new();
    if toks.len() < n {
        return m;
    }
    for i in 0..=toks.len() - n {
        *m.entry(toks[i..i + n].to_vec()).or_default() += 1;
    }
    m
}

fn modified_precision(hyp: &[String], reference: &[String], n: usize) -> (usize, usize) {
    let hyp_c = ngram_counts(hyp, n);
    let ref_c = ngram_counts(reference, n);
    if hyp_c.is_empty() {
        return (0, 0);
    }
    let mut clipped = 0usize;
    let mut total = 0usize;
    for (ng, &cnt) in &hyp_c {
        total += cnt;
        clipped += cnt.min(*ref_c.get(ng).unwrap_or(&0));
    }
    (clipped, total)
}

fn brevity_penalty(hyp_len: usize, ref_len: usize) -> f64 {
    if hyp_len == 0 {
        return 0.0;
    }
    if hyp_len > ref_len {
        1.0
    } else {
        (1.0 - ref_len as f64 / hyp_len as f64).exp()
    }
}

/// Sentence BLEU-4 (geometric mean of precisions 1..=4 × BP).
pub fn bleu(hypothesis: &str, reference: &str) -> f64 {
    corpus_bleu(&[hypothesis], &[reference])
}

/// Corpus BLEU-4 over parallel hyp/ref lists (same length).
pub fn corpus_bleu(hyps: &[&str], refs: &[&str]) -> f64 {
    assert_eq!(hyps.len(), refs.len());
    if hyps.is_empty() {
        return 0.0;
    }
    let mut clip = [0usize; 4];
    let mut tot = [0usize; 4];
    let mut hyp_len = 0usize;
    let mut ref_len = 0usize;
    for (h, r) in hyps.iter().zip(refs) {
        let ht = tokens(h);
        let rt = tokens(r);
        hyp_len += ht.len();
        ref_len += rt.len();
        for n in 1..=4 {
            let (c, t) = modified_precision(&ht, &rt, n);
            clip[n - 1] += c;
            tot[n - 1] += t;
        }
    }
    let mut log_sum = 0.0;
    let mut used = 0usize;
    for n in 0..4 {
        if tot[n] == 0 {
            continue;
        }
        let p = clip[n] as f64 / tot[n] as f64;
        if p <= 0.0 {
            return 0.0;
        }
        log_sum += p.ln();
        used += 1;
    }
    if used == 0 {
        return 0.0;
    }
    let geo = (log_sum / used as f64).exp();
    geo * brevity_penalty(hyp_len, ref_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_high() {
        let s = "Il produit 950 chevaux.";
        assert!(bleu(s, s) > 0.99);
    }
}
