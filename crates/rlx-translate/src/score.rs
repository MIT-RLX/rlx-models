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

//! Scoring a translation against a reference.
//!
//! Word-level overlap is not comparable across the languages this model
//! covers: Chinese, Japanese and Thai are unspaced, so a *correct* translation
//! scores zero on it. chrF — character n-gram F1 — is the standard answer and
//! is script-agnostic, which is why it is the metric used everywhere here.

/// Character n-grams of `s` with whitespace removed.
///
/// Whitespace goes, case stays — which is what chrF specifies and what
/// sacreBLEU does by default. This used to lowercase as well, one of three
/// quiet deviations from the metric it claimed to be.
fn ngrams(s: &str, n: usize) -> Vec<String> {
    let c: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    if c.len() < n {
        return Vec::new();
    }
    (0..=c.len() - n)
        .map(|i| c[i..i + n].iter().collect())
        .collect()
}

/// chrF (Popović 2015), in `0.0..=1.0`.
///
/// Character n-grams to order 6, precision and recall averaged over the orders
/// that both strings populate, then combined with `beta = 2` so recall counts
/// four times precision. That is the definition, and it is **not** what this
/// function computed for most of the port's life: it used orders 1..=4, `F1`,
/// and lowercased both sides, so every "chrF" figure reported here was
/// internally consistent and comparable to nothing published.
///
/// Verified against sacreBLEU 2.6.0 (`chrF2`, the default) — see the tests.
/// Both sides being empty scores 1.0 rather than 0.0, because two empty strings
/// agree.
pub fn chrf(got: &str, want: &str) -> f64 {
    const ORDER: usize = 6;
    const BETA2: f64 = 4.0;
    let (mut sum_p, mut sum_r, mut effective) = (0.0f64, 0.0f64, 0usize);
    for n in 1..=ORDER {
        let (g, w) = (ngrams(got, n), ngrams(want, n));
        if g.is_empty() || w.is_empty() {
            // sacreBLEU's effective-order smoothing: an order neither string can
            // populate is skipped rather than scored zero, which otherwise
            // punishes short segments for being short.
            continue;
        }
        let mut pool = g.clone();
        let mut hits = 0usize;
        for t in &w {
            if let Some(i) = pool.iter().position(|x| x == t) {
                pool.remove(i);
                hits += 1;
            }
        }
        sum_p += hits as f64 / g.len() as f64;
        sum_r += hits as f64 / w.len() as f64;
        effective += 1;
    }
    if effective == 0 {
        // No order is shared. Two empty strings agree; anything else does not.
        let empty = |s: &str| s.chars().all(char::is_whitespace);
        return f64::from(u8::from(empty(got) && empty(want)));
    }
    let (p, r) = (sum_p / effective as f64, sum_r / effective as f64);
    if p + r == 0.0 {
        return 0.0;
    }
    (1.0 + BETA2) * p * r / (BETA2 * p + r)
}

/// Cosine similarity of two vectors, in `-1.0..=1.0`.
///
/// Paired with [`crate::decode::Nmt::sentence_embedding`] this is a *semantic*
/// score, where [`chrf`] is a surface one. The two disagree exactly where it
/// matters: `我爱夏天` and `我喜欢夏季` are both correct renderings of "I love
/// the summer" and share two characters, so chrF calls them 0.11.
///
/// **Centre the vectors first** — see [`center`] and [`cosine_centered`].
/// Post-norm encoder states are strongly anisotropic, so one common direction
/// dominates every pair and raw cosine is squashed into a narrow band near 1.0.
/// Measured on the shipped model, raw cosine separates paraphrase from
/// unrelated text by 0.983 against 0.942; after centring the same pairs read
/// 0.596 against -0.166. The ordering survives either way, but only the
/// centred number can be *reported* without misleading.
///
/// Treat it as a second opinion, not a verdict. Mean-pooled encoder states
/// ignore word order, so a sentence and its reversal score identically. It
/// tells you whether an answer is *about the same thing*; it cannot tell you
/// whether it is correct.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Mean of a set of vectors: the common direction to subtract.
///
/// A handful of sentences in the target language is enough; it is the shared
/// component that matters, not a precise estimate.
pub fn center(vectors: &[Vec<f32>]) -> Vec<f32> {
    let Some(first) = vectors.first() else {
        return Vec::new();
    };
    let mut mean = vec![0.0f32; first.len()];
    let n = vectors.len() as f32;
    for v in vectors {
        if v.len() != mean.len() {
            continue;
        }
        for (m, x) in mean.iter_mut().zip(v) {
            *m += *x / n;
        }
    }
    mean
}

/// [`cosine`] after subtracting `mean` from both vectors.
///
/// This is the form worth reporting; see [`cosine`] for why the raw number is
/// not.
pub fn cosine_centered(a: &[f32], b: &[f32], mean: &[f32]) -> f64 {
    if mean.len() != a.len() || mean.len() != b.len() {
        return cosine(a, b);
    }
    let ca: Vec<f32> = a.iter().zip(mean).map(|(x, m)| x - m).collect();
    let cb: Vec<f32> = b.iter().zip(mean).map(|(x, m)| x - m).collect();
    cosine(&ca, &cb)
}

/// BERTScore F1 between a candidate and a reference, from token embeddings.
///
/// chrF compares characters and [`cosine`] compares one pooled vector per
/// sentence. Both miss things: chrF calls two correct Chinese renderings 0.11,
/// and pooling throws away word order and lets a long sentence hide a wrong
/// clause. BERTScore is the standard middle ground — greedily match every
/// candidate token to its closest reference token and vice versa, then take the
/// harmonic mean.
///
/// Inputs are the contextual embeddings of each side's tokens, in order,
/// **without** the model's special tokens: `[CLS]` and `[SEP]` match each other
/// perfectly in every pair and would inflate every score.
///
/// This takes vectors rather than text so the library stays free of an
/// embedding-model dependency; the caller supplies them.
pub fn bertscore(candidate: &[Vec<f32>], reference: &[Vec<f32>]) -> f64 {
    if candidate.is_empty() || reference.is_empty() {
        return 0.0;
    }
    let best = |v: &[f32], against: &[Vec<f32>]| {
        against
            .iter()
            .map(|w| cosine(v, w))
            .fold(f64::MIN, f64::max)
    };
    let precision =
        candidate.iter().map(|c| best(c, reference)).sum::<f64>() / candidate.len() as f64;
    let recall = reference.iter().map(|r| best(r, candidate)).sum::<f64>() / reference.len() as f64;
    if precision + recall <= 0.0 {
        return 0.0;
    }
    2.0 * precision * recall / (precision + recall)
}

#[cfg(test)]
mod tests {

    /// Values from sacreBLEU 2.6.0 `CHRF()` (order 6, beta 2, no whitespace,
    /// no lowercasing), divided by 100.
    #[test]
    fn chrf_matches_sacrebleu() {
        for (got, want, expect) in [
            ("the small grey cat", "the small gray cat", 0.711_971),
            ("hello world", "hello world", 1.0),
            ("abc", "xyz", 0.0),
            (
                "Nous avons marche le long de la riviere",
                "Nous avons marche le long de la riviere jusqu",
                0.880_294,
            ),
            ("a", "ab", 0.555_556),
            ("Machina na vulytsi.", "Machina nadvori.", 0.538_983),
        ] {
            let got_score = super::chrf(got, want);
            assert!(
                (got_score - expect).abs() < 5e-4,
                "chrf({got:?}, {want:?}) = {got_score:.6}, sacreBLEU says {expect:.6}"
            );
        }
    }

    #[test]
    fn chrf_is_recall_weighted_and_case_sensitive() {
        // beta = 2 means dropping content hurts more than adding it.
        let short = super::chrf("the cat", "the cat sat on the mat");
        let long = super::chrf("the cat sat on the mat", "the cat");
        assert!(
            short < long,
            "recall should dominate: {short:.3} vs {long:.3}"
        );
        // Case is part of the string, as the metric specifies.
        assert!(super::chrf("Chat", "chat") < 1.0);
        assert_eq!(super::chrf("", ""), 1.0);
    }
    use super::*;

    #[test]
    fn bertscore_is_one_for_identical_token_sequences() {
        let a = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
        assert!((bertscore(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bertscore_rewards_a_reordering_more_than_a_replacement() {
        let x = vec![1.0f32, 0.0];
        let y = vec![0.0f32, 1.0];
        let z = vec![-1.0f32, 0.0];
        // Same tokens, different order: greedy matching still pairs them up.
        let reordered = bertscore(&[y.clone(), x.clone()], &[x.clone(), y.clone()]);
        // One token replaced by its opposite.
        let replaced = bertscore(&[x.clone(), z], &[x, y]);
        assert!(
            reordered > replaced,
            "reordered {reordered:.3} should beat replaced {replaced:.3}"
        );
    }

    #[test]
    fn bertscore_handles_empty_input() {
        assert_eq!(bertscore(&[], &[vec![1.0, 0.0]]), 0.0);
        assert_eq!(bertscore(&[vec![1.0, 0.0]], &[]), 0.0);
    }

    #[test]
    fn centring_removes_a_shared_component() {
        // Two vectors that differ only in a small direction, on top of a large
        // shared one: raw cosine calls them nearly identical, centred does not.
        let a = vec![10.0f32, 1.0, 0.0];
        let b = vec![10.0f32, 0.0, 1.0];
        let mean = center(&[a.clone(), b.clone()]);
        assert!(cosine(&a, &b) > 0.98, "raw cosine should be squashed high");
        assert!(
            cosine_centered(&a, &b, &mean) < 0.0,
            "centring should expose that they differ"
        );
    }

    #[test]
    fn cosine_centered_falls_back_on_a_length_mismatch() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0];
        assert!((cosine_centered(&a, &b, &[0.0]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_of_a_vector_with_itself_is_one() {
        let v = [0.5f32, -1.0, 2.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-9);
        assert!((cosine(&v, &[-0.5, 1.0, -2.0]) + 1.0).abs() < 1e-9);
        assert_eq!(
            cosine(&v, &[1.0, 2.0]),
            0.0,
            "length mismatch must not panic"
        );
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0, "zero vector");
    }

    #[test]
    fn identical_strings_score_one() {
        assert!((chrf("La femme de mes rêves", "La femme de mes rêves") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn spacing_is_ignored_but_case_is_not() {
        // Whitespace is removed before n-gramming, so these are the same string.
        assert!((chrf("un livre", "un  livre") - 1.0).abs() < 1e-9);
        assert!((chrf("un livre", "unlivre") - 1.0).abs() < 1e-9);
        // Case is not: chrF is case-sensitive, and this test asserted the
        // opposite while the implementation lowercased both sides.
        assert!(chrf("un livre", "Un livre") < 1.0);
    }

    #[test]
    fn unrelated_strings_score_low() {
        assert!(chrf("Me encanta el verano", "夏が大好きです") < 0.05);
    }

    /// The reason this exists rather than word overlap.
    #[test]
    fn unspaced_scripts_are_scored_at_all() {
        // Two correct renderings of "I love the summer". They share only two
        // characters and no longer n-gram, so the score is genuinely low
        // (~0.11) — but it is not *zero*, which is what splitting on
        // whitespace would give for text that contains none.
        let a = chrf("我爱夏天", "我喜欢夏季");
        assert!(
            a > 0.0,
            "unspaced text scored 0; word overlap's failure mode"
        );
        assert!(a < 0.3, "scored {a}: these share only 2 of 4 characters");
    }

    #[test]
    fn a_trailing_stop_is_a_small_penalty_not_a_failure() {
        let s = chrf("저는 여름을 사랑합니다", "저는 여름을 사랑합니다.");
        assert!(
            s > 0.5,
            "a trailing period should not read as a wrong answer"
        );
        assert!(s < 1.0);
    }
}
