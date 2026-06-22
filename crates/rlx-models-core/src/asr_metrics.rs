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

//! ASR evaluation metrics shared across model crates: word/character error
//! rate, a corpus accumulator (micro- and macro-averaged WER), the real-time
//! factor (RTFx), and the batch-to-stream factor (BSF).
//!
//! These mirror the metrics used in the on-device streaming ASR literature
//! (e.g. arXiv:2604.14493): WER per dataset plus an unweighted average, RTFx
//! = audio_duration / processing_time, and BSF = streaming_WER / batch_WER as
//! a measure of how much accuracy a model loses when moved to chunked / real-
//! time operation (BSF == 1.0 means streaming is free).
//!
//! Backend-agnostic and dependency-free — accuracy reporting only.

/// Substitution / insertion / deletion breakdown of a single hypothesis vs
/// its reference, computed at the token (word) level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EditCounts {
    pub substitutions: usize,
    pub insertions: usize,
    pub deletions: usize,
    /// Reference length in tokens — the WER denominator.
    pub ref_len: usize,
}

impl EditCounts {
    /// Total edit operations (`S + I + D`), i.e. the Levenshtein distance.
    pub fn errors(&self) -> usize {
        self.substitutions + self.insertions + self.deletions
    }

    /// Error rate `errors / ref_len`, clamped to `[0, 1]`. An empty reference
    /// scores `0.0` when the hypothesis is also empty, else `1.0`.
    pub fn error_rate(&self) -> f64 {
        if self.ref_len == 0 {
            return if self.errors() == 0 { 0.0 } else { 1.0 };
        }
        (self.errors() as f64 / self.ref_len as f64).clamp(0.0, 1.0)
    }
}

/// Lowercase, split on whitespace, strip leading/trailing non-alphanumerics,
/// and drop empties — a conventional WER normalization for English ASR.
pub fn normalize_words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Generic Levenshtein distance between two slices (two-row DP, `O(min) space`).
pub fn edit_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    // Iterate so the inner (allocated) row is the shorter of the two.
    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ai) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, bj) in b.iter().enumerate() {
            let cost = if ai == bj { 0 } else { 1 };
            cur[j + 1] = (cur[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Token-level S/I/D breakdown via a full DP matrix with backtrace. Reference
/// runs down the rows, hypothesis across the columns.
pub fn word_edit_counts(reference: &str, hypothesis: &str) -> EditCounts {
    let r = normalize_words(reference);
    let h = normalize_words(hypothesis);
    let (n, m) = (r.len(), h.len());
    if n == 0 {
        return EditCounts {
            insertions: m,
            ref_len: 0,
            ..Default::default()
        };
    }
    if m == 0 {
        return EditCounts {
            deletions: n,
            ref_len: n,
            ..Default::default()
        };
    }

    // dp[i][j] = edit distance between r[..i] and h[..j].
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = if r[i - 1] == h[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    // Backtrace, preferring diagonal (match/sub) so counts are stable.
    let (mut i, mut j) = (n, m);
    let mut counts = EditCounts {
        ref_len: n,
        ..Default::default()
    };
    while i > 0 || j > 0 {
        let cost = if i > 0 && j > 0 && r[i - 1] == h[j - 1] {
            0
        } else {
            1
        };
        if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + cost {
            if cost == 1 {
                counts.substitutions += 1;
            }
            i -= 1;
            j -= 1;
        } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            counts.deletions += 1; // reference token absent from hypothesis
            i -= 1;
        } else {
            counts.insertions += 1; // extra hypothesis token
            j -= 1;
        }
    }
    counts
}

/// Word error rate of `hypothesis` against `reference` in `[0, 1]`.
pub fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    word_edit_counts(reference, hypothesis).error_rate()
}

/// Character error rate in `[0, 1]`. Whitespace is collapsed to single spaces
/// and text is lowercased before the character-level comparison.
pub fn character_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let norm = |s: &str| -> Vec<char> {
        s.to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .collect()
    };
    let r = norm(reference);
    let h = norm(hypothesis);
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    (edit_distance(&r, &h) as f64 / r.len() as f64).clamp(0.0, 1.0)
}

/// Accumulates references and hypotheses across a corpus to report both a
/// micro-average (pool all words, then divide) and a macro-average (mean of
/// per-utterance WERs). Benchmarks typically report the micro-average per
/// dataset and an unweighted mean across datasets.
#[derive(Debug, Clone, Default)]
pub struct WerAccumulator {
    total_errors: usize,
    total_ref_words: usize,
    per_utt: Vec<f64>,
}

impl WerAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one (reference, hypothesis) pair and return its WER.
    pub fn push(&mut self, reference: &str, hypothesis: &str) -> f64 {
        let counts = word_edit_counts(reference, hypothesis);
        self.total_errors += counts.errors();
        self.total_ref_words += counts.ref_len;
        let wer = counts.error_rate();
        self.per_utt.push(wer);
        wer
    }

    /// Number of utterances accumulated.
    pub fn len(&self) -> usize {
        self.per_utt.len()
    }

    pub fn is_empty(&self) -> bool {
        self.per_utt.is_empty()
    }

    /// Corpus WER: total errors over total reference words (length-weighted).
    pub fn micro_wer(&self) -> f64 {
        if self.total_ref_words == 0 {
            return 0.0;
        }
        self.total_errors as f64 / self.total_ref_words as f64
    }

    /// Mean of per-utterance WERs (each utterance weighted equally).
    pub fn macro_wer(&self) -> f64 {
        if self.per_utt.is_empty() {
            return 0.0;
        }
        self.per_utt.iter().sum::<f64>() / self.per_utt.len() as f64
    }
}

/// Real-time factor: `audio_duration / processing_time`. A value of `5.0`
/// means audio is processed 5× faster than real-time; values `> 1.0` keep up
/// with a live stream. Returns `0.0` for non-positive processing time.
pub fn rtfx(audio_seconds: f64, processing_seconds: f64) -> f64 {
    if processing_seconds <= 0.0 {
        return 0.0;
    }
    audio_seconds / processing_seconds
}

/// Batch-to-stream factor: `streaming_wer / batch_wer`. `1.0` means no
/// accuracy loss from streaming; `> 1.0` quantifies the degradation. Returns
/// `1.0` when the batch WER is zero (no degradation definable).
pub fn batch_to_stream_factor(streaming_wer: f64, batch_wer: f64) -> f64 {
    if batch_wer <= 0.0 {
        return 1.0;
    }
    streaming_wer / batch_wer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wer_zero_on_exact_match_after_normalization() {
        assert!(word_error_rate("Hello, world!", "hello world") < 1e-12);
    }

    #[test]
    fn wer_counts_one_substitution() {
        // 3 reference words, 1 wrong -> 1/3.
        let wer = word_error_rate("the quick fox", "the slow fox");
        assert!((wer - 1.0 / 3.0).abs() < 1e-9, "got {wer}");
    }

    #[test]
    fn edit_counts_classify_sid() {
        // ref: a b c d   hyp: a x c d e
        //   b -> x  (substitution), trailing e (insertion).
        let c = word_edit_counts("a b c d", "a x c d e");
        assert_eq!(c.substitutions, 1);
        assert_eq!(c.insertions, 1);
        assert_eq!(c.deletions, 0);
        assert_eq!(c.ref_len, 4);
        assert_eq!(c.errors(), 2);
    }

    #[test]
    fn deletion_counted_when_hypothesis_drops_a_word() {
        let c = word_edit_counts("one two three", "one three");
        assert_eq!(c.deletions, 1);
        assert_eq!(c.substitutions, 0);
        assert_eq!(c.insertions, 0);
    }

    #[test]
    fn empty_hypothesis_is_all_deletions() {
        let c = word_edit_counts("a b c", "");
        assert_eq!(c.deletions, 3);
        assert!((c.error_rate() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn edit_distance_matches_counts() {
        let r = normalize_words("a b c d");
        let h = normalize_words("a x c d e");
        assert_eq!(
            edit_distance(&r, &h),
            word_edit_counts("a b c d", "a x c d e").errors()
        );
    }

    #[test]
    fn cer_below_wer_for_single_char_typo() {
        // One char wrong out of many -> small CER; whole word wrong -> larger WER.
        let cer = character_error_rate("kitten sequence", "kittin sequence");
        let wer = word_error_rate("kitten sequence", "kittin sequence");
        assert!(cer < wer, "cer {cer} should be < wer {wer}");
    }

    #[test]
    fn accumulator_micro_vs_macro() {
        let mut acc = WerAccumulator::new();
        // Utt 1: 1 error / 1 word -> 1.0.  Utt 2: 1 error / 9 words -> 0.111.
        acc.push("a", "b");
        acc.push(
            "one two three four five six seven eight nine",
            "one two three four five six seven eight WRONG",
        );
        assert_eq!(acc.len(), 2);
        // micro = 2 errors / 10 words = 0.2
        assert!((acc.micro_wer() - 0.2).abs() < 1e-9, "{}", acc.micro_wer());
        // macro = (1.0 + 1/9) / 2 ~= 0.5556
        assert!((acc.macro_wer() - (1.0 + 1.0 / 9.0) / 2.0).abs() < 1e-9);
    }

    #[test]
    fn rtfx_and_bsf_basics() {
        assert!((rtfx(10.0, 2.0) - 5.0).abs() < 1e-12);
        assert_eq!(rtfx(10.0, 0.0), 0.0);
        assert!((batch_to_stream_factor(0.10, 0.08) - 1.25).abs() < 1e-12);
        assert_eq!(batch_to_stream_factor(0.10, 0.0), 1.0);
    }
}
