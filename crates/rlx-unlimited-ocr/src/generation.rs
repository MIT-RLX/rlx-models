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

//! Greedy / temperature-sampled decode with the checkpoint's sliding-window
//! n-gram repeat guard ([`crate::ngram::SlidingWindowNoRepeatNgramProcessor`]).
//!
//! HF README recommends `no_repeat_ngram_size=35` for both configs, with a
//! `ngram_window` matched to how long a single "OCR unit" (a table row, a
//! paragraph line) can run before legitimately repeating structure:
//! `128` for single-page Gundam/Base, `1024` for multi-page/PDF batches.

use crate::embed::argmax_token;
use crate::ngram::SlidingWindowNoRepeatNgramProcessor;

/// Decoding knobs, defaulting to the HF README's single-page (Gundam) recipe.
#[derive(Debug, Clone)]
pub struct SampleOpts {
    pub temperature: f32,
    pub repetition_penalty: f32,
    /// `0` disables the sliding-window n-gram guard.
    pub no_repeat_ngram_size: usize,
    /// Trailing-token lookback window for the n-gram guard.
    pub ngram_window: usize,
    pub max_new_tokens: usize,
}

impl Default for SampleOpts {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 35,
            ngram_window: 128,
            max_new_tokens: 32_768,
        }
    }
}

impl SampleOpts {
    /// HF README's multi-page/PDF recipe (`ngram_window=1024`).
    pub fn multi_page() -> Self {
        Self {
            ngram_window: 1024,
            ..Self::default()
        }
    }

    fn ngram_processor(&self) -> SlidingWindowNoRepeatNgramProcessor {
        SlidingWindowNoRepeatNgramProcessor::new(self.no_repeat_ngram_size, self.ngram_window)
    }
}

/// Greedy (or temperature-scaled) sample from a single logits row `[vocab]`,
/// given the token `history` so far (repetition penalty + n-gram guard both
/// look backward over it).
pub fn sample_token(logits: &[f32], opts: &SampleOpts, history: &[u32]) -> u32 {
    debug_assert!(!logits.is_empty());
    let mut scores: Vec<f32> = logits.to_vec();

    if opts.repetition_penalty != 1.0 {
        for &tok in history {
            let i = tok as usize;
            if i < scores.len() {
                if scores[i] > 0.0 {
                    scores[i] /= opts.repetition_penalty;
                } else {
                    scores[i] *= opts.repetition_penalty;
                }
            }
        }
    }

    if opts.no_repeat_ngram_size > 0 {
        opts.ngram_processor().call(history, &mut scores);
    }

    if opts.temperature > 0.0 {
        for s in &mut scores {
            *s /= opts.temperature;
        }
        return sample_stochastic(&scores, opts.temperature);
    }
    argmax_token(&scores)
}

/// Multinomial sample from raw (pre-softmax) `scores`; `temperature` has
/// already been divided in — kept as a parameter only to seed the RNG
/// distinctly per call (avoids identical draws in tight decode loops).
fn sample_stochastic(scores: &[f32], seed_mix: f32) -> u32 {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return argmax_token(scores);
    }
    let mut probs: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return argmax_token(scores);
    }
    for p in &mut probs {
        *p /= sum;
    }
    let r = rand_uniform(seed_mix);
    let mut cum = 0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return i as u32;
        }
    }
    argmax_token(scores)
}

fn rand_uniform(seed_mix: f32) -> f32 {
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    SystemTime::now().hash(&mut h);
    seed_mix.to_bits().hash(&mut h);
    (h.finish() % 1_000_000) as f32 / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax_when_no_repeat_guard() {
        let opts = SampleOpts {
            no_repeat_ngram_size: 0,
            ..Default::default()
        };
        let logits = [0.1, 0.9, 0.2];
        assert_eq!(sample_token(&logits, &opts, &[]), 1);
    }

    #[test]
    fn ngram_guard_avoids_repeating_trigram_completion() {
        let opts = SampleOpts {
            no_repeat_ngram_size: 3,
            ngram_window: 128,
            ..Default::default()
        };
        // History "1 2 3 1 2" already followed "1 2" with 3; greedily
        // repeating 3 should be blocked even though it has the top logit.
        let history = [1u32, 2, 3, 1, 2];
        let mut logits = vec![0.0f32; 4];
        logits[3] = 10.0;
        let picked = sample_token(&logits, &opts, &history);
        assert_ne!(picked, 3);
    }

    #[test]
    fn ngram_guard_respects_window() {
        let opts = SampleOpts {
            no_repeat_ngram_size: 3,
            ngram_window: 2, // too short to see the "1 2 3" occurrence
            ..Default::default()
        };
        let history = [1u32, 2, 3, 1, 2];
        let mut logits = vec![0.0f32; 4];
        logits[3] = 10.0;
        assert_eq!(sample_token(&logits, &opts, &history), 3);
    }

    #[test]
    fn multi_page_defaults_use_wider_window() {
        let opts = SampleOpts::multi_page();
        assert_eq!(opts.ngram_window, 1024);
        assert_eq!(opts.no_repeat_ngram_size, 35);
    }

    #[test]
    fn repetition_penalty_demotes_history_tokens() {
        let opts = SampleOpts {
            no_repeat_ngram_size: 0,
            repetition_penalty: 2.0,
            ..Default::default()
        };
        let logits = [1.0, 1.0];
        // Token 0 already appeared -> its penalized score (0.5) loses to
        // token 1's untouched score (1.0).
        assert_eq!(sample_token(&logits, &opts, &[0]), 1);
    }
}
