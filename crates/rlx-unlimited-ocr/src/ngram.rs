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

//! Port of `SlidingWindowNoRepeatNgramProcessor` from `modeling_unlimitedocr.py`
//! ("Aligned with SGLang `DeepseekOCRNoRepeatNGramLogitProcessor`").
//!
//! Long OCR documents can send greedy decoding into a repeat loop (the same
//! line/table cell forever); this blocks any token that would complete an
//! `ngram_size`-gram already seen within the trailing `window` tokens of the
//! sequence — cheaper than global `no_repeat_ngram_size` (which never lets a
//! legitimately-repeated short n-gram, e.g. common table separators, reappear
//! *ever*, even pages later).
//!
//! Reference (HF `modeling_unlimitedocr.py`):
//! ```python
//! class SlidingWindowNoRepeatNgramProcessor:
//!     def __init__(self, ngram_size, window, whitelist_token_ids=None):
//!         self.ngram_size = ngram_size
//!         self.window = window
//!         self.whitelist = set(whitelist_token_ids) if whitelist_token_ids else set()
//!
//!     def __call__(self, input_ids, scores):
//!         for batch_idx in range(input_ids.shape[0]):
//!             sequence = input_ids[batch_idx].tolist()
//!             if len(sequence) < self.ngram_size:
//!                 continue
//!             search_start = max(0, len(sequence) - self.window)
//!             search_end = len(sequence) - self.ngram_size + 1
//!             if search_end <= search_start:
//!                 continue
//!             if self.ngram_size > 1:
//!                 current_prefix = tuple(sequence[-(self.ngram_size - 1):])
//!             else:
//!                 current_prefix = tuple()
//!             banned = set()
//!             for idx in range(search_start, search_end):
//!                 ngram = sequence[idx:idx + self.ngram_size]
//!                 if self.ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
//!                     banned.add(ngram[-1])
//!             banned.difference_update(self.whitelist)
//!             for token_id in banned:
//!                 scores[batch_idx, token_id] = float('-inf')
//!         return scores
//! ```

use std::collections::HashSet;

/// Block `ngram_size`-gram repetitions within the trailing `window` tokens
/// of a decode sequence (single-sequence / batch size 1 — RLX decode loops
/// run one sequence at a time; wrap in a loop over batch for larger batches).
#[derive(Debug, Clone)]
pub struct SlidingWindowNoRepeatNgramProcessor {
    pub ngram_size: usize,
    pub window: usize,
    pub whitelist: HashSet<u32>,
}

impl SlidingWindowNoRepeatNgramProcessor {
    pub fn new(ngram_size: usize, window: usize) -> Self {
        Self {
            ngram_size,
            window,
            whitelist: HashSet::new(),
        }
    }

    pub fn with_whitelist(
        ngram_size: usize,
        window: usize,
        whitelist_token_ids: impl IntoIterator<Item = u32>,
    ) -> Self {
        Self {
            ngram_size,
            window,
            whitelist: whitelist_token_ids.into_iter().collect(),
        }
    }

    /// Disabled when `ngram_size == 0` (matches HF's `no_repeat_ngram_size > 0` gate).
    pub fn is_enabled(&self) -> bool {
        self.ngram_size > 0
    }

    /// Token ids that would complete an `ngram_size`-gram already present in
    /// `sequence` within the trailing `window` tokens, minus the whitelist.
    pub fn banned_tokens(&self, sequence: &[u32]) -> HashSet<u32> {
        let mut banned = HashSet::new();
        if !self.is_enabled() || sequence.len() < self.ngram_size {
            return banned;
        }
        let search_start = sequence.len().saturating_sub(self.window);
        // Exclusive end of the (inclusive) `idx` range HF iterates.
        let search_end = sequence.len() + 1 - self.ngram_size;
        if search_end <= search_start {
            return banned;
        }
        let current_prefix: &[u32] = if self.ngram_size > 1 {
            &sequence[sequence.len() - (self.ngram_size - 1)..]
        } else {
            &[]
        };
        for idx in search_start..search_end {
            let ngram = &sequence[idx..idx + self.ngram_size];
            let (prefix, last) = ngram.split_at(ngram.len() - 1);
            if self.ngram_size == 1 || prefix == current_prefix {
                banned.insert(last[0]);
            }
        }
        for tok in &self.whitelist {
            banned.remove(tok);
        }
        banned
    }

    /// `__call__(input_ids, scores)` in HF — sets banned logits to `-inf` in place.
    pub fn call(&self, sequence: &[u32], scores: &mut [f32]) {
        for tok in self.banned_tokens(sequence) {
            if let Some(slot) = scores.get_mut(tok as usize) {
                *slot = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_ban_when_sequence_shorter_than_ngram() {
        let proc = SlidingWindowNoRepeatNgramProcessor::new(3, 128);
        assert!(proc.banned_tokens(&[1, 2]).is_empty());
    }

    #[test]
    fn bans_completion_of_seen_trigram_within_window() {
        // History: 1 2 3 1 2 -> continuing with 3 would repeat "1 2 3".
        let proc = SlidingWindowNoRepeatNgramProcessor::new(3, 128);
        let banned = proc.banned_tokens(&[1, 2, 3, 1, 2]);
        assert_eq!(banned, HashSet::from([3]));
    }

    #[test]
    fn window_limits_lookback() {
        let proc = SlidingWindowNoRepeatNgramProcessor::new(3, 3);
        // search_start = max(0, 5-3) = 2; the "1 2 3" occurrence starts at
        // idx 0, which is now outside the window, so no ban.
        let banned = proc.banned_tokens(&[1, 2, 3, 1, 2]);
        assert!(banned.is_empty());
    }

    #[test]
    fn whitelist_is_never_banned() {
        let proc = SlidingWindowNoRepeatNgramProcessor::with_whitelist(3, 128, [3u32]);
        let banned = proc.banned_tokens(&[1, 2, 3, 1, 2]);
        assert!(banned.is_empty());
    }

    #[test]
    fn ngram_size_one_bans_every_recent_token() {
        let proc = SlidingWindowNoRepeatNgramProcessor::new(1, 128);
        let banned = proc.banned_tokens(&[7, 8, 9]);
        assert_eq!(banned, HashSet::from([7, 8, 9]));
    }

    #[test]
    fn call_masks_scores_to_neg_infinity_in_place() {
        let proc = SlidingWindowNoRepeatNgramProcessor::new(3, 128);
        let sequence = [1u32, 2, 3, 1, 2];
        let mut scores = vec![0f32; 4];
        proc.call(&sequence, &mut scores);
        assert_eq!(scores[3], f32::NEG_INFINITY);
        assert_eq!(scores[0], 0.0);
        assert_eq!(scores[1], 0.0);
        assert_eq!(scores[2], 0.0);
    }

    #[test]
    fn disabled_when_ngram_size_zero() {
        let proc = SlidingWindowNoRepeatNgramProcessor::new(0, 128);
        assert!(!proc.is_enabled());
        assert!(proc.banned_tokens(&[1, 2, 3]).is_empty());
    }

    #[test]
    fn matches_hf_multi_gram_example() {
        // sequence = [5, 6, 7, 6, 7]; current prefix (last n-1=2 tokens) is
        // "6 7". Of the 3 candidate trigrams ("5 6 7", "6 7 6", "7 6 7"),
        // only "6 7 6" (idx 1) has prefix "6 7" -> ban its completion, 6.
        let proc = SlidingWindowNoRepeatNgramProcessor::new(3, 128);
        let banned = proc.banned_tokens(&[5, 6, 7, 6, 7]);
        assert_eq!(banned, HashSet::from([6]));
    }
}
