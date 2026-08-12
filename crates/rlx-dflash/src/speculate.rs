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

//! Speculative accept/reject — the model-agnostic half of DFlash.
//!
//! Kept free of any runner so it can be unit-tested without weights: the draft
//! loop's only subtle part is the acceptance rule, and that is pure logic.

/// Longest prefix of `draft` the target agrees with, under greedy decoding.
///
/// Greedy speculative decoding is exact: accept `draft[i]` iff it equals the
/// target's argmax at position `i`. The target's forward over the drafted block
/// yields `target_argmax[i]` = what the target would emit having consumed
/// `draft[..i]`, so agreement up to the first mismatch is exactly the set of
/// tokens we may keep.
///
/// Returns the accepted count `a` (`0 ..= draft.len()`). The caller then emits
/// `draft[..a]` PLUS `target_argmax[a]` — the bonus token. That bonus is what
/// makes the scheme a strict win: even a fully-rejected block still advances by
/// one, so speculation never loses ground versus plain decoding.
pub fn accepted_prefix(draft: &[u32], target_argmax: &[u32]) -> usize {
    let n = draft.len().min(target_argmax.len());
    (0..n).take_while(|&i| draft[i] == target_argmax[i]).count()
}

/// Tokens committed for one speculative step: the accepted prefix plus the
/// target's own next token.
///
/// `target_argmax` must have `draft.len() + 1` entries — one per drafted
/// position plus the bonus. With a fully-accepted block of 16 this commits 17
/// tokens for one target forward.
pub fn commit_step(draft: &[u32], target_argmax: &[u32]) -> Vec<u32> {
    let a = accepted_prefix(draft, target_argmax);
    let mut out = draft[..a].to_vec();
    if let Some(&bonus) = target_argmax.get(a) {
        out.push(bonus);
    }
    out
}

/// Running acceptance statistics — the number that decides whether speculation
/// is paying for itself.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpecStats {
    pub steps: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub committed: usize,
}

impl SpecStats {
    pub fn record(&mut self, drafted: usize, accepted: usize, committed: usize) {
        self.steps += 1;
        self.drafted += drafted;
        self.accepted += accepted;
        self.committed += committed;
    }

    /// Mean accepted draft tokens per step.
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted == 0 {
            return 0.0;
        }
        self.accepted as f64 / self.drafted as f64
    }

    /// Tokens committed per TARGET forward — the actual speedup multiplier over
    /// plain decoding, which commits exactly 1. Below 1.0 is impossible (the
    /// bonus token guarantees it); near 1.0 means the drafter is not earning the
    /// extra draft cost.
    pub fn tokens_per_target_forward(&self) -> f64 {
        if self.steps == 0 {
            return 0.0;
        }
        self.committed as f64 / self.steps as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_acceptance_commits_block_plus_bonus() {
        let draft = [10, 11, 12, 13];
        let target = [10, 11, 12, 13, 99];
        assert_eq!(accepted_prefix(&draft, &target), 4);
        // 4 drafted + 1 bonus = 5 tokens from ONE target forward.
        assert_eq!(commit_step(&draft, &target), vec![10, 11, 12, 13, 99]);
    }

    #[test]
    fn stops_at_first_mismatch_not_longest_common_subsequence() {
        // Position 1 diverges; the later match at index 3 must NOT be credited —
        // the target's continuation after a rejection is a different sequence.
        let draft = [10, 11, 12, 13];
        let target = [10, 77, 12, 13, 5];
        assert_eq!(accepted_prefix(&draft, &target), 1);
        assert_eq!(commit_step(&draft, &target), vec![10, 77]);
    }

    #[test]
    fn total_rejection_still_advances_by_one() {
        // The guarantee that makes speculation safe: never worse than plain decode.
        let draft = [10, 11];
        let target = [42, 43, 44];
        assert_eq!(accepted_prefix(&draft, &target), 0);
        assert_eq!(commit_step(&draft, &target), vec![42]);
    }

    #[test]
    fn stats_multiplier_is_tokens_per_target_forward() {
        let mut s = SpecStats::default();
        // 16-token blocks: one fully accepted (17 committed), one fully rejected (1).
        s.record(16, 16, 17);
        s.record(16, 0, 1);
        assert_eq!(s.steps, 2);
        assert!((s.acceptance_rate() - 0.5).abs() < 1e-9);
        assert!((s.tokens_per_target_forward() - 9.0).abs() < 1e-9);
    }

    #[test]
    fn empty_draft_is_plain_decoding() {
        assert_eq!(accepted_prefix(&[], &[7]), 0);
        assert_eq!(commit_step(&[], &[7]), vec![7]);
    }
}
