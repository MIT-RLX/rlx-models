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

//! Host-side logits sampler.
//!
//! Operates on a single `[vocab]` slice — caller is responsible for
//! pulling the last position's row out of `[B, S, vocab]` logits.
//!
//! Sampling is a host-side step (not a graph op) for now because:
//!   - The decision tree (temperature → top-k → top-p → multinomial)
//!     is branchy and cheap; no win from baking it into the graph.
//!   - Keeping it out of the graph lets a downstream `Speculator`
//!     impl call the same sampler for both the draft and the
//!     verifier without graph surgery.
//!
//! Determinism: backed by `rlx_ir::Philox4x32`, same RNG already used
//! by `rlx-runtime/src/spec_decode.rs`. Same seed → same sequence.

use rlx_ir::Philox4x32;

/// Sampling configuration. Construct via [`SampleOpts::greedy`] /
/// [`SampleOpts::temperature`] or build manually.
///
/// Order of operations matches HF `transformers` defaults:
///   1. `temperature` divides logits (skipped if `<= 0` or `1.0`).
///   2. `top_k` truncates to the K highest-logit tokens (0 = disabled).
///   3. `top_p` truncates by nucleus cumulative-mass cutoff (1.0 = disabled).
///   4. Softmax + multinomial sample (or argmax when greedy).
#[derive(Debug, Clone, Copy)]
pub struct SampleOpts {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
    pub greedy: bool,
}

impl SampleOpts {
    pub fn greedy() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
            greedy: true,
        }
    }

    pub fn temperature(temp: f32, seed: u64) -> Self {
        Self {
            temperature: temp,
            top_k: 0,
            top_p: 1.0,
            seed,
            greedy: false,
        }
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }
}

/// Sample one token id from a `[vocab]` logits slice. Returns the
/// chosen index. Stateless w.r.t. prior calls — the RNG is seeded
/// per-call from `opts.seed` so repeated calls with the same seed
/// and logits yield the same token.
pub fn sample_token(logits: &[f32], opts: SampleOpts) -> usize {
    assert!(!logits.is_empty(), "sample_token: empty logits");

    if opts.greedy {
        return argmax(logits);
    }

    // 1. temperature: divide logits, in place on a working copy.
    let mut work: Vec<f32> = if opts.temperature > 0.0 && opts.temperature != 1.0 {
        logits.iter().map(|&l| l / opts.temperature).collect()
    } else {
        logits.to_vec()
    };

    // 2. top_k: mask everything outside the K highest logits.
    if opts.top_k > 0 && opts.top_k < work.len() {
        let mut indexed: Vec<(usize, f32)> =
            work.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        // Partial sort: nth_element-style, descending.
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let cutoff = indexed[opts.top_k - 1].1;
        for v in work.iter_mut() {
            if *v < cutoff {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // 3. softmax (numerically stable).
    let max = work.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = work.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    } else {
        // All -inf (shouldn't happen post-softmax): fall back to greedy.
        return argmax(logits);
    }

    // 4. top_p: nucleus cutoff over sorted-descending probability.
    if opts.top_p < 1.0 && opts.top_p > 0.0 {
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cum = 0.0f32;
        let mut keep = vec![false; probs.len()];
        for &i in &order {
            cum += probs[i];
            keep[i] = true;
            if cum >= opts.top_p {
                break;
            }
        }
        let mut renorm = 0.0f32;
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep[i] {
                *p = 0.0;
            } else {
                renorm += *p;
            }
        }
        if renorm > 0.0 {
            for p in probs.iter_mut() {
                *p /= renorm;
            }
        }
    }

    // 5. multinomial sample.
    let mut rng = Philox4x32::new(opts.seed);
    let u = rng.next_f32();
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return i;
        }
    }
    probs.len() - 1
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Numerically-stable softmax over a logits row. Exposed so
/// `Speculator` implementations can hand the resulting probability
/// vector to `rlx-runtime::spec_decode` without re-implementing it.
pub fn softmax_logits(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = p.iter().sum();
    if sum > 0.0 {
        for v in p.iter_mut() {
            *v /= sum;
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_matches_argmax() {
        let logits = vec![0.1, 0.5, 0.2, -1.0, 0.49];
        let t = sample_token(&logits, SampleOpts::greedy());
        assert_eq!(t, 1);
    }

    #[test]
    fn top_k_one_equals_greedy() {
        let logits = vec![0.1, 0.5, 0.2, -1.0, 0.49];
        let opts = SampleOpts::temperature(1.0, 42).with_top_k(1);
        assert_eq!(sample_token(&logits, opts), 1);
    }

    #[test]
    fn top_p_full_equals_unrestricted_multinomial() {
        // With top_p=1.0 the nucleus mask is a no-op; sampling should
        // still be deterministic given the seed and produce a valid id.
        let logits = vec![1.0, 2.0, 0.5, 0.0];
        let opts = SampleOpts::temperature(1.0, 7).with_top_p(1.0);
        let t = sample_token(&logits, opts);
        assert!(t < logits.len());
    }

    #[test]
    fn deterministic_for_same_seed() {
        let logits: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
        let opts = SampleOpts::temperature(0.7, 123).with_top_k(4);
        let a = sample_token(&logits, opts);
        let b = sample_token(&logits, opts);
        assert_eq!(a, b);
    }

    #[test]
    fn top_p_truncates_low_mass() {
        // One token has nearly all the mass; top_p=0.5 should keep
        // only that token and pick it regardless of RNG.
        let mut logits = vec![-10.0f32; 16];
        logits[7] = 10.0;
        let opts = SampleOpts::temperature(1.0, 999).with_top_p(0.5);
        assert_eq!(sample_token(&logits, opts), 7);
    }

    #[test]
    fn high_temperature_still_returns_valid_id() {
        let logits = vec![0.0; 10];
        let opts = SampleOpts::temperature(100.0, 1);
        let t = sample_token(&logits, opts);
        assert!(t < 10);
    }
}
