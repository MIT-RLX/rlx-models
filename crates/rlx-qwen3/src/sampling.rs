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
use rlx_runtime::samplers::{
    Dry, DynamicTemperature, MirostatV1, MirostatV2, RepetitionPenalty, SamplerChain, SamplerState,
    Temperature, TopK, TopNSigma, TopP, TypicalP, Xtc,
};

/// Mirostat variant — mirrors `rlx_runtime::lm::MirostatMode` so callers
/// don't need to import two enums.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MirostatMode {
    #[default]
    Off,
    V1,
    V2,
}

/// Sampling configuration. Construct via [`SampleOpts::greedy`] /
/// [`SampleOpts::temperature`] or build manually.
///
/// Order of operations matches HF `transformers` defaults:
///   1. `temperature` divides logits (skipped if `<= 0` or `1.0`).
///   2. `top_k` truncates to the K highest-logit tokens (0 = disabled).
///   3. `top_p` truncates by nucleus cumulative-mass cutoff (1.0 = disabled).
///   4. Softmax + multinomial sample (or argmax when greedy).
///
/// Advanced samplers (Mirostat, DRY, XTC, etc.) live on the struct
/// behind explicit fields that default to off; when any of them is
/// engaged the sampler routes through `rlx_runtime::SamplerChain`
/// instead of the inline top-k/top-p path. See [`SampleOpts::is_classic`].
/// Maximum number of DRY-sampler sequence-break tokens supported by
/// the fixed-size [`SampleOpts::dry_sequence_breakers`] buffer.
/// 32 covers every special-token set in current tokenizers (BOS,
/// EOS, padding, role-tag tokens, newline, period). Storing it
/// inline keeps `SampleOpts` `Copy`; the whole model-runner
/// ecosystem assumes that — moving away cascades into 30+ call sites.
pub const DRY_BREAKERS_MAX: usize = 32;

#[derive(Debug, Clone, Copy)]
pub struct SampleOpts {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
    pub greedy: bool,

    // ── advanced samplers (all default to off / no-op) ───────────
    pub dynamic_temp: Option<(f32, f32)>,
    pub dynamic_temp_exponent: f32,
    pub typical_p: f32,
    pub top_n_sigma: f32,
    pub xtc_threshold: f32,
    pub xtc_prob: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_max_ngram: usize,
    /// DRY sequence-break tokens packed into a fixed-size array.
    /// Use [`SampleOpts::dry_breakers`] to read the slice or
    /// [`SampleOpts::with_dry_breakers`] to set it.
    pub dry_sequence_breakers: [u32; DRY_BREAKERS_MAX],
    pub dry_sequence_breakers_len: u8,
    pub mirostat: MirostatMode,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub mirostat_m: usize,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub repetition_window: usize,
    pub min_keep: usize,
}

impl SampleOpts {
    pub fn greedy() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
            greedy: true,
            dynamic_temp: None,
            dynamic_temp_exponent: 1.0,
            typical_p: 1.0,
            top_n_sigma: 0.0,
            xtc_threshold: 0.0,
            xtc_prob: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_max_ngram: 32,
            dry_sequence_breakers: [0; DRY_BREAKERS_MAX],
            dry_sequence_breakers_len: 0,
            mirostat: MirostatMode::Off,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            mirostat_m: 100,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_window: 64,
            min_keep: 1,
        }
    }

    pub fn temperature(temp: f32, seed: u64) -> Self {
        Self {
            temperature: temp,
            top_k: 0,
            top_p: 1.0,
            seed,
            greedy: false,
            ..Self::greedy()
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

    // ── advanced sampler builders (chainable) ────────────────────

    pub fn with_dynamic_temp(mut self, min: f32, max: f32) -> Self {
        self.dynamic_temp = Some((min, max));
        self.greedy = false;
        self
    }

    pub fn with_typical_p(mut self, p: f32) -> Self {
        self.typical_p = p;
        self.greedy = false;
        self
    }

    pub fn with_top_n_sigma(mut self, n: f32) -> Self {
        self.top_n_sigma = n;
        self.greedy = false;
        self
    }

    pub fn with_xtc(mut self, threshold: f32, prob: f32) -> Self {
        self.xtc_threshold = threshold;
        self.xtc_prob = prob;
        self.greedy = false;
        self
    }

    pub fn with_dry(mut self, multiplier: f32, base: f32, allowed_length: usize) -> Self {
        self.dry_multiplier = multiplier;
        self.dry_base = base;
        self.dry_allowed_length = allowed_length;
        self.greedy = false;
        self
    }

    pub fn with_mirostat_v1(mut self, tau: f32, eta: f32) -> Self {
        self.mirostat = MirostatMode::V1;
        self.mirostat_tau = tau;
        self.mirostat_eta = eta;
        self.greedy = false;
        self
    }

    pub fn with_mirostat_v2(mut self, tau: f32, eta: f32) -> Self {
        self.mirostat = MirostatMode::V2;
        self.mirostat_tau = tau;
        self.mirostat_eta = eta;
        self.greedy = false;
        self
    }

    pub fn with_repetition_penalty(mut self, p: f32) -> Self {
        self.repetition_penalty = p;
        self
    }

    pub fn with_frequency_presence(mut self, frequency: f32, presence: f32) -> Self {
        self.frequency_penalty = frequency;
        self.presence_penalty = presence;
        self
    }

    /// True when only classic top-k/top-p/temperature/greedy is active —
    /// callers can stay on the cheap inline path. Returns false the
    /// moment any advanced sampler is engaged so they get routed
    /// through `SamplerChain`.
    pub fn is_classic(&self) -> bool {
        self.dynamic_temp.is_none()
            && self.typical_p >= 1.0
            && self.top_n_sigma <= 0.0
            && self.xtc_prob <= 0.0
            && self.dry_multiplier <= 0.0
            && self.mirostat == MirostatMode::Off
            && self.frequency_penalty == 0.0
            && self.presence_penalty == 0.0
            && (self.repetition_penalty - 1.0).abs() < f32::EPSILON
    }

    /// Build a `rlx_runtime::SamplerChain` from these options. See
    /// `rlx_runtime::SampleOpts::into_chain` — same ordering.
    pub fn into_chain(&self) -> SamplerChain {
        let mut b = SamplerChain::builder();
        if (self.repetition_penalty - 1.0).abs() > f32::EPSILON
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
        {
            b = b.push(RepetitionPenalty {
                penalty: self.repetition_penalty,
                frequency: self.frequency_penalty,
                presence: self.presence_penalty,
                last_n: self.repetition_window,
            });
        }
        if self.dry_multiplier > 0.0 {
            b = b.push(Dry {
                multiplier: self.dry_multiplier,
                base: self.dry_base,
                allowed_length: self.dry_allowed_length,
                max_ngram: self.dry_max_ngram,
                sequence_breakers: self.dry_sequence_breakers
                    [..self.dry_sequence_breakers_len as usize]
                    .to_vec(),
            });
        }
        if self.mirostat == MirostatMode::Off {
            if let Some((mn, mx)) = self.dynamic_temp {
                b = b.push(DynamicTemperature {
                    min: mn,
                    max: mx,
                    exponent: self.dynamic_temp_exponent,
                });
            } else if self.temperature > 0.0 && (self.temperature - 1.0).abs() > f32::EPSILON {
                b = b.push(Temperature {
                    t: self.temperature,
                });
            }
        }
        if self.top_k > 0 {
            b = b.push(TopK { k: self.top_k });
        }
        if self.typical_p < 1.0 && self.typical_p > 0.0 {
            b = b.push(TypicalP {
                p: self.typical_p,
                min_keep: self.min_keep,
            });
        }
        if self.top_p < 1.0 && self.top_p > 0.0 {
            b = b.push(TopP {
                p: self.top_p,
                min_keep: self.min_keep,
            });
        }
        if self.top_n_sigma > 0.0 {
            b = b.push(TopNSigma {
                n: self.top_n_sigma,
            });
        }
        if self.xtc_prob > 0.0 && self.xtc_threshold > 0.0 {
            b = b.push(Xtc {
                threshold: self.xtc_threshold,
                prob: self.xtc_prob,
                min_keep: self.min_keep,
            });
        }
        match self.mirostat {
            MirostatMode::Off => {}
            MirostatMode::V1 => {
                b = b.push(MirostatV1 {
                    tau: self.mirostat_tau,
                    eta: self.mirostat_eta,
                    m: self.mirostat_m,
                });
            }
            MirostatMode::V2 => {
                b = b.push(MirostatV2 {
                    tau: self.mirostat_tau,
                    eta: self.mirostat_eta,
                });
            }
        }
        b.build()
    }
}

/// Sample one token id from a `[vocab]` logits slice. Returns the
/// chosen index. Uses `opts.seed` only — prefer [`sample_token_at`]
/// when generating multiple tokens from the same seed.
pub fn sample_token(logits: &[f32], opts: SampleOpts) -> usize {
    sample_token_at(logits, opts, 0)
}

/// Like [`sample_token`] but derives RNG from `opts.seed.wrapping_add(step)`.
/// Callers should pass the zero-based index of the token being sampled
/// (0 for the first post-prefill token, then 1, 2, …).
pub fn sample_token_at(logits: &[f32], opts: SampleOpts, step: u64) -> usize {
    // Fast classic path: untouched legacy behaviour for callers that
    // don't engage any advanced sampler. Saves a Vec clone + chain build.
    if opts.is_classic() {
        return sample_token_inner(logits, opts, step);
    }
    sample_token_with_history(logits, &opts, &[], step) as usize
}

/// Sample one token using the full advanced sampler chain, optionally
/// conditioned on `history` (newest token last). Required for
/// `DRY` / `RepetitionPenalty` which use the token history. The chain
/// is rebuilt per call — cheap (microseconds) compared to the kernel
/// step that produced the logits. Callers that want amortised chain
/// state (e.g. Mirostat's μ updates) should keep their own
/// `SamplerState` across steps; see [`sample_token_stateful`].
pub fn sample_token_with_history(
    logits: &[f32],
    opts: &SampleOpts,
    history: &[u32],
    step: u64,
) -> u32 {
    let mut state = SamplerState::new();
    sample_token_stateful(logits, opts, history, step, &mut state)
}

/// Like [`sample_token_with_history`] but threads the `SamplerState`
/// across calls. Use this when running Mirostat or other stateful
/// samplers across a decode loop so μ persists between steps.
pub fn sample_token_stateful(
    logits: &[f32],
    opts: &SampleOpts,
    history: &[u32],
    step: u64,
    state: &mut SamplerState,
) -> u32 {
    let mut work = logits.to_vec();
    let chain = opts.into_chain();
    let mut rng = Philox4x32::new(opts.seed.wrapping_add(step));
    chain.sample(&mut work, history, state, &mut rng)
}

fn sample_token_inner(logits: &[f32], opts: SampleOpts, step: u64) -> usize {
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
    let mut rng = Philox4x32::new(opts.seed.wrapping_add(step));
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

/// HF-style repetition penalty using per-token occurrence counts.
pub fn apply_repetition_penalty(
    logits: &mut [f32],
    counts: &std::collections::HashMap<u32, u32>,
    penalty: f32,
) {
    if penalty <= 1.0 || counts.is_empty() {
        return;
    }
    for (&tok, &count) in counts {
        let idx = tok as usize;
        if idx >= logits.len() {
            continue;
        }
        let factor = penalty.powi(count as i32);
        if logits[idx] > 0.0 {
            logits[idx] /= factor;
        } else {
            logits[idx] *= factor;
        }
    }
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

    #[test]
    fn sample_token_at_varies_rng_by_step() {
        let logits: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05).collect();
        let opts = SampleOpts::temperature(0.9, 100).with_top_p(1.0);
        let a = sample_token_at(&logits, opts, 0);
        let b = sample_token_at(&logits, opts, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn mirostat_v2_routes_through_chain() {
        // Engaging Mirostat flips `is_classic()` and routes through the
        // SamplerChain path. Check it produces a valid token without
        // diverging.
        let logits: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let opts = SampleOpts::temperature(0.7, 42).with_mirostat_v2(5.0, 0.1);
        assert!(!opts.is_classic());
        let t = sample_token(&logits, opts);
        assert!(t < 32);
    }

    #[test]
    fn dynamic_temp_routes_through_chain() {
        let logits: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let opts = SampleOpts::temperature(1.0, 7).with_dynamic_temp(0.5, 1.5);
        assert!(!opts.is_classic());
        let t = sample_token(&logits, opts);
        assert!(t < 32);
    }

    #[test]
    fn classic_path_unchanged_with_new_fields() {
        // Greedy without any advanced sampler must still be classic
        // (no behavior change for existing callers).
        let opts = SampleOpts::greedy();
        assert!(opts.is_classic());
        let opts = SampleOpts::temperature(0.7, 1)
            .with_top_k(40)
            .with_top_p(0.9);
        assert!(opts.is_classic());
    }

    #[test]
    fn sample_token_with_history_threads_dry() {
        // History [0,1,0,1,0] + DRY → token 1 (continuing the 0-1 pattern)
        // should be penalised.
        let history = vec![0u32, 1, 0, 1, 0];
        let logits = vec![0.0, 0.0];
        let opts = SampleOpts::temperature(1.0, 0).with_dry(1.0, 2.0, 2);
        let mut state = SamplerState::new();
        let t = sample_token_stateful(&logits, &opts, &history, 0, &mut state);
        assert!(t < 2);
    }
}
