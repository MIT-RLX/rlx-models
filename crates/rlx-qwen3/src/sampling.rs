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
use rlx_runtime::samplers::{SamplerChain, SamplerState};

// Re-export the host-side logit-bias processor so callers (e.g. the server)
// can apply OpenAI `logit_bias` to a raw logits row before sampling.
pub use rlx_runtime::samplers::apply_logit_bias;

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
    /// Min-p cutoff (Nguyen et al. 2024): keep tokens with prob ≥ `min_p` ·
    /// p_max. 0 ⇒ off.
    pub min_p: f32,
    pub xtc_threshold: f32,
    pub xtc_prob: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_max_ngram: usize,
    /// DRY sequence-break tokens packed into a fixed-size array.
    /// Use `SampleOpts::dry_breakers` to read the slice or
    /// `SampleOpts::with_dry_breakers` to set it.
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
            min_p: 0.0,
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

    pub fn with_min_p(mut self, p: f32) -> Self {
        self.min_p = p;
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
            && self.min_p <= 0.0
            && self.xtc_prob <= 0.0
            && self.dry_multiplier <= 0.0
            && self.mirostat == MirostatMode::Off
            && self.frequency_penalty == 0.0
            && self.presence_penalty == 0.0
            && (self.repetition_penalty - 1.0).abs() < f32::EPSILON
    }

    /// Build a `rlx_runtime::SamplerChain` from these options: the chain of
    /// [`rlx_runtime::lm::SampleOpts::into_chain`], same ordering.
    pub fn into_chain(&self) -> SamplerChain {
        self.to_runtime().into_chain()
    }

    /// The same settings as [`rlx_runtime::lm::SampleOpts`] (which owns
    /// `dry_sequence_breakers` as a `Vec`, so is not `Copy`).
    pub fn to_runtime(&self) -> rlx_runtime::lm::SampleOpts {
        rlx_runtime::lm::SampleOpts {
            // Temperature `<= 0` is "no scaling" here; greediness is the
            // separate `greedy` flag. The runtime reads `<= 0` as greedy and
            // adds a 1e-6 temperature, so pass the neutral value instead.
            temperature: if self.temperature > 0.0 {
                self.temperature
            } else {
                1.0
            },
            top_p: self.top_p,
            top_k: (self.top_k > 0).then_some(self.top_k as u32),
            repetition_penalty: self.repetition_penalty,
            dynamic_temp: self.dynamic_temp,
            dynamic_temp_exponent: self.dynamic_temp_exponent,
            typical_p: self.typical_p,
            top_n_sigma: self.top_n_sigma,
            min_p: self.min_p,
            xtc_threshold: self.xtc_threshold,
            xtc_prob: self.xtc_prob,
            dry_multiplier: self.dry_multiplier,
            dry_base: self.dry_base,
            dry_allowed_length: self.dry_allowed_length,
            dry_max_ngram: self.dry_max_ngram,
            dry_sequence_breakers: self.dry_sequence_breakers
                [..self.dry_sequence_breakers_len as usize]
                .to_vec(),
            mirostat: match self.mirostat {
                MirostatMode::Off => rlx_runtime::lm::MirostatMode::Off,
                MirostatMode::V1 => rlx_runtime::lm::MirostatMode::V1,
                MirostatMode::V2 => rlx_runtime::lm::MirostatMode::V2,
            },
            mirostat_tau: self.mirostat_tau,
            mirostat_eta: self.mirostat_eta,
            mirostat_m: self.mirostat_m,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            repetition_window: self.repetition_window,
            min_keep: self.min_keep,
        }
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
    // Plain greedy: temperature, top-k and top-p never drop the top token, so
    // the chain would end on the same argmax; skip building it.
    if opts.greedy && opts.is_classic() {
        assert!(!logits.is_empty(), "sample_token: empty logits");
        return argmax(logits);
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
    // `greedy` is not a chain step — it changes how the chain *ends*. Without
    // this branch the chain always finished with a multinomial draw, so a
    // caller asking for `temperature = 0` got a sample from the full
    // distribution: usually the top token, because the distribution is
    // usually sharp, and occasionally not. Deterministic decoding was not
    // deterministic, and the difference only showed up as the odd
    // inexplicable word.
    if opts.greedy {
        chain.sample_argmax(&mut work, history, state, &mut rng)
    } else {
        chain.sample(&mut work, history, state, &mut rng)
    }
}

fn argmax(xs: &[f32]) -> usize {
    rlx_runtime::search::argmax(xs)
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
    fn min_p_routes_through_chain_and_truncates() {
        // min_p engages the chain (not classic) and, on a peaked dist,
        // keeps only the dominant token.
        let mut logits = vec![-10.0f32; 16];
        logits[7] = 10.0;
        let opts = SampleOpts::temperature(1.0, 5).with_min_p(0.1);
        assert!(!opts.is_classic());
        assert_eq!(sample_token(&logits, opts), 7);
    }

    #[test]
    fn logit_bias_steers_greedy() {
        // Bias a non-max token above the current argmax → it wins.
        let mut logits = vec![0.1, 0.5, 0.2, -1.0];
        apply_logit_bias(&mut logits, &[(2, 1.0)]);
        assert_eq!(sample_token(&logits, SampleOpts::greedy()), 2);
    }

    /// A distribution with a clear winner that is nonetheless easy to miss:
    /// the top token holds about 73% of the mass, so a multinomial draw
    /// picks something else roughly one time in four.
    fn soft_logits() -> Vec<f32> {
        let mut v = vec![0.0f32; 64];
        v[13] = 3.0;
        v[7] = 1.5;
        v[42] = 1.0;
        v[5] = 0.5;
        v
    }

    #[test]
    fn zero_temperature_is_deterministic_through_the_chain() {
        // `greedy` is not a chain step, and the chain always ended in a
        // multinomial draw — so this asked for determinism and got a sample.
        // With a sharp distribution it looked right most of the time, which
        // is why it survived.
        let opts = SampleOpts::greedy();
        let logits = soft_logits();
        for step in 0..200u64 {
            let got = sample_token_with_history(&logits, &opts, &[], step);
            assert_eq!(got, 13, "step {step} did not take the top token");
        }
    }

    #[test]
    fn greedy_still_honours_a_repetition_penalty() {
        // Deterministic means "do not roll dice", not "ignore my other
        // settings": a penalised token must still lose.
        let mut opts = SampleOpts::greedy();
        opts.repetition_penalty = 10.0;
        opts.repetition_window = 8;
        let logits = soft_logits();
        // 13 has already been produced, so the penalty should push it below 7.
        let got = sample_token_with_history(&logits, &opts, &[13], 0);
        assert_eq!(got, 7, "the penalty did not move the greedy choice");
    }

    #[test]
    fn a_positive_temperature_still_samples() {
        // The other half of the contract: this one is *meant* to vary.
        let mut opts = SampleOpts::greedy();
        opts.greedy = false;
        opts.temperature = 2.0;
        let logits = soft_logits();
        let picks: std::collections::HashSet<u32> = (0..200u64)
            .map(|s| sample_token_with_history(&logits, &opts, &[], s))
            .collect();
        assert!(picks.len() > 1, "sampling returned one token every time");
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
