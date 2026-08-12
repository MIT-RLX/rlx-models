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

//! Entropy-bounded block-diffusion sampling — the host half of generation.
//!
//! Each denoising step the model scores the whole canvas at once. The sampler
//! decides which of those predictions to *keep*: it accepts the `k` lowest-entropy
//! positions such that
//!
//! ```text
//! Σᵢ₌₁..ₖ Hᵢ − max(H₁..Hₖ) ≤ entropy_bound
//! ```
//!
//! which upper-bounds the joint mutual information between the accepted tokens,
//! so they are approximately independent and can be committed in parallel
//! (<https://arxiv.org/pdf/2505.24857>). Everything not accepted is re-noised to
//! a fresh uniform token and reconsidered next step. That is what buys ~15–20
//! tokens per forward pass instead of one.
//!
//! Because the entropies are sorted ascending, `Σ − max` is just "the sum of all
//! strictly smaller entropies", so the accepted set is always a prefix of the
//! sorted order.

/// Small seedable RNG (`xoshiro256**`) so canvas initialization and multinomial
/// draws are reproducible across runs and backends.
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn seed_from_u64(seed: u64) -> Self {
        // SplitMix64 expansion, the standard way to seed xoshiro from one word.
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Uniform in `[0, n)`.
    pub fn below(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        (self.next_u64() % n as u64) as u32
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        // 24 bits of mantissa — enough for a categorical draw.
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Shannon entropy of one logits row, in nats — `torch.distributions.Categorical
/// (logits=…).entropy()`, which normalizes the logits first.
pub fn row_entropy(logits: &[f32]) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return 0.0;
    }
    let mut sum_exp = 0f64;
    for &l in logits {
        sum_exp += ((l - max) as f64).exp();
    }
    let lse = max as f64 + sum_exp.ln();
    // H = -Σ p·logp with logp = l - lse.
    let mut h = 0f64;
    for &l in logits {
        let logp = l as f64 - lse;
        h -= logp.exp() * logp;
    }
    h as f32
}

/// Index of the largest logit (ties → smallest index, matching `torch.argmax`).
pub fn row_argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// One categorical draw from `softmax(logits)`.
pub fn row_sample(logits: &[f32], rng: &mut Rng) -> u32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    for &l in logits {
        sum += ((l - max) as f64).exp();
    }
    let target = rng.unit() as f64 * sum;
    let mut acc = 0f64;
    for (i, &l) in logits.iter().enumerate() {
        acc += ((l - max) as f64).exp();
        if acc >= target {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

/// Per-step view of the denoiser output for one canvas.
#[derive(Debug, Clone)]
pub struct StepScores {
    /// Per-position entropy of the processed logits.
    pub entropy: Vec<f32>,
    /// Per-position argmax — the "draft" the model is most confident about.
    pub argmax: Vec<u32>,
    /// Per-position multinomial draw.
    pub sampled: Vec<u32>,
}

impl StepScores {
    /// Reduce a `[canvas, vocab]` block of processed logits.
    pub fn from_logits(logits: &[f32], canvas: usize, vocab: usize, rng: &mut Rng) -> Self {
        assert_eq!(
            logits.len(),
            canvas * vocab,
            "logits must be [canvas, vocab]"
        );
        let mut entropy = Vec::with_capacity(canvas);
        let mut argmax = Vec::with_capacity(canvas);
        let mut sampled = Vec::with_capacity(canvas);
        for c in 0..canvas {
            let row = &logits[c * vocab..(c + 1) * vocab];
            entropy.push(row_entropy(row));
            argmax.push(row_argmax(row));
            sampled.push(row_sample(row, rng));
        }
        Self {
            entropy,
            argmax,
            sampled,
        }
    }
}

/// `EntropyBoundSampler` — accept/renoise over a single canvas.
#[derive(Debug, Clone)]
pub struct EntropyBoundSampler {
    pub entropy_bound: f32,
    pub canvas_length: usize,
    pub vocab_size: usize,
    /// Positions accepted by the most recent [`Self::accept`].
    accepted: Vec<bool>,
}

impl EntropyBoundSampler {
    pub fn new(entropy_bound: f32, canvas_length: usize, vocab_size: usize) -> Self {
        Self {
            entropy_bound,
            canvas_length,
            vocab_size,
            accepted: vec![false; canvas_length],
        }
    }

    /// Fresh canvas of uniform random token ids.
    pub fn initialize_canvas(&self, rng: &mut Rng) -> Vec<u32> {
        (0..self.canvas_length)
            .map(|_| rng.below(self.vocab_size as u32))
            .collect()
    }

    /// Which positions the bound accepts, given per-position entropies.
    ///
    /// Accepts the `k` smallest entropies while the sum of the *strictly
    /// smaller* ones stays within `entropy_bound`. Ties break by position so the
    /// result is deterministic.
    pub fn acceptance_mask(&self, entropy: &[f32]) -> Vec<bool> {
        let mut order: Vec<usize> = (0..entropy.len()).collect();
        order.sort_by(|&a, &b| {
            entropy[a]
                .partial_cmp(&entropy[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut mask = vec![false; entropy.len()];
        // `cumulative_entropy - sorted_entropy` = sum of everything before this
        // element in sorted order, since the list is ascending.
        let mut prefix = 0f64;
        for &idx in &order {
            if prefix <= self.entropy_bound as f64 {
                mask[idx] = true;
            }
            prefix += entropy[idx] as f64;
        }
        mask
    }

    /// Keep the denoiser's tokens where accepted, the current canvas elsewhere.
    pub fn accept(&mut self, current: &[u32], denoiser: &[u32], entropy: &[f32]) -> Vec<u32> {
        self.accepted = self.acceptance_mask(entropy);
        current
            .iter()
            .zip(denoiser)
            .zip(&self.accepted)
            .map(|((&cur, &den), &acc)| if acc { den } else { cur })
            .collect()
    }

    /// Re-noise every position the last [`Self::accept`] rejected.
    pub fn renoise(&self, accepted_canvas: &[u32], rng: &mut Rng) -> Vec<u32> {
        accepted_canvas
            .iter()
            .zip(&self.accepted)
            .map(|(&tok, &acc)| {
                if acc {
                    tok
                } else {
                    rng.below(self.vocab_size as u32)
                }
            })
            .collect()
    }

    /// Positions accepted by the last [`Self::accept`].
    pub fn accepted_mask(&self) -> &[bool] {
        &self.accepted
    }
}

/// `StableAndConfidentStoppingCriteria` — stop denoising early once the draft
/// stops moving *and* the model is confident.
#[derive(Debug, Clone)]
pub struct StableAndConfident {
    pub stability_threshold: usize,
    pub confidence_threshold: f32,
    history: Vec<Vec<u32>>,
}

impl StableAndConfident {
    pub fn new(stability_threshold: usize, confidence_threshold: f32) -> Self {
        Self {
            stability_threshold,
            confidence_threshold,
            history: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// `stable && confident` for this step, updating the rolling history.
    pub fn should_stop(&mut self, argmax_canvas: &[u32], entropy: &[f32]) -> bool {
        let stable = if self.stability_threshold == 0 {
            true
        } else {
            if self.history.is_empty() {
                // HF seeds the history with -1, which can never match a token id.
                self.history = vec![Vec::new(); self.stability_threshold];
            }
            let stable = self.history.iter().all(|h| h.as_slice() == argmax_canvas);
            self.history.remove(0);
            self.history.push(argmax_canvas.to_vec());
            stable
        };

        let mean_entropy =
            entropy.iter().map(|&e| e as f64).sum::<f64>() / entropy.len().max(1) as f64;
        let confident = (mean_entropy as f32) < self.confidence_threshold;
        stable && confident
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits_from_probs(p: &[f32]) -> Vec<f32> {
        p.iter().map(|&x| x.ln()).collect()
    }

    #[test]
    fn entropy_matches_closed_forms() {
        // Uniform over 4 → ln 4.
        let uniform = logits_from_probs(&[0.25, 0.25, 0.25, 0.25]);
        assert!((row_entropy(&uniform) - 4f32.ln()).abs() < 1e-5);

        // One-hot → 0.
        let peaked = vec![50.0f32, 0.0, 0.0, 0.0];
        assert!(row_entropy(&peaked) < 1e-6);

        // Entropy is shift-invariant (softmax is).
        let shifted: Vec<f32> = uniform.iter().map(|l| l + 13.5).collect();
        assert!((row_entropy(&shifted) - row_entropy(&uniform)).abs() < 1e-5);

        // Known asymmetric case: p = [0.5, 0.25, 0.25] → 1.5·ln 2.
        let p = logits_from_probs(&[0.5, 0.25, 0.25]);
        assert!((row_entropy(&p) - 1.5 * 2f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn acceptance_is_a_prefix_of_the_sorted_order() {
        let s = EntropyBoundSampler::new(0.1, 5, 100);
        // Sum of everything strictly smaller must stay <= 0.1.
        // sorted: 0.0 (prefix 0.0 ✓), 0.02 (0.0 ✓), 0.05 (0.02 ✓),
        //         0.3 (0.07 ✓), 0.9 (0.37 ✗)
        let e = vec![0.3, 0.0, 0.9, 0.02, 0.05];
        let mask = s.acceptance_mask(&e);
        assert_eq!(mask, vec![true, true, false, true, true]);
    }

    #[test]
    fn a_zero_bound_still_accepts_the_single_lowest() {
        // prefix before the first element is 0.0, and 0.0 <= 0.0.
        let s = EntropyBoundSampler::new(0.0, 3, 10);
        let mask = s.acceptance_mask(&[0.4, 0.1, 0.7]);
        assert_eq!(mask, vec![false, true, false]);
    }

    #[test]
    fn a_huge_bound_accepts_everything() {
        let s = EntropyBoundSampler::new(1e9, 4, 10);
        assert_eq!(s.acceptance_mask(&[3.0, 1.0, 2.0, 9.0]), vec![true; 4]);
    }

    /// The bound is on the *already-selected* set, not on each token, so a
    /// high-entropy position is still accepted when everything sorted before it
    /// was cheap. Cross-checked against `torch.sort`/`cumsum`/`scatter`:
    /// `cum - sorted = [0.0, 0.0, 0.01, 5.01]` → the first three pass at 0.1.
    #[test]
    fn accept_then_renoise_only_touches_rejected_slots() {
        let mut s = EntropyBoundSampler::new(0.1, 4, 1000);
        let current = vec![10, 11, 12, 13];
        let denoiser = vec![20, 21, 22, 23];
        let entropy = vec![0.0, 5.0, 0.01, 6.0];
        let accepted = s.accept(&current, &denoiser, &entropy);
        assert_eq!(accepted, vec![20, 21, 22, 13]);

        let mut rng = Rng::seed_from_u64(7);
        let renoised = s.renoise(&accepted, &mut rng);
        assert_eq!(renoised[0], 20, "accepted slots survive renoising");
        assert_eq!(renoised[2], 22);
        // Only the rejected slot is redrawn.
        assert_eq!(renoised[1], 21);
        assert!(renoised[3] < 1000);
    }

    #[test]
    fn canvas_init_is_seeded_and_in_range() {
        let s = EntropyBoundSampler::new(0.1, 64, 257);
        let a = s.initialize_canvas(&mut Rng::seed_from_u64(42));
        let b = s.initialize_canvas(&mut Rng::seed_from_u64(42));
        let c = s.initialize_canvas(&mut Rng::seed_from_u64(43));
        assert_eq!(a, b, "same seed → same canvas");
        assert_ne!(a, c, "different seed → different canvas");
        assert_eq!(a.len(), 64);
        assert!(a.iter().all(|&t| t < 257));
    }

    #[test]
    fn stopping_needs_both_stability_and_confidence() {
        let mut s = StableAndConfident::new(1, 0.005);
        let canvas = vec![1u32, 2, 3];
        let low = vec![0.001f32; 3];
        let high = vec![1.0f32; 3];

        // First call: history is empty-seeded, so it cannot be stable yet.
        assert!(!s.should_stop(&canvas, &low));
        // Second call with the same draft and low entropy: stable + confident.
        assert!(s.should_stop(&canvas, &low));
        // Confident but the draft moved → not stable.
        assert!(!s.should_stop(&[9, 9, 9], &low));
        // Stable but not confident.
        s.reset();
        s.should_stop(&canvas, &high);
        assert!(!s.should_stop(&canvas, &high));
    }

    #[test]
    fn zero_stability_threshold_only_checks_confidence() {
        let mut s = StableAndConfident::new(0, 0.005);
        assert!(s.should_stop(&[1, 2, 3], &[0.001; 3]));
        assert!(!s.should_stop(&[1, 2, 3], &[0.9; 3]));
    }

    #[test]
    fn sampling_follows_the_distribution() {
        // p ≈ [0.7, 0.2, 0.1]
        let logits = logits_from_probs(&[0.7, 0.2, 0.1]);
        let mut rng = Rng::seed_from_u64(1234);
        let mut counts = [0usize; 3];
        for _ in 0..20_000 {
            counts[row_sample(&logits, &mut rng) as usize] += 1;
        }
        let f0 = counts[0] as f32 / 20_000.0;
        let f1 = counts[1] as f32 / 20_000.0;
        assert!((f0 - 0.7).abs() < 0.02, "p0 = {f0}");
        assert!((f1 - 0.2).abs() < 0.02, "p1 = {f1}");
        assert_eq!(row_argmax(&logits), 0);
    }

    #[test]
    fn step_scores_reduce_a_logits_block() {
        let canvas = 3;
        let vocab = 4;
        let mut logits = vec![0f32; canvas * vocab];
        // Row 0: sharply peaked on token 2. Row 1: uniform. Row 2: peaked on 0.
        logits[2] = 20.0; // row 0 → token 2
        logits[2 * vocab] = 20.0; // row 2 → token 0
        let mut rng = Rng::seed_from_u64(5);
        let s = StepScores::from_logits(&logits, canvas, vocab, &mut rng);
        assert_eq!(s.argmax, vec![2, 0, 0]);
        assert!(s.entropy[0] < 1e-5);
        assert!((s.entropy[1] - 4f32.ln()).abs() < 1e-5);
        assert!(s.sampled.iter().all(|&t| t < vocab as u32));
        // A peaked row samples its mode.
        assert_eq!(s.sampled[0], 2);
    }
}
