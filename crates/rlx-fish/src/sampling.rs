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

//! Token sampling as the two fish models do it.
//!
//! Both rank by logit, keep the top-ranked token whatever the limits say, and
//! draw with the exponential race (`argmax p / Exp(1)`). They differ in one
//! place that matters: S2 computes the top-p cut on the *raw* logits and
//! applies the temperature afterwards; Drama 3 divides by the temperature
//! first.

/// Sampling settings. Defaults are the Fish Audio SDK's (`top_k` 30 for both
/// models, which the SDK does not expose).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.7,
            top_k: 30,
        }
    }
}

impl SampleParams {
    /// Always the most likely token.
    pub fn greedy() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
        }
    }
}

/// Where the temperature enters (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavor {
    /// S2: top-p over `softmax(logits)`, then `/ temperature`.
    S2,
    /// Drama 3: `/ temperature`, then top-p.
    Drama3,
}

const MIN_TEMPERATURE: f32 = 1e-5;

/// SplitMix64 — the sampler's only source of randomness, so a seed fixes
/// the whole utterance.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]`.
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    /// `Exp(1)`.
    fn exponential(&mut self) -> f64 {
        -self.uniform().ln()
    }
}

fn softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = x.iter().map(|&v| (v as f64 - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// Draw an index of `logits`.
pub fn sample(logits: &[f32], p: SampleParams, flavor: Flavor, rng: &mut Rng) -> usize {
    let t = p.temperature.max(MIN_TEMPERATURE);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    // Descending, ties to the lower index (deterministic greedy).
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    if p.top_k <= 1 {
        return order[0];
    }
    let sorted: Vec<f32> = order.iter().map(|&i| logits[i]).collect();
    let scaled: Vec<f32> = sorted.iter().map(|v| v / t).collect();
    let cut_basis = match flavor {
        Flavor::S2 => softmax(&sorted),
        Flavor::Drama3 => softmax(&scaled),
    };
    let mut cum = 0f64;
    let mut keep = 0usize;
    for (rank, pr) in cut_basis.iter().enumerate() {
        cum += pr;
        let dropped = rank > 0 && (cum > p.top_p as f64 || rank >= p.top_k);
        if dropped {
            break;
        }
        keep = rank + 1;
    }
    let probs = softmax(&scaled[..keep]);
    let mut best = (f64::NEG_INFINITY, 0usize);
    for (rank, pr) in probs.iter().enumerate() {
        let score = pr / rng.exponential();
        if score > best.0 {
            best = (score, rank);
        }
    }
    order[best.1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_argmax() {
        let l = [0.1, 3.0, -1.0, 2.9];
        assert_eq!(
            sample(&l, SampleParams::greedy(), Flavor::S2, &mut Rng::new(1)),
            1
        );
    }

    #[test]
    fn top_p_keeps_only_the_head() {
        // One dominant token: top_p 0.5 leaves just it.
        let l = [10.0, 0.0, 0.0, 0.0];
        let p = SampleParams {
            temperature: 1.0,
            top_p: 0.5,
            top_k: 30,
        };
        let mut rng = Rng::new(7);
        for _ in 0..100 {
            assert_eq!(sample(&l, p, Flavor::S2, &mut rng), 0);
        }
    }

    #[test]
    fn flat_logits_spread_out() {
        let l = [0.0f32; 8];
        let p = SampleParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 8,
        };
        let mut rng = Rng::new(3);
        let mut seen = [0usize; 8];
        for _ in 0..800 {
            seen[sample(&l, p, Flavor::Drama3, &mut rng)] += 1;
        }
        assert!(seen.iter().all(|&n| n > 50), "{seen:?}");
    }
}
