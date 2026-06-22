//! Sampling primitives for Kyutai TTS.
//!
//! The model is trained with *distilled* classifier-free guidance — the LUT
//! conditioner `cfg` (7 bins, 1.0..4.0) tells the model the CFG strength up
//! front, so single-pass generation already produces guidance-strengthened
//! logits at the requested ratio. No batch-doubling, no second forward pass.
//!
//! This module wraps the standard temperature + top-k + multinomial path and
//! exposes a thin [`CfgSampler`] that captures the chosen strength so callers
//! can record / propagate it.

use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Argmax over a logits vector.
pub fn argmax(logits: &Array1<f32>) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

/// Temperature + top-k multinomial sampler with a seeded RNG.
#[derive(Debug, Clone)]
pub struct LogitsProcessor {
    pub temperature: f32,
    pub top_k: usize,
    rng: StdRng,
}

impl LogitsProcessor {
    pub fn new(temperature: f32, top_k: usize, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample one token id from a `[card]` logits vector.
    pub fn sample(&mut self, logits: &Array1<f32>) -> u32 {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        let mut scored: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v / self.temperature))
            .collect();
        // Top-k: partial sort by descending score.
        if self.top_k > 0 && self.top_k < scored.len() {
            scored.select_nth_unstable_by(self.top_k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(self.top_k);
        }
        // Stable softmax over the candidate set.
        let max = scored
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (_, v) in scored.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for (_, v) in scored.iter_mut() {
            *v *= inv;
        }
        // Cumulative draw.
        let u: f32 = self.rng.r#gen::<f32>();
        let mut acc = 0.0f32;
        for (id, p) in &scored {
            acc += *p;
            if u <= acc {
                return *id as u32;
            }
        }
        scored.last().map(|(id, _)| *id as u32).unwrap_or(0)
    }
}

/// Distilled-CFG sampler — records the strength so it can be plumbed to the
/// `cfg` LUT conditioner before each generation.
#[derive(Debug, Clone)]
pub struct CfgSampler {
    pub inner: LogitsProcessor,
    /// Strength to look up in the LUT `cfg` conditioner (one of
    /// `"1.0", "1.5", …, "4.0"` per the published config).
    pub alpha: f32,
}

impl CfgSampler {
    pub fn new(inner: LogitsProcessor, alpha: f32) -> Self {
        Self { inner, alpha }
    }

    /// `possible_values` key for the LUT conditioner (e.g. `2.0_f32 → "2.0"`).
    pub fn cfg_lut_key(&self) -> String {
        format!("{:.1}", self.alpha)
    }

    pub fn sample(&mut self, logits: &Array1<f32>) -> u32 {
        self.inner.sample(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn argmax_picks_largest() {
        let l = array![0.1, 5.0, 2.0, -1.0];
        assert_eq!(argmax(&l), 1);
    }

    #[test]
    fn zero_temperature_is_argmax() {
        let mut lp = LogitsProcessor::new(0.0, 0, 0);
        let l = array![-1.0, 3.0, 2.0];
        for _ in 0..5 {
            assert_eq!(lp.sample(&l), 1);
        }
    }

    #[test]
    fn top_k_one_collapses_to_argmax() {
        let mut lp = LogitsProcessor::new(1.0, 1, 42);
        let l = array![0.0, 10.0, 1.0, 2.0];
        for _ in 0..10 {
            assert_eq!(lp.sample(&l), 1);
        }
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let l = array![1.0, 2.0, 3.0, 4.0];
        let mut a = LogitsProcessor::new(1.0, 0, 123);
        let mut b = LogitsProcessor::new(1.0, 0, 123);
        for _ in 0..20 {
            assert_eq!(a.sample(&l), b.sample(&l));
        }
    }

    #[test]
    fn cfg_key_formats_one_decimal() {
        let lp = LogitsProcessor::new(1.0, 0, 0);
        assert_eq!(CfgSampler::new(lp.clone(), 1.5).cfg_lut_key(), "1.5");
        assert_eq!(CfgSampler::new(lp.clone(), 4.0).cfg_lut_key(), "4.0");
    }
}
