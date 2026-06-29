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

/// Moshi runs the LM in bf16 and casts to f32 before sampling (`logits.float()`).
#[inline]
fn f32_from_bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

/// Argmax over a logits vector (bf16-rounded, matching Moshi tie-break).
pub fn argmax(logits: &Array1<f32>) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        let v = f32_from_bf16(v);
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

/// Moshi `LMGen` uses one global RNG for text and DepFormer audio draws.
#[derive(Debug, Clone)]
pub struct StreamSampler {
    rng: StdRng,
    pub text_temperature: f32,
    pub audio_temperature: f32,
}

impl StreamSampler {
    pub fn new(seed: u64, text_temperature: f32, audio_temperature: f32) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            text_temperature,
            audio_temperature,
        }
    }

    pub fn sample_text(&mut self, logits: &Array1<f32>) -> u32 {
        sample_with(&mut self.rng, logits, self.text_temperature, 25)
    }

    pub fn sample_audio(&mut self, logits: &Array1<f32>) -> u32 {
        sample_with(&mut self.rng, logits, self.audio_temperature, 250)
    }
}

fn sample_with(rng: &mut StdRng, logits: &Array1<f32>, temperature: f32, top_k: usize) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    // Moshi `sample_token`: softmax(logits / temp), then `sample_top_k` on probs.
    let inv_t = 1.0 / temperature;
    let max = logits
        .iter()
        .map(|&v| f32_from_bf16(v))
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let v = f32_from_bf16(v);
            let p = ((v * inv_t) - max * inv_t).exp();
            (i, p)
        })
        .collect();
    let mut sum = 0.0f32;
    for (_, p) in probs.iter_mut() {
        sum += *p;
    }
    let inv_sum = 1.0 / sum;
    for (_, p) in probs.iter_mut() {
        *p *= inv_sum;
    }
    if top_k > 0 && top_k < probs.len() {
        probs.select_nth_unstable_by(top_k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        probs.truncate(top_k);
        let sub_sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        let inv_sub = 1.0 / sub_sum;
        for (_, p) in probs.iter_mut() {
            *p *= inv_sub;
        }
    }
    // Gumbel-max (`probs / exponential()`), matching Moshi `multinomial`.
    let mut best_id = 0u32;
    let mut best_score = f32::NEG_INFINITY;
    for (id, p) in probs {
        let u: f32 = rng.r#gen::<f32>().max(1e-10);
        let score = p / (-u.ln());
        if score > best_score {
            best_score = score;
            best_id = id as u32;
        }
    }
    best_id
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
        sample_with(&mut self.rng, logits, self.temperature, self.top_k)
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
