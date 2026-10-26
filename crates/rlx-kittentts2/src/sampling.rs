// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Decode-time sampler, in upstream's order (`kittenml.kittentts2.logits`):
//!
//! 1. **token-run penalty** — once one codec token has repeated `grace` times
//!    in a row, subtract `(run − grace + 1) · strength` from it; the guard
//!    against sticking on silence or a held phoneme;
//! 2. **repetition penalty** over the last `window` *generated* tokens, never
//!    penalizing `<speech_end>` / `[STOP]` (that would push the model away from
//!    closing the segment);
//! 3. temperature, top-k, top-p, min-p — `transformers`' warpers, including
//!    their tie and minimum-keep rules;
//! 4. multinomial draw (or argmax when greedy).

use crate::config::{DecodePreset, TokenMap};

/// Every knob of one generation's sampler.
#[derive(Debug, Clone)]
pub struct SamplerOpts {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    /// Scan only the last N generated tokens; `None` scans the whole sequence.
    pub repetition_window: Option<usize>,
    pub token_run_penalty: f32,
    pub token_run_grace: usize,
    /// Argmax after the processors instead of a draw.
    pub greedy: bool,
}

impl SamplerOpts {
    /// Upstream's defaults (`ADVANCED_OPTIONS`) around a decode preset.
    pub fn from_preset(p: DecodePreset, repetition_window: Option<usize>) -> Self {
        Self {
            temperature: p.temperature,
            top_k: p.top_k,
            top_p: p.top_p,
            min_p: p.min_p,
            repetition_penalty: 1.1,
            repetition_window,
            token_run_penalty: 1.3,
            token_run_grace: 10,
            greedy: false,
        }
    }
}

/// SplitMix64 — tiny, seedable, and the same stream on every platform.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}

/// Apply the penalties in place. `history` is prompt + generated so far;
/// `prompt_len` marks where generation began.
pub fn apply_penalties(
    scores: &mut [f32],
    history: &[u32],
    prompt_len: usize,
    tm: &TokenMap,
    o: &SamplerOpts,
) {
    if o.token_run_penalty != 0.0
        && let Some(&last) = history.last()
    {
        let run = history.iter().rev().take_while(|&&t| t == last).count();
        if tm.is_audio(last) && run >= o.token_run_grace {
            scores[last as usize] -= (run - o.token_run_grace + 1) as f32 * o.token_run_penalty;
        }
    }
    if o.repetition_penalty != 1.0 {
        let start = match o.repetition_window {
            Some(w) if w > 0 => prompt_len.max(history.len().saturating_sub(w)),
            _ => 0,
        };
        let exempt = [tm.speech_end_id, tm.stop_id];
        let mut seen = std::collections::HashSet::new();
        for &t in &history[start.min(history.len())..] {
            if exempt.contains(&t) || !seen.insert(t) {
                continue;
            }
            let s = &mut scores[t as usize];
            *s = if *s < 0.0 {
                *s * o.repetition_penalty
            } else {
                *s / o.repetition_penalty
            };
        }
    }
}

/// Temperature → top-k → top-p → min-p, then draw. Returns the token id.
pub fn sample(scores: &[f32], o: &SamplerOpts, rng: &mut Rng) -> u32 {
    let inv_t = if o.temperature != 1.0 && o.temperature > 0.0 {
        1.0 / o.temperature
    } else {
        1.0
    };
    // Candidates as (id, scaled score), best first.
    let mut cand: Vec<(u32, f32)> = scores
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_finite())
        .map(|(i, &s)| (i as u32, s * inv_t))
        .collect();
    if cand.is_empty() {
        return 0;
    }
    let by_score = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if o.top_k > 0 && o.top_k < cand.len() {
        // `TopKLogitsWarper` keeps everything scoring at least the k-th value.
        cand.select_nth_unstable_by(o.top_k - 1, by_score);
        let kth = cand[o.top_k - 1].1;
        cand.retain(|c| c.1 >= kth);
    }
    cand.sort_by(by_score);
    if o.top_p < 1.0 {
        // `TopPLogitsWarper`: drop the ascending-order prefix whose cumulative
        // probability is ≤ 1 − top_p, always keeping the single best.
        let p = softmax(&cand);
        let mut cum = 0.0f64;
        let mut keep = cand.len();
        for (i, &pi) in p.iter().enumerate().rev() {
            cum += pi;
            if cum <= (1.0 - o.top_p) as f64 && i > 0 {
                keep = i;
            } else {
                break;
            }
        }
        cand.truncate(keep.max(1));
    }
    if o.min_p > 0.0 {
        let p = softmax(&cand);
        let cut = p[0] * o.min_p as f64;
        let n = p.iter().take_while(|&&x| x >= cut).count().max(1);
        cand.truncate(n);
    }
    if o.greedy {
        return cand[0].0;
    }
    let p = softmax(&cand);
    let mut r = rng.uniform();
    for (c, pi) in cand.iter().zip(&p) {
        if r < *pi {
            return c.0;
        }
        r -= pi;
    }
    cand.last().map_or(0, |c| c.0)
}

fn softmax(cand: &[(u32, f32)]) -> Vec<f64> {
    let m = cand.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = cand.iter().map(|c| (c.1 as f64 - m).exp()).collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|x| x / z).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tm() -> TokenMap {
        serde_json::from_value(serde_json::json!({
            "audio_id_base": 4, "num_audio_tokens": 4, "speech_start_id": 0,
            "speech_end_id": 1, "text_start_id": 2, "text_end_id": 3, "start_id": 8,
            "stop_id": 9
        }))
        .unwrap()
    }

    fn opts() -> SamplerOpts {
        SamplerOpts {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 2.0,
            repetition_window: Some(3),
            token_run_penalty: 1.5,
            token_run_grace: 2,
            greedy: true,
        }
    }

    #[test]
    fn penalties_follow_upstream() {
        let tm = tm();
        let mut s = vec![1.0f32; 10];
        s[1] = 4.0;
        s[6] = -2.0;
        // prompt [8, 6], generated [6, 6, 1]: token 6 runs? no — last is 1.
        apply_penalties(&mut s, &[8, 6, 6, 6, 1], 2, &tm, &opts());
        assert_eq!(s[1], 4.0, "speech_end is exempt");
        assert_eq!(s[6], -4.0, "negative score is multiplied");
        assert_eq!(
            s[8], 1.0,
            "prompt tokens outside the window are not penalized"
        );

        let mut s = vec![1.0f32; 10];
        apply_penalties(&mut s, &[8, 5, 5, 5], 1, &tm, &opts());
        // run of 3 ≥ grace 2 → −(3 − 2 + 1)·1.5, then the repetition divide.
        assert_eq!(s[5], (1.0 - 3.0) * 2.0);
    }

    #[test]
    fn top_k_keeps_ties_and_top_p_keeps_the_best() {
        let mut o = opts();
        o.greedy = false;
        o.top_k = 2;
        let s = [0.0, 3.0, 3.0, 3.0, 1.0];
        let mut rng = Rng::new(7);
        for _ in 0..50 {
            assert!([1, 2, 3].contains(&sample(&s, &o, &mut rng)));
        }
        o.top_k = 0;
        o.top_p = 0.01;
        for _ in 0..20 {
            assert_eq!(sample(&[0.0, 9.0, 1.0], &o, &mut rng), 1);
        }
    }
}
