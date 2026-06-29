// RLX models — OpenAI-compatible server.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Map OpenAI sampling parameters onto rlx's [`SampleOpts`] plus a host-side
//! `logit_bias` table (applied to the raw logits before sampling, since
//! `SampleOpts` is `Copy` and carries no map).

use rlx_qwen3::SampleOpts;
use std::collections::HashMap;

/// The OpenAI-style sampling knobs we accept. All optional; `None` ⇒ rlx
/// default. `top_k`/`min_p`/`repetition_penalty` are accepted extensions.
#[derive(Debug, Default, Clone)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
}

/// Build a [`SampleOpts`] plus a `(token_id, bias)` table from request
/// params. `temperature == 0` (or absent) ⇒ greedy.
pub fn to_sample_opts(
    p: &SamplingParams,
    logit_bias: Option<&HashMap<String, f32>>,
) -> (SampleOpts, Vec<(u32, f32)>) {
    let seed = p.seed.unwrap_or(0);
    let temp = p.temperature.unwrap_or(0.0);
    let mut opts = if temp > 0.0 {
        SampleOpts::temperature(temp, seed)
    } else {
        SampleOpts::greedy()
    };
    if let Some(tp) = p.top_p {
        opts = opts.with_top_p(tp);
    }
    if let Some(tk) = p.top_k {
        opts = opts.with_top_k(tk);
    }
    if let Some(mp) = p.min_p {
        opts = opts.with_min_p(mp);
    }
    if p.frequency_penalty.is_some() || p.presence_penalty.is_some() {
        opts = opts.with_frequency_presence(
            p.frequency_penalty.unwrap_or(0.0),
            p.presence_penalty.unwrap_or(0.0),
        );
    }
    if let Some(rp) = p.repetition_penalty {
        opts = opts.with_repetition_penalty(rp);
    }

    let bias = logit_bias
        .map(|m| {
            m.iter()
                .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
                .collect()
        })
        .unwrap_or_default();
    (opts, bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_temperature_is_greedy() {
        let (opts, _) = to_sample_opts(&SamplingParams::default(), None);
        assert!(opts.greedy);
    }

    #[test]
    fn maps_temperature_and_filters() {
        let p = SamplingParams {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            min_p: Some(0.05),
            seed: Some(123),
            ..Default::default()
        };
        let (opts, _) = to_sample_opts(&p, None);
        assert!(!opts.greedy);
        assert_eq!(opts.seed, 123);
        assert_eq!(opts.top_p, 0.9);
        assert_eq!(opts.top_k, 40);
        assert_eq!(opts.min_p, 0.05);
        assert!(!opts.is_classic()); // min_p routes through the chain
    }

    #[test]
    fn parses_logit_bias_ids() {
        let mut m = HashMap::new();
        m.insert("100".to_string(), 5.0);
        m.insert("not-an-id".to_string(), 1.0);
        let (_, bias) = to_sample_opts(&SamplingParams::default(), Some(&m));
        assert_eq!(bias, vec![(100u32, 5.0)]);
    }
}
