// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Speed dimension: prefill throughput, decode throughput, time-to-first-token
//! and peak RSS — the four numbers you compare across models and backends.
//!
//! All measured through the model-agnostic `LmRunner` surface:
//! - **prefill** is timed via a dedicated `prefill_logits` call when the runner
//!   supports it (F32 path); packed/quantized runners that don't are reported
//!   with `prefill_toks_s = 0` and folded into TTFT instead.
//! - **decode** is the steady-state rate of a `generate` run: total minus the
//!   first token, over the post-first-token wall time — so compile/prefill cost
//!   lands in TTFT, not the throughput figure.

use std::time::Instant;

use anyhow::{Result, bail};

use crate::metrics::peak_rss_mb;
use crate::model::BenchModel;

/// What to measure. Provide an explicit `prompt_ids` for a realistic prompt, or
/// leave it empty and set `prompt_len` for a synthetic one.
#[derive(Clone, Debug)]
pub struct SpeedConfig {
    /// Explicit prompt token ids. Empty ⇒ synthesize `prompt_len` ids.
    pub prompt_ids: Vec<u32>,
    /// Length of the synthetic prompt when `prompt_ids` is empty.
    pub prompt_len: usize,
    /// Number of tokens to generate for the decode measurement.
    pub decode_tokens: usize,
    /// Run one short throwaway generation first to warm compile caches so the
    /// measured run reflects steady state, not first-compile latency.
    pub warmup: bool,
}

impl Default for SpeedConfig {
    fn default() -> Self {
        Self {
            prompt_ids: Vec::new(),
            prompt_len: 64,
            decode_tokens: 64,
            warmup: true,
        }
    }
}

/// One speed measurement.
#[derive(Clone, Debug)]
pub struct SpeedResult {
    pub prompt_tokens: usize,
    pub decode_tokens: usize,
    /// Wall time of the dedicated prefill forward (0 if unsupported).
    pub prefill_s: f64,
    /// `prompt_tokens / prefill_s` (0 if prefill timing unsupported).
    pub prefill_toks_s: f64,
    /// Time to first generated token (prefill + one decode step), milliseconds.
    pub ttft_ms: f64,
    /// Wall time of steady-state decoding (after the first token).
    pub decode_s: f64,
    /// `(decode_tokens - 1) / decode_s` — the headline generation rate.
    pub decode_toks_s: f64,
    /// End-to-end generation wall time.
    pub total_s: f64,
    /// Process peak resident memory, MB.
    pub peak_rss_mb: u64,
}

impl SpeedResult {
    /// Machine-readable one-liner for log scraping.
    pub fn bench_line(&self, name: &str, device: &str) -> String {
        format!(
            "LLMBENCH kind=speed model={name} device={device} prompt_toks={} \
             prefill_toks_s={:.1} decode_toks_s={:.1} ttft_ms={:.1} rss_mb={}",
            self.prompt_tokens,
            self.prefill_toks_s,
            self.decode_toks_s,
            self.ttft_ms,
            self.peak_rss_mb
        )
    }
}

fn resolve_prompt(model: &BenchModel, cfg: &SpeedConfig) -> Vec<u32> {
    if !cfg.prompt_ids.is_empty() {
        return cfg.prompt_ids.clone();
    }
    let vocab = model.vocab_size().max(1);
    // Deterministic, in-range synthetic ids. Start at 1 to avoid leaning on a
    // model's padding/BOS id at position 0.
    (0..cfg.prompt_len)
        .map(|i| ((i % (vocab - 1).max(1)) + 1) as u32)
        .collect()
}

/// Run the speed measurement described by `cfg` against `model`.
pub fn run_speed(model: &mut BenchModel, cfg: &SpeedConfig) -> Result<SpeedResult> {
    let prompt = resolve_prompt(model, cfg);
    if prompt.is_empty() {
        bail!("speed bench needs a non-empty prompt (set prompt_ids or prompt_len > 0)");
    }
    if cfg.warmup {
        // One token, ignore the result — just pay compile costs up front.
        let _ = model.runner.generate(&prompt, 1, &mut |_| false);
    }

    // Dedicated prefill timing (best effort — packed runners may not support it).
    let (prefill_s, prefill_toks_s) = {
        let t = Instant::now();
        match model.runner.prefill_logits(&prompt) {
            Ok(_) => {
                let s = t.elapsed().as_secs_f64();
                let rate = if s > 0.0 {
                    prompt.len() as f64 / s
                } else {
                    0.0
                };
                (s, rate)
            }
            Err(_) => (0.0, 0.0),
        }
    };

    // End-to-end generation with time-to-first-token capture.
    let n = cfg.decode_tokens.max(1);
    let start = Instant::now();
    let mut ttft: Option<f64> = None;
    let mut produced = 0usize;
    model.runner.generate(&prompt, n, &mut |_tok| {
        if ttft.is_none() {
            ttft = Some(start.elapsed().as_secs_f64());
        }
        produced += 1;
        true
    })?;
    let total_s = start.elapsed().as_secs_f64();
    let ttft_s = ttft.unwrap_or(total_s);
    let decode_s = (total_s - ttft_s).max(0.0);
    let decode_toks_s = if produced > 1 && decode_s > 0.0 {
        (produced - 1) as f64 / decode_s
    } else {
        0.0
    };

    Ok(SpeedResult {
        prompt_tokens: prompt.len(),
        decode_tokens: produced,
        prefill_s,
        prefill_toks_s,
        ttft_ms: ttft_s * 1000.0,
        decode_s,
        decode_toks_s,
        total_s,
        peak_rss_mb: peak_rss_mb(),
    })
}
