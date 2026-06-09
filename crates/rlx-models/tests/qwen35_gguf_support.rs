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

//! Shared helpers for env-gated Qwen3.5 real-GGUF tests/benches.

#![allow(dead_code)]

mod compile_support;

use rlx_models::qwen3::SampleOpts;
use rlx_models::{Qwen35Runner, Qwen35RunnerBuilder};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const DEFAULT_GGUF: &str = "/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf";

pub const WARMUP_DECODE_TOKENS: usize = 4;
pub const BENCH_DECODE_TOKENS: usize = 16;
pub const BENCH_DECODE_TOKENS_SHORT_ROW: usize = 8;

/// Resolve real weights: `QWEN35_GGUF_PATH` or [`DEFAULT_GGUF`].
pub fn gguf_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("QWEN35_GGUF_PATH") {
        if !p.is_empty() {
            let path = PathBuf::from(p);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    let path = PathBuf::from(DEFAULT_GGUF);
    path.is_file().then_some(path)
}

/// Two prompts with different lengths (8 vs 7 tokens) for batch=2 padding tests.
pub fn heterogeneous_prompts_batch2() -> [Vec<u32>; 2] {
    [(1u32..=8).collect(), (4u32..=10).collect()]
}

pub fn heterogeneous_prompts_batch2_vecs() -> Vec<Vec<u32>> {
    heterogeneous_prompts_batch2().into_iter().collect()
}

fn batch1_prompt() -> Vec<u32> {
    (1u32..=8).collect()
}

pub struct HetBatchBenchReport {
    pub device: Device,
    pub prefill_batch1_ms: f64,
    pub prefill_batch2_ms: f64,
    pub decode_batch1_per_stream_tok_s: f64,
    pub decode_batch2_uniform_aggregate_tok_s: f64,
    pub decode_batch2_per_row_limits_aggregate_tok_s: f64,
}

impl HetBatchBenchReport {
    pub fn batch2_uniform_per_stream_equiv(&self) -> f64 {
        self.decode_batch2_uniform_aggregate_tok_s / 2.0
    }

    pub fn batch2_per_row_limits_per_stream_equiv(&self) -> f64 {
        self.decode_batch2_per_row_limits_aggregate_tok_s / 2.0
    }

    pub fn batch2_uniform_efficiency(&self) -> f64 {
        if self.decode_batch1_per_stream_tok_s <= 0.0 {
            return 0.0;
        }
        self.decode_batch2_uniform_aggregate_tok_s / (2.0 * self.decode_batch1_per_stream_tok_s)
    }

    pub fn log(&self) {
        eprintln!(
            "qwen35 het-batch bench {:?}: \
             prefill b1={:.1}ms b2={:.1}ms | \
             decode b1={:.2} tok/s/stream | \
             decode b2 uniform agg={:.2} tok/s ({:.2}/stream, eff={:.2}x) | \
             decode b2 per-row-limits agg={:.2} tok/s ({:.2}/stream)",
            self.device,
            self.prefill_batch1_ms,
            self.prefill_batch2_ms,
            self.decode_batch1_per_stream_tok_s,
            self.decode_batch2_uniform_aggregate_tok_s,
            self.batch2_uniform_per_stream_equiv(),
            self.batch2_uniform_efficiency(),
            self.decode_batch2_per_row_limits_aggregate_tok_s,
            self.batch2_per_row_limits_per_stream_equiv(),
        );
    }
}

fn build_runner(path: &Path, device: Device, batch: usize, max_seq: usize) -> Qwen35Runner {
    Qwen35RunnerBuilder::default()
        .weights(path)
        .device(device)
        .batch(batch)
        .packed_weights(true)
        .max_seq(max_seq.max(16))
        .last_logits_only(true)
        .build()
        .expect("build qwen35 runner")
}

fn bench_prefill_batch1_ms(path: &Path, device: Device) -> f64 {
    let prompt = batch1_prompt();
    let mut runner = build_runner(path, device, 1, prompt.len() + 4);
    let _ = runner.predict_logits(&prompt).expect("prefill warmup");
    let t0 = Instant::now();
    let _ = runner.predict_logits(&prompt).expect("prefill steady");
    t0.elapsed().as_secs_f64() * 1000.0
}

fn bench_prefill_batch2_ms(path: &Path, device: Device) -> f64 {
    let prompts = heterogeneous_prompts_batch2_vecs();
    let max_seq = prompts.iter().map(|p| p.len()).max().unwrap() + 4;
    let mut runner = build_runner(path, device, 2, max_seq);
    let _ = runner
        .predict_logits_batch(&prompts)
        .expect("batch2 prefill warmup");
    let t0 = Instant::now();
    let _ = runner
        .predict_logits_batch(&prompts)
        .expect("batch2 prefill steady");
    t0.elapsed().as_secs_f64() * 1000.0
}

fn bench_decode_per_stream_tok_s(path: &Path, device: Device, new_tokens: usize) -> f64 {
    let prompt = batch1_prompt();
    let max_seq = prompt.len() + new_tokens + WARMUP_DECODE_TOKENS;
    let mut runner = build_runner(path, device, 1, max_seq);
    let _ = runner
        .generate_with_opts(&prompt, WARMUP_DECODE_TOKENS, SampleOpts::greedy(), |_| {
            true
        })
        .expect("decode warmup");
    let t0 = Instant::now();
    let _ = runner
        .generate_with_opts(&prompt, new_tokens, SampleOpts::greedy(), |_| true)
        .expect("decode bench");
    let secs = t0.elapsed().as_secs_f64();
    if secs <= 0.0 {
        return 0.0;
    }
    new_tokens as f64 / secs
}

fn bench_decode_batch2_uniform_aggregate_tok_s(
    path: &Path,
    device: Device,
    new_tokens: usize,
) -> f64 {
    let prompts = heterogeneous_prompts_batch2_vecs();
    let max_seq =
        prompts.iter().map(|p| p.len()).max().unwrap() + new_tokens + WARMUP_DECODE_TOKENS;
    let mut runner = build_runner(path, device, 2, max_seq);
    let _ = runner
        .generate_batch_with_opts(
            &prompts,
            WARMUP_DECODE_TOKENS,
            None,
            SampleOpts::greedy(),
            |_, _| true,
        )
        .expect("batch2 decode warmup");
    let t0 = Instant::now();
    let _ = runner
        .generate_batch_with_opts(&prompts, new_tokens, None, SampleOpts::greedy(), |_, _| {
            true
        })
        .expect("batch2 decode bench");
    let secs = t0.elapsed().as_secs_f64();
    if secs <= 0.0 {
        return 0.0;
    }
    (new_tokens * 2) as f64 / secs
}

fn bench_decode_batch2_per_row_limits_aggregate_tok_s(path: &Path, device: Device) -> f64 {
    let prompts = heterogeneous_prompts_batch2_vecs();
    let limits = [BENCH_DECODE_TOKENS, BENCH_DECODE_TOKENS_SHORT_ROW];
    let total_tokens: usize = limits.iter().sum();
    let max_seq = prompts.iter().map(|p| p.len()).max().unwrap()
        + *limits.iter().max().unwrap()
        + WARMUP_DECODE_TOKENS;
    let mut runner = build_runner(path, device, 2, max_seq);
    let _ = runner
        .generate_batch_with_opts(
            &prompts,
            WARMUP_DECODE_TOKENS,
            None,
            SampleOpts::greedy(),
            |_, _| true,
        )
        .expect("batch2 limits warmup");
    let t0 = Instant::now();
    let out = runner
        .generate_batch_with_opts(
            &prompts,
            BENCH_DECODE_TOKENS,
            Some(&limits),
            SampleOpts::greedy(),
            |_, _| true,
        )
        .expect("batch2 limits bench");
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(out[0].len(), limits[0]);
    assert_eq!(out[1].len(), limits[1]);
    if secs <= 0.0 {
        return 0.0;
    }
    total_tokens as f64 / secs
}

/// Full heterogeneous batch=2 benchmark on real GGUF weights.
pub fn bench_heterogeneous_batch(path: &Path, device: Device) -> HetBatchBenchReport {
    HetBatchBenchReport {
        device,
        prefill_batch1_ms: bench_prefill_batch1_ms(path, device),
        prefill_batch2_ms: bench_prefill_batch2_ms(path, device),
        decode_batch1_per_stream_tok_s: bench_decode_per_stream_tok_s(
            path,
            device,
            BENCH_DECODE_TOKENS,
        ),
        decode_batch2_uniform_aggregate_tok_s: bench_decode_batch2_uniform_aggregate_tok_s(
            path,
            device,
            BENCH_DECODE_TOKENS,
        ),
        decode_batch2_per_row_limits_aggregate_tok_s:
            bench_decode_batch2_per_row_limits_aggregate_tok_s(path, device),
    }
}

pub fn assert_finite_positive(report: &HetBatchBenchReport) {
    assert!(report.prefill_batch1_ms.is_finite() && report.prefill_batch1_ms > 0.0);
    assert!(report.prefill_batch2_ms.is_finite() && report.prefill_batch2_ms > 0.0);
    assert!(
        report.decode_batch1_per_stream_tok_s.is_finite()
            && report.decode_batch1_per_stream_tok_s > 0.0
    );
    assert!(
        report.decode_batch2_uniform_aggregate_tok_s.is_finite()
            && report.decode_batch2_uniform_aggregate_tok_s > 0.0
    );
    assert!(
        report
            .decode_batch2_per_row_limits_aggregate_tok_s
            .is_finite()
            && report.decode_batch2_per_row_limits_aggregate_tok_s > 0.0
    );
}
