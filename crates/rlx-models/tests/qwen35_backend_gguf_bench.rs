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

// Env-gated: real 0.8B GGUF decode perf on CPU / Metal / MLX (batch=1).
//
//   QWEN35_GGUF_PATH=/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf \
//     cargo test -p rlx-models --test qwen35_backend_gguf_bench --features "metal,mlx" --release -- --nocapture
//
// Heterogeneous batch=2 benchmarks: see `qwen35_batch_gguf_bench`.

#[path = "qwen35_gguf_support.rs"]
mod support;

use rlx_models::Qwen35RunnerBuilder;
use rlx_models::qwen3::SampleOpts;
use rlx_runtime::Device;
use std::path::Path;
use std::time::Instant;
use support::{BENCH_DECODE_TOKENS, WARMUP_DECODE_TOKENS, gguf_path};

fn bench_generate_tok_s(path: &Path, device: Device, new_tokens: usize) -> f64 {
    let prompt: Vec<u32> = (1..=8).collect();
    let max_seq = prompt.len() + new_tokens + WARMUP_DECODE_TOKENS;
    let mut runner = Qwen35RunnerBuilder::default()
        .weights(path)
        .device(device)
        .batch(1)
        .packed_weights(true)
        .max_seq(max_seq.max(16))
        .last_logits_only(true)
        .build()
        .expect("build runner");
    let _ = runner
        .generate_with_opts(&prompt, WARMUP_DECODE_TOKENS, SampleOpts::greedy(), |_| {
            true
        })
        .expect("warmup generate");
    let t0 = Instant::now();
    let _ = runner
        .generate_with_opts(&prompt, new_tokens, SampleOpts::greedy(), |_| true)
        .expect("bench generate");
    let secs = t0.elapsed().as_secs_f64();
    if secs <= 0.0 {
        return 0.0;
    }
    new_tokens as f64 / secs
}

fn bench_prefill_ms(path: &Path, device: Device) -> f64 {
    let prompt: Vec<u32> = vec![1, 2, 3];
    let max_seq = 8usize;
    let mut runner = Qwen35RunnerBuilder::default()
        .weights(path)
        .device(device)
        .packed_weights(true)
        .max_seq(max_seq)
        .last_logits_only(true)
        .build()
        .expect("build runner");
    let _ = runner.predict_logits(&prompt).expect("warmup predict");
    let t0 = Instant::now();
    let _ = runner.predict_logits(&prompt).expect("steady predict");
    t0.elapsed().as_secs_f64() * 1000.0
}

fn report(label: &str, prefill_ms: f64, tok_s: f64) {
    eprintln!(
        "qwen35 perf {label}: prefill_steady={prefill_ms:.1}ms generate={tok_s:.2} tok/s ({BENCH_DECODE_TOKENS} tok after {WARMUP_DECODE_TOKENS} warmup)"
    );
}

#[test]
fn qwen35_real_gguf_bench_cpu() {
    let path = match gguf_path() {
        Some(p) => p,
        None => {
            eprintln!("skip qwen35_backend_gguf_bench: set QWEN35_GGUF_PATH");
            return;
        }
    };
    let prefill = bench_prefill_ms(&path, Device::Cpu);
    let tok_s = bench_generate_tok_s(&path, Device::Cpu, BENCH_DECODE_TOKENS);
    report("cpu", prefill, tok_s);
    assert!(tok_s.is_finite() && tok_s > 0.0);
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn qwen35_real_gguf_bench_metal() {
    let path = match gguf_path() {
        Some(p) => p,
        None => return,
    };
    let prefill = bench_prefill_ms(&path, Device::Metal);
    let tok_s = bench_generate_tok_s(&path, Device::Metal, BENCH_DECODE_TOKENS);
    report("metal", prefill, tok_s);
    assert!(tok_s.is_finite() && tok_s > 0.0);
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn qwen35_real_gguf_bench_mlx() {
    let path = match gguf_path() {
        Some(p) => p,
        None => return,
    };
    let prefill = bench_prefill_ms(&path, Device::Mlx);
    let tok_s = bench_generate_tok_s(&path, Device::Mlx, BENCH_DECODE_TOKENS);
    report("mlx", prefill, tok_s);
    assert!(tok_s.is_finite() && tok_s > 0.0);
}
