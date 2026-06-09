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

// Env-gated: heterogeneous batch=2 correctness on real GGUF weights.
//
//   QWEN35_GGUF_PATH=/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf \
//     cargo test -p rlx-models --test qwen35_batch_gguf_quick_check --release -- --nocapture

#[path = "qwen35_gguf_support.rs"]
mod support;

use rlx_models::Qwen35RunnerBuilder;
use rlx_models::qwen3::SampleOpts;
use rlx_runtime::Device;
use std::path::Path;
use support::{
    BENCH_DECODE_TOKENS, BENCH_DECODE_TOKENS_SHORT_ROW, gguf_path,
    heterogeneous_prompts_batch2_vecs,
};

#[test]
fn qwen35_batch2_heterogeneous_prefill_and_generate() {
    let path = match gguf_path() {
        Some(p) => p,
        None => {
            eprintln!("skip qwen35_batch_gguf_quick_check: set QWEN35_GGUF_PATH");
            return;
        }
    };
    run_quick_check(&path);
}

fn run_quick_check(path: &Path) {
    let prompts = heterogeneous_prompts_batch2_vecs();
    let max_seq = prompts.iter().map(|p| p.len()).max().unwrap() + BENCH_DECODE_TOKENS + 4;
    let mut runner = Qwen35RunnerBuilder::default()
        .weights(path)
        .device(Device::Cpu)
        .batch(2)
        .packed_weights(true)
        .max_seq(max_seq)
        .last_logits_only(true)
        .build()
        .expect("batch=2 runner");

    let prefill = runner
        .predict_logits_batch(&prompts)
        .expect("heterogeneous batch prefill");
    assert_eq!(prefill.len(), 2);
    for row in &prefill {
        assert!(!row.logits.is_empty());
        assert!(row.logits.iter().all(|v| v.is_finite()));
    }

    let generated = runner
        .generate_batch_with_opts(&prompts, 3, None, SampleOpts::greedy(), |_, _| true)
        .expect("heterogeneous batch generate (uniform limits)");
    assert_eq!(generated.len(), 2);
    assert_eq!(generated[0].len(), 3);
    assert_eq!(generated[1].len(), 3);

    let limits = [BENCH_DECODE_TOKENS, BENCH_DECODE_TOKENS_SHORT_ROW];
    runner.reset_decode_cache();
    let per_row = runner
        .generate_batch_with_opts(
            &prompts,
            BENCH_DECODE_TOKENS,
            Some(&limits),
            SampleOpts::greedy(),
            |_, _| true,
        )
        .expect("heterogeneous batch generate (per-row limits)");
    assert_eq!(per_row[0].len(), limits[0]);
    assert_eq!(per_row[1].len(), limits[1]);

    eprintln!(
        "qwen35 batch2 heterogeneous quick check ok: uniform={:?} per_row_lens=[{}, {}]",
        generated, limits[0], limits[1]
    );
}
