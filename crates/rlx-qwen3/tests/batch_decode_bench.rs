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

#![cfg(feature = "export")]

//! What decoding several sequences together costs: an exported Qwen3
//! package's `generate_batch` at batch 1, 2, 4 and 8 on `RLX_TEST_DEVICE`,
//! per step and per token. Opt-in: `RLX_BENCH_BATCH=1` and
//! `RLX_TEST_QWEN3_DIR`.
//!
//! Each batch decodes `RLX_BENCH_STEPS` (48) tokens after a one-token
//! prompt (no stop), once to compile, then best of three.
//! `RLX_BENCH_BATCHES=1,4` picks the batch sizes; `RLX_BENCH_BUILD=1` runs
//! the package built for the device; `RLX_BENCH_PREFILL_BATCHES=0` leaves
//! out the batched prompt functions.

use rlx_qwen3::export::{ExportOptions, export_rlxp};
use serde_json::json;

#[test]
fn batched_decode_cost() {
    let (Some(dir), Some(_)) = (
        std::env::var_os("RLX_TEST_QWEN3_DIR"),
        std::env::var_os("RLX_BENCH_BATCH"),
    ) else {
        eprintln!("skip: set RLX_BENCH_BATCH=1 and RLX_TEST_QWEN3_DIR");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let device = std::env::var("RLX_TEST_DEVICE").unwrap_or_else(|_| "metal".into());
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("qwen3.rlxp");
    // `RLX_BENCH_PREFILL_BATCHES=0`: no batched prompt functions, so a
    // batch's prompts prefill one after another.
    let opts = ExportOptions {
        uppers: vec![64, 512],
        prefill_batches: if std::env::var("RLX_BENCH_PREFILL_BATCHES").as_deref() == Ok("0") {
            Vec::new()
        } else {
            ExportOptions::default().prefill_batches
        },
        ..ExportOptions::default()
    };
    export_rlxp(&dir, &out, &opts).unwrap();
    // `RLX_BENCH_BUILD=1`: the package built for the device (`rlx-pkg build`;
    // on Metal, f16 weights).
    let out = if std::env::var_os("RLX_BENCH_BUILD").is_some() {
        let built = tmp.path().join("qwen3.built.rlxp");
        let recipe = rlx_runtime::model::rlxp::TargetRecipe::for_target(&device).unwrap();
        rlx_runtime::model::rlxp::build_for_targets(&out, &built, &[(device.as_str(), recipe)])
            .unwrap();
        built
    } else {
        out
    };
    rlx_text::install_model_tokenizer();
    // `RLX_BENCH_PROMPT=<n>`: prompts of about n tokens each (one token by
    // default), and `RLX_BENCH_BATCH_PREFILL=0` runs them a token a step.
    let prompt_words: usize = std::env::var("RLX_BENCH_PROMPT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let batch_prefill = std::env::var("RLX_BENCH_BATCH_PREFILL").as_deref() != Ok("0");
    let mut m = rlx_runtime::model::open_source(
        out.to_str().unwrap(),
        json!({"device": device, "batch_prefill": batch_prefill}),
    )
    .unwrap();
    // No stop ids: every sequence runs all `steps`.
    let steps: usize = std::env::var("RLX_BENCH_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let batches: Vec<usize> = std::env::var("RLX_BENCH_BATCHES")
        .ok()
        .map(|v| v.split(',').filter_map(|b| b.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8]);
    let opts = json!({"max_tokens": steps, "stop_ids": []});
    let words = ["The", "A", "In", "One", "When", "It", "We", "This"];
    eprintln!(
        "device {device}, prompts of ~{prompt_words} tokens, batch prefill {batch_prefill}: \
         batch | ms/step | ms/token | tokens/s"
    );
    let mut single = 0.0;
    for b in batches {
        let prompts: Vec<String> = words[..b]
            .iter()
            .map(|w| {
                let mut p = w.to_string();
                for i in 1..prompt_words {
                    p.push_str([" river", " stone", " light", " north"][i % 4]);
                }
                p
            })
            .collect();
        let prompts: Vec<&str> = prompts.iter().map(String::as_str).collect();
        let mut time = || {
            let t = std::time::Instant::now();
            let outs = if b == 1 {
                vec![m.generate_text(prompts[0], &opts).unwrap()]
            } else {
                m.generate_batch(&prompts, &opts).unwrap()
            };
            assert_eq!(outs.len(), b);
            t.elapsed().as_secs_f64() * 1e3
        };
        time();
        let ms = (0..3).map(|_| time()).fold(f64::MAX, f64::min);
        let step = ms / steps as f64;
        if b == 1 {
            single = step;
        }
        eprintln!(
            "  {b:>5} | {step:>7.2} | {:>8.2} | {:>8.1}  ({:.2}× a single step)",
            step / b as f64,
            (b * steps) as f64 / (ms / 1e3),
            step / single
        );
    }
}
