// RLX models — Qwen3.5.
// Copyright (C) 2026 Eugene Hauptmann.
// SPDX-License-Identifier: GPL-3.0-only

//! The predict graph must be compiled once, not twice.
//!
//! `compile_hir_for_config` used to specialize into `predict_hir_cache` and
//! throw the result away, then — on every non-CPU device — compile the whole
//! graph again to have something to return. `predict_hir_cache` is read only on
//! the CPU path, so on a GPU the first compile was unreachable work.
//!
//! Both are full compiles, and the arena is the cost: measured on
//! Qwen3.5-0.8B-Q4_K_M on Metal, rlx-metal ran its pipeline twice over the same
//! 1648 nodes — byte-identical plans (66298834948 B planned, 9777934340 B
//! accepted), identical scratch offsets — allocating 10.80 GB of Metal buffers
//! each time. That one dead arena was 42% of the 25.9 GB of buffers the process
//! held for a 532 MB model.
//!
//! Run with `RLX_VERBOSE=1` and count `compiling graph:` in the output; this
//! test asserts the cheap observable (one compile's worth of wall clock is not
//! something to assert on, but the graph count is printed for the operator).
//!
//! `QWEN35_GGUF=/path/Qwen3.5-0.8B-Q4_K_M.gguf cargo test -p rlx-qwen35 \
//!   --features metal --test predict_compiles_once -- --ignored --nocapture`

#![cfg(target_os = "macos")]

use rlx_runtime::Device;
use rlx_runtime::lm::LmRunner;

#[test]
#[ignore = "needs a Qwen3.5 GGUF (QWEN35_GGUF) + Metal"]
fn predict_logits_compiles_the_graph_once() {
    let gguf = std::env::var("QWEN35_GGUF").unwrap_or_else(|_| {
        "/Volumes/FOUR/weights/Qwen3.5-0.8B-gguf/Qwen3.5-0.8B-Q4_K_M.gguf".into()
    });
    if !std::path::Path::new(&gguf).is_file() {
        eprintln!("skip: no Qwen3.5 GGUF at {gguf}");
        return;
    }
    // Keep the graph small: the point is how MANY compiles happen, not how big.
    let mut r = rlx_qwen35::Qwen35Runner::builder()
        .weights(&gguf)
        .device(Device::Metal)
        .packed_weights(true)
        .max_seq(512)
        .skip_warm(true)
        .build()
        .expect("build Qwen3.5 runner on Metal");

    let prompt: Vec<u32> = (0..16).map(|i| (i % 2000) as u32 + 10).collect();
    let logits = LmRunner::predict_logits(&mut r, &prompt).expect("predict_logits");
    assert!(
        logits.len() > 1024,
        "expected a full logits row, got {} values",
        logits.len()
    );
    eprintln!(
        "[predict-once] ok — grep the RLX_VERBOSE output above for 'compiling graph:'; \
         it must appear ONCE for the predict graph, not twice"
    );
}
