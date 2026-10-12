// RLX models — Qwen3.
// Copyright (C) 2026 Eugene Hauptmann.
// SPDX-License-Identifier: GPL-3.0-only

//! The KV context store must survive a second `generate()`.
//!
//! `step_cached` required `tokens.len() > past_seq`, on the reasoning that the
//! cache is a prefix of the token history. The retrieval splice in
//! `apply_retention` breaks that: it rebuilds the cache from kept + retrieved
//! rows and sets `past_len = entries.len()`, so the row count tracks the
//! RETRIEVED context rather than the history. A second turn with a short prompt
//! against a long retained context then tripped the check.
//!
//! The shape that caught it: NeuroSkill's daemon re-prefills the whole chat on
//! every turn, so turn 2 of a session arrives as a fresh `generate()` with a
//! modest prompt while the store still holds the previous turn's blocks.
//! Measured over HTTP on Qwen3-0.6B-Q4_K_M: with the store on, 0 of 8 turns
//! survived (`cache invariant violated: tokens.len() 38 <= past_seq 260`); with
//! it off, 8 of 8. The feature was effectively unusable for chat.
//!
//! Needs a real Qwen3 GGUF; token ids are fed directly, so no tokenizer.
//! Run: `QWEN3_GGUF=/path/Qwen3-0.6B-Q4_K_M.gguf cargo test -p rlx-qwen3 \
//!   --features mmap-kv --test kv_store_multi_turn -- --ignored --nocapture`

#![cfg(feature = "mmap-kv")]

use rlx_runtime::Device;

fn gguf() -> Option<String> {
    let p = std::env::var("QWEN3_GGUF").unwrap_or_else(|_| {
        "/Users/Shared/rlx-models/weights/lm/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf".to_string()
    });
    std::path::Path::new(&p).is_file().then_some(p)
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n).map(|i| (i % 2000) as u32 + 10).collect()
}

#[test]
#[ignore = "needs a Qwen3 GGUF (QWEN3_GGUF)"]
fn a_store_backed_runner_answers_more_than_one_turn() {
    let Some(g) = gguf() else {
        eprintln!("skip: no Qwen3 GGUF");
        return;
    };
    let mut r = rlx_qwen3::Qwen3Runner::builder()
        .weights(&g)
        .device(Device::Cpu)
        .packed_weights(true)
        // Big enough that the bucket ladder covers every prompt below — this
        // test is about the store, not the context cap.
        .max_seq(2048)
        .build()
        .expect("build packed runner");
    r.enable_kv_store(
        rlx_qwen3::KvStoreConfig::new()
            .capacity_tokens(100_000)
            .topk(16),
    )
    .expect("enable KV store");

    // Turn 1: long enough that blocks get offloaded and the store has something
    // to retrieve on the next turn.
    let t1 = r
        .generate(&prompt(400), 4, |_| {})
        .expect("turn 1 must generate");
    assert_eq!(t1.len(), 4, "turn 1 token count");

    // Turn 2: a SHORT prompt, which is what the daemon sends on a fresh chat and
    // what used to trip the invariant.
    let t2 = r
        .generate(&prompt(38), 4, |_| {})
        .expect("turn 2 with a short prompt must generate, not bail on the cache invariant");
    assert_eq!(t2.len(), 4, "turn 2 token count");

    // Turn 3: short again, to show it is not a one-off recovery.
    let t3 = r
        .generate(&prompt(24), 4, |_| {})
        .expect("turn 3 must generate");
    assert_eq!(t3.len(), 4, "turn 3 token count");
    eprintln!("[kv-multi-turn] ok: {t1:?} / {t2:?} / {t3:?}");
}
