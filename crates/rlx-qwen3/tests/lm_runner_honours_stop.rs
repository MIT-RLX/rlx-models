// RLX models — Qwen3.
// Copyright (C) 2026 Eugene Hauptmann.
// SPDX-License-Identifier: GPL-3.0-only

//! `LmRunner::generate` must honour the callback's stop signal.
//!
//! The trait callback is `FnMut(u32) -> bool`, where `false` means stop. This
//! impl used to discard it (`let _ = on_token(tok)`), so every reason a caller
//! has to end a turn early was thrown away — chat EOS, user stop strings, a
//! stray `</think>`, a degenerate-repetition guard — and generation always ran
//! the full `n_new`.
//!
//! What that looked like downstream: NeuroSkill asked Qwen3-0.6B for the
//! capital of France and got 303 tokens of "The capital of France's capital of
//! France's …". Its streamer's repetition guard had returned `false` around the
//! tenth token; nothing was listening. `finish_reason` was also permanently
//! "length", and each turn paid for tokens the caller had already rejected.
//!
//! Needs a real Qwen3 GGUF (token ids are fed directly; no tokenizer).
//! Run: `QWEN3_GGUF=/path/Qwen3-0.6B-Q4_K_M.gguf cargo test -p rlx-qwen3 \
//!   --test lm_runner_honours_stop -- --ignored --nocapture`

use rlx_runtime::Device;
use rlx_runtime::lm::LmRunner;

fn gguf() -> Option<String> {
    let p = std::env::var("QWEN3_GGUF").unwrap_or_else(|_| {
        "/Users/Shared/rlx-models/weights/lm/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf".to_string()
    });
    std::path::Path::new(&p).is_file().then_some(p)
}

#[test]
#[ignore = "needs a Qwen3 GGUF (QWEN3_GGUF)"]
fn stopping_after_three_tokens_stops_generation() {
    let Some(g) = gguf() else {
        eprintln!("skip: no Qwen3 GGUF");
        return;
    };
    let mut r = rlx_qwen3::Qwen3Runner::builder()
        .weights(&g)
        .device(Device::Cpu)
        .packed_weights(true)
        .max_seq(512)
        .build()
        .expect("build packed runner");

    let prompt: Vec<u32> = (0..24).map(|i| (i % 2000) as u32 + 10).collect();
    // Ask for 64 but refuse everything after the third token.
    let mut seen = 0usize;
    let out = LmRunner::generate(&mut r, &prompt, 64, &mut |_tok| {
        seen += 1;
        seen < 3
    })
    .expect("generate");

    eprintln!(
        "[stop] callback saw {seen} token(s), runner returned {}",
        out.len()
    );
    assert!(
        seen <= 4,
        "callback kept being called after it said stop: saw {seen} of a 64-token request"
    );
    assert!(
        out.len() <= 4,
        "generation ran past the stop signal: produced {} tokens for a request that stopped at 3",
        out.len()
    );
}
