// RLX models — Qwen3.
// Copyright (C) 2026 Eugene Hauptmann.
// SPDX-License-Identifier: GPL-3.0-only

//! A packed runner asked to decode past its bucket ladder must say what is
//! actually wrong.
//!
//! `Qwen3Generator::step_cached` picks a compiled bucket for `past_seq`, and
//! when no bucket covers it, falls back to the one-shot decode path. That
//! fallback rebuilds the graph from `weights_cache` — which the native-packed
//! constructor leaves EMPTY, because the projections lower to
//! `Op::DequantMatMul` over the GGUF blob instead. So the fallback could never
//! work for a packed runner; the only thing it produced was
//! `weight not found: model.embed_tokens.weight` from the first
//! `WeightMap::take`.
//!
//! That message names a tensor and hides the cause. Downstream (NeuroSkill's
//! daemon) it looked exactly like a GGUF-vs-safetensors naming bug and was
//! chased as one, when the real cause was the ladder: `max_seq` defaults to
//! 128, `with_decode_cache(max_seq + 64)` makes it cover a past of 256, and
//! every chat with a system prompt is longer than that — so *every* request
//! failed, while a toy 60-token prompt worked.
//!
//! Needs a real Qwen3 GGUF (token ids are fed directly; no tokenizer).
//! Run: `QWEN3_GGUF=/path/Qwen3-0.6B-Q4_K_M.gguf cargo test -p rlx-qwen3 \
//!   --test packed_decode_beyond_buckets -- --ignored --nocapture`

use rlx_runtime::Device;

fn gguf() -> Option<String> {
    let p = std::env::var("QWEN3_GGUF").unwrap_or_else(|_| {
        "/Users/Shared/rlx-models/weights/lm/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf".to_string()
    });
    std::path::Path::new(&p).is_file().then_some(p)
}

/// Token ids only — the point is the bucket ladder, not the text.
fn prompt(n: usize) -> Vec<u32> {
    (0..n).map(|i| (i % 2000) as u32 + 10).collect()
}

fn runner(gguf: &str, max_seq: usize) -> rlx_qwen3::Qwen3Runner {
    rlx_qwen3::Qwen3Runner::builder()
        .weights(gguf)
        .device(Device::Cpu) // the defect is device-independent; CPU keeps it cheap
        .packed_weights(true)
        .max_seq(max_seq)
        .build()
        .expect("build packed Qwen3 runner")
}

#[test]
#[ignore = "needs a Qwen3 GGUF (QWEN3_GGUF)"]
fn past_the_ladder_names_the_context_cap_not_a_missing_tensor() {
    let Some(g) = gguf() else {
        eprintln!("skip: no Qwen3 GGUF");
        return;
    };
    // max_seq 64 → `with_decode_cache(128)` → ladder covers a past of 1..=128.
    // A 200-token prompt therefore lands outside it on the first decode step.
    let mut r = runner(&g, 64);
    let err = r
        .generate(&prompt(200), 2, |_| {})
        .expect_err("decoding past the ladder must fail, not silently misbehave")
        .to_string();
    eprintln!("[err] {err}");

    assert!(
        !err.contains("weight not found"),
        "packed decode past the ladder still reports a missing tensor, which sends \
         callers hunting through GGUF tensor names: {err}"
    );
    assert!(
        err.contains("outside the decode buckets"),
        "error should say the past length is outside the decode buckets: {err}"
    );
    assert!(
        err.contains("max_seq"),
        "error should point at the knob that fixes it: {err}"
    );
}

/// The same prompt must still generate once the ladder actually covers it —
/// the guard must refuse only the impossible case, not the working one.
#[test]
#[ignore = "needs a Qwen3 GGUF (QWEN3_GGUF)"]
fn inside_the_ladder_still_generates() {
    let Some(g) = gguf() else {
        eprintln!("skip: no Qwen3 GGUF");
        return;
    };
    let mut r = runner(&g, 512);
    let mut got = 0usize;
    let out = r
        .generate(&prompt(200), 2, |_| got += 1)
        .expect("a 200-token prompt inside a 512-token context must generate");
    assert_eq!(out.len(), 2, "expected 2 tokens, got {out:?}");
    assert_eq!(got, 2, "callback should fire per token");
}
