// RLX models — Qwen3.
// Copyright (C) 2026 Eugene Hauptmann.
// SPDX-License-Identifier: GPL-3.0-only

//! The batched packed decode paths must mask their bucket padding.
//!
//! A bucketed decode graph is compiled for `upper` cached rows and fed
//! `past_seq ≤ upper` real ones, the rest zero. The packed batched builders
//! took no `mask` input, so the one-query causal mask hid nothing and the zero
//! rows took part in attention — while the callers computed and fed a mask the
//! graph silently ignored. Reference: `decode_get_logits`, which compiles for
//! the exact length (no padding), from the same cache.
//!
//! Needs a real Qwen3 GGUF: `QWEN3_GGUF=/path/Qwen3-0.6B-Q4_K_M.gguf cargo test
//! -p rlx-qwen3 --release --test packed_batched_decode_mask -- --ignored`

use rlx_runtime::Device;

fn gguf() -> Option<String> {
    let p = std::env::var("QWEN3_GGUF").unwrap_or_else(|_| {
        "/Users/Shared/rlx-models/weights/lm/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf".to_string()
    });
    std::path::Path::new(&p).is_file().then_some(p)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
#[ignore = "needs a Qwen3 GGUF (QWEN3_GGUF)"]
fn batched_packed_decode_ignores_bucket_padding() {
    let Some(g) = gguf() else {
        eprintln!("skip: no Qwen3 GGUF");
        return;
    };
    let mut qg = rlx_qwen3::Qwen3Runner::builder()
        .weights(&g)
        .device(Device::Cpu)
        .packed_weights(true)
        .max_seq(64)
        .build()
        .expect("packed runner")
        .into_generator()
        .expect("generator");
    // 20 cached rows → a 32-row bucket: 12 zero rows to mask.
    let prompt: Vec<u32> = (0..20).map(|i| 100 + 37 * i).collect();
    let past = prompt.len();
    qg.prefill_get_last_logits(&prompt).unwrap();
    let (kv, _) = qg.export_cache().unwrap();
    let tok = 9707u32;
    let want = qg.decode_get_logits(tok).unwrap();

    // Same weights, same cache: only the padded rows can tell the two apart.
    // Masked, the batched graphs agree with the unpadded one to f32 noise
    // (|Δ| ≈ 1e-5 on CPU); attending 12 zero rows moves logits by ~0.1.
    let check = |what: &str, got: &[f32]| {
        let d = got
            .iter()
            .zip(&want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!(
            "[{what}] max |Δ logit| {d:.2e}, cosine {:.7}, argmax {} vs {}",
            cosine(got, &want),
            argmax(got),
            argmax(&want)
        );
        assert!(d < 1e-2, "{what}: max |Δ logit| {d} vs the unpadded decode");
        assert_eq!(argmax(got), argmax(&want), "{what}: next token differs");
    };

    let out = qg
        .decode_batched_uniform(&[(tok, &kv), (tok, &kv)], past, past)
        .unwrap();
    for (i, (logits, _)) in out.iter().enumerate() {
        check(&format!("uniform[{i}]"), logits);
    }

    qg.decode_batched_resident_init(&[&kv, &kv]);
    let out = qg
        .decode_batched_resident_step(&[tok, tok], past, past)
        .unwrap();
    for (i, logits) in out.iter().enumerate() {
        check(&format!("resident[{i}]"), logits);
    }
}
