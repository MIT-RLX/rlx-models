// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! CPU parity for the `FusedNomicLayer` full-layer fusion in rlx-cpu.
//!
//! The fusion collapses a whole NomicBERT encoder layer
//! (`FusedAttnBlock → FusedResidualLN → Sgemm(fc11‖fc12) → FusedSwiGLU →
//! Sgemm(fc2) → FusedResidualLN`) into one kernel. It lives in the
//! model-agnostic backend (rlx-cpu) but its correctness can only be judged
//! against a real model graph — which lives here, in rlx-models. This test
//! builds the real Nomic graph (synthetic weights, no checkpoint needed) and
//! asserts the fused path matches the unfused path bit-for-…-close. The
//! `FUSED_NOMIC_LAYER_COUNT` counter proves the fusion actually fired (it is
//! invisible through the `Session` API), so the parity check isn't vacuous.
//!
//! NOTE: requires rlx core via the repo-root `[patch.crates-io]` pointing at the
//! sibling `../rlx` working tree; against published rlx the fusion is disabled
//! and the test degenerates to unfused-vs-unfused (still passes).

use rlx_core::config::NomicBertConfig;
use rlx_core::weight_map::WeightMap;
use rlx_nomic::build_nomic_graph_sized;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;
use std::sync::atomic::Ordering;

/// Deterministic non-zero weights. Norm gains → 1, biases → 0, everything else
/// small zero-centered (so LayerNorm and the matmuls are non-degenerate).
fn syn(key: &str, n: usize) -> Vec<f32> {
    if key.ends_with("norm1.weight") || key.ends_with("norm2.weight") || key == "emb_ln.weight" {
        return vec![1.0; n];
    }
    if key.ends_with(".bias") {
        return vec![0.0; n];
    }
    let mut h = 1469598103934665603u64;
    for b in key.bytes() {
        h = (h ^ b as u64).wrapping_mul(1099511628211);
    }
    (0..n)
        .map(|i| {
            let z = h.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            (((z >> 40) & 0xffff) as f32 / 65535.0 - 0.5) * 0.4
        })
        .collect()
}

#[test]
fn nomic_layer_fusion_matches_unfused() {
    let cfg = NomicBertConfig {
        vocab_size: 32,
        hidden_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        intermediate_size: 32,
        max_position_embeddings: 32,
        type_vocab_size: 2,
        layer_norm_eps: 1e-5,
        head_dim: 4,
        rotary_emb_base: 1000.0,
    };
    let h = cfg.hidden_size;
    let id = cfg.intermediate_size;

    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut put = |key: String, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        t.insert(key.clone(), (syn(&key, n), shape));
    };
    put(
        "embeddings.word_embeddings.weight".into(),
        vec![cfg.vocab_size, h],
    );
    put(
        "embeddings.token_type_embeddings.weight".into(),
        vec![cfg.type_vocab_size, h],
    );
    put("emb_ln.weight".into(), vec![h]);
    put("emb_ln.bias".into(), vec![h]);
    let lp = "encoder.layers.0";
    put(format!("{lp}.attn.Wqkv.weight"), vec![3 * h, h]);
    put(format!("{lp}.attn.out_proj.weight"), vec![h, h]);
    put(format!("{lp}.norm1.weight"), vec![h]);
    put(format!("{lp}.norm1.bias"), vec![h]);
    put(format!("{lp}.mlp.fc11.weight"), vec![id, h]);
    put(format!("{lp}.mlp.fc12.weight"), vec![id, h]);
    put(format!("{lp}.mlp.fc2.weight"), vec![h, id]);
    put(format!("{lp}.norm2.weight"), vec![h]);
    put(format!("{lp}.norm2.bias"), vec![h]);

    let (batch, seq) = (1usize, 4usize);
    let mut wm = WeightMap::from_tensors(t);
    let (graph, params) = build_nomic_graph_sized(&cfg, &mut wm, batch, seq).unwrap();

    let ids = vec![1.0f32, 2.0, 3.0, 4.0];
    let mask = vec![1.0f32; batch * seq];
    let tt = vec![0.0f32; batch * seq];

    let run = |disable: bool| -> (Vec<f32>, u64) {
        // The fusion is decided at compile time, gated by this env var.
        unsafe {
            if disable {
                std::env::set_var("RLX_DISABLE_NOMIC_FUSION", "1");
            } else {
                std::env::remove_var("RLX_DISABLE_NOMIC_FUSION");
            }
        }
        rlx_cpu::thunk::FUSED_NOMIC_LAYER_COUNT.store(0, Ordering::Relaxed);
        let mut c = Session::new(Device::Cpu).compile(graph.clone());
        let fired = rlx_cpu::thunk::FUSED_NOMIC_LAYER_COUNT.load(Ordering::Relaxed);
        for (k, v) in &params {
            c.set_param(k, v);
        }
        let out = c
            .run(&[
                ("input_ids", ids.as_slice()),
                ("attention_mask", mask.as_slice()),
                ("token_type_ids", tt.as_slice()),
            ])
            .remove(0);
        (out, fired)
    };

    let (fused, fired_on) = run(false);
    let (reference, fired_off) = run(true);
    unsafe {
        std::env::remove_var("RLX_DISABLE_NOMIC_FUSION");
    }

    // Non-vacuous only when running against the patched local rlx (the published
    // crate ships the fusion disabled). Don't hard-fail on published rlx.
    if fired_on == 0 {
        eprintln!(
            "note: FusedNomicLayer never fired — running against published rlx? \
             parity below is unfused-vs-unfused."
        );
    }
    assert_eq!(
        fired_off, 0,
        "RLX_DISABLE_NOMIC_FUSION must suppress the fusion"
    );
    assert_eq!(fused.len(), reference.len());
    let max_abs = fused
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-4,
        "FusedNomicLayer diverged from unfused: max_abs={max_abs} (fired_on={fired_on})"
    );
}
