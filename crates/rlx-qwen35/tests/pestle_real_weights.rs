// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Structural checks against a real `Doses-AI/Pestle-27B-Ternary-GGUF`.
//!
//! `tests/pestle_projection.rs` pins the *math* of a Pestle projection
//! against synthetic fixtures. What it cannot cover is the part that is a
//! property of the checkpoint rather than the graph: that the slot→tensor
//! mapping is the one the `mortar.cpp` fork uses, that the file really is
//! mixed (Q2_0 factor pairs + one dense BF16 block + G8_0 embed tables),
//! and that every projection came back with the rank and dims the config
//! implies. Getting any of that wrong still loads, still runs, and still
//! emits fluent text — so it needs an assertion, not an eyeball.
//!
//! The fixture is an 7.9 GiB download, so the test self-skips without it:
//!
//! ```sh
//! just fetch-pestle-27b
//! RLX_QWEN35_PESTLE_GGUF=weights/pestle-27b-ternary-gguf/pestle-27b-ternary.gguf \
//!   cargo test -p rlx-qwen35 --test pestle_real_weights -- --nocapture
//! ```

use rlx_qwen35::{Qwen35Config, Qwen35LayerFfn, Qwen35TrunkLayer, Qwen35Weights};
use std::path::PathBuf;

fn fixture() -> Option<PathBuf> {
    std::env::var_os("RLX_QWEN35_PESTLE_GGUF").map(PathBuf::from)
}

#[test]
fn pestle_27b_loads_with_the_expected_mixed_layout() {
    let Some(path) = fixture() else {
        eprintln!("skip: set RLX_QWEN35_PESTLE_GGUF to a Pestle GGUF (just fetch-pestle-27b)");
        return;
    };
    let raw = rlx_gguf::GgufFile::from_path(&path).expect("parse GGUF");
    let cfg = Qwen35Config::from_gguf(&raw).expect("parse qwen35 config");

    // Pestle is Qwen3.6-27B's topology verbatim, minus the MTP head.
    assert_eq!(cfg.num_hidden_layers, 64);
    assert_eq!(cfg.nextn_predict_layers, 0, "Pestle ships no MTP head");
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.full_attention_interval, 4);

    let mut loader =
        rlx_core::weight_loader::GgufLoader::from_file(path.to_str().expect("utf-8 path"))
            .expect("open GGUF loader");
    let weights = Qwen35Weights::from_loader_packed(&mut loader, &cfg).expect("load weights");

    let mut pestle_layers = 0usize;
    let mut dense_layers = 0usize;
    let mut factor_mats = 0usize;
    for (il, layer) in weights.trunk_layers.iter().enumerate() {
        let (projs, ffn): (Vec<_>, _) = match layer {
            Qwen35TrunkLayer::Linear(l) => (
                vec![
                    &l.attn_qkv,
                    &l.attn_gate,
                    &l.ssm_beta,
                    &l.ssm_alpha,
                    &l.ssm_out,
                ],
                &l.ffn,
            ),
            Qwen35TrunkLayer::FullAttn(f) => (
                vec![&f.attn_q_gate, &f.attn_k, &f.attn_v, &f.attn_output],
                &f.ffn,
            ),
        };
        let Qwen35LayerFfn::Dense { gate, up, down } = ffn else {
            panic!("blk.{il}: Pestle-27B is a dense-FFN model");
        };
        let all: Vec<_> = projs.into_iter().chain([gate, up, down]).collect();

        // A layer is Pestle or dense as a whole — never half of each.
        let n_pestle = all.iter().filter(|p| p.is_pestle()).count();
        assert!(
            n_pestle == 0 || n_pestle == all.len(),
            "blk.{il}: mixed within one layer — {n_pestle}/{} projections factorized",
            all.len()
        );
        if n_pestle == 0 {
            dense_layers += 1;
            continue;
        }
        pestle_layers += 1;
        for p in &all {
            // Every factor must be Q2_0-packed: dequantizing them to F32
            // would turn 7.9 GiB of weights into ~100 GB.
            for m in p.mats() {
                assert!(
                    matches!(m.scheme(), Some(rlx_ir::quant::QuantScheme::GgufQ2_0)),
                    "blk.{il}: factor is not Q2_0-packed ({:?})",
                    m.scheme()
                );
                factor_mats += 1;
            }
        }
    }

    // 63 factorized blocks + one dense BF16 block ("matching-parent final
    // decoder block"), and it is the last one.
    assert_eq!(pestle_layers, 63, "expected 63 Pestle blocks");
    assert_eq!(dense_layers, 1, "expected exactly one dense block");
    assert!(
        !matches!(&weights.trunk_layers[63], Qwen35TrunkLayer::Linear(l) if l.attn_qkv.is_pestle()),
        "the dense block should be the last one"
    );
    // 48 linear-attn blocks × 8 slots + 15 full-attn blocks × 7 slots,
    // two matrices each.
    assert_eq!(factor_mats, 2 * (48 * 8 + 15 * 7), "factor matrix count");

    // Untied embed/lm_head, both in the G8_0 exact-ternary format.
    //
    // Width comes from the tables, NOT `cfg.vocab_size`: the GGUF carries no
    // `qwen35.vocab_size` key, so the config falls back to Qwen's nominal
    // 151936 while the real tables are 248320 rows. `lm_vocab_size()` exists
    // for exactly this; asserting against `cfg.vocab_size` here would pin the
    // fallback rather than the checkpoint.
    let n_vocab = weights.lm_vocab_size(&cfg);
    assert_eq!(n_vocab, 248_320, "embedding table width");
    assert!(
        n_vocab > cfg.vocab_size,
        "expected the table to be wider than the nominal cfg.vocab_size"
    );
    // The EOS the runner stops on must be addressable, or generation would
    // index past the LM head.
    assert!(248_046 < n_vocab, "<|im_end|> must be inside the LM head");

    let out = weights
        .output
        .as_ref()
        .expect("Pestle has an untied lm_head");
    assert_eq!(
        out.scheme(),
        Some(rlx_ir::quant::QuantScheme::GgufG8_0),
        "lm_head should stay G8_0-packed"
    );
    assert_eq!(
        out.shape(),
        [n_vocab, cfg.hidden_size],
        "lm_head must match the embedding table width"
    );
    // A G8_0 table decoded with a single block-wide scale (or with the
    // scales read as f16 instead of bf16) collapses toward zero.
    let rms = (weights
        .token_embd
        .iter()
        .take(1 << 20)
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        / (1u64 << 20) as f64)
        .sqrt();
    assert!(
        rms > 1e-4 && rms < 1.0,
        "token_embd RMS {rms} is implausible — check the G8_0 bf16 scales"
    );
}
