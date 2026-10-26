// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! End-to-end checks against a real `prism-ml/Ternary-Bonsai-2-27B-gguf`.
//!
//! `tests/bonsai2_manifest.rs` pins the header contract against a 175 KB
//! fixture and `tests/prism_hadamard_projection.rs` pins the rotation
//! math against synthetic weights. Neither touches tensor *data*, so
//! neither can catch the thing that actually decides whether this model
//! works: that the ternary bytes decode to the values the publisher
//! wrote, and that the rotation composed with them produces a sane
//! forward pass rather than fluent noise.
//!
//! The fixture is a 5.95 GB download, so the test self-skips without it:
//!
//! ```sh
//! just fetch-bonsai2-27b
//! RLX_QWEN35_BONSAI2_GGUF=weights/lm/ternary-bonsai-2-27b-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
//!   cargo test -p rlx-qwen35 --test bonsai2_real_weights -- --nocapture
//! ```

use rlx_qwen35::prism_hadamard::{FoldCtx, PrismHadamard, apply_inverse_transform};
use rlx_qwen35::{Qwen35Config, Qwen35TrunkLayer, Qwen35Weights};
use std::path::PathBuf;

fn fixture() -> Option<PathBuf> {
    rlx_ir::env::var_os("RLX_QWEN35_BONSAI2_GGUF").map(PathBuf::from)
}

macro_rules! skip_without_fixture {
    () => {
        match fixture() {
            Some(p) => p,
            None => {
                eprintln!(
                    "skip: set RLX_QWEN35_BONSAI2_GGUF to a Ternary Bonsai 2 GGUF \
                     (just fetch-bonsai2-27b)"
                );
                return;
            }
        }
    };
}

/// The ternary payload must decode to exactly `{−d, 0, +d}` per group.
///
/// This is the check that the base-3 unpacking is right on real bytes
/// rather than on bytes this repo generated. A wrong digit extraction
/// still yields values in range — it just yields the wrong *ones* — so
/// the assertion is on the shape of the distribution, not just finiteness.
#[test]
fn ternary_payload_decodes_to_three_values_per_group() {
    let path = skip_without_fixture!();
    let raw = rlx_gguf::GgufFile::from_path_mmap(&path).expect("mmap GGUF");

    let name = "blk.3.attn_k.weight";
    let t = raw.get(name).expect(name);
    assert_eq!(t.dtype, rlx_gguf::GgmlType::PTQ1_0);
    let bytes = raw.tensor_bytes(t).expect("tensor bytes");

    // One block is 28 bytes / 128 weights; check a few thousand of them.
    const BLOCKS: usize = 4000;
    let vals = rlx_gguf::ptq1_dequant::dequant_ptq1_0(&bytes[..BLOCKS * 28], BLOCKS * 128)
        .expect("decode PTQ1_0");

    let mut zero = 0usize;
    let mut pos = 0usize;
    let mut neg = 0usize;
    for (b, blk) in vals.as_chunks::<128>().0.iter().enumerate() {
        let d = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
        for &v in blk {
            assert!(
                v == 0.0 || v == d || v == -d,
                "block {b}: {v} is not one of {{0, ±{d}}} — base-3 unpack is wrong"
            );
            match v.partial_cmp(&0.0).unwrap() {
                std::cmp::Ordering::Less => neg += 1,
                std::cmp::Ordering::Equal => zero += 1,
                std::cmp::Ordering::Greater => pos += 1,
            }
        }
    }
    let n = (BLOCKS * 128) as f64;
    eprintln!(
        "{name}: -1 {:.1}% / 0 {:.1}% / +1 {:.1}%",
        100.0 * neg as f64 / n,
        100.0 * zero as f64 / n,
        100.0 * pos as f64 / n
    );
    // A trained ternary checkpoint is close to balanced. A mis-shifted
    // digit collapses toward one code, which this catches; exact
    // thirds are not expected, so the band is wide.
    for (label, c) in [("-1", neg), ("0", zero), ("+1", pos)] {
        let frac = c as f64 / n;
        assert!(
            (0.15..0.55).contains(&frac),
            "trit {label} is {:.1}% of the tensor — the digit extraction is skewed",
            100.0 * frac
        );
    }
}

/// Load the whole model and check the rotation metadata resolved
/// against the tensors it names.
#[test]
fn loads_packed_with_every_folded_weight_resolved() {
    let path = skip_without_fixture!();
    let raw = rlx_gguf::GgufFile::header_from_path(&path).expect("parse header");
    let cfg = Qwen35Config::from_gguf(&raw).expect("parse config");
    assert_eq!(cfg.num_hidden_layers, 64);
    assert_eq!(cfg.nextn_predict_layers, 0, "Bonsai 2 ships no MTP head");

    let h = PrismHadamard::from_gguf(&raw)
        .expect("parse prism.hadamard")
        .expect("Bonsai 2 declares a rotation");
    let ctx = FoldCtx::new(h.clone(), cfg.ssm_time_step_rank, cfg.ssm_group_count);

    let mut loader = rlx_core::weight_loader::GgufLoader::from_file(path.to_str().unwrap())
        .expect("open loader");
    let w = Qwen35Weights::from_loader_packed(&mut loader, &cfg).expect("load weights");

    // Every projection the file declares folded must have come back as
    // `Proj::Folded`. A `Dense` here means the weight reaches the matmul
    // without its transform, which does not fail — it just makes the
    // model wrong — so it has to be asserted.
    let mut folded = 0usize;
    let mut permuted = 0usize;
    for (il, layer) in w.trunk_layers.iter().enumerate() {
        let named: Vec<(&str, &rlx_qwen35::Proj)> = match layer {
            Qwen35TrunkLayer::Linear(l) => vec![
                ("attn_qkv", &l.attn_qkv),
                ("attn_gate", &l.attn_gate),
                ("ssm_out", &l.ssm_out),
            ],
            Qwen35TrunkLayer::FullAttn(l) => vec![
                ("attn_q", &l.attn_q_gate),
                ("attn_k", &l.attn_k),
                ("attn_v", &l.attn_v),
                ("attn_output", &l.attn_output),
            ],
        };
        for (slot, proj) in named {
            let key = format!("blk.{il}.{slot}.weight");
            if !h.is_folded(&key) {
                continue;
            }
            match proj {
                rlx_qwen35::Proj::Folded(f) => {
                    folded += 1;
                    assert_eq!(f.fold.block_size, 1024);
                    assert!(f.fold.signs.is_some(), "{key} lost its sign vector");
                    if f.fold.perm.is_some() {
                        permuted += 1;
                    }
                }
                other => panic!("{key} is declared folded but loaded as {other:?}"),
            }
        }
    }
    eprintln!("folded projections loaded: {folded} ({permuted} permuted)");
    assert_eq!(
        permuted, 48,
        "one permuted ssm_out per linear-attention layer"
    );
    assert!(
        w.output_fold.is_some(),
        "output.weight is folded and must carry its transform"
    );
    let _ = ctx;
}

/// The embedding table is stored rotated; after the inverse transform
/// its rows must look like embeddings, not like the rotated latents.
///
/// A Hadamard rotation of a sparse-ish ternary row spreads its mass —
/// so the rotated and un-rotated forms differ sharply in kurtosis. That
/// makes "did the inverse actually run, in the right direction" testable
/// without a reference implementation.
#[test]
fn token_embedding_rows_are_unrotated_at_load() {
    let path = skip_without_fixture!();
    let raw = rlx_gguf::GgufFile::from_path_mmap(&path).expect("mmap GGUF");
    let cfg = Qwen35Config::from_gguf(&raw).expect("parse config");
    let h = PrismHadamard::from_gguf(&raw).unwrap().unwrap();
    assert!(h.is_inverse("token_embd.weight"));

    let n_embd = cfg.hidden_size;
    let embd = raw.get("token_embd.weight").expect("token_embd.weight");
    let bytes = raw.tensor_bytes(embd).expect("embd bytes");
    let row_bytes = n_embd / 128 * 28;
    // One row, straight off disk: still in the rotated basis.
    let stored =
        rlx_gguf::ptq1_dequant::dequant_ptq1_0(&bytes[..row_bytes], n_embd).expect("decode row");
    assert!(
        stored.iter().all(|v| v.is_finite()),
        "stored embedding row is not finite"
    );

    let ctx = FoldCtx::new(h.clone(), cfg.ssm_time_step_rank, cfg.ssm_group_count);
    let fold = ctx.inverse_fold(n_embd).expect("inverse fold");
    assert!(
        fold.signs.is_some(),
        "the inverse table does carry a sign flip — omitting it is what made \
         the model emit EOS on its first token"
    );
    let mut restored = stored.clone();
    apply_inverse_transform(&mut restored, n_embd, &fold);

    let kurt = |x: &[f32]| {
        let n = x.len() as f64;
        let m = x.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = x.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / n;
        let m4 = x.iter().map(|&v| (v as f64 - m).powi(4)).sum::<f64>() / n;
        m4 / (var * var)
    };
    let (k_stored, k_restored) = (kurt(&stored), kurt(&restored));
    eprintln!("token_embd row kurtosis: stored {k_stored:.2} -> un-rotated {k_restored:.2}");
    assert!(
        k_restored > k_stored * 1.5,
        "un-rotating did not concentrate the row (kurtosis {k_stored:.2} -> {k_restored:.2}); \
         the transform may be a no-op or applied in the wrong direction"
    );
}

/// The lazy (packed) embedding gather must return exactly what the
/// materialized F32 table holds.
///
/// Leaving `token_embd` packed avoids expanding a 278 MB table to 4.74 GiB,
/// but it moves the dequant *and* the `prism.hadamard` inverse from a
/// one-shot pass at load onto a per-row gather. Those are two different code
/// paths over the same bytes, and a mismatch would be a quietly wrong
/// embedding for every token — so compare them row for row on real weights.
#[test]
fn lazy_embed_rows_match_the_materialized_table() {
    let path = skip_without_fixture!();
    let p = path.to_str().unwrap();
    let cfg =
        Qwen35Config::from_gguf(&rlx_gguf::GgufFile::header_from_path(&path).expect("header"))
            .expect("config");
    let n_embd = cfg.hidden_size;

    let mut l_dense = rlx_core::weight_loader::GgufLoader::from_file(p).expect("loader");
    rlx_ir::env::set("RLX_QWEN35_NO_LAZY_EMBED", "1");
    let dense = Qwen35Weights::from_loader_packed(&mut l_dense, &cfg).expect("dense load");
    rlx_ir::env::remove("RLX_QWEN35_NO_LAZY_EMBED");
    assert!(!dense.embed_is_lazy(), "control bundle should be dense");

    let mut l_lazy = rlx_core::weight_loader::GgufLoader::from_file(p).expect("loader");
    let lazy = Qwen35Weights::from_loader_packed(&mut l_lazy, &cfg).expect("lazy load");
    assert!(
        lazy.embed_is_lazy(),
        "this checkpoint is packed and untied, so the table should stay packed"
    );

    let tbl = dense.token_embd();
    let mut row = vec![0f32; n_embd];
    // First, last, and a spread through the middle of the 248320-row table.
    let n_vocab = lazy.lm_vocab_size(&cfg);
    for id in [0u32, 1, 760, 6511, 9338, 100_000, (n_vocab - 1) as u32] {
        lazy.embed_row_into(Some(&l_lazy), id, &mut row)
            .unwrap_or_else(|e| panic!("gather row {id}: {e}"));
        let want = &tbl[id as usize * n_embd..(id as usize + 1) * n_embd];
        let worst = row
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-5,
            "lazy row {id} differs from the materialized table by {worst}"
        );
        // Guard the guard: an all-zero row would match a zeroed table.
        assert!(
            row.iter().any(|v| v.abs() > 1e-6),
            "row {id} is all zeros — the gather is not returning data"
        );
    }
}
