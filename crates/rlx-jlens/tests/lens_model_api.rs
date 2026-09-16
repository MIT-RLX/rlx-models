//! The model-agnostic path: fit a per-block Jacobian through [`LensModel`].
//!
//! Nothing here mentions attention, RoPE or delta-nets — it drives whatever
//! `LensModel` it is handed. Qwen3.5/3.6 is the implementation under test; a
//! second model should pass this file unchanged with one line swapped.

#![cfg(feature = "qwen35")]

use rlx_jlens::models::qwen35::{BlockKind, Qwen35LensModel};
use rlx_jlens::{BlockLens, FitConfig, LensModel};
use rlx_qwen35::synth::{synth_weights, tiny_cfg};
use rlx_runtime::Device;

const SEQ: usize = 6;
const ATTN_LAYER: usize = 2;

fn model() -> Qwen35LensModel {
    let cfg = tiny_cfg();
    let weights = synth_weights(&cfg);
    Qwen35LensModel::new(cfg, weights)
}

/// tiny_cfg has a 6-token sequence, so the production skip of 16 leading
/// positions would leave nothing to average.
fn config() -> FitConfig {
    FitConfig {
        dim_batch: 4,
        skip_first: 1,
        device: Device::Cpu,
    }
}

fn residual(model: &Qwen35LensModel) -> Vec<f32> {
    (0..SEQ * model.d_model())
        .map(|i| {
            let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x ^= x >> 29;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 32;
            0.5 * (((x >> 40) as f32) / 8_388_608.0 - 1.0)
        })
        .collect()
}

#[test]
fn model_reports_its_own_shape() {
    let m = model();
    // tiny_cfg declares num_hidden_layers = 4, but one of those is the
    // multi-token-prediction head, not a trunk block. The lens taps the trunk,
    // so `n_layers` counts trunk blocks — 3 — rather than the config value.
    assert_eq!(m.n_layers(), 3);
    assert_eq!(m.d_model(), 16);
    // full_attention_interval = 3 ⇒ every 3rd layer is attention.
    assert_eq!(m.block_kind(0), BlockKind::GatedDeltaNet);
    assert_eq!(m.block_kind(1), BlockKind::GatedDeltaNet);
    assert_eq!(m.block_kind(2), BlockKind::FullAttention);
}

#[test]
fn out_of_range_layer_is_rejected() {
    let m = model();
    let Err(err) = BlockLens::new(&m, 99, SEQ, config()) else {
        panic!("layer 99 of a 4-layer model should be rejected");
    };
    assert!(err.to_string().contains("out of range"), "{err}");
}

/// A sequence too short to leave any averaging positions must fail loudly
/// rather than silently fitting on nothing.
#[test]
fn too_short_a_sequence_is_rejected() {
    let m = model();
    let cfg = FitConfig {
        skip_first: 16,
        ..config()
    };
    assert!(BlockLens::new(&m, ATTN_LAYER, SEQ, cfg).is_err());
}

#[test]
fn fits_a_block_jacobian_through_the_trait() {
    let m = model();
    let mut lens = BlockLens::new(&m, ATTN_LAYER, SEQ, config()).expect("build block lens");

    assert_eq!(lens.positions(), &[1, 2, 3, 4]);
    // d_model = 16 over dim_batch = 4.
    assert_eq!(lens.passes(), 4);

    let h = residual(&m);
    let batched = lens.replicate(&h).unwrap();
    let j = lens.jacobian(&batched).expect("fit jacobian");

    assert_eq!(j.d_model, m.d_model());
    assert_eq!(j.values.len(), m.d_model() * m.d_model());
    assert!(
        j.values.iter().all(|v| v.is_finite()),
        "Jacobian contains non-finite entries"
    );
    let magnitude = j.values.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    assert!(magnitude > 0.1, "Jacobian is ~zero (max {magnitude})");
    // A block is residual, so J should sit near the identity: the diagonal
    // should dominate a typical off-diagonal entry.
    let diag = (0..j.d_model)
        .map(|i| j.values[i * j.d_model + i].abs())
        .fold(f32::MAX, f32::min);
    assert!(
        diag > 0.5,
        "smallest diagonal entry is {diag}; a residual block's Jacobian should be near-identity"
    );
    assert!(j.scaled_norm() > 0.0);
}

#[test]
fn transport_applies_the_jacobian() {
    let m = model();
    let mut lens = BlockLens::new(&m, ATTN_LAYER, SEQ, config()).unwrap();
    let h = residual(&m);
    let j = lens.jacobian(&lens.replicate(&h).unwrap()).unwrap();

    let d = j.d_model;
    let row = &h[..d];
    let got = j.transport(row);
    assert_eq!(got.len(), d);
    for i in 0..d {
        let want: f32 = (0..d).map(|k| j.values[i * d + k] * row[k]).sum();
        assert!(
            (got[i] - want).abs() < 1e-4,
            "transport[{i}]: {} vs {want}",
            got[i]
        );
    }
}

/// `fit` is the running mean over a corpus, and must report each prompt's own
/// Jacobian on the way through.
#[test]
fn fit_averages_over_a_corpus() {
    let m = model();
    let mut lens = BlockLens::new(&m, ATTN_LAYER, SEQ, config()).unwrap();

    let a = residual(&m);
    let b: Vec<f32> = a.iter().map(|v| v * 0.5 + 0.05).collect();
    let ja = lens.jacobian(&lens.replicate(&a).unwrap()).unwrap();
    let jb = lens.jacobian(&lens.replicate(&b).unwrap()).unwrap();

    let mut seen = Vec::new();
    let mean = lens
        .fit(&[a, b], |i, j| seen.push((i, j.scaled_norm())))
        .unwrap()
        .expect("non-empty corpus");

    assert_eq!(seen.len(), 2, "observer should see every prompt");
    assert_eq!(seen[0].0, 0);
    assert_eq!(seen[1].0, 1);
    for i in 0..mean.values.len() {
        let want = 0.5 * (ja.values[i] + jb.values[i]);
        assert!(
            (mean.values[i] - want).abs() < 1e-5,
            "mean[{i}]: {} vs {want}",
            mean.values[i]
        );
    }
    // The two prompts must actually differ, or the average proves nothing.
    let differ = ja
        .values
        .iter()
        .zip(&jb.values)
        .any(|(x, y)| (x - y).abs() > 1e-4);
    assert!(
        differ,
        "both prompts gave the same Jacobian — test is vacuous"
    );
}

#[test]
fn empty_corpus_yields_no_jacobian() {
    let m = model();
    let mut lens = BlockLens::new(&m, ATTN_LAYER, SEQ, config()).unwrap();
    assert!(lens.fit(&[], |_, _| {}).unwrap().is_none());
}
