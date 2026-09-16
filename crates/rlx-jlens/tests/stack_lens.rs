//! The lens proper: `J_l = ∂h_target/∂h_l` for every layer, from one forward.
//!
//! `BlockLens` differentiates one block in isolation; this differentiates the
//! residual at the target layer with respect to the residual entering each
//! source layer, over a real prompt. That is the object the lens is defined by.
//!
//! The correctness argument here does not lean on finite differences. Tapping
//! the *last* layer's entry residual makes `J` the Jacobian of exactly one
//! block, which `BlockLens` computes independently — so the two must agree.
//! That check covers tap discovery, the `Wrt::Output` addressing, the packing
//! of many taps into one backward, and the row accumulation, against a path
//! that shares none of it.

#![cfg(feature = "qwen35")]

use rlx_jlens::models::qwen35::Qwen35LensModel;
use rlx_jlens::{BlockLens, FitConfig, LensModel, StackLens};
use rlx_qwen35::synth::{synth_weights, tiny_cfg};
use rlx_runtime::Device;

const SEQ: usize = 6;

fn model() -> Qwen35LensModel {
    let cfg = tiny_cfg();
    let weights = synth_weights(&cfg);
    Qwen35LensModel::new(cfg, weights)
}

/// tiny_cfg's 6-token prompt cannot afford the production skip of 16.
fn config() -> FitConfig {
    FitConfig {
        dim_batch: 4,
        skip_first: 1,
        device: Device::Cpu,
    }
}

/// Token ids within `tiny_cfg`'s 32-entry vocabulary.
fn prompt() -> Vec<f32> {
    (0..SEQ).map(|i| ((i * 7 + 3) % 32) as f32).collect()
}

#[test]
fn fits_every_layer_from_one_forward() {
    let m = model();
    let target = m.n_layers() - 1;
    let layers: Vec<usize> = (0..=target).collect();
    let mut lens = StackLens::new(&m, &layers, target, SEQ, config()).expect("build stack lens");

    assert_eq!(lens.layers(), layers.as_slice());
    assert_eq!(lens.positions(), &[1, 2, 3, 4]);
    assert_eq!(lens.passes(), m.d_model() / 4);

    let tokens = lens.replicate_tokens(&prompt()).unwrap();
    let js = lens.jacobians(&tokens).expect("fit jacobians");

    assert_eq!(js.len(), layers.len(), "one Jacobian per tapped layer");
    // One forward for the whole sweep, and one backward per pass covering every
    // layer at once — that is the point of tapping them together.
    let t = lens.timing();
    assert_eq!(t.forward_runs, 1);
    assert_eq!(t.replay_runs, lens.passes());
    eprintln!("{} layers | {}", js.len(), t.summary());

    for (l, j) in layers.iter().zip(&js) {
        assert_eq!(j.d_model, m.d_model());
        assert!(
            j.values.iter().all(|v| v.is_finite()),
            "layer {l}: non-finite entries"
        );
        let magnitude = j.values.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        assert!(magnitude > 1e-4, "layer {l}: J is ~zero (max {magnitude})");
        eprintln!("  layer {l}: ||J||/sqrt(d) = {:.4}", j.scaled_norm());
    }
}

/// Tapping the last layer's entry makes `J` that block's own Jacobian, which
/// `BlockLens` computes by a completely different route.
#[test]
fn last_layer_tap_matches_the_block_lens() {
    let m = model();
    let target = m.n_layers() - 1;
    let cfg = config();

    // Tap the layer *before* the target: taps sit at each layer's exit (the
    // convention the Python reference uses), so tapping `target` itself would
    // transport the target's own output to the target and give the identity.
    // One layer earlier is the residual entering `target`, and transporting that
    // to `target`'s output is exactly one block's Jacobian.
    let source = target - 1;
    let mut stack = StackLens::new(&m, &[source], target, SEQ, cfg).expect("stack lens");
    let tokens = stack.replicate_tokens(&prompt()).unwrap();
    let stack_j = stack.jacobians(&tokens).expect("stack jacobians").remove(0);

    // Block: the same block, fed the residual the trunk actually produces there.
    // Recover it by running the stack's own forward — `h_final` is output 0 of
    // the save half, but the tap's *value* is what the block needs, so rebuild
    // the trunk truncated one layer earlier.
    let stack_graph = m
        .stack(&[source], target, cfg.dim_batch, SEQ)
        .expect("stack graph");
    let mut fwd = rlx_runtime::Session::new(cfg.device).compile(stack_graph.tapped.graph().clone());
    for (name, data) in &stack_graph.params {
        fwd.set_param(name, data);
    }
    let outs = fwd.run(&[(stack_graph.token_input.as_str(), &tokens[..])]);
    // outputs = [h_target, tap]; the tap leaves `source`, i.e. enters `target`.
    let entry_residual = outs[1].clone();

    let mut block = BlockLens::new(&m, target, SEQ, cfg).expect("block lens");
    // The stack graph is built at `dim_batch`, so this residual is already
    // `[dim_batch, seq, d_model]` — exactly what BlockLens wants.
    let block_j = block.jacobian(&entry_residual).expect("block jacobian");

    assert_eq!(stack_j.d_model, block_j.d_model);
    let mut worst = 0.0f32;
    for (a, b) in stack_j.values.iter().zip(&block_j.values) {
        worst = worst.max((a - b).abs());
    }
    let magnitude = stack_j.values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        magnitude > 0.1,
        "J is ~zero (max {magnitude}) — check is vacuous"
    );
    assert!(
        worst < 1e-4,
        "stack lens and block lens disagree on the last layer: worst {worst}"
    );
    eprintln!("last-layer tap vs BlockLens: worst |Δ| = {worst:.2e}, max |J| = {magnitude:.4}");
}

#[test]
fn rejects_a_source_after_the_target() {
    let m = model();
    let Err(err) = StackLens::new(&m, &[2], 1, SEQ, config()) else {
        panic!("a source layer after the target should be rejected");
    };
    assert!(err.to_string().contains("after the target"), "{err}");
}

#[test]
fn fit_averages_over_prompts() {
    let m = model();
    let target = m.n_layers() - 1;
    let mut lens = StackLens::new(&m, &[0, target], target, SEQ, config()).unwrap();

    let a = prompt();
    let b: Vec<f32> = (0..SEQ).map(|i| ((i * 11 + 5) % 32) as f32).collect();
    let ja = lens.jacobians(&lens.replicate_tokens(&a).unwrap()).unwrap();
    let jb = lens.jacobians(&lens.replicate_tokens(&b).unwrap()).unwrap();

    let mut seen = Vec::new();
    let mean = lens
        .fit(&[a, b], |i, js| seen.push((i, js.len())))
        .unwrap()
        .expect("non-empty corpus");

    assert_eq!(seen, vec![(0, 2), (1, 2)]);
    assert_eq!(mean.len(), 2);
    for (l, m_j) in mean.iter().enumerate() {
        for i in 0..m_j.values.len() {
            let want = 0.5 * (ja[l].values[i] + jb[l].values[i]);
            assert!(
                (m_j.values[i] - want).abs() < 1e-5,
                "layer slot {l} entry {i}: {} vs {want}",
                m_j.values[i]
            );
        }
    }
    // The two prompts must differ, or the average proves nothing.
    let differ = ja[0]
        .values
        .iter()
        .zip(&jb[0].values)
        .any(|(x, y)| (x - y).abs() > 1e-4);
    assert!(
        differ,
        "both prompts gave the same Jacobian — test is vacuous"
    );
}

#[test]
fn empty_corpus_yields_nothing() {
    let m = model();
    let target = m.n_layers() - 1;
    let mut lens = StackLens::new(&m, &[target], target, SEQ, config()).unwrap();
    assert!(lens.fit(&[], |_, _| {}).unwrap().is_none());
}
