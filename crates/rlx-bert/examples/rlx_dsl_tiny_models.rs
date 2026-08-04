// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Small models declared with the `rlx!` graph DSL, compiled and run on CPU.
//!
//! This sits next to the crate's real `build_bert_graph_sized` (which uses the
//! `rlx-flow` builder) as a *demonstration* that the compact `rlx!` little
//! language is expressive enough for the smaller models — an MLP classifier, a
//! (weight-tied) BERT-style Transformer encoder, and an RNN — end to end.
//!
//! Run it (from the rlx-models workspace, with local RLX patched in):
//!
//! ```text
//! cargo run -p rlx-bert --example rlx_dsl_tiny_models
//! ```
//!
//! It exercises the DSL surface real models lean on: `@` matmul, `+`/`*`,
//! activation sugar (`relu`/`gelu`/`silu`/`tanh`), the `fn` subgraph +
//! `repeat` stacking, the `scan` compact loop (one `Op::Scan`, not unrolled),
//! and the `.method(..)` escape hatch for `attention` / `layer_norm`.

use rlx::rlx;
use rlx::runtime::{Device, Session};

/// Small deterministic weights in `[-0.1, 0.1]` — enough to produce finite,
/// non-degenerate activations without loading real checkpoints.
fn fill(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + 3) % 11) as f32 / 11.0 - 0.5) * 0.2)
        .collect()
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}
fn zeros(n: usize) -> Vec<f32> {
    vec![0.0; n]
}

fn main() {
    mlp_classifier();
    bert_encoder();
    rnn_scan();
    println!("\nAll three rlx! models compiled and ran on CPU. ✔");
}

/// The simplest complete small model: a 2-layer MLP classifier.
/// `logits = relu(x·W1 + b1)·W2 + b2`, then argmax.
fn mlp_classifier() {
    let g = rlx! {
        graph "mlp";
        input x: [1, 8];
        param w1: [8, 16];   param b1: [16];
        param w2: [16, 4];   param b2: [4];
        let h = relu(x @ w1 + b1);
        let logits = h @ w2 + b2;
        out logits;
    };

    let mut m = Session::new(Device::Cpu).compile(g);
    m.set_param("w1", &fill(8 * 16));
    m.set_param("b1", &zeros(16));
    m.set_param("w2", &fill(16 * 4));
    m.set_param("b2", &zeros(4));

    let out = m.run(&[("x", &fill(8)[..])]);
    let logits = &out[0];
    let class = argmax(logits);
    println!("[mlp]     logits = {logits:?}  → class {class}");
    assert_eq!(logits.len(), 4);
}

/// A BERT-style Transformer encoder: a `fn` block (bidirectional self-attention,
/// post-LayerNorm, and a GELU feed-forward), stacked with `repeat` (weight-tied
/// across layers, so one set of parameters). This is the same architecture
/// family as this crate's real BERT, written in a dozen lines.
fn bert_encoder() {
    // dim = num_heads (2) * head_dim (4) = 8; seq = 6.
    let g = rlx! {
        graph "encoder";
        input x: [1, 6, 8];
        param wq: [8, 8];  param wk: [8, 8];  param wv: [8, 8];  param wo: [8, 8];
        param wi: [8, 16]; param wff: [16, 8];
        param ln1g: [8]; param ln1b: [8];
        param ln2g: [8]; param ln2b: [8];

        // One encoder layer: attn → residual+LN → FFN(GELU) → residual+LN.
        fn layer(x, wq, wk, wv, wo, wi, wff, ln1g, ln1b, ln2g, ln2b) {
            let q = x @ wq;
            let k = x @ wk;
            let v = x @ wv;
            let a = q.attention(k, v, 2, 4, MaskKind::None);
            let x = (x + a @ wo).layer_norm(ln1g, ln1b, 1e-5f32);
            let ff = gelu(x @ wi) @ wff;
            let y = (x + ff).layer_norm(ln2g, ln2b, 1e-5f32);
        }

        // Three stacked encoder layers (weight-tied).
        repeat 3 {
            let x = layer(x, wq, wk, wv, wo, wi, wff, ln1g, ln1b, ln2g, ln2b);
        }
        out x;
    };

    let mut m = Session::new(Device::Cpu).compile(g);
    for p in ["wq", "wk", "wv", "wo"] {
        m.set_param(p, &fill(8 * 8));
    }
    m.set_param("wi", &fill(8 * 16));
    m.set_param("wff", &fill(16 * 8));
    for p in ["ln1g", "ln2g"] {
        m.set_param(p, &ones(8));
    }
    for p in ["ln1b", "ln2b"] {
        m.set_param(p, &zeros(8));
    }

    let out = m.run(&[("x", &fill(6 * 8)[..])]);
    let hidden = &out[0];
    println!(
        "[encoder] out shape [1, 6, 8] = {} values; first row = {:?}",
        hidden.len(),
        &hidden[..8]
    );
    assert_eq!(hidden.len(), 6 * 8);
    assert!(hidden.iter().all(|v| v.is_finite()));
}

/// A recurrence as a compact `scan` (`Op::Scan`): `hₜ₊₁ = tanh(hₜ·W)`, 5 steps.
/// Unlike a `repeat`-unrolled loop, this is a single scan node — O(1) IR
/// regardless of the step count.
fn rnn_scan() {
    let g = rlx! {
        graph "rnn";
        input h0: [1, 8];
        param w: [8, 8];
        scan h = h0 for 5 {
            let h = tanh(h @ w);
        }
        out h;
    };

    let mut m = Session::new(Device::Cpu).compile(g);
    m.set_param("w", &fill(8 * 8));
    let out = m.run(&[("h0", &ones(8)[..])]);
    let h = &out[0];
    println!("[rnn]     final carry (5 steps) = {h:?}");
    assert_eq!(h.len(), 8);
    assert!(h.iter().all(|v| v.is_finite()));
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}
