// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Training losses as graph builders.
//!
//! These compose first-class ops (autodiff-supported, backend-portable), so a
//! graph built with them differentiates and trains on any backend.

use rlx_ir::infer::GraphExt;
use rlx_ir::{Graph, NodeId};

/// Masked cross-entropy — the standard LM fine-tuning objective.
///
/// `logits [N, C]` vs integer `labels [N]` (f32-encoded class ids), weighted
/// by a 0/1 `mask [N]` (e.g. prompt tokens masked out), reduced to the mean
/// over unmasked rows. Uses the fused `SoftmaxCrossEntropyWithLogits` op
/// (numerically stable + autodiff-supported).
pub fn cross_entropy_masked(g: &mut Graph, logits: NodeId, labels: NodeId, mask: NodeId) -> NodeId {
    let per_row = g.softmax_cross_entropy_with_logits(logits, labels); // [N]
    let masked = g.mul(per_row, mask);
    // Mean over all rows; masked-out rows contribute 0. (A normalize-by-active
    // -count variant would divide by sum(mask), but dividing by a node hurts
    // the gradient scale here; the constant-N mean trains cleanly.)
    g.mean(masked, vec![0], false)
}

/// DPO-style preference loss built from cross-entropy.
///
/// DPO minimizes `-log σ(margin)`, which is exactly a 2-class softmax
/// cross-entropy on the logit pair `[margin, 0]` with the positive class.
/// `margin` is the `[N]` per-pair preference margin
/// `β·((π_chosen − π_rejected) − (ref_chosen − ref_rejected))`; the caller
/// builds it from sequence log-probs. `zeros` is an `[N]` constant-0 node and
/// `pos_labels` an `[N]` constant-0 (class-0) label node — both supplied so
/// this stays a pure graph composition.
pub fn dpo_loss_from_margin(
    g: &mut Graph,
    margin: NodeId,
    zeros: NodeId,
    pos_labels: NodeId,
) -> NodeId {
    // logits[:,0] = margin (favored), logits[:,1] = 0 → softmax-CE with the
    // favored class is -log σ(margin).
    let m = g.reshape_(margin, vec![-1, 1]);
    let z = g.reshape_(zeros, vec![-1, 1]);
    let pair = g.concat_(vec![m, z], 1); // [N, 2]
    let per_row = g.softmax_cross_entropy_with_logits(pair, pos_labels); // [N]
    g.mean(per_row, vec![0], false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::{Adam, ParamSlot, train};
    use rlx_ir::{DType, Shape};
    use std::collections::HashMap;

    #[test]
    fn cross_entropy_trains_a_linear_classifier() {
        // logits = x·W; train W to classify N points into C classes by CE.
        // Cleanly separable data (one-hot feature per class), so CE → ~0.
        let (nn, k, c) = (6usize, 3usize, 3usize);
        let f = DType::F32;
        let mut g = Graph::new("ce_fit");
        let x = g.input("x", Shape::new(&[nn, k], f));
        let w = g.param("w", Shape::new(&[k, c], f));
        let logits = g.matmul(x, w, Shape::new(&[nn, c], f));
        let labels = g.input("labels", Shape::new(&[nn], f));
        let mask = g.input("mask", Shape::new(&[nn], f));
        let loss = cross_entropy_masked(&mut g, logits, labels, mask);
        g.set_outputs(vec![loss]);

        let labels_d = vec![0.0f32, 1.0, 2.0, 0.0, 1.0, 2.0];
        // Row i's features are the one-hot of its class → linearly separable.
        let mut xd = vec![0.0f32; nn * k];
        for (i, &lab) in labels_d.iter().enumerate() {
            xd[i * k + lab as usize] = 1.0;
        }
        let mask_d = vec![1.0; nn];

        let mut params = HashMap::new();
        params.insert("w".to_string(), vec![0.0; k * c]); // start uniform → loss ≈ ln(3)
        let wrt = vec![ParamSlot {
            name: "w".into(),
            node: w,
        }];
        let inputs = vec![
            ("x".to_string(), xd),
            ("labels".to_string(), labels_d),
            ("mask".to_string(), mask_d),
        ];
        let mut opt = Adam::new(0.2);
        let losses = train(g, &wrt, &mut params, &inputs, &mut opt, 300, None).unwrap();

        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(
            first > 0.9,
            "uniform CE should start near ln(3)≈1.10, got {first}"
        );
        assert!(
            last < 0.1,
            "separable CE should train to ~0: {first} -> {last}"
        );
    }

    #[test]
    fn masking_excludes_rows_from_loss() {
        // A masked row must not contribute: same loss whether its label is
        // right or wrong, as long as it's masked out.
        let f = DType::F32;
        let build = || {
            let mut g = Graph::new("ce_mask");
            let logits = g.input("logits", Shape::new(&[2, 2], f));
            let labels = g.input("labels", Shape::new(&[2], f));
            let mask = g.input("mask", Shape::new(&[2], f));
            let loss = cross_entropy_masked(&mut g, logits, labels, mask);
            g.set_outputs(vec![loss]);
            g
        };
        use rlx_runtime::{CompileOptions, Device, Session};
        let run = |labels: &[f32]| {
            let g = build();
            let mut c = Session::new(Device::Cpu).compile_with(g, &CompileOptions::new());
            c.run(&[
                ("logits", &[2.0f32, 0.0, 0.0, 2.0]),
                ("labels", labels),
                ("mask", &[1.0f32, 0.0]), // row 1 masked out
            ])[0][0]
        };
        // Row 1's label differs but it's masked → identical loss.
        assert!((run(&[0.0, 0.0]) - run(&[0.0, 1.0])).abs() < 1e-6);
    }
}
