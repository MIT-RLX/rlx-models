// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! DINO cross-view consistency loss, built in-graph.
//!
//! `H(a, b) = −Σ a·log softmax(b)` between a stop-grad teacher target
//! distribution `a` (host-computed with centering + sharpening, fed as a graph
//! input) and the student log-softmax. Summed over all cross-view pairs
//! `(teacher global t, student view s)` with `s ≠ t`.

use rlx_ir::infer::GraphExt;
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, NodeId};

const F: DType = DType::F32;

/// L2-normalize the last axis (of width `dim`): `x / ‖x‖₂`.
///
/// Implemented on the first-class `rms_norm` op — `RMS(x) = √(mean(x²)+ε)` and
/// `‖x‖₂ = √dim · RMS(x)`, so `x/‖x‖₂ = rms_norm(x)/√dim`. This reuses the
/// backend-verified RmsNorm backward instead of a hand-rolled `sqrt`+broadcast-`div`
/// (whose backward is NaN on Metal).
pub fn l2_normalize(g: &mut Graph, x: NodeId, dim: usize) -> NodeId {
    let gamma = g.full(&[dim], 1.0, F);
    let beta = g.zeros(&[dim], F);
    let rms = g.rms_norm(x, gamma, beta, 1e-12); // x / √(mean(x²)+ε)
    let scale = g.constant((1.0 / (dim as f32).sqrt()) as f64, F);
    g.mul(rms, scale)
}

/// Numerically-stable `log softmax(x)` over the last axis: `log(softmax(x) + ε)`.
/// The `softmax` is already max-shifted; the ε floors its output so
/// `log` never sees an exact `0` — a peaked (low-temperature) softmax can
/// underflow entries to `0` on some backends (Metal), making `log(0)`'s
/// gradient NaN.
fn log_softmax(g: &mut Graph, x: NodeId) -> NodeId {
    let sm = g.sm(x, -1);
    let shape = g.shape(sm).clone();
    let eps = g.constant(1e-9, F);
    let sm_eps = g.add(sm, eps);
    g.activation(Activation::Log, sm_eps, shape)
}

/// Build the DINO cross-view loss (a scalar node).
///
/// - `student_logits` `[Ns, K]` — student projections of all `Ns` views.
/// - `teacher_targets` `[Nt, K]` — stop-grad teacher target distributions
///   (an `Op::Input`; host-computed via [`super::teacher::teacher_targets`]).
/// - `pair_mask` `[Nt, Ns]` — `1` for included `(t, s)` pairs, else `0`
///   (an `Op::Input`; build with [`pair_mask`]).
/// - `temp_s` — student temperature.
/// - `active_pairs` — number of `1`s in `pair_mask` (the mean denominator).
pub fn build_dino_loss(
    g: &mut Graph,
    student_logits: NodeId,
    teacher_targets: NodeId,
    pair_mask: NodeId,
    temp_s: f32,
    active_pairs: usize,
) -> NodeId {
    let inv_ts = g.constant((1.0 / temp_s) as f64, F);
    let scaled = g.mul(student_logits, inv_ts);
    let logsm = log_softmax(g, scaled);
    let logsm_t = g.transpose_(logsm, vec![1, 0]); // [K, Ns]
    let prod = g.mm(teacher_targets, logsm_t); // [Nt, Ns] = Σ_k a·log b
    let masked = g.mul(prod, pair_mask);
    let s = g.sum(masked, vec![0, 1], false); // scalar Σ over pairs
    let neg = g.constant(-1.0 / active_pairs.max(1) as f64, F);
    g.mul(s, neg) // −mean over active pairs
}

/// Aligned DINO cross-entropy: `−mean_n Σ_k target[n,k]·log softmax(student[n]/τ_s)`,
/// where `student` and `target` are row-aligned `[N, K]` (used by GLARE's
/// global/local/regional terms, where correspondence is by position).
pub fn dino_ce_aligned(
    g: &mut Graph,
    student_logits: NodeId,
    targets: NodeId,
    rows: usize,
    temp_s: f32,
) -> NodeId {
    let inv_ts = g.constant((1.0 / temp_s) as f64, F);
    let scaled = g.mul(student_logits, inv_ts);
    let logsm = log_softmax(g, scaled);
    let prod = g.mul(targets, logsm);
    let s = g.sum(prod, vec![0, 1], false);
    let neg = g.constant(-1.0 / rows.max(1) as f64, F);
    g.mul(s, neg)
}

/// Cross-view pair mask `[n_global, n_crops]`: include every (teacher global,
/// student view) pair except the identical view (same crop index). Returns the
/// flat mask and the count of included pairs.
pub fn pair_mask(n_global: usize, n_crops: usize) -> (Vec<f32>, usize) {
    let mut m = vec![1.0f32; n_global * n_crops];
    let mut active = n_global * n_crops;
    for t in 0..n_global {
        // Student view index t is the same crop as teacher global t.
        m[t * n_crops + t] = 0.0;
        active -= 1;
    }
    (m, active)
}
