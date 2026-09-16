//! The lens VJP against a real Qwen3.5/3.6 decoder block.
//!
//! `rlx-autodiff`'s own tests verify each op's VJP in isolation (attention,
//! RoPE, RMSNorm by rank). This verifies the *composition*: a full trunk block
//! as the model builder assembles it — RMSNorm, Q/K per-head norm, RoPE,
//! grouped-query attention or a gated delta-net, SwiGLU FFN, residuals.
//!
//! `build_qwen35_layer_probe_graph` takes the incoming residual as a named
//! graph input (`trunk_h`), so a single block needs only `Wrt::Leaf`. The
//! interior taps that need `Wrt::Output` come with the full stack.
//!
//! # On finite differences here
//!
//! Qwen3 applies RMSNorm **per attention head**. RMSNorm's gradient carries a
//! `1/rms` factor, so a head whose components happen to be nearly equal — RMS
//! ~4e-4 against components of ~4e-4 — amplifies that block's gradient by a
//! factor of thousands. At such a point *no* finite-difference step is valid:
//! small enough to be a local derivative and f32 rounding swamps it; large
//! enough to clear rounding and it is a several-hundred-percent perturbation.
//! The autodiff is right there and the finite difference is not.
//!
//! So the oracle screens itself: each central difference is computed at two
//! step sizes, and a point where they disagree is reported as *unusable* rather
//! than compared. `head_dim` is 4 in `tiny_cfg`, which makes near-degenerate
//! heads common, so a skip rate is expected — the tests assert it stays low
//! enough that the check has not gone vacuous.

use rlx_ir::Graph;
use rlx_jlens::{ResidualShape, Tap, TappedGraph, fill_onehot_cotangent, write_rows};
use rlx_qwen35::synth::{synth_weights, tiny_cfg};
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;

const SEQ: usize = 6;
/// tiny_cfg's `full_attention_interval` is 3, so `(il + 1) % 3 == 0` selects a
/// full-attention block: layer 0 is a gated delta-net, layer 2 is attention.
const GDN_LAYER: usize = 0;
const ATTN_LAYER: usize = 2;

/// Agreement required between autodiff and a *usable* finite difference.
///
/// This is a composition test, and its precision is bounded by the finite
/// difference, not by the gradient. A block's f32 forward carries roughly 1e-5
/// of relative error; a central difference divides that by `2·eps`, putting the
/// noise floor near 1e-3 at the step used here. Gradients in this block are
/// O(1), so 2e-2 is a ~2% check — loose in absolute terms but far tighter than
/// any real defect: the ops verified in `rlx-autodiff`'s own tests agree to
/// 1e-5, and the ill-conditioning investigated above showed up as 20–300%.
const GRAD_TOL: f64 = 2e-2;
/// Two central differences an octave apart must agree this closely for the
/// point to count as usable.
const FD_CONSISTENCY_TOL: f64 = 5e-3;
/// Step size — large enough to clear the f32 noise floor, small enough that the
/// octave check below still rejects points with strong local curvature.
const FD_EPS: f32 = 5e-3;
/// Below this fraction of usable points the oracle is too weak to trust.
const MIN_USABLE_FRACTION: f64 = 0.6;

struct Probe {
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
    d_model: usize,
}

/// Deterministic pseudo-random scalar from a seed and index.
fn hashed(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 40) as f32) / 8_388_608.0 - 1.0
}

/// Replace every parameter with full-rank pseudo-random values.
///
/// `synth_weights` builds each projection from `ramp()` — a strictly linear
/// sequence — so the weight matrices are near rank-2 and every per-head Q/K
/// vector comes out as an arithmetic progression with a tiny RMS. That makes
/// the per-head RMSNorm ill-conditioned at *every* position rather than a few.
/// Rotary tables are randomized along with the rest: `Rope` with arbitrary
/// cos/sin is still a well-defined linear op, and its gradient does not care
/// whether `cos² + sin² = 1`.
fn randomize_params(params: &mut HashMap<String, Vec<f32>>) {
    for (name, data) in params.iter_mut() {
        let seed = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x1000_0000_01b3)
        });
        for (i, v) in data.iter_mut().enumerate() {
            *v = 0.25 * hashed(seed, i);
        }
    }
}

fn probe(layer: usize, batch: usize) -> Probe {
    let cfg = tiny_cfg();
    let weights = synth_weights(&cfg);
    let (graph, mut params, packed) =
        rlx_qwen35::build_qwen35_layer_probe_graph(&cfg, weights, layer, batch, SEQ, false)
            .expect("build layer probe");
    assert!(
        packed.is_empty(),
        "synthetic weights should not produce packed params; \
         a packed path would need set_param_typed feeding"
    );
    randomize_params(&mut params);
    Probe {
        graph,
        params,
        d_model: cfg.hidden_size,
    }
}

fn compile(graph: Graph, params: &HashMap<String, Vec<f32>>) -> CompiledGraph {
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    for (name, data) in params {
        compiled.set_param(name, data);
    }
    compiled
}

fn residual_input(shape: ResidualShape) -> Vec<f32> {
    (0..shape.elements())
        .map(|i| 0.5 * hashed(0x5eed, i))
        .collect()
}

/// Replica `b` of a `[batch, seq, d_model]` buffer.
fn replica(buf: &[f32], shape: ResidualShape, b: usize) -> &[f32] {
    &buf[b * shape.seq * shape.d_model..][..shape.seq * shape.d_model]
}

/// Central difference of `f` along coordinate `i`, screened for usability.
///
/// Returns `None` where the estimate is not stable across a halving of the
/// step — the signature of a point where curvature or rounding dominates and
/// the finite difference is not measuring a derivative.
fn fd_checked(f: &mut dyn FnMut(&[f32]) -> f64, x: &[f32], i: usize, eps: f32) -> Option<f64> {
    let mut at = |step: f32| -> f64 {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += step;
        xm[i] -= step;
        (f(&xp) - f(&xm)) / (2.0 * step as f64)
    };
    let coarse = at(eps);
    let fine = at(eps * 0.5);
    ((coarse - fine).abs() <= FD_CONSISTENCY_TOL).then_some(fine)
}

/// Assert `grad` matches the finite-difference oracle wherever the oracle is
/// usable, and that enough of it was usable to mean something.
fn compare(grad: &[f32], f: &mut dyn FnMut(&[f32]) -> f64, x: &[f32], label: &str) {
    let mut usable = 0usize;
    let mut worst = 0.0f64;
    for i in 0..x.len() {
        let Some(fd) = fd_checked(f, x, i, FD_EPS) else {
            continue;
        };
        usable += 1;
        let delta = (fd - grad[i] as f64).abs();
        worst = worst.max(delta);
        assert!(
            delta < GRAD_TOL,
            "{label}[{i}]: autodiff {} vs finite-difference {fd} (delta {delta:.2e})",
            grad[i]
        );
    }
    let fraction = usable as f64 / x.len() as f64;
    let pct = fraction * 100.0;
    assert!(
        fraction >= MIN_USABLE_FRACTION,
        "{label}: only {usable}/{} points had a usable finite difference \
         ({pct:.0}%) — the oracle is too weak to trust",
        x.len()
    );
    let magnitude = grad.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        magnitude > 1e-3,
        "{label}: gradient is ~zero (max {magnitude}) — agreement is vacuous"
    );
    eprintln!(
        "{label}: {usable}/{} usable ({pct:.0}%), worst delta {worst:.2e}, \
         max |grad| {magnitude:.4}",
        x.len()
    );
}

/// The `dim_batch` trick puts one output dimension per batch element, which is
/// only valid if replicas don't interact. If they ever did, every Jacobian row
/// would be silently polluted by its neighbours, so assert it directly.
#[test]
fn batch_replicas_are_independent() {
    let batch = 3;
    let p = probe(ATTN_LAYER, batch);
    let shape = ResidualShape::new(batch, SEQ, p.d_model);
    let mut compiled = compile(p.graph, &p.params);

    let base = residual_input(shape);
    let out_base = compiled.run(&[("trunk_h", &base[..])])[0].clone();

    let mut perturbed = base.clone();
    perturbed[7] += 1.0; // replica 0 only
    let out_perturbed = compiled.run(&[("trunk_h", &perturbed[..])])[0].clone();

    for b in 1..batch {
        assert_eq!(
            replica(&out_base, shape, b),
            replica(&out_perturbed, shape, b),
            "replica {b} moved when replica 0 was perturbed — \
             batch elements are not independent, so the dim_batch trick is invalid"
        );
    }
    assert_ne!(
        replica(&out_base, shape, 0),
        replica(&out_perturbed, shape, 0),
        "replica 0 did not move — the perturbation had no effect, test is vacuous"
    );
}

/// A general VJP with an arbitrary (non-one-hot) cotangent.
fn vjp_matches_finite_differences(layer: usize, label: &str) {
    let p = probe(layer, 1);
    let shape = ResidualShape::new(1, SEQ, p.d_model);
    let n = shape.elements();

    let tapped = TappedGraph::new(p.graph.clone(), vec![Tap::at_input(layer, "trunk_h")])
        .expect("tap trunk_h");
    let mut bwd = compile(tapped.vjp_graph(), &p.params);

    let h0 = residual_input(shape);
    // Non-uniform cotangent — a uniform one hides index-order bugs — but
    // supported on a single target position. A cotangent spread over the whole
    // output makes the finite-difference probe `Σ v·y` a sum of ~100 O(1)
    // terms, and the f32 forward's relative error accumulates across all of
    // them before being divided by `2·eps`. Concentrating it keeps the probe
    // O(1) and the oracle usable, while still exercising an arbitrary cotangent.
    let target_pos = SEQ / 2;
    let mut v = vec![0.0f32; n];
    for d in 0..p.d_model {
        v[target_pos * p.d_model + d] = 0.5 + 0.25 * ((d % 7) as f32);
    }
    let grad =
        bwd.run(&[("trunk_h", &h0[..]), ("d_output", &v[..])])[tapped.grad_output_index(0)].clone();
    assert_eq!(grad.len(), n);

    let mut fwd = compile(p.graph, &p.params);
    let mut weighted = |h: &[f32]| -> f64 {
        fwd.run(&[("trunk_h", h)])[0]
            .iter()
            .zip(&v)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    };
    compare(&grad, &mut weighted, &h0, label);
}

#[test]
fn attention_block_vjp_matches_finite_differences() {
    vjp_matches_finite_differences(ATTN_LAYER, "attention block");
}

/// Regression: this block once lost **all** cross-position gradient.
///
/// The VJP through a gated-delta-net block used to be exactly zero for every
/// `(target, source)` pair with `source < target`, while the block is genuinely
/// sensitive there — the same-position gradient stayed correct, so nothing else
/// revealed it. It was not any single op: `Attention`, `Rope`, `RmsNorm`,
/// `GatedDeltaNet` and the depthwise causal conv each verify against finite
/// differences in `rlx-autodiff`'s own tests.
///
/// The cause was in the fusion pipeline, which runs over a backward graph like
/// any other: `Rewriter::copy_node` re-copied nodes that `ensure_mapped` had
/// already hoisted, leaving two `Op::Param` nodes per weight. Binding is by
/// name and reaches one node, so the duplicate read zeros and every gradient
/// term through it vanished. A forward graph never showed it — there the
/// duplicate is dead code. Fixed in `rlx-fusion`; pinned upstream by
/// `rlx-autodiff/tests/fused_shared_input_matmul_grad.rs` and
/// `rlx-fusion/tests/rewriter_no_duplicate_leaves.rs`.
///
/// Keep this test: it is the end-to-end statement of the property that broke,
/// on the block that exposed it.
#[test]
fn gated_delta_net_block_vjp_matches_finite_differences() {
    vjp_matches_finite_differences(GDN_LAYER, "gated-delta-net block");
}

/// The whole estimator, end to end: build `J` for one block with the one-hot
/// cotangent sweep, and compare against a Jacobian assembled by finite
/// differences under the *same* reduction (sum over valid target positions,
/// mean over source positions).
#[test]
fn block_jacobian_matches_finite_differences() {
    // Ragged on purpose: d_model = 16 over dim_batch = 5 gives passes of
    // 5/5/5/1, so the short final pass is exercised.
    let dim_batch = 5;
    let p = probe(ATTN_LAYER, dim_batch);
    let d_model = p.d_model;
    let shape = ResidualShape::new(dim_batch, SEQ, d_model);
    // tiny_cfg can't afford the default skip of 16 on a 6-token prompt.
    let positions: Vec<usize> = (1..SEQ - 1).collect();

    let tapped = TappedGraph::new(p.graph.clone(), vec![Tap::at_input(ATTN_LAYER, "trunk_h")])
        .expect("tap trunk_h");
    let mut bwd = compile(tapped.vjp_graph(), &p.params);

    let h0 = residual_input(shape);
    let mut cotangent = vec![0.0f32; shape.elements()];
    let mut jacobian = vec![0.0f32; d_model * d_model];

    let mut dim_start = 0;
    while dim_start < d_model {
        let n_dims = dim_batch.min(d_model - dim_start);
        fill_onehot_cotangent(&mut cotangent, shape, dim_start, n_dims, &positions).unwrap();
        let outs = bwd.run(&[("trunk_h", &h0[..]), ("d_output", &cotangent[..])]);
        write_rows(
            &outs[tapped.grad_output_index(0)],
            shape,
            dim_start,
            n_dims,
            &positions,
            &mut jacobian,
        )
        .unwrap();
        dim_start += dim_batch;
    }

    // Finite-difference oracle under the estimator's own reduction: perturb one
    // (source position, input dim) at a time in replica 0, sum the response over
    // valid target positions, average over source positions.
    let mut fwd = compile(p.graph, &p.params);
    let mut reference = vec![0.0f64; d_model * d_model];
    let mut usable = 0usize;
    let mut total = 0usize;
    for &src in &positions {
        for j in 0..d_model {
            let idx = src * d_model + j; // replica 0
            for i in 0..d_model {
                total += 1;
                let mut summed = |h: &[f32]| -> f64 {
                    let out = &fwd.run(&[("trunk_h", h)])[0];
                    positions
                        .iter()
                        .map(|&tgt| out[tgt * d_model + i] as f64)
                        .sum()
                };
                if let Some(fd) = fd_checked(&mut summed, &h0, idx, FD_EPS) {
                    usable += 1;
                    reference[i * d_model + j] += fd;
                } else {
                    // Mark unusable so the comparison below skips it.
                    reference[i * d_model + j] = f64::NAN;
                }
            }
        }
    }

    let mut compared = 0usize;
    let mut worst = 0.0f64;
    for i in 0..d_model {
        for j in 0..d_model {
            let k = i * d_model + j;
            if reference[k].is_nan() {
                continue;
            }
            let expected = reference[k] / positions.len() as f64;
            let delta = (expected - jacobian[k] as f64).abs();
            worst = worst.max(delta);
            compared += 1;
            assert!(
                delta < GRAD_TOL,
                "J[{i}][{j}]: estimator {} vs finite-difference {expected}",
                jacobian[k]
            );
        }
    }
    let fraction = usable as f64 / total as f64;
    let pct = fraction * 100.0;
    assert!(
        fraction >= MIN_USABLE_FRACTION,
        "only {usable}/{total} finite differences were usable ({pct:.0}%)"
    );
    let magnitude = jacobian.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        magnitude > 0.1,
        "fitted J is ~zero (max |J| = {magnitude}) — agreement is vacuous"
    );
    eprintln!(
        "block Jacobian: {compared} entries compared, {pct:.0}% of differences usable, \
         max |J| = {magnitude:.4}, worst delta = {worst:.2e}"
    );
}
