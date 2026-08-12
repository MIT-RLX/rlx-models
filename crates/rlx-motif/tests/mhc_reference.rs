// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! One [`rlx_motif::mhc`] block against a host reference, on `RLX_TEST_DEVICE`.
//!
//! MHC replaces the residual stream itself, so an error here is silent — the
//! model still runs, it just mixes the wrong streams. The three things worth
//! pinning: the gates are computed from an RMSNorm of the *flattened* `[E, D]`
//! state (not per stream), the mixing matrix is genuinely doubly stochastic
//! after the Sinkhorn iterations, and the `einsum("bsij,bsjd->bsid")` combine
//! contracts over the right index (`h_res · x`, not `xᵀ · h_res`).

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_motif::mhc::{MhcDims, apply_h_pre, combine, emit_mhc_gates, sinkhorn_reference};
use rlx_runtime::Device;
use std::collections::HashMap;

const H: usize = 5;
const E: usize = 4;
const SEQ: usize = 3;
const ITERS: usize = 20;
const EPS: f32 = 1e-6;

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn fill(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * amp
        })
        .collect()
}

fn dims() -> MhcDims {
    MhcDims {
        hidden: H,
        expansion: E,
        sinkhorn_iters: ITERS,
        h_post_coeff: 1.0,
        seq: SEQ,
    }
}

fn tensors() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut t = HashMap::new();
    t.insert(
        "m.rms_norm.weight".into(),
        (fill(E * H, 2, 1.5), vec![E * H]),
    );
    t.insert(
        "m.proj_pre.weight".into(),
        (fill(E * E * H, 3, 1.0), vec![E, E * H]),
    );
    t.insert(
        "m.proj_post.weight".into(),
        (fill(E * E * H, 4, 1.0), vec![E, E * H]),
    );
    t.insert(
        "m.proj_res.weight".into(),
        (fill(E * E * E * H, 5, 1.0), vec![E * E, E * H]),
    );
    t.insert("m.bias_pre".into(), (fill(E, 6, 1.0), vec![E]));
    t.insert("m.bias_post".into(), (fill(E, 7, 1.0), vec![E]));
    t.insert("m.bias_res".into(), (fill(E * E, 8, 1.0), vec![E, E]));
    t.insert("m.alpha_pre".into(), (vec![0.7], vec![1]));
    t.insert("m.alpha_post".into(), (vec![-0.4], vec![1]));
    t.insert("m.alpha_res".into(), (vec![1.3], vec![1]));
    t
}

/// `(h_pre, h_post, h_res)` for one token row of the `[E, H]` state.
fn reference_gates(
    t: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    x_row: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let flat = E * H;
    let gamma = &t["m.rms_norm.weight"].0;
    let ms = x_row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / flat as f64;
    let inv = 1.0 / (ms + EPS as f64).sqrt();
    let xn: Vec<f64> = x_row
        .iter()
        .zip(gamma)
        .map(|(v, g)| *v as f64 * inv * *g as f64)
        .collect();

    let proj = |key: &str, out: usize| -> Vec<f64> {
        let w = &t[key].0;
        (0..out)
            .map(|o| (0..flat).map(|i| xn[i] * w[o * flat + i] as f64).sum())
            .collect()
    };
    let sig = |v: f64| 1.0 / (1.0 + (-v.clamp(-10.0, 10.0)).exp());

    let a_pre = t["m.alpha_pre"].0[0] as f64;
    let a_post = t["m.alpha_post"].0[0] as f64;
    let a_res = t["m.alpha_res"].0[0] as f64;
    let b_pre = &t["m.bias_pre"].0;
    let b_post = &t["m.bias_post"].0;
    let b_res = &t["m.bias_res"].0;

    let h_pre: Vec<f32> = proj("m.proj_pre.weight", E)
        .into_iter()
        .enumerate()
        .map(|(i, v)| sig(a_pre * v + b_pre[i] as f64) as f32)
        .collect();
    let h_post: Vec<f32> = proj("m.proj_post.weight", E)
        .into_iter()
        .enumerate()
        .map(|(i, v)| sig(a_post * v + b_post[i] as f64) as f32)
        .collect();
    let m: Vec<f32> = proj("m.proj_res.weight", E * E)
        .into_iter()
        .enumerate()
        .map(|(i, v)| (a_res * v + b_res[i] as f64) as f32)
        .collect();
    (h_pre, h_post, sinkhorn_reference(&m, E, ITERS))
}

/// Emit the gates and pack them into one `[1, SEQ, 2E + E²]` output.
fn run_gates(x: &[f32]) -> Vec<f32> {
    let f = DType::F32;
    let width = 2 * E + E * E;
    let mut wm = WeightMap::from_tensors(tensors());
    let d = dims();
    let built = ModelFlow::new("mhc_gates")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", Shape::new(&[1, SEQ, E, H], f))
        .plugin_named("gates", move |emit, _prev| {
            let x = emit.flow_input("x")?.hir_id();
            let g = emit_mhc_gates(emit, "m", x, d)?;
            let mut gb = HirMut::new(emit.hir());
            let res = gb.reshape_(g.h_res, vec![1, SEQ as i64, (E * E) as i64]);
            let packed = gb.concat_(vec![g.h_pre, g.h_post, res], 2);
            Ok(Some(emit.wrap(packed, Shape::new(&[1, SEQ, width], f))))
        })
        .output("y")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build mhc gates");
    compile_built(built, dev())
        .expect("compile mhc gates")
        .run(&[("x", x)])
        .into_iter()
        .next()
        .expect("output")
}

/// Full sublayer wrap: `combine(x, branch)` after `apply_h_pre`.
fn run_combine(x: &[f32], branch: &[f32]) -> Vec<f32> {
    let f = DType::F32;
    let mut wm = WeightMap::from_tensors(tensors());
    let d = dims();
    let built = ModelFlow::new("mhc_combine")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", Shape::new(&[1, SEQ, E, H], f))
        .input("branch", Shape::new(&[1, SEQ, H], f))
        .plugin_named("wrap", move |emit, _prev| {
            let x = emit.flow_input("x")?.hir_id();
            let b = emit.flow_input("branch")?.hir_id();
            let g = emit_mhc_gates(emit, "m", x, d)?;
            let mut gb = HirMut::new(emit.hir());
            let out = combine(&mut gb, x, b, g, d);
            Ok(Some(emit.wrap(out, Shape::new(&[1, SEQ, E, H], f))))
        })
        .output("y")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build mhc combine");
    compile_built(built, dev())
        .expect("compile mhc combine")
        .run(&[("x", x), ("branch", branch)])
        .into_iter()
        .next()
        .expect("output")
}

/// `apply_h_pre` alone: `[1, s, E, H] → [1, s, H]`.
fn run_reduce(x: &[f32]) -> Vec<f32> {
    let f = DType::F32;
    let mut wm = WeightMap::from_tensors(tensors());
    let d = dims();
    let built = ModelFlow::new("mhc_reduce")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", Shape::new(&[1, SEQ, E, H], f))
        .plugin_named("reduce", move |emit, _prev| {
            let x = emit.flow_input("x")?.hir_id();
            let g = emit_mhc_gates(emit, "m", x, d)?;
            let mut gb = HirMut::new(emit.hir());
            let out = apply_h_pre(&mut gb, x, g.h_pre, d);
            Ok(Some(emit.wrap(out, Shape::new(&[1, SEQ, H], f))))
        })
        .output("y")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build mhc reduce");
    compile_built(built, dev())
        .expect("compile mhc reduce")
        .run(&[("x", x)])
        .into_iter()
        .next()
        .expect("output")
}

fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    let m = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    m / want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6)
}

#[test]
fn mhc_gates_match_host_reference() {
    let x = fill(SEQ * E * H, 101, 1.2);
    let got = run_gates(&x);
    let width = 2 * E + E * E;
    let t = tensors();
    for s in 0..SEQ {
        let (pre, post, res) = reference_gates(&t, &x[s * E * H..(s + 1) * E * H]);
        let row = &got[s * width..(s + 1) * width];
        assert!(
            max_rel(&row[..E], &pre) < 1e-5,
            "h_pre row {s}: {:?} vs {pre:?}",
            &row[..E]
        );
        assert!(
            max_rel(&row[E..2 * E], &post) < 1e-5,
            "h_post row {s}: {:?} vs {post:?}",
            &row[E..2 * E]
        );
        assert!(
            max_rel(&row[2 * E..], &res) < 1e-4,
            "h_res row {s}: {:?} vs {res:?}",
            &row[2 * E..]
        );
    }
}

/// The whole point of the Sinkhorn iterations: `h_res` is doubly stochastic, so
/// the mixing neither grows nor shrinks the residual stream on average.
#[test]
fn h_res_is_doubly_stochastic() {
    let x = fill(SEQ * E * H, 101, 1.2);
    let got = run_gates(&x);
    let width = 2 * E + E * E;
    for s in 0..SEQ {
        let res = &got[s * width + 2 * E..(s + 1) * width];
        for i in 0..E {
            let row: f32 = (0..E).map(|j| res[i * E + j]).sum();
            let col: f32 = (0..E).map(|j| res[j * E + i]).sum();
            assert!((row - 1.0).abs() < 1e-3, "token {s} row {i} sums to {row}");
            assert!((col - 1.0).abs() < 1e-3, "token {s} col {i} sums to {col}");
        }
        assert!(
            res.iter().all(|v| *v >= 0.0),
            "Sinkhorn output must be >= 0"
        );
    }
}

/// `x_reduced = Σ_e h_pre[e]·x[e]`.
#[test]
fn apply_h_pre_matches_host_reference() {
    let x = fill(SEQ * E * H, 101, 1.2);
    let got = run_reduce(&x);
    let t = tensors();
    let mut want = vec![0f32; SEQ * H];
    for s in 0..SEQ {
        let (pre, _, _) = reference_gates(&t, &x[s * E * H..(s + 1) * E * H]);
        for e in 0..E {
            for h in 0..H {
                want[s * H + h] += pre[e] * x[s * E * H + e * H + h];
            }
        }
    }
    assert!(max_rel(&got, &want) < 1e-5, "{got:?} vs {want:?}");
}

/// `out[i] = Σ_j h_res[i,j]·x[j] + h_post[i]·branch`. Contracting the wrong
/// index of a doubly stochastic matrix is invisible to the row/col-sum check,
/// so it needs its own reference.
#[test]
fn combine_matches_host_reference() {
    let x = fill(SEQ * E * H, 101, 1.2);
    let branch = fill(SEQ * H, 202, 0.8);
    let got = run_combine(&x, &branch);
    let t = tensors();
    let mut want = vec![0f32; SEQ * E * H];
    for s in 0..SEQ {
        let (_, post, res) = reference_gates(&t, &x[s * E * H..(s + 1) * E * H]);
        for i in 0..E {
            for h in 0..H {
                let mixed: f32 = (0..E)
                    .map(|j| res[i * E + j] * x[s * E * H + j * H + h])
                    .sum();
                want[s * E * H + i * H + h] = mixed + post[i] * branch[s * H + h];
            }
        }
    }
    assert!(
        max_rel(&got, &want) < 1e-4,
        "combine disagrees with reference"
    );
}
