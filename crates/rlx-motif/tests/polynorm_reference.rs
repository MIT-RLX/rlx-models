// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! [`rlx_motif::polynorm::emit_poly_norm_mul`] against its host reference, on
//! `RLX_TEST_DEVICE`.
//!
//! PolyNorm is the one activation in this model with no analogue anywhere else
//! in the repo, and it has three ways to go quietly wrong: the norm is over the
//! *powers* of the gate (so `n(x³)` is not `n(x)³`), the two upstream variants
//! differ in whether the product is clamped, and the routed-expert variant takes
//! a different coefficient row per token. All three are checked here.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};
use rlx_motif::polynorm::{PolyNormSpec, emit_poly_norm_mul, poly_norm_row};
use rlx_runtime::Device;
use std::collections::HashMap;

const ROWS: usize = 4;
const WIDTH: usize = 6;

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

/// Run one PolyNorm on `[ROWS, WIDTH]` with a `[coeff_rows, 4]` coefficient
/// table (1 = shared, ROWS = per-token, as the expert gather produces).
fn run(gate: &[f32], up: &[f32], coeff: &[f32], coeff_rows: usize, spec: PolyNormSpec) -> Vec<f32> {
    let f = DType::F32;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert("c".into(), (coeff.to_vec(), vec![coeff_rows, 4]));
    let mut wm = WeightMap::from_tensors(t);
    let built = ModelFlow::new("polynorm")
        .with_profile(CompileProfile::llama32_prefill())
        .input("gate", Shape::new(&[ROWS, WIDTH], f))
        .input("up", Shape::new(&[ROWS, WIDTH], f))
        .plugin_named("poly", move |emit, _prev| {
            let g = emit.flow_input("gate")?.hir_id();
            let u = emit.flow_input("up")?.hir_id();
            let c = emit.load_param("c", false)?;
            let y = emit_poly_norm_mul(emit, "p", g, u, c, WIDTH, spec)?;
            Ok(Some(emit.wrap(y, Shape::new(&[ROWS, WIDTH], f))))
        })
        .output("y")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build polynorm");
    compile_built(built, dev())
        .expect("compile polynorm")
        .run(&[("gate", gate), ("up", up)])
        .into_iter()
        .next()
        .expect("output")
}

fn reference(
    gate: &[f32],
    up: &[f32],
    coeff: &[f32],
    coeff_rows: usize,
    spec: PolyNormSpec,
) -> Vec<f32> {
    (0..ROWS)
        .flat_map(|r| {
            let c = if coeff_rows == 1 { 0 } else { r };
            poly_norm_row(
                &gate[r * WIDTH..(r + 1) * WIDTH],
                &up[r * WIDTH..(r + 1) * WIDTH],
                [
                    coeff[c * 4],
                    coeff[c * 4 + 1],
                    coeff[c * 4 + 2],
                    coeff[c * 4 + 3],
                ],
                spec,
            )
        })
        .collect()
}

fn check(name: &str, coeff_rows: usize, spec: PolyNormSpec, amp: f32) {
    let gate = fill(ROWS * WIDTH, 11, amp);
    let up = fill(ROWS * WIDTH, 29, amp);
    let coeff = fill(coeff_rows * 4, 47, 1.6);
    let want = reference(&gate, &up, &coeff, coeff_rows, spec);
    let got = run(&gate, &up, &coeff, coeff_rows, spec);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    eprintln!(
        "polynorm[{name}]: max |Δ| {max_abs:.3e} (rel {:.2e})",
        max_abs / scale
    );
    assert!(
        max_abs / scale < 1e-5,
        "{name}: PolyNorm graph disagrees with the host reference — max |Δ| {max_abs:.3e}"
    );
}

/// `PolyNormTorch` (dense MLP / shared expert): shared coefficients, no clamp on
/// the product.
#[test]
fn dense_polynorm_matches_reference() {
    check(
        "dense",
        1,
        PolyNormSpec {
            eps: 1e-6,
            hidden_clamp: Some(1e6),
            output_scale: 0.5,
            clamp_result: false,
        },
        1.2,
    );
}

/// `GroupedPolyNorm` (routed experts): one coefficient row per token, product
/// clamped.
#[test]
fn per_expert_polynorm_matches_reference() {
    check(
        "per-expert",
        ROWS,
        PolyNormSpec {
            eps: 1e-6,
            hidden_clamp: Some(1e6),
            output_scale: 0.5,
            clamp_result: true,
        },
        1.2,
    );
}

/// With a clamp tight enough to bite, the two variants must *disagree* — this is
/// the test that would fail if `clamp_result` were ignored.
#[test]
fn result_clamp_is_not_cosmetic() {
    let spec = PolyNormSpec {
        eps: 1e-6,
        hidden_clamp: Some(0.05),
        output_scale: 1.0,
        clamp_result: true,
    };
    let loose = PolyNormSpec {
        clamp_result: false,
        ..spec
    };
    let gate = fill(ROWS * WIDTH, 3, 4.0);
    let up = fill(ROWS * WIDTH, 5, 4.0);
    // Coefficients large enough that |poly| ≫ 1, so poly·up escapes ±0.05 even
    // though both operands were already clamped to it.
    let coeff = vec![3.0, 3.0, 3.0, 2.0];

    check("tight-clamp", 1, spec, 4.0);
    let clamped = run(&gate, &up, &coeff, 1, spec);
    let unclamped = run(&gate, &up, &coeff, 1, loose);
    assert!(
        clamped
            .iter()
            .zip(&unclamped)
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "a ±0.05 clamp on the product changed nothing — clamp_result is being dropped"
    );
    assert!(
        clamped.iter().all(|v| v.abs() <= 0.05 + 1e-6),
        "clamped output escaped the bound"
    );
}

/// `hidden_clamp = None` must skip the clamp nodes entirely and still match.
#[test]
fn unclamped_polynorm_matches_reference() {
    check(
        "unclamped",
        ROWS,
        PolyNormSpec {
            eps: 1e-6,
            hidden_clamp: None,
            output_scale: 1.0,
            clamp_result: true,
        },
        1.2,
    );
}
