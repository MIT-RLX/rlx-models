// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! One mHC site against a host f64 transcription of
//! `Glm5NextTextHyperConnection.forward` plus the decoder layer's re-expansion.
//!
//! mHC is where GLM-5.3-Flash's residual stream lives, so an error here is an
//! error in every layer, and it is easy to make: rlx already carries a *second*
//! mHC implementation (`rlx_motif::mhc`) whose gates differ in three ways —
//! weighted vs unweighted input norm, sigmoid vs softmax `comb`, and a
//! symmetric vs column-first Sinkhorn schedule. This test pins GLM's variant.
//!
//! The sublayer is the identity, so the emitted value is
//! `post ⊗ collapsed + combᵀ · x`, which depends on all three gates at once.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_glm5next::mhc::{MhcDims, emit_mhc_expand, emit_mhc_gates};
use rlx_ir::{DType, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

const HIDDEN: usize = 6;
const H: usize = 4; // hc_mult
const SEQ: usize = 5;
const ITERS: usize = 20;
const HC_EPS: f32 = 1e-6;
const NORM_EPS: f32 = 1e-5;

const MIX: usize = (2 + H) * H;
const FLAT: usize = H * HIDDEN;

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 1.5
        })
        .collect()
}

fn dims() -> MhcDims {
    MhcDims {
        hidden: HIDDEN,
        streams: H,
        sinkhorn_iters: ITERS,
        eps: HC_EPS,
        norm_eps: NORM_EPS,
        seq: SEQ,
    }
}

struct Params {
    /// `[MIX, FLAT]`, as the checkpoint stores it.
    w_fn: Vec<f32>,
    base: Vec<f32>,
    scale: Vec<f32>,
}

fn params() -> Params {
    Params {
        w_fn: fill(MIX * FLAT, 11),
        base: fill(MIX, 23),
        // Deliberately not all-ones: a dropped `scale` would otherwise pass.
        scale: vec![0.7, 1.3, -0.4],
    }
}

/// `Glm5NextTextHyperConnection.forward` + the decoder layer's re-expansion,
/// transcribed in f64.
fn reference(p: &Params, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; SEQ * H * HIDDEN];
    for t in 0..SEQ {
        let row = &x[t * FLAT..(t + 1) * FLAT];

        // Unweighted RMSNorm over the flattened streams.
        let ms: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / FLAT as f64;
        let inv = 1.0 / (ms + NORM_EPS as f64).sqrt();
        let normed: Vec<f64> = row.iter().map(|v| *v as f64 * inv).collect();

        // mix = normed @ fnᵀ
        let mix: Vec<f64> = (0..MIX)
            .map(|o| {
                (0..FLAT)
                    .map(|i| normed[i] * p.w_fn[o * FLAT + i] as f64)
                    .sum()
            })
            .collect();

        let (s_pre, s_post, s_comb) = (p.scale[0] as f64, p.scale[1] as f64, p.scale[2] as f64);
        let sig = |v: f64| 1.0 / (1.0 + (-v).exp());

        // pre = σ(w·scale + b) + ε
        let pre: Vec<f64> = (0..H)
            .map(|i| sig(mix[i] * s_pre + p.base[i] as f64) + HC_EPS as f64)
            .collect();
        // post = 2·σ(w·scale + b)
        let post: Vec<f64> = (0..H)
            .map(|i| 2.0 * sig(mix[H + i] * s_post + p.base[H + i] as f64))
            .collect();

        // comb = softmax_row(w·scale + b), + ε, then Sinkhorn.
        let mut comb = vec![0f64; H * H];
        for r in 0..H {
            let logits: Vec<f64> = (0..H)
                .map(|c| {
                    let k = r * H + c;
                    mix[2 * H + k] * s_comb + p.base[2 * H + k] as f64
                })
                .collect();
            let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = logits.iter().map(|v| (v - m).exp()).collect();
            let z: f64 = ex.iter().sum();
            for c in 0..H {
                comb[r * H + c] = ex[c] / z + HC_EPS as f64;
            }
        }
        // One column pass, then `iters - 1` rounds of (row, column).
        let norm_axis = |m: &mut [f64], rows: bool| {
            if rows {
                for r in 0..H {
                    let s: f64 = (0..H).map(|c| m[r * H + c]).sum::<f64>() + HC_EPS as f64;
                    for c in 0..H {
                        m[r * H + c] /= s;
                    }
                }
            } else {
                for c in 0..H {
                    let s: f64 = (0..H).map(|r| m[r * H + c]).sum::<f64>() + HC_EPS as f64;
                    for r in 0..H {
                        m[r * H + c] /= s;
                    }
                }
            }
        };
        norm_axis(&mut comb, false);
        for _ in 1..ITERS {
            norm_axis(&mut comb, true);
            norm_axis(&mut comb, false);
        }

        // collapsed = Σ_h pre[h] · x[h]
        let collapsed: Vec<f64> = (0..HIDDEN)
            .map(|d| (0..H).map(|h| pre[h] * row[h * HIDDEN + d] as f64).sum())
            .collect();

        // out = post ⊗ collapsed + combᵀ · x   (identity sublayer)
        for h in 0..H {
            for d in 0..HIDDEN {
                let mixed: f64 = (0..H)
                    .map(|hp| comb[hp * H + h] * row[hp * HIDDEN + d] as f64)
                    .sum();
                out[(t * H + h) * HIDDEN + d] = (post[h] * collapsed[d] + mixed) as f32;
            }
        }
    }
    out
}

fn emitted(p: &Params, x: &[f32]) -> Vec<f32> {
    let d = dims();
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert("m_fn.weight".into(), (p.w_fn.clone(), vec![MIX, FLAT]));
    t.insert("m_base.weight".into(), (p.base.clone(), vec![MIX]));
    t.insert("m_scale.weight".into(), (p.scale.clone(), vec![3]));
    let mut wm = WeightMap::from_tensors(t);

    let stream = Shape::new(&[1, SEQ, H, HIDDEN], DType::F32);
    let flow = ModelFlow::new("mhc")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", stream.clone())
        .plugin_named("site", move |emit, _prev| {
            let x = emit.flow_input("x")?.hir_id();
            let gates = emit_mhc_gates(emit, "m", x, d)?;
            // Identity sublayer: feed the collapsed streams straight back.
            let out = emit_mhc_expand(emit, gates.collapsed, x, gates, d);
            Ok(Some(emit.wrap(out, stream.clone())))
        })
        .output("out");

    let built = flow
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build mhc flow");
    let mut compiled = compile_built(built, dev()).expect("compile");
    let mut outs = compiled.run(&[("x", x)]);
    outs.pop().expect("out")
}

#[test]
fn mhc_site_matches_the_reference() {
    let p = params();
    let x = fill(SEQ * FLAT, 77);
    let got = emitted(&p, &x);
    let want = reference(&p, &x);
    assert_eq!(got.len(), want.len());

    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(1e-3, f32::max);
    assert!(
        worst / scale < 2e-5,
        "mHC site diverges: max |Δ| = {worst} over scale {scale}"
    );
}

/// The column-first Sinkhorn schedule leaves `comb` column-stochastic — the
/// last operation of every round is a column normalization. A row-last schedule
/// would not, so this pins the exact variant the reference runs rather than
/// just "some Sinkhorn".
///
/// The normalization is `c / (sum + hc_eps)`, so columns sum to
/// `sum / (sum + hc_eps) = 1 - O(hc_eps)`, not exactly 1. That residual is the
/// reference's, not a numerical artifact.
#[test]
fn sinkhorn_leaves_columns_normalized() {
    let p = params();
    let x = fill(SEQ * FLAT, 91);
    // With `post = 0` the emitted value is purely `combᵀ · x`; instead of
    // rigging the weights, recompute `comb` in the reference and check it
    // directly — the previous test already ties the reference to the graph.
    let mut comb_cols_ok = 0;
    for t in 0..SEQ {
        let row = &x[t * FLAT..(t + 1) * FLAT];
        let comb = reference_comb(&p, row);
        for c in 0..H {
            let s: f64 = (0..H).map(|r| comb[r * H + c]).sum();
            let dev = (s - 1.0).abs();
            assert!(
                dev < 4.0 * HC_EPS as f64,
                "column {c} of comb at t={t} sums to {s}; expected 1 - O(hc_eps)"
            );
            comb_cols_ok += 1;
        }
    }
    assert_eq!(comb_cols_ok, SEQ * H);
}

/// `comb` alone, extracted from [`reference`] so the stochasticity check can
/// look at it without going through the whole site.
fn reference_comb(p: &Params, row: &[f32]) -> Vec<f64> {
    let ms: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / FLAT as f64;
    let inv = 1.0 / (ms + NORM_EPS as f64).sqrt();
    let normed: Vec<f64> = row.iter().map(|v| *v as f64 * inv).collect();
    let s_comb = p.scale[2] as f64;

    let mut comb = vec![0f64; H * H];
    for r in 0..H {
        let logits: Vec<f64> = (0..H)
            .map(|c| {
                let k = r * H + c;
                let w: f64 = (0..FLAT)
                    .map(|i| normed[i] * p.w_fn[(2 * H + k) * FLAT + i] as f64)
                    .sum();
                w * s_comb + p.base[2 * H + k] as f64
            })
            .collect();
        let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ex: Vec<f64> = logits.iter().map(|v| (v - m).exp()).collect();
        let z: f64 = ex.iter().sum();
        for c in 0..H {
            comb[r * H + c] = ex[c] / z + HC_EPS as f64;
        }
    }
    for c in 0..H {
        let s: f64 = (0..H).map(|r| comb[r * H + c]).sum::<f64>() + HC_EPS as f64;
        for r in 0..H {
            comb[r * H + c] /= s;
        }
    }
    for _ in 1..ITERS {
        for r in 0..H {
            let s: f64 = (0..H).map(|c| comb[r * H + c]).sum::<f64>() + HC_EPS as f64;
            for c in 0..H {
                comb[r * H + c] /= s;
            }
        }
        for c in 0..H {
            let s: f64 = (0..H).map(|r| comb[r * H + c]).sum::<f64>() + HC_EPS as f64;
            for r in 0..H {
                comb[r * H + c] /= s;
            }
        }
    }
    comb
}
