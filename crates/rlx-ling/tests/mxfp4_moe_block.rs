// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! **MXFP4 expert-bank layout gate**, at the level where it can actually bite.
//!
//! A whole-model cosine cannot police this. Measured on the 4-layer synthetic
//! Ling in `mxfp4_model.rs`, max relative deviation from the f32 model is:
//!
//! ```text
//!   correct        1.87e-3
//!   gate/up swap   1.95e-3     ← a real layout bug, indistinguishable
//!   expert shift   3.23e-3
//!   all experts 0  2.63e-2
//! ```
//!
//! The routed experts are ~2.6% of that model's output, and `silu(g)·u` vs
//! `silu(u)·g` on symmetric random weights are statistically alike, so the
//! classic swap hides. Here the MoE block **is** the output, and the reference
//! is computed on the *round-tripped* weights — so quantization error cancels
//! and the correct answer lands at f32-accumulation noise. Every layout mistake
//! then shows up as an O(1) deviation, three orders of magnitude away.
//!
//! `RLX_TEST_DEVICE=metal|mlx|cuda|…` runs it on a backend.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::mxfp4_pack::{GROUP_SIZE, round_trip};
use rlx_core::weight_map::WeightMap;
use rlx_deepseek::moe::{DeepseekMoeDims, emit_deepseek_moe};
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};
use rlx_ling::weights::{PackedBanks, upload_packed_banks};
use rlx_runtime::Device;
use std::collections::HashMap;

// Multiples of the MXFP4 group size (32) — a smaller H/INTER takes the dense
// fallback and tests nothing.
const H: usize = 64;
const INTER: usize = 32;
const E: usize = 8;
const TOPK: usize = 2;
/// Default row count. `MANY_ROWS` deliberately exceeds CUDA's amortized-kernel
/// register cap (16 rows) so the row-chunking path is covered — that kernel
/// silently computes only its first 16 rows if handed more.
const SEQ: usize = 5;
const MANY_ROWS: usize = 40;
const SCALING: f32 = 2.5;

/// Expert weights are stored **already round-tripped through MXFP4**, so the
/// host reference and the packed kernel are quantizing identical values and the
/// only residual difference is f32 accumulation order.
fn tensors() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut t = HashMap::new();
    t.insert("m.gate.weight".into(), (fill(E * H, 5), vec![E, H]));
    t.insert("m.gate.expert_bias".into(), (fill(E, 9), vec![E]));
    for ei in 0..E {
        for (part, seed, rows, k) in [
            ("gate_proj", 20 + ei as u64, INTER, H),
            ("up_proj", 60 + ei as u64, INTER, H),
            ("down_proj", 100 + ei as u64, H, INTER),
        ] {
            let w = round_trip(&fill(rows * k, seed), rows, k, GROUP_SIZE);
            t.insert(format!("m.experts.{ei}.{part}.weight"), (w, vec![rows, k]));
        }
    }
    // The shared expert stays dense f32 in both paths.
    for (k, seed, shape) in [
        ("gate_proj", 201u64, vec![INTER, H]),
        ("up_proj", 202, vec![INTER, H]),
        ("down_proj", 203, vec![H, INTER]),
    ] {
        let n: usize = shape.iter().product();
        t.insert(
            format!("m.shared_experts.{k}.weight"),
            (fill(n, seed), shape),
        );
    }
    t
}

fn swiglu_down(
    t: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    xr: &[f32],
    out: &mut [f32],
    weight: f64,
) {
    let g = &t[&format!("{prefix}.gate_proj.weight")].0;
    let u = &t[&format!("{prefix}.up_proj.weight")].0;
    let d = &t[&format!("{prefix}.down_proj.weight")].0;
    let mut hid = [0f64; INTER];
    for (j, h) in hid.iter_mut().enumerate() {
        let (mut gg, mut uu) = (0f64, 0f64);
        for (i, &xi) in xr.iter().enumerate() {
            gg += xi as f64 * g[j * H + i] as f64;
            uu += xi as f64 * u[j * H + i] as f64;
        }
        *h = (gg / (1.0 + (-gg).exp())) * uu; // silu(gate) * up
    }
    for (o, slot) in out.iter_mut().enumerate() {
        let mut acc = 0f64;
        for (j, &hj) in hid.iter().enumerate() {
            acc += hj * d[o * INTER + j] as f64;
        }
        *slot += (acc * weight) as f32;
    }
}

/// `n_group = topk_group = 1`, so the group limit is inert and this is plain top-k.
fn reference(t: &HashMap<String, (Vec<f32>, Vec<usize>)>, x: &[f32], seq: usize) -> Vec<f32> {
    let router = &t["m.gate.weight"].0;
    let bias = &t["m.gate.expert_bias"].0;
    let mut out = vec![0f32; seq * H];
    for r in 0..seq {
        let xr = &x[r * H..(r + 1) * H];
        let scores: Vec<f64> = (0..E)
            .map(|e| {
                let mut d = 0f64;
                for (i, &xi) in xr.iter().enumerate() {
                    d += xi as f64 * router[e * H + i] as f64;
                }
                1.0 / (1.0 + (-d).exp())
            })
            .collect();
        let mut order: Vec<usize> = (0..E).collect();
        order.sort_by(|&a, &b| {
            (scores[b] + bias[b] as f64)
                .partial_cmp(&(scores[a] + bias[a] as f64))
                .unwrap()
        });
        let picked = &order[..TOPK];
        let sum: f64 = picked.iter().map(|&e| scores[e]).sum::<f64>() + 1e-20;
        let row = &mut out[r * H..(r + 1) * H];
        for &e in picked {
            let w = scores[e] / sum * SCALING as f64;
            swiglu_down(t, &format!("m.experts.{e}"), xr, row, w);
        }
        swiglu_down(t, "m.shared_experts", xr, row, 1.0);
    }
    out
}

/// Pack the block's experts into the `[E, N, K]` banks the op reads.
fn pack(t: &HashMap<String, (Vec<f32>, Vec<usize>)>) -> PackedBanks {
    use rlx_core::mxfp4_pack::quantize_rows;
    let mut b = PackedBanks {
        gate_up_codes: Vec::new(),
        gate_up_scales: Vec::new(),
        down_codes: Vec::new(),
        down_scales: Vec::new(),
    };
    for ei in 0..E {
        for part in ["gate_proj", "up_proj"] {
            let q = quantize_rows(
                &t[&format!("m.experts.{ei}.{part}.weight")].0,
                INTER,
                H,
                GROUP_SIZE,
            );
            b.gate_up_codes.extend_from_slice(&q.codes);
            b.gate_up_scales.extend_from_slice(&q.scales_bf16());
        }
        let q = quantize_rows(
            &t[&format!("m.experts.{ei}.down_proj.weight")].0,
            H,
            INTER,
            GROUP_SIZE,
        );
        b.down_codes.extend_from_slice(&q.codes);
        b.down_scales.extend_from_slice(&q.scales_bf16());
    }
    b
}

fn run(device: Device, x: &[f32], banks: &PackedBanks, seq: usize) -> Vec<f32> {
    let f = DType::F32;
    let mut wm = WeightMap::from_tensors(tensors());
    let (b, s) = wm.take("m.gate.expert_bias").unwrap();
    wm.insert("m.gate.e_score_correction_bias", b, s);

    let dims = DeepseekMoeDims {
        hidden: H,
        moe_inter: INTER,
        n_routed: E,
        top_k: TOPK,
        n_group: 1,
        topk_group: 1,
        routed_scaling: SCALING,
        shared_inter: INTER,
        seq,
        experts_pretransposed: false,
        mxfp4_group: Some(GROUP_SIZE as u32),
    };
    let built = ModelFlow::new("mxfp4_moe_block")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", Shape::new(&[1, seq, H], f))
        .plugin_named("moe", move |emit, _prev| {
            let x = emit.flow_input("x")?.hir_id();
            let y = emit_deepseek_moe(emit, "m", x, dims)?;
            Ok(Some(emit.wrap(y, Shape::new(&[1, seq, H], f))))
        })
        .output("y")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build mxfp4 moe block");
    let mut compiled = compile_built(built, device).expect("compile mxfp4 moe block");
    upload_packed_banks(&mut compiled, "m", banks);
    compiled
        .run(&[("x", x)])
        .into_iter()
        .next()
        .expect("output")
}

/// Tolerance for "the layout is right". f32 accumulation lands at ~2e-7; Metal
/// stages the activation as `half` inside its MXFP4 kernels (measured worth 11%
/// of prefill there, and the weight it multiplies carries only ~3.3 mantissa
/// bits) which costs ~1e-4. Both sit far below the smallest layout mistake this
/// file can produce — a gate/up swap shows at 3.8e-3 — so the gate keeps its
/// teeth either way; `layout_mistakes_are_caught` asserts that margin directly
/// rather than trusting this constant.
fn layout_tol(device: Device) -> f32 {
    if matches!(device, Device::Metal) {
        5e-4
    } else {
        1e-5
    }
}

fn rel_dev(device: Device, banks: &PackedBanks) -> f32 {
    rel_dev_seq(device, banks, SEQ)
}

fn rel_dev_seq(device: Device, banks: &PackedBanks, seq: usize) -> f32 {
    let x = fill(seq * H, 777);
    let want = reference(&tensors(), &x, seq);
    let got = run(device, &x, banks, seq);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    max_abs / scale
}

#[test]
fn packed_bank_layout_matches_the_reference() {
    let d = dev();
    let rel = rel_dev(d, &pack(&tensors()));
    eprintln!("MXFP4 MoE block on {d:?}: rel dev = {rel:.3e}");
    assert!(
        rel < layout_tol(d),
        "MXFP4 MoE block on {d:?} disagrees with the host reference by {rel:.3e} — \
         the weights are pre-round-tripped, so this is a layout bug, not quantization"
    );
}

/// Each mutation is a layout mistake the packed path could plausibly make.
/// They must all be caught by a wide margin — that margin is what makes the
/// threshold above meaningful.
#[test]
fn layout_mistakes_are_caught() {
    let d = dev();
    let ok = rel_dev(d, &pack(&tensors()));
    let per_expert_gu = 2 * INTER * (H / 2);

    let mut swapped = pack(&tensors());
    for e in 0..E {
        let base = e * per_expert_gu;
        let (g, u) =
            swapped.gate_up_codes[base..base + per_expert_gu].split_at_mut(per_expert_gu / 2);
        g.swap_with_slice(u);
    }

    let mut shifted = pack(&tensors());
    shifted.gate_up_codes.rotate_left(per_expert_gu);

    let mut transposed = pack(&tensors());
    // Feeding a `[E, K, N]` bank where `[E, N, K]` is expected: for the square-ish
    // down bank a byte-level transpose is the same corruption the wrong
    // orientation produces.
    transposed.down_codes.reverse();

    for (name, banks) in [
        ("gate/up halves swapped", swapped),
        ("expert stride shifted", shifted),
        ("down bank transposed", transposed),
    ] {
        let rel = rel_dev(d, &banks);
        eprintln!("  {name}: rel dev = {rel:.3e} (correct {ok:.3e})");
        assert!(
            rel > 5.0 * layout_tol(d),
            "{name} moved the block output by only {rel:.3e} (correct {ok:.3e}, \
             pass threshold {:.1e}) — this gate cannot see layout bugs",
            layout_tol(d)
        );
    }
}

/// **Row-chunking guard.** CUDA's amortized MXFP4 kernel holds one accumulator
/// per row in registers and caps at 16 (`MLX_AMORT_MAXR`); handed more, it
/// computes only the first 16 and leaves the rest as whatever the arena held.
/// The launcher chunks rows to stay under that cap. Every other test here runs
/// 5 rows, so without this one the chunking is never executed and a real prefill
/// (seq 64) would return garbage rows 16.. on CUDA.
#[test]
fn many_rows_are_all_computed() {
    let d = dev();
    let rel = rel_dev_seq(d, &pack(&tensors()), MANY_ROWS);
    eprintln!("MXFP4 MoE block, {MANY_ROWS} rows on {d:?}: rel dev = {rel:.3e}");
    assert!(
        rel < layout_tol(d),
        "{MANY_ROWS} rows on {d:?}: rel dev {rel:.3e} — rows past the amortized \
         kernel's 16-row cap were not computed"
    );
}

/// Block-level weights want a wider spread than the whole-model tests.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    fill_spread(n, seed, 0.6)
}

include!("common/metrics.rs");
