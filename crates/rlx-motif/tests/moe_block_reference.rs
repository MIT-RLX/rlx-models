// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! One [`rlx_motif::moe::emit_motif_moe`] block against a host reference, on
//! `RLX_TEST_DEVICE`.
//!
//! The block is fed through [`rlx_motif::prepare_checkpoint`] from stock
//! upstream tensor names/layouts, so this also pins the contract between the
//! host-side rewrite (coefficient fold + `[E,N,K] → [E,K,N]`) and what the graph
//! reads back — a mismatch there is silent, since both sides are "valid".
//!
//! What makes this router non-generic: selection uses `sigmoid + expert_bias`
//! but the *weights* are the unbiased sigmoids, normalized then scaled by
//! `route_scale`. And each expert carries its own PolyNorm coefficients, which
//! the graph gathers per token — upstream can only do that with an eager Python
//! loop over experts.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};
use rlx_motif::config::POLYNORM_EPS;
use rlx_motif::moe::{MotifMoeDims, emit_motif_moe};
use rlx_motif::polynorm::{PolyNormSpec, poly_norm_row};
use rlx_motif::{MotifConfig, prepare_checkpoint};
use rlx_runtime::Device;
use std::collections::HashMap;

const H: usize = 16;
const INTER: usize = 6;
const E: usize = 8;
const TOPK: usize = 2;
const SEQ: usize = 5;
const PREFIX: &str = "model.layers.0.moe";

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

fn cfg(route_scale: f32) -> MotifConfig {
    MotifConfig::from_json_str(&format!(
        r#"{{"hidden_size":{H},"moe_intermediate_size":{INTER},"num_experts":{E},
            "experts_top_k":{TOPK},"num_hidden_layers":1,"n_dense_first_layers":0,
            "interleave_moe_layer_step":1,"num_shared_experts":1,
            "load_balance_coeff":0.0001,"route_norm":true,"route_scale":{route_scale},
            "score_func":"sigmoid","hidden_act":"poly_norm","polynorm_output_scale":0.5,
            "polynorm_bias_clamp":0.5,"hidden_clamp":1000000.0}}"#
    ))
    .expect("parse moe config")
}

fn tensors() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut t = HashMap::new();
    t.insert(
        format!("{PREFIX}.router.gate.weight"),
        (fill(E * H, 5, 0.6), vec![E, H]),
    );
    t.insert(format!("{PREFIX}.expert_bias"), (fill(E, 9, 0.6), vec![E]));
    t.insert(
        format!("{PREFIX}.experts.gate_up_proj"),
        (fill(E * 2 * INTER * H, 20, 0.6), vec![E, 2 * INTER, H]),
    );
    t.insert(
        format!("{PREFIX}.experts.down_proj"),
        (fill(E * H * INTER, 60, 0.6), vec![E, H, INTER]),
    );
    t.insert(
        format!("{PREFIX}.experts.act_fn.weight"),
        (fill(E * 3, 71, 3.0), vec![E, 3]),
    );
    t.insert(
        format!("{PREFIX}.experts.act_fn.bias"),
        // Deliberately wider than ±bias_clamp so the clamp has to bite.
        (fill(E, 73, 3.0), vec![E, 1]),
    );
    for (k, seed, shape) in [
        ("gate_proj", 201u64, vec![INTER, H]),
        ("up_proj", 202, vec![INTER, H]),
        ("down_proj", 203, vec![H, INTER]),
    ] {
        let n: usize = shape.iter().product();
        t.insert(
            format!("{PREFIX}.shared_experts.{k}.weight"),
            (fill(n, seed, 0.6), shape),
        );
    }
    t.insert(
        format!("{PREFIX}.shared_experts.act_fn.weight"),
        (fill(3, 204, 3.0), vec![3]),
    );
    t.insert(
        format!("{PREFIX}.shared_experts.act_fn.bias"),
        (fill(1, 205, 3.0), vec![1]),
    );
    t
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn expert_spec() -> PolyNormSpec {
    PolyNormSpec {
        eps: POLYNORM_EPS,
        hidden_clamp: Some(1e6),
        output_scale: 0.5,
        clamp_result: true,
    }
}

/// `x @ W_gate_upᵀ → PolyNorm → @ W_downᵀ`, accumulated into `out` with `weight`.
#[allow(clippy::too_many_arguments)]
fn expert_ffn(
    gate_up: &[f32],
    down: &[f32],
    coeff: [f32; 4],
    spec: PolyNormSpec,
    xr: &[f32],
    out: &mut [f32],
    weight: f32,
) {
    // gate_up is [2*INTER, H] (row-major, [out, in]); rows 0..INTER are the gate.
    let proj: Vec<f32> = (0..2 * INTER)
        .map(|o| (0..H).map(|i| xr[i] * gate_up[o * H + i]).sum())
        .collect();
    let act = poly_norm_row(&proj[..INTER], &proj[INTER..], coeff, spec);
    for (o, slot) in out.iter_mut().enumerate() {
        let acc: f32 = (0..INTER).map(|j| act[j] * down[o * INTER + j]).sum();
        *slot += acc * weight;
    }
}

fn reference(t: &HashMap<String, (Vec<f32>, Vec<usize>)>, x: &[f32], route_scale: f32) -> Vec<f32> {
    let router = &t[&format!("{PREFIX}.router.gate.weight")].0;
    let bias = &t[&format!("{PREFIX}.expert_bias")].0;
    let gate_up = &t[&format!("{PREFIX}.experts.gate_up_proj")].0;
    let down = &t[&format!("{PREFIX}.experts.down_proj")].0;
    let aw = &t[&format!("{PREFIX}.experts.act_fn.weight")].0;
    let ab = &t[&format!("{PREFIX}.experts.act_fn.bias")].0;
    let s_aw = &t[&format!("{PREFIX}.shared_experts.act_fn.weight")].0;
    let s_ab = &t[&format!("{PREFIX}.shared_experts.act_fn.bias")].0;

    let mut out = vec![0f32; SEQ * H];
    for r in 0..SEQ {
        let xr = &x[r * H..(r + 1) * H];
        let scores: Vec<f32> = (0..E)
            .map(|e| sigmoid((0..H).map(|i| xr[i] * router[e * H + i]).sum()))
            .collect();
        let mut order: Vec<usize> = (0..E).collect();
        order.sort_by(|&a, &b| {
            (scores[b] + bias[b])
                .partial_cmp(&(scores[a] + bias[a]))
                .unwrap()
        });
        let picked = &order[..TOPK];
        let sum: f32 = picked.iter().map(|&e| scores[e]).sum::<f32>() + 1e-20;
        let row = &mut out[r * H..(r + 1) * H];
        for &e in picked {
            let coeff = [
                sigmoid(aw[e * 3]),
                sigmoid(aw[e * 3 + 1]),
                sigmoid(aw[e * 3 + 2]),
                ab[e].clamp(-0.5, 0.5),
            ];
            expert_ffn(
                &gate_up[e * 2 * INTER * H..(e + 1) * 2 * INTER * H],
                &down[e * H * INTER..(e + 1) * H * INTER],
                coeff,
                expert_spec(),
                xr,
                row,
                scores[e] / sum * route_scale,
            );
        }
        // Shared expert: separate gate/up tensors, PolyNormTorch (bias unclamped,
        // product unclamped).
        let g = &t[&format!("{PREFIX}.shared_experts.gate_proj.weight")].0;
        let u = &t[&format!("{PREFIX}.shared_experts.up_proj.weight")].0;
        let d = &t[&format!("{PREFIX}.shared_experts.down_proj.weight")].0;
        let mut fused = vec![0f32; 2 * INTER * H];
        fused[..INTER * H].copy_from_slice(g);
        fused[INTER * H..].copy_from_slice(u);
        expert_ffn(
            &fused,
            d,
            [
                sigmoid(s_aw[0]),
                sigmoid(s_aw[1]),
                sigmoid(s_aw[2]),
                s_ab[0],
            ],
            PolyNormSpec {
                clamp_result: false,
                ..expert_spec()
            },
            xr,
            row,
            1.0,
        );
    }
    out
}

fn run(x: &[f32], route_scale: f32) -> Vec<f32> {
    let f = DType::F32;
    let c = cfg(route_scale);
    let mut wm = WeightMap::from_tensors(tensors());
    prepare_checkpoint(&c, &mut wm).expect("prepare checkpoint");

    let dims = MotifMoeDims {
        hidden: H,
        moe_inter: INTER,
        num_experts: E,
        top_k: TOPK,
        route_scale,
        has_expert_bias: true,
        has_shared_expert: true,
        poly: expert_spec(),
        seq: SEQ,
    };
    let built = ModelFlow::new("motif_moe_block")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", Shape::new(&[1, SEQ, H], f))
        .plugin_named("moe", move |emit, _prev| {
            let x = emit.flow_input("x")?.hir_id();
            let y = emit_motif_moe(emit, PREFIX, x, dims)?;
            Ok(Some(emit.wrap(y, Shape::new(&[1, SEQ, H], f))))
        })
        .output("y")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build moe block");
    compile_built(built, dev())
        .expect("compile moe block")
        .run(&[("x", x)])
        .into_iter()
        .next()
        .expect("output")
}

fn check(route_scale: f32) {
    let x = fill(SEQ * H, 777, 0.6);
    let want = reference(&tensors(), &x, route_scale);
    let got = run(&x, route_scale);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    eprintln!(
        "motif-moe [{:?}, route_scale={route_scale}]: max |Δ| {max_abs:.3e} (rel {:.2e})",
        dev(),
        max_abs / scale
    );
    assert!(
        max_abs / scale < 1e-5,
        "MoE block (route_scale={route_scale}) disagrees with the host reference — \
         max |Δ| {max_abs:.3e}, rel {:.2e}",
        max_abs / scale
    );
}

#[test]
fn moe_block_matches_host_reference() {
    check(2.0);
}

/// Control: `route_scale = 0` zeroes the routed experts, leaving only the shared
/// one — so a failure here is in the shared PolyNorm MLP, not the router.
#[test]
fn shared_expert_matches_host_reference() {
    check(0.0);
}

/// The routed experts must actually contribute — otherwise the control above
/// would pass for the wrong reason.
#[test]
fn routed_experts_contribute() {
    let x = fill(SEQ * H, 777, 0.6);
    let full = run(&x, 2.0);
    let shared_only = run(&x, 0.0);
    let delta = full
        .iter()
        .zip(&shared_only)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        delta > 1e-3,
        "routed path contributed nothing (Δ {delta:.3e})"
    );
}
