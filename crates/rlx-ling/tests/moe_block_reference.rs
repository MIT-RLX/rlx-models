// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! One [`rlx_deepseek::moe::emit_deepseek_moe`] block against a host f64
//! reference, on `RLX_TEST_DEVICE`.
//!
//! Ling, DeepSeek-V3 and Kimi-K3 all share that emitter, and it is where
//! cross-backend divergence has actually lived: the routed expert path fuses a
//! narrow+SwiGLU over a `GroupedMatMul` output, which several backends got wrong
//! (they dropped `FusedSwiGLU::gate_first` and computed `gate · silu(up)`).
//! Checking the block in isolation localises that in one test instead of a
//! whole-model cosine.
//!
//! `routed_scaling = 0` is run as a control: it zeroes the routed experts, so the
//! block reduces to its shared expert and any failure there is elsewhere.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_deepseek::moe::{DeepseekMoeDims, emit_deepseek_moe};
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

const H: usize = 16;
const INTER: usize = 8;
const E: usize = 8;
const TOPK: usize = 2;
const SEQ: usize = 5;

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
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.6
        })
        .collect()
}

fn tensors() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut t = HashMap::new();
    t.insert("m.gate.weight".into(), (fill(E * H, 5), vec![E, H]));
    t.insert("m.gate.expert_bias".into(), (fill(E, 9), vec![E]));
    for ei in 0..E {
        t.insert(
            format!("m.experts.{ei}.gate_proj.weight"),
            (fill(INTER * H, 20 + ei as u64), vec![INTER, H]),
        );
        t.insert(
            format!("m.experts.{ei}.up_proj.weight"),
            (fill(INTER * H, 60 + ei as u64), vec![INTER, H]),
        );
        t.insert(
            format!("m.experts.{ei}.down_proj.weight"),
            (fill(H * INTER, 100 + ei as u64), vec![H, INTER]),
        );
    }
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
fn reference(t: &HashMap<String, (Vec<f32>, Vec<usize>)>, x: &[f32], scaling: f32) -> Vec<f32> {
    let router = &t["m.gate.weight"].0;
    let bias = &t["m.gate.expert_bias"].0;
    let mut out = vec![0f32; SEQ * H];
    for r in 0..SEQ {
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
            let w = scores[e] / sum * scaling as f64;
            swiglu_down(t, &format!("m.experts.{e}"), xr, row, w);
        }
        swiglu_down(t, "m.shared_experts", xr, row, 1.0);
    }
    out
}

fn run(device: Device, x: &[f32], scaling: f32) -> Vec<f32> {
    run_opt(device, x, scaling, false)
}

fn run_opt(device: Device, x: &[f32], scaling: f32, pre_norm: bool) -> Vec<f32> {
    let f = DType::F32;
    let mut wm = WeightMap::from_tensors(tensors());
    // Stack per-expert tensors exactly as rlx_ling::prepare_checkpoint does.
    let (mut gate_up, mut down) = (Vec::new(), Vec::new());
    for ei in 0..E {
        for part in ["gate_proj", "up_proj"] {
            gate_up
                .extend_from_slice(&wm.take(&format!("m.experts.{ei}.{part}.weight")).unwrap().0);
        }
        down.extend_from_slice(
            &wm.take(&format!("m.experts.{ei}.down_proj.weight"))
                .unwrap()
                .0,
        );
    }
    wm.insert("m.experts.gate_up_proj", gate_up, vec![E, 2 * INTER, H]);
    wm.insert("m.experts.down_proj", down, vec![E, H, INTER]);
    let (b, s) = wm.take("m.gate.expert_bias").unwrap();
    wm.insert("m.gate.e_score_correction_bias", b, s);

    let dims = DeepseekMoeDims {
        hidden: H,
        moe_inter: INTER,
        n_routed: E,
        top_k: TOPK,
        n_group: 1,
        topk_group: 1,
        routed_scaling: scaling,
        shared_inter: INTER,
        seq: SEQ,
        experts_pretransposed: false,
        mxfp4_group: None,
    };
    let built = ModelFlow::new("moe_block")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", Shape::new(&[1, SEQ, H], f))
        .plugin_named("moe", move |emit, _prev| {
            let mut x = emit.flow_input("x")?.hir_id();
            if pre_norm {
                // Feed the MoE from an RMSNorm, as a real decoder layer does —
                // this is what turns on fuse_residual_rms_norm / fuse_rms_norm_reshape
                // upstream of the expert path.
                let gamma = emit.synth_param("m.pn.w", vec![1.0; H], Shape::new(&[H], f));
                let zb = emit.synth_param("m.pn.zb", vec![0.0; H], Shape::new(&[H], f));
                let mut gb = rlx_ir::hir::HirMut::new(emit.hir());
                use rlx_ir::HirGraphExt;
                x = gb.rms_norm(x, gamma, zb, 1e-6);
            }
            let y = emit_deepseek_moe(emit, "m", x, dims)?;
            Ok(Some(emit.wrap(y, Shape::new(&[1, SEQ, H], f))))
        })
        .output("y")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build moe block");
    compile_built(built, device)
        .expect("compile moe block")
        .run(&[("x", x)])
        .into_iter()
        .next()
        .expect("output")
}

fn check(device: Device, scaling: f32) {
    let x = fill(SEQ * H, 777);
    let want = reference(&tensors(), &x, scaling);
    let got = run(device, &x, scaling);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    eprintln!(
        "moe-block [{device:?}, routed_scaling={scaling}]: max |Δ| {max_abs:.3e} (rel {:.2e})",
        max_abs / scale
    );
    assert!(
        max_abs / scale < 1e-5,
        "MoE block on {device:?} (routed_scaling={scaling}) disagrees with the host \
         reference — max |Δ| {max_abs:.3e}, rel {:.2e}",
        max_abs / scale
    );
}

#[test]
fn moe_block_matches_host_reference() {
    check(dev(), 2.5);
}

/// Control: routed experts contribute nothing, so only the shared expert runs.
#[test]
fn moe_shared_expert_matches_host_reference() {
    check(dev(), 0.0);
}

/// Same block, but fed from an RMSNorm the way a decoder layer feeds it.
/// wgpu and ROCm are exact on the bare block yet diverge in the full model, so
/// this checks whether the norm upstream of the expert path is what breaks them.
#[test]
fn moe_block_after_rmsnorm_matches_host_reference() {
    let device = dev();
    let x = fill(SEQ * H, 777);
    // rms_norm with unit gamma, matching the graph.
    let mut normed = vec![0f32; SEQ * H];
    for r in 0..SEQ {
        let row = &x[r * H..(r + 1) * H];
        let ms = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / H as f64;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for (j, v) in row.iter().enumerate() {
            normed[r * H + j] = (*v as f64 * inv) as f32;
        }
    }
    let want = reference(&tensors(), &normed, 2.5);
    let got = run_opt(device, &x, 2.5, true);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    eprintln!(
        "moe-block-after-rmsnorm [{device:?}]: max |Δ| {max_abs:.3e} (rel {:.2e})",
        max_abs / scale
    );
    assert!(
        max_abs / scale < 1e-5,
        "MoE block behind an RMSNorm on {device:?} disagrees with the host reference \
         — max |Δ| {max_abs:.3e}, rel {:.2e}",
        max_abs / scale
    );
}
