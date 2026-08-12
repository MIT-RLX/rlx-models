// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! One [`rlx_motif::gdla::emit_gdla_attention`] block against a host reference,
//! on `RLX_TEST_DEVICE`.
//!
//! GDLA stacks four things that are each individually easy to get subtly wrong,
//! and none of which shows up as a crash:
//!
//! * the Q/K split into a position-free half and a RoPE half, with **one** RoPE
//!   head shared across all KV heads,
//! * GQA by `repeat_interleave` (KV head `g` serves the `H/KV` *consecutive* Q
//!   heads of group `g`) — tiling instead would pair heads with the wrong KV,
//! * the differential regroup: within each group the leading heads are signal
//!   and the last is noise, subtracted with a per-signal-head λ,
//! * V narrower than the scores, which rides zero-padded attention.
//!
//! The test uses `H=6, KV=noise=2, grouped_ratio=2, v_head_dim=6 < head_dim=8`
//! so every one of those is asymmetric enough to catch a transposition.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};
use rlx_motif::gdla::{GdlaDims, ROPE_COS, ROPE_SIN, emit_gdla_attention};
use rlx_runtime::Device;
use std::collections::HashMap;

const HID: usize = 12;
const HEADS: usize = 6;
const KV: usize = 2;
const RATIO: usize = 2; // signal heads per group
const GS: usize = RATIO + 1; // heads per group
const SIG: usize = RATIO * KV;
const HD: usize = 8;
const ROPE: usize = 4;
const NOPE: usize = HD - ROPE;
const VD: usize = 6;
const QL: usize = 5;
const KVL: usize = 7;
const SEQ: usize = 5;
const EPS: f32 = 1e-6;
const SCALE: f32 = 0.4;
const PREFIX: &str = "attn";

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

fn tensors() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut t = HashMap::new();
    let mut put = |k: &str, shape: Vec<usize>, seed: u64| {
        let n: usize = shape.iter().product();
        t.insert(format!("{PREFIX}.{k}"), (fill(n, seed, 0.8), shape));
    };
    put("wq_a.weight", vec![QL, HID], 1);
    put("q_norm.weight", vec![QL], 2);
    put("wq_b.weight", vec![HEADS * HD, QL], 3);
    put("wq_b_gate.weight", vec![SIG * VD, QL], 4);
    put("wkv_a.weight", vec![KVL + ROPE, HID], 5);
    put("kv_norm.weight", vec![KVL], 6);
    put("wkv_b.weight", vec![KV * (NOPE + VD), KVL], 7);
    put("lambda_proj.weight", vec![SIG, HID], 8);
    put("wo.weight", vec![HID, SIG * VD], 9);
    t
}

/// `[seq, ROPE/2]` tables, same shape the model builder produces.
fn rope_tables() -> (Vec<f32>, Vec<f32>) {
    let half = ROPE / 2;
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for pos in 0..SEQ {
        for j in 0..half {
            let inv = 1000f64.powf(-2.0 * j as f64 / ROPE as f64);
            let a = pos as f64 * inv;
            cos.push(a.cos() as f32);
            sin.push(a.sin() as f32);
        }
    }
    (cos, sin)
}

fn matvec(w: &[f32], x: &[f64], out_dim: usize, in_dim: usize) -> Vec<f64> {
    (0..out_dim)
        .map(|o| (0..in_dim).map(|i| x[i] * w[o * in_dim + i] as f64).sum())
        .collect()
}

fn rms(x: &[f64], gamma: &[f32]) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + EPS as f64).sqrt();
    x.iter()
        .zip(gamma)
        .map(|(v, g)| v * inv * *g as f64)
        .collect()
}

/// NeoX half-split rotation over a `ROPE`-wide slice at position `pos`.
fn rope_apply(v: &mut [f64], cos: &[f32], sin: &[f32], pos: usize) {
    let half = ROPE / 2;
    for i in 0..half {
        let (c, s) = (cos[pos * half + i] as f64, sin[pos * half + i] as f64);
        let (a, b) = (v[i], v[i + half]);
        v[i] = a * c - b * s;
        v[i + half] = b * c + a * s;
    }
}

fn reference(
    t: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    x: &[f32],
    window: Option<usize>,
) -> Vec<f32> {
    let (cos, sin) = rope_tables();
    let g = |k: &str| &t[&format!("{PREFIX}.{k}")].0;

    // Per-token projections.
    let mut q = vec![vec![0f64; HEADS * HD]; SEQ]; // [s][h*HD]
    let mut k = vec![vec![0f64; KV * HD]; SEQ];
    let mut v = vec![vec![0f64; KV * VD]; SEQ];
    let mut gate = vec![vec![0f64; SIG * VD]; SEQ];
    let mut lambda = vec![vec![0f64; SIG]; SEQ];
    for s in 0..SEQ {
        let xr: Vec<f64> = x[s * HID..(s + 1) * HID]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let q_lat = rms(&matvec(g("wq_a.weight"), &xr, QL, HID), g("q_norm.weight"));
        let qf = matvec(g("wq_b.weight"), &q_lat, HEADS * HD, QL);
        gate[s] = matvec(g("wq_b_gate.weight"), &q_lat, SIG * VD, QL);
        lambda[s] = matvec(g("lambda_proj.weight"), &xr, SIG, HID);

        let ckv = matvec(g("wkv_a.weight"), &xr, KVL + ROPE, HID);
        let kv_lat = rms(&ckv[..KVL], g("kv_norm.weight"));
        let mut k_pe = ckv[KVL..].to_vec();
        rope_apply(&mut k_pe, &cos, &sin, s);
        let kv_up = matvec(g("wkv_b.weight"), &kv_lat, KV * (NOPE + VD), KVL);

        for h in 0..HEADS {
            let mut head = qf[h * HD..(h + 1) * HD].to_vec();
            let mut pe = head[NOPE..].to_vec();
            rope_apply(&mut pe, &cos, &sin, s);
            head[NOPE..].copy_from_slice(&pe);
            q[s][h * HD..(h + 1) * HD].copy_from_slice(&head);
        }
        for j in 0..KV {
            let slab = &kv_up[j * (NOPE + VD)..(j + 1) * (NOPE + VD)];
            k[s][j * HD..j * HD + NOPE].copy_from_slice(&slab[..NOPE]);
            // The single RoPE head is shared by every KV head.
            k[s][j * HD + NOPE..(j + 1) * HD].copy_from_slice(&k_pe);
            v[s][j * VD..(j + 1) * VD].copy_from_slice(&slab[NOPE..]);
        }
    }

    // Attention, per query token and head.
    let mut ctx = vec![vec![0f64; HEADS * VD]; SEQ];
    for i in 0..SEQ {
        for h in 0..HEADS {
            let kvh = h / (HEADS / KV);
            let lo = match window {
                Some(w) => i.saturating_sub(w),
                None => 0,
            };
            let logits: Vec<f64> = (lo..=i)
                .map(|j| {
                    SCALE as f64
                        * (0..HD)
                            .map(|d| q[i][h * HD + d] * k[j][kvh * HD + d])
                            .sum::<f64>()
                })
                .collect();
            let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
            let z: f64 = exps.iter().sum();
            for (n, j) in (lo..=i).enumerate() {
                let p = exps[n] / z;
                for d in 0..VD {
                    ctx[i][h * VD + d] += p * v[j][kvh * VD + d];
                }
            }
        }
    }

    // Differential regroup + output gate + projection.
    let mut out = vec![0f32; SEQ * HID];
    for s in 0..SEQ {
        let mut flat = [0f64; SIG * VD];
        for grp in 0..KV {
            for c in 0..RATIO {
                let si = grp * RATIO + c;
                let sig_head = grp * GS + c;
                let noise_head = grp * GS + RATIO;
                let lam = 1.0 / (1.0 + (-lambda[s][si]).exp());
                for d in 0..VD {
                    let diff = ctx[s][sig_head * VD + d] - lam * ctx[s][noise_head * VD + d];
                    let gg = 1.0 / (1.0 + (-gate[s][si * VD + d]).exp());
                    flat[si * VD + d] = diff * gg;
                }
            }
        }
        let wo = &t[&format!("{PREFIX}.wo.weight")].0;
        for o in 0..HID {
            let acc: f64 = (0..SIG * VD)
                .map(|i| flat[i] * wo[o * SIG * VD + i] as f64)
                .sum();
            out[s * HID + o] = acc as f32;
        }
    }
    out
}

fn run(x: &[f32], window: Option<usize>) -> Vec<f32> {
    let f = DType::F32;
    let half = ROPE / 2;
    let mut wm = WeightMap::from_tensors(tensors());
    let dims = GdlaDims {
        hidden: HID,
        num_heads: HEADS,
        num_kv_heads: KV,
        grouped_ratio: RATIO,
        head_dim: HD,
        qk_rope_head_dim: ROPE,
        v_head_dim: VD,
        q_lora_rank: QL,
        kv_lora_rank: KVL,
        window,
        score_scale: SCALE,
        eps: EPS,
        seq: SEQ,
    };
    // The emitter reads the SWA tables on windowed layers; feed both names from
    // the same tables so this test isolates the mask, not the RoPE base.
    let (cos_name, sin_name) = if window.is_some() {
        (rlx_motif::SWA_ROPE_COS, rlx_motif::SWA_ROPE_SIN)
    } else {
        (ROPE_COS, ROPE_SIN)
    };
    let built = ModelFlow::new("gdla_block")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", Shape::new(&[1, SEQ, HID], f))
        .input(cos_name, Shape::new(&[SEQ, half], f))
        .input(sin_name, Shape::new(&[SEQ, half], f))
        .plugin_named("gdla", move |emit, _prev| {
            let x = emit.flow_input("x")?.hir_id();
            let y = emit_gdla_attention(emit, PREFIX, x, dims)?;
            Ok(Some(emit.wrap(y, Shape::new(&[1, SEQ, HID], f))))
        })
        .output("y")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build gdla block");
    let (cos, sin) = rope_tables();
    compile_built(built, dev())
        .expect("compile gdla block")
        .run(&[
            ("x", x),
            (cos_name, cos.as_slice()),
            (sin_name, sin.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("output")
}

fn check(name: &str, window: Option<usize>) {
    let x = fill(SEQ * HID, 313, 0.9);
    let want = reference(&tensors(), &x, window);
    let got = run(&x, window);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    eprintln!(
        "gdla[{name}, {:?}]: max |Δ| {max_abs:.3e} (rel {:.2e})",
        dev(),
        max_abs / scale
    );
    // The differential recombination subtracts two nearly-equal attention
    // outputs, so this block is cancellation-prone by construction and its
    // relative error tracks the backend's accumulation order rather than the
    // op count. Measured: CPU / Metal / MLX / CUDA / Vulkan / macOS-wgpu land at
    // ≤4e-7, ROCm at ~1e-4. A genuine failure is not subtle — Linux wgpu under
    // its arena-reuse bug reads 1.2 (rel 3.2) — so this threshold separates the
    // two by three orders of magnitude.
    assert!(
        max_abs / scale < 5e-4,
        "{name}: GDLA disagrees with the host reference — max |Δ| {max_abs:.3e}, \
         rel {:.2e}",
        max_abs / scale
    );
}

#[test]
fn gdla_global_matches_host_reference() {
    check("global", None);
}

/// Sliding-window layers keep keys `[q − w, q]`; `w = 2` over a 5-token prompt
/// drops real keys, so this fails loudly if the window is off by one or absent.
#[test]
fn gdla_sliding_window_matches_host_reference() {
    check("sliding", Some(2));
}

/// The window must actually change the result — otherwise the check above would
/// pass even if `MaskKind::SlidingWindow` were being ignored.
#[test]
fn window_changes_the_output() {
    let x = fill(SEQ * HID, 313, 0.9);
    let full = run(&x, None);
    let windowed = run(&x, Some(2));
    let delta = full
        .iter()
        .zip(&windowed)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        delta > 1e-4,
        "a 2-key window changed nothing (Δ {delta:.3e})"
    );
}
