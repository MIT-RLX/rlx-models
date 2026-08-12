// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! The KDA decode step, run one token at a time, must reproduce what the prefill
//! path produces for the whole sequence.
//!
//! This is the property that makes incremental generation trustworthy, and it is
//! entirely self-contained — no real weights, no reference implementation. If the
//! carried short-conv state or the resumed delta-net recurrence is off by even a
//! token, the two disagree immediately.
//!
//! State threading: every piece of state is an explicit graph input that comes
//! back out as part of the packed output, and the harness feeds it forward.
//!
//! That includes the delta-net scan state. `Op::GatedDeltaNet { carry_state }`
//! documents an in-place update, and CPU/Metal/wgpu do mutate the buffer — but
//! MLX substitutes the new state into its evaluation env
//! (`env.insert(node.inputs[5], …)`), which does not survive to the next
//! `run()`. Binding the state to a persistent param therefore silently loses it
//! on MLX. Reading the state node as an output works on all of them, because
//! after the op that node *is* the updated state on every backend.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};
use rlx_ling::kda::{KdaDims, KdaState, emit_kda_attention, emit_kda_decode};
use rlx_runtime::Device;
use std::collections::HashMap;

const HIDDEN: usize = 16;
const HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const KERNEL: usize = 4;
const SEQ: usize = 6;

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
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.5
        })
        .collect()
}

fn dims(seq: usize) -> KdaDims {
    KdaDims {
        hidden: HIDDEN,
        num_heads: HEADS,
        head_dim: HEAD_DIM,
        conv_kernel: KERNEL,
        no_lora: true,
        // The shipped Ling config sets this, so exercise the sigmoid gate branch.
        lower_bound: Some(-5.0),
        eps: 1e-6,
        seq,
        quant: Default::default(),
    }
}

fn weights() -> WeightMap {
    let proj = HEADS * HEAD_DIM;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 7u64;
    let mut put = |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
                   k: &str,
                   shape: Vec<usize>,
                   offset: f32| {
        let n: usize = shape.iter().product();
        seed += 5;
        let v = fill(n, seed).iter().map(|x| offset + x).collect();
        t.insert(k.to_string(), (v, shape));
    };
    for p in ["q_proj", "k_proj", "v_proj", "f_proj", "g_proj"] {
        put(&mut t, &format!("kda.{p}.weight"), vec![proj, HIDDEN], 0.0);
    }
    for c in ["q_conv1d", "k_conv1d", "v_conv1d"] {
        put(
            &mut t,
            &format!("kda.{c}.weight"),
            vec![proj, 1, KERNEL],
            0.0,
        );
    }
    put(&mut t, "kda.b_proj.weight", vec![HEADS, HIDDEN], 0.0);
    // A_log ≈ log U(1,16) so the gate actually saturates.
    put(&mut t, "kda.A_log", vec![HEADS], 2.0);
    put(&mut t, "kda.dt_bias", vec![proj], 0.0);
    put(&mut t, "kda.o_norm.weight", vec![HEAD_DIM], 1.0);
    put(&mut t, "kda.o_proj.weight", vec![HIDDEN, proj], 0.0);
    WeightMap::from_tensors(t)
}

/// Prefill: the whole sequence in one shot.
fn run_prefill(device: Device, x: &[f32]) -> Vec<f32> {
    let f = DType::F32;
    let mut wm = weights();
    let d = dims(SEQ);
    let built = ModelFlow::new("kda_prefill")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", Shape::new(&[1, SEQ, HIDDEN], f))
        .plugin_named("kda", move |emit, _prev| {
            let x = emit.flow_input("x")?.hir_id();
            let y = emit_kda_attention(emit, "kda", x, d)?;
            Ok(Some(emit.wrap(y, Shape::new(&[1, SEQ, HIDDEN], f))))
        })
        .output("y")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build prefill");
    compile_built(built, device)
        .expect("compile prefill")
        .run(&[("x", x)])
        .into_iter()
        .next()
        .expect("prefill output")
}

/// Decode: one token per `run()`, conv states threaded through I/O and the scan
/// state living in a param that the kernel updates in place.
fn run_decode(device: Device, x: &[f32]) -> Vec<f32> {
    let f = DType::F32;
    let proj = HEADS * HEAD_DIM;
    let cstate = Shape::new(&[1, KERNEL - 1, proj], f);
    let mut wm = weights();
    let d = dims(1);

    let built = ModelFlow::new("kda_decode")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", Shape::new(&[1, 1, HIDDEN], f))
        .input("cq", cstate.clone())
        .input("ck", cstate.clone())
        .input("cv", cstate.clone())
        .input("scan", Shape::new(&[1, HEADS, HEAD_DIM, HEAD_DIM], f))
        .plugin_named("kda", move |emit, _prev| {
            let x = emit.flow_input("x")?.hir_id();
            let scan = emit.flow_input("scan")?.hir_id();
            let st = KdaState {
                conv_q: emit.flow_input("cq")?.hir_id(),
                conv_k: emit.flow_input("ck")?.hir_id(),
                conv_v: emit.flow_input("cv")?.hir_id(),
                scan,
            };
            let (y, next) = emit_kda_decode(emit, "kda", x, st, d)?;
            // Concatenate [out | next conv states] into one output so the harness
            // can read them back without multi-output plumbing.
            let mut gb = rlx_ir::hir::HirMut::new(emit.hir());
            use rlx_ir::HirGraphExt;
            let y2 = gb.reshape_(y, vec![1, HIDDEN as i64]);
            let flat = |gb: &mut rlx_ir::hir::HirMut<'_>, n: rlx_ir::HirNodeId| {
                gb.reshape_(n, vec![1, ((KERNEL - 1) * proj) as i64])
            };
            let (a, b, c) = (
                flat(&mut gb, next.conv_q),
                flat(&mut gb, next.conv_k),
                flat(&mut gb, next.conv_v),
            );
            // `scan` after the GDN op is the *updated* state on every backend.
            let scan_flat = gb.reshape_(scan, vec![1, (HEADS * HEAD_DIM * HEAD_DIM) as i64]);
            let packed = gb.concat_(vec![y2, a, b, c, scan_flat], 1);
            let width = HIDDEN + 3 * (KERNEL - 1) * proj + HEADS * HEAD_DIM * HEAD_DIM;
            Ok(Some(emit.wrap(packed, Shape::new(&[1, width], f))))
        })
        .output("packed")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build decode");
    let mut compiled = compile_built(built, device).expect("compile decode");

    let cw = (KERNEL - 1) * proj;
    let mut cq = vec![0f32; cw];
    let mut ck = vec![0f32; cw];
    let mut cv = vec![0f32; cw];
    let sw = HEADS * HEAD_DIM * HEAD_DIM;
    let mut scan = vec![0f32; sw];
    let mut out = Vec::with_capacity(SEQ * HIDDEN);
    for t in 0..SEQ {
        let tok = &x[t * HIDDEN..(t + 1) * HIDDEN];
        let packed = compiled
            .run(&[
                ("x", tok),
                ("cq", cq.as_slice()),
                ("ck", ck.as_slice()),
                ("cv", cv.as_slice()),
                ("scan", scan.as_slice()),
            ])
            .into_iter()
            .next()
            .expect("decode output");
        out.extend_from_slice(&packed[..HIDDEN]);
        cq.copy_from_slice(&packed[HIDDEN..HIDDEN + cw]);
        ck.copy_from_slice(&packed[HIDDEN + cw..HIDDEN + 2 * cw]);
        cv.copy_from_slice(&packed[HIDDEN + 2 * cw..HIDDEN + 3 * cw]);
        scan.copy_from_slice(&packed[HIDDEN + 3 * cw..HIDDEN + 3 * cw + sw]);
    }
    out
}

#[test]
fn kda_decode_matches_prefill() {
    let device = dev();
    let x = fill(SEQ * HIDDEN, 99);
    let want = run_prefill(device, &x);
    let got = run_decode(device, &x);
    assert_eq!(got.len(), want.len());

    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    eprintln!(
        "kda decode vs prefill [{device:?}]: max |Δ| {max_abs:.3e} (rel {:.2e})",
        max_abs / scale
    );

    // Per-position, so a state bug that only shows up after the conv window
    // fills (t >= kernel-1) cannot hide in an aggregate.
    for t in 0..SEQ {
        let a = &got[t * HIDDEN..(t + 1) * HIDDEN];
        let b = &want[t * HIDDEN..(t + 1) * HIDDEN];
        let m = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            m / scale < 1e-5,
            "token {t}: decode diverges from prefill (max |Δ| {m:.3e}) — carried \
             conv state or resumed scan state is wrong"
        );
    }
}

// ── MLA: cached decode must reproduce full-prompt prefill ───────────────────

use rlx_ling::config::AttnGate;
use rlx_ling::mla::{MlaCache, MlaDims, ROPE_COS, ROPE_SIN, emit_mla_attention, emit_mla_decode};

const Q_LORA: usize = 12;
const KV_LORA: usize = 10;
const NOPE: usize = 8;
const ROPE: usize = 4;
const VD: usize = 8;

fn mla_dims(seq: usize) -> MlaDims {
    MlaDims {
        hidden: HIDDEN,
        num_heads: HEADS,
        q_lora_rank: Some(Q_LORA),
        kv_lora_rank: KV_LORA,
        qk_nope_head_dim: NOPE,
        qk_rope_head_dim: ROPE,
        v_head_dim: VD,
        gate: AttnGate::HeadWise,
        eps: 1e-6,
        seq,
        quant: Default::default(),
    }
}

fn mla_weights() -> WeightMap {
    let qk = NOPE + ROPE;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 31u64;
    let mut put = |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
                   k: &str,
                   shape: Vec<usize>,
                   offset: f32| {
        let n: usize = shape.iter().product();
        seed += 5;
        let v = fill(n, seed).iter().map(|x| offset + x).collect();
        t.insert(k.to_string(), (v, shape));
    };
    put(&mut t, "mla.q_a_proj.weight", vec![Q_LORA, HIDDEN], 0.0);
    put(&mut t, "mla.q_a_layernorm.weight", vec![Q_LORA], 1.0);
    put(&mut t, "mla.q_b_proj.weight", vec![HEADS * qk, Q_LORA], 0.0);
    put(
        &mut t,
        "mla.kv_a_proj_with_mqa.weight",
        vec![KV_LORA + ROPE, HIDDEN],
        0.0,
    );
    put(&mut t, "mla.kv_a_layernorm.weight", vec![KV_LORA], 1.0);
    put(
        &mut t,
        "mla.kv_b_proj.weight",
        vec![HEADS * (NOPE + VD), KV_LORA],
        0.0,
    );
    put(&mut t, "mla.g_proj.weight", vec![HEADS, HIDDEN], 0.0);
    put(&mut t, "mla.dense.weight", vec![HIDDEN, HEADS * VD], 0.0);
    WeightMap::from_tensors(t)
}

/// `(cos, sin)` for positions `0..seq`, GPT-J layout — same table the model uses.
fn rope_tables(seq: usize) -> (Vec<f32>, Vec<f32>) {
    let half = ROPE / 2;
    let (mut c, mut s) = (Vec::new(), Vec::new());
    for pos in 0..seq {
        for j in 0..half {
            let inv = 600000f64.powf(-2.0 * j as f64 / ROPE as f64);
            let a = pos as f64 * inv;
            c.push(a.cos() as f32);
            s.push(a.sin() as f32);
        }
    }
    (c, s)
}

fn mla_prefill(device: Device, x: &[f32]) -> Vec<f32> {
    let f = DType::F32;
    let half = ROPE / 2;
    let mut wm = mla_weights();
    let d = mla_dims(SEQ);
    let built = ModelFlow::new("mla_prefill")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", Shape::new(&[1, SEQ, HIDDEN], f))
        .input(ROPE_COS, Shape::new(&[SEQ, half], f))
        .input(ROPE_SIN, Shape::new(&[SEQ, half], f))
        .plugin_named("mla", move |emit, _prev| {
            let x = emit.flow_input("x")?.hir_id();
            let y = emit_mla_attention(emit, "mla", x, d)?;
            Ok(Some(emit.wrap(y, Shape::new(&[1, SEQ, HIDDEN], f))))
        })
        .output("y")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build mla prefill");
    let (c, s) = rope_tables(SEQ);
    compile_built(built, device)
        .expect("compile mla prefill")
        .run(&[("x", x), ("rope_cos", &c), ("rope_sin", &s)])
        .into_iter()
        .next()
        .expect("mla prefill output")
}

fn mla_decode(device: Device, x: &[f32]) -> Vec<f32> {
    let f = DType::F32;
    let qk = NOPE + ROPE;
    let half = ROPE / 2;
    let cap = SEQ; // one slot per prompt position
    let (kw, vw) = (HEADS * qk, HEADS * VD);
    let mut wm = mla_weights();
    let d = mla_dims(1);

    let built = ModelFlow::new("mla_decode")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", Shape::new(&[1, 1, HIDDEN], f))
        .input(ROPE_COS, Shape::new(&[1, half], f))
        .input(ROPE_SIN, Shape::new(&[1, half], f))
        .input("kc", Shape::new(&[1, cap, kw], f))
        .input("vc", Shape::new(&[1, cap, vw], f))
        .input("mask", Shape::new(&[1, cap + 1], f))
        .plugin_named("mla", move |emit, _prev| {
            let x = emit.flow_input("x")?.hir_id();
            let cache = MlaCache {
                k: emit.flow_input("kc")?.hir_id(),
                v: emit.flow_input("vc")?.hir_id(),
                mask: emit.flow_input("mask")?.hir_id(),
                cap,
            };
            let (y, k_new, v_new) = emit_mla_decode(emit, "mla", x, cache, d)?;
            let mut gb = rlx_ir::hir::HirMut::new(emit.hir());
            use rlx_ir::HirGraphExt;
            let y2 = gb.reshape_(y, vec![1, HIDDEN as i64]);
            let k2 = gb.reshape_(k_new, vec![1, kw as i64]);
            let v2 = gb.reshape_(v_new, vec![1, vw as i64]);
            let packed = gb.concat_(vec![y2, k2, v2], 1);
            Ok(Some(
                emit.wrap(packed, Shape::new(&[1, HIDDEN + kw + vw], f)),
            ))
        })
        .output("packed")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build mla decode");
    let mut compiled = compile_built(built, device).expect("compile mla decode");

    let (call, sall) = rope_tables(SEQ);
    let mut kc = vec![0f32; cap * kw];
    let mut vc = vec![0f32; cap * vw];
    let mut mask = vec![0f32; cap + 1];
    mask[cap] = 1.0; // the current token is always valid
    let mut out = Vec::with_capacity(SEQ * HIDDEN);
    for t in 0..SEQ {
        let tok = &x[t * HIDDEN..(t + 1) * HIDDEN];
        let c = &call[t * half..(t + 1) * half];
        let s = &sall[t * half..(t + 1) * half];
        let packed = compiled
            .run(&[
                ("x", tok),
                ("rope_cos", c),
                ("rope_sin", s),
                ("kc", kc.as_slice()),
                ("vc", vc.as_slice()),
                ("mask", mask.as_slice()),
            ])
            .into_iter()
            .next()
            .expect("mla decode output");
        out.extend_from_slice(&packed[..HIDDEN]);
        // Commit this token's k/v into slot t, then let position t attend to it.
        kc[t * kw..(t + 1) * kw].copy_from_slice(&packed[HIDDEN..HIDDEN + kw]);
        vc[t * vw..(t + 1) * vw].copy_from_slice(&packed[HIDDEN + kw..HIDDEN + kw + vw]);
        mask[t] = 1.0;
    }
    out
}

#[test]
fn mla_decode_matches_prefill() {
    let device = dev();
    let x = fill(SEQ * HIDDEN, 4242);
    let want = mla_prefill(device, &x);
    let got = mla_decode(device, &x);
    assert_eq!(got.len(), want.len());

    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "mla decode vs prefill [{device:?}]: max |Δ| {max_abs:.3e} (rel {:.2e})",
        max_abs / scale
    );
    for t in 0..SEQ {
        let a = &got[t * HIDDEN..(t + 1) * HIDDEN];
        let b = &want[t * HIDDEN..(t + 1) * HIDDEN];
        let m = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            m / scale < 1e-5,
            "token {t}: cached decode diverges from prefill (max |Δ| {m:.3e}) — \
             KV cache contents, the validity mask, or the decode RoPE position is wrong"
        );
    }
}

// ── whole model: token-by-token decode vs one-shot prefill ──────────────────

use rlx_ling::flow_decode::{DecodeNames, DecodeSession, ScanState, build_ling_decode_flow_with};
use rlx_ling::{LingConfig, build_ling_text_flow, prepare_checkpoint};

include!("common/model_weights.rs");

#[test]
fn whole_model_decode_matches_prefill() {
    check_whole_model(ScanState::Portable);
}

/// `ScanState::InPlace` keeps the 18.9 MB/token scan state in a param that the
/// GDN kernel mutates, skipping the round-trip. Correct only where that in-place
/// update survives to the next `run()` — this is the test that says whether a
/// given backend qualifies.
#[test]
fn whole_model_decode_matches_prefill_inplace_scan() {
    // MLX and CoreML do not persist `Op::GatedDeltaNet`'s in-place state update
    // across `run()` calls (they substitute it into a per-evaluation env), so
    // `InPlace` is not applicable there — the state would freeze at zero. Skip
    // rather than leave a permanently red test; `ScanState`'s docs say which
    // backends qualify, and the Portable variant above covers all of them.
    if matches!(dev(), Device::Mlx | Device::Ane) {
        eprintln!(
            "skipping InPlace scan on {:?}: no persistent in-place carry",
            dev()
        );
        return;
    }
    check_whole_model(ScanState::InPlace);
}

fn check_whole_model(scan_mode: ScanState) {
    let device = dev();
    let cfg = tiny_model_config();
    let seq = 5usize;
    let ids: Vec<u32> = (0..seq)
        .map(|i| ((i * 7) % cfg.vocab_size) as u32)
        .collect();
    let ids_f: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let v = cfg.vocab_size;

    // Reference: one-shot prefill over the whole prompt.
    let want = {
        let mut wm = model_weights(&cfg);
        prepare_checkpoint(&cfg, &mut wm).expect("prepare");
        let built = build_ling_text_flow(&cfg, &mut wm, seq, true).expect("build prefill");
        let mut c = compile_built(built, device).expect("compile prefill");
        let (cos, sin) = cfg.rope_tables(seq);
        c.run(&[
            ("input_ids", ids_f.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("prefill logits")
    };

    // Decode the same prompt one token at a time.
    let mut wm = model_weights(&cfg);
    prepare_checkpoint(&cfg, &mut wm).expect("prepare");
    let (built, layout) =
        build_ling_decode_flow_with(&cfg, &mut wm, seq, true, scan_mode).expect("build decode");
    let mut compiled = compile_built(built, device).expect("compile decode");
    let names = DecodeNames::new(&cfg);
    let mut sess = DecodeSession::new(&cfg, layout, seq);
    let (cos_all, sin_all) = cfg.rope_tables(seq);
    let half = cfg.qk_rope_head_dim / 2;

    let mut got = Vec::with_capacity(seq * v);
    for t in 0..seq {
        let tok = [ids_f[t]];
        let cos = &cos_all[t * half..(t + 1) * half];
        let sin = &sin_all[t * half..(t + 1) * half];
        let mut inputs: Vec<(&str, &[f32])> = vec![
            ("input_ids", &tok[..]),
            ("rope_cos", cos),
            ("rope_sin", sin),
        ];
        let state_inputs = sess.inputs(&cfg, &names);
        inputs.extend(state_inputs.iter().copied());
        let packed = compiled
            .run(&inputs)
            .into_iter()
            .next()
            .expect("decode output");
        drop(inputs);
        let logits = sess.commit(&packed).expect("commit state");
        got.extend_from_slice(logits);
    }

    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "whole-model decode vs prefill [{device:?}, scan={scan_mode:?}]: max |Δ| {max_abs:.3e} (rel {:.2e}), \
         state {:.1} KB/token",
        max_abs / scale,
        sess.state_bytes() as f64 / 1024.0
    );
    for t in 0..seq {
        let a = &got[t * v..(t + 1) * v];
        let b = &want[t * v..(t + 1) * v];
        let m = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let am = |s: &[f32]| {
            s.iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        assert!(
            m / scale < 1e-4,
            "token {t}: whole-model decode diverges from prefill (max |Δ| {m:.3e}, scan={scan_mode:?})"
        );
        assert_eq!(am(a), am(b), "token {t}: argmax differs");
    }
}
