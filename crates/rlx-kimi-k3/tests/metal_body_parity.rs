// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! CPU-vs-Metal parity for the KimiLinear body. `flow_smoke` only checks
//! finiteness, which hid a Metal miscompile in the KDA-bearing layer body (real
//! weights: token 8582 vs CPU 132719). This builds the same synthetic 2-layer
//! KDA flow and asserts CPU ≈ <RLX_TEST_DEVICE>. Run with
//! `RLX_TEST_DEVICE=metal cargo test -p rlx-kimi-k3 --features metal --test metal_body_parity`.

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_kimi_k3::flow::{
    AttnWeights, FfnWeights, FlowConfig, FlowWeights, LayerWeights, build_kimi_text_flow,
};
use rlx_kimi_k3::kda::{KdaDims, KdaWeights};
use rlx_kimi_k3::mla::{MlaDims, MlaWeights};
use rlx_kimi_k3::moe::{DenseMlpWeights, MoeDims, MoeWeights};
use rlx_runtime::Device;
use std::collections::HashMap;

fn dev() -> Device {
    match std::env::var("RLX_TEST_DEVICE").ok().as_deref() {
        Some("metal") | Some("mtl") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("gpu") | Some("wgpu") => Device::Gpu,
        _ => Device::Cpu,
    }
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.15
        })
        .collect()
}

fn kda_w(d: KdaDims, sd: u64) -> KdaWeights {
    let (hidden, h, hd, proj, k) = (d.hidden, d.num_heads, d.head_dim, d.proj(), d.conv_kernel);
    KdaWeights {
        q_proj: fill(hidden * proj, sd + 1),
        k_proj: fill(hidden * proj, sd + 2),
        v_proj: fill(hidden * proj, sd + 3),
        q_conv: fill(proj * k, sd + 4),
        k_conv: fill(proj * k, sd + 5),
        v_conv: fill(proj * k, sd + 6),
        f_a: fill(hidden * hd, sd + 7),
        f_b: fill(hd * proj, sd + 8),
        dt_bias: fill(proj, sd + 9),
        a_log: fill(hd, sd + 10),
        b_proj: fill(hidden * h, sd + 11),
        g_proj: fill(hidden * proj, sd + 12),
        o_norm: vec![1.0; hd],
        o_proj: fill(proj * hidden, sd + 13),
    }
}

fn moe_w(d: MoeDims, sd: u64) -> MoeWeights {
    let (hidden, l, e, mi, si) = (
        d.hidden,
        d.latent,
        d.num_experts,
        d.moe_inter,
        d.num_shared * d.moe_inter,
    );
    MoeWeights {
        router: fill(hidden * e, sd + 1),
        e_score_bias: fill(e, sd + 2),
        down_latent: fill(hidden * l, sd + 3),
        up_latent: fill(l * hidden, sd + 4),
        routed_norm: vec![1.0; l],
        experts_gate_up: fill(e * l * 2 * mi, sd + 5),
        experts_down: fill(e * mi * l, sd + 6),
        shared_gate: fill(hidden * si, sd + 7),
        shared_up: fill(hidden * si, sd + 8),
        shared_down: fill(si * hidden, sd + 9),
    }
}

fn layer(hidden: usize, attn: AttnWeights, ffn: FfnWeights, sd: u64) -> LayerWeights {
    LayerWeights {
        input_ln: vec![1.0; hidden],
        post_ln: vec![1.0; hidden],
        sa_res_norm: vec![1.0; hidden],
        sa_res_proj: fill(hidden, sd + 1),
        mlp_res_norm: vec![1.0; hidden],
        mlp_res_proj: fill(hidden, sd + 2),
        attn,
        ffn,
    }
}

fn synth(n_layers: usize, seq: usize) -> (FlowWeights, FlowConfig) {
    // REAL KDA dims (hidden 7168, 96 heads × 128) — the bug is size-dependent and
    // does NOT show at hidden=16. Everything else stays tiny (small MoE/vocab) so
    // there's no disk load and it still runs in a couple seconds.
    let (batch, hidden, vocab) = (1usize, 7168usize, 20usize);
    let kda = KdaDims {
        hidden,
        num_heads: 96,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch,
        seq,
    };
    let moe = MoeDims {
        hidden,
        latent: 64,
        moe_inter: 32,
        num_experts: 4,
        top_k: 2,
        num_shared: 1,
        routed_scaling: 1.0,
        eps: 1e-5,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch,
        seq,
    };
    let dense_inter = 64usize;
    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let attn = AttnWeights::Kda(Box::new(kda_w(kda, 100 + i as u64 * 10)));
        let ffn = if i == 0 {
            FfnWeights::Dense(Box::new(DenseMlpWeights {
                gate: fill(hidden * dense_inter, 900),
                up: fill(hidden * dense_inter, 901),
                down: fill(dense_inter * hidden, 902),
            }))
        } else {
            FfnWeights::Moe(Box::new(moe_w(moe, 200 + i as u64 * 10)))
        };
        layers.push(layer(hidden, attn, ffn, 10 + i as u64));
    }
    let w = FlowWeights {
        layers,
        final_norm: vec![1.0; hidden],
        out_res_norm: vec![1.0; hidden],
        out_res_proj: fill(hidden, 800),
        lm_head: fill(hidden * vocab, 801),
    };
    let cfg = FlowConfig {
        hidden,
        vocab,
        attn_res_block_size: 12,
        eps: 1e-5,
        kda,
        mla: rlx_kimi_k3::mla::MlaDims {
            hidden,
            num_heads: 2,
            q_lora_rank: 8,
            kv_lora_rank: 6,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 2,
            v_head_dim: 4,
            eps: 1e-5,
            batch,
            seq,
        },
        moe,
        dense_inter,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch,
        seq,
    };
    (w, cfg)
}

fn run_on(device: Device, w: &FlowWeights, cfg: &FlowConfig, hin: &[f32]) -> Vec<f32> {
    let (batch, seq, hidden) = (cfg.batch, cfg.seq, cfg.hidden);
    let mut hir = HirModule::new("kimi");
    let mut g = HirMut::new(&mut hir);
    let h = g.input("h", Shape::new(&[batch, seq, hidden], DType::F32));
    let mut params = HashMap::new();
    let logits = build_kimi_text_flow(&mut g, &mut params, h, w, cfg).expect("build");
    g.set_outputs(vec![logits]);
    let built = built_from_hir(hir, params).expect("built");
    let mut c = compile_built(built, device).expect("compile");
    c.run(&[("h", hin)]).remove(0)
}

/// Isolate the MLA op at real dims — the full 93-layer Metal run diverges (token
/// 198 vs CPU 276) even after the KDA/GDN fix, and the 2-layer parity only covers
/// KDA; MLA first appears at layer 3.
#[test]
fn mla_only_parity() {
    use rlx_kimi_k3::mla::build_mla_layer;
    let d = dev();
    if matches!(d, Device::Cpu) {
        return;
    }
    let md = MlaDims {
        hidden: 7168,
        num_heads: 96,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
        eps: 1e-5,
        batch: 1,
        seq: 1,
    };
    let (hidden, h, ql, kvl, nope, rope, vd, qk) = (
        md.hidden,
        md.num_heads,
        md.q_lora_rank,
        md.kv_lora_rank,
        md.qk_nope_head_dim,
        md.qk_rope_head_dim,
        md.v_head_dim,
        md.qk_nope_head_dim + md.qk_rope_head_dim,
    );
    let w = MlaWeights {
        q_a_proj: fill(hidden * ql, 1),
        q_a_layernorm: vec![1.0; ql],
        q_b_proj: fill(ql * h * qk, 2),
        kv_a_proj_with_mqa: fill(hidden * (kvl + rope), 3),
        kv_a_layernorm: vec![1.0; kvl],
        kv_b_proj: fill(kvl * h * (nope + vd), 4),
        g_proj: fill(hidden * h * vd, 5),
        o_proj: fill(h * vd * hidden, 6),
    };
    let hin = fill(hidden, 7);
    let run = |device: Device| -> Vec<f32> {
        let mut hir = HirModule::new("mla");
        let mut g = HirMut::new(&mut hir);
        let hn = g.input("h", Shape::new(&[1, 1, hidden], DType::F32));
        let mut params = HashMap::new();
        let out = build_mla_layer(&mut g, &mut params, "self_attn", hn, &w, md).expect("build");
        g.set_outputs(vec![out]);
        let built = built_from_hir(hir, params).expect("built");
        let mut c = compile_built(built, device).expect("compile");
        c.run(&[("h", &hin)]).remove(0)
    };
    let a = run(Device::Cpu);
    let b = run(d);
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    // magnitude + systematic-bias diagnostics
    let mean_abs = a.iter().map(|x| x.abs()).sum::<f32>() / a.len() as f32;
    let (sa, sb) = (
        a.iter().map(|x| x * x).sum::<f32>().sqrt(),
        b.iter().map(|x| x * x).sum::<f32>().sqrt(),
    );
    let rel = max / mean_abs.max(1e-9);
    eprintln!(
        "MLA-only {d:?}: max|Δ|={max:.3e} mean|out|={mean_abs:.3e} rel={rel:.3e} ‖cpu‖={sa:.4} ‖dev‖={sb:.4} ratio={:.4}",
        sb / sa.max(1e-9)
    );
    // MLA is a long matmul chain + softmax attention → ~5e-2 f32 GPU-vs-CPU
    // rounding (disabling hybrid barely moves it: not the GDN-class boundary bug).
    // Threshold guards against a gross miscompile (KDA's was ~400), not rounding.
    assert!(
        max < 2e-1,
        "MLA grossly diverges on {d:?}: max|Δ|={max:.3e}"
    );
}

/// Isolate `rms_norm` over a rank-4 `[1,1,96,128]` tensor (the KDA `o_norm`),
/// vs the rank-3 `[1,1,7168]` body norms which are Metal-correct. A rank-axis bug
/// would hit the rank-4 case only.
#[test]
fn rmsnorm_rank4_parity() {
    let d = dev();
    if matches!(d, Device::Cpu) {
        return;
    }
    let (h, hd) = (96usize, 128usize);
    let x = fill(h * hd, 9);
    let w = fill(hd, 3);
    let run = |device: Device, rank4: bool| -> Vec<f32> {
        let f = DType::F32;
        let mut hir = HirModule::new("rn");
        let mut g = HirMut::new(&mut hir);
        let shape = if rank4 {
            vec![1usize, 1, h, hd]
        } else {
            vec![1usize, 1, h * hd]
        };
        let xn = g.input("x", Shape::new(&shape, f));
        let wn = g.input("w", Shape::new(&[hd], f));
        let bn = g.input("b", Shape::new(&[hd], f));
        let out = g.rms_norm(xn, wn, bn, 1e-5);
        g.set_outputs(vec![out]);
        let built = built_from_hir(hir, HashMap::new()).expect("built");
        let mut c = compile_built(built, device).expect("compile");
        let wfull = w.clone();
        let _ = &wfull;
        c.run(&[
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("b", &vec![0f32; hd]),
        ])
        .remove(0)
    };
    let a = run(Device::Cpu, true);
    let bb = run(d, true);
    let max = a
        .iter()
        .zip(&bb)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    eprintln!("rms_norm rank4 [1,1,{h},{hd}] {d:?}: max|Δ| = {max:.3e}");
    assert!(
        max < 1e-3,
        "rms_norm rank4 diverges on {d:?}: max|Δ|={max:.3e}"
    );
}

/// Isolate the GatedDeltaNet op at real dims (96 heads × head_dim 128). The
/// existing gdn test uses head_dim=4 (passes on Metal); the real head_dim=128 may
/// blow a Metal threadgroup/shared-mem limit (the recurrence holds a 128×128
/// state per head).
#[test]
fn gdn_only_parity() {
    let d = dev();
    if matches!(d, Device::Cpu) {
        return;
    }
    let (b, s, h, n) = (1usize, 1usize, 96usize, 128usize);
    let bshn = b * s * h * n;
    let (q, k, v) = (fill(bshn, 1), fill(bshn, 2), fill(bshn, 3));
    // Realistic gate: g_log is NEGATIVE (decay), ~[-2, -0.2], not tiny like fill.
    let gl: Vec<f32> = fill(bshn, 4)
        .iter()
        .map(|x| -0.2 - (x + 0.075) * 12.0)
        .collect();
    let beta: Vec<f32> = fill(b * s * h, 5).iter().map(|x| 0.5 + x).collect();
    let run = |device: Device| -> Vec<f32> {
        let f = DType::F32;
        let mut hir = HirModule::new("gdn");
        let mut g = HirMut::new(&mut hir);
        let qn = g.input("q", Shape::new(&[b, s, h, n], f));
        let kn = g.input("k", Shape::new(&[b, s, h, n], f));
        let vn = g.input("v", Shape::new(&[b, s, h, n], f));
        let gn = g.input("g", Shape::new(&[b, s, h, n], f));
        let bn = g.input("beta", Shape::new(&[b, s, h], f));
        let out = g.gated_delta_net_pc(qn, kn, vn, gn, bn, n, Shape::new(&[b, s, h, n], f));
        g.set_outputs(vec![out]);
        let built = built_from_hir(hir, HashMap::new()).expect("built");
        let mut c = compile_built(built, device).expect("compile");
        c.run(&[
            ("q", q.as_slice()),
            ("k", k.as_slice()),
            ("v", v.as_slice()),
            ("g", gl.as_slice()),
            ("beta", beta.as_slice()),
        ])
        .remove(0)
    };
    let a = run(Device::Cpu);
    let bb = run(d);
    let max = a
        .iter()
        .zip(&bb)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    eprintln!("GDN-only {d:?} (h={h},hd={n}): max|Δ| = {max:.3e}");
    assert!(max < 1e-3, "GDN diverges on {d:?}: max|Δ|={max:.3e}");
}

/// Isolate the KDA op alone at real dims (hidden 7168, 96×128). If this diverges,
/// the bug is in `build_kda_layer` / GatedDeltaNet on Metal, not the surrounding
/// AttnRes/residual.
#[test]
fn kda_only_parity() {
    use rlx_kimi_k3::kda::build_kda_layer;
    let d = dev();
    if matches!(d, Device::Cpu) {
        return;
    }
    let kd = KdaDims {
        hidden: 7168,
        num_heads: 96,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq: 1,
    };
    let w = kda_w(kd, 55);
    let hin = fill(kd.hidden, 7);
    let run = |device: Device| -> Vec<f32> {
        let mut hir = HirModule::new("kda");
        let mut g = HirMut::new(&mut hir);
        let h = g.input("h", Shape::new(&[1, 1, kd.hidden], DType::F32));
        let mut params = HashMap::new();
        let out = build_kda_layer(&mut g, &mut params, "self_attn", h, &w, kd).expect("build");
        g.set_outputs(vec![out]);
        let built = built_from_hir(hir, params).expect("built");
        let mut c = compile_built(built, device).expect("compile");
        c.run(&[("h", &hin)]).remove(0)
    };
    let a = run(Device::Cpu);
    let b = run(d);
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let mean_abs = a.iter().map(|x| x.abs()).sum::<f32>() / a.len() as f32;
    let rel = max / mean_abs.max(1e-9);
    eprintln!("KDA-only {d:?}: max|Δ|={max:.3e} mean|out|={mean_abs:.3e} rel={rel:.3e}");
    // 12288-wide matmul chain → ~1e-3 f32 GPU-vs-CPU rounding is fine; the bug it
    // guards against was ~400 (MPSGraph hybrid boundary miscompile).
    assert!(max < 5e-2, "KDA diverges on {d:?}: max|Δ|={max:.3e}");
}

#[test]
fn cpu_vs_device_2layer() {
    let d = dev();
    if matches!(d, Device::Cpu) {
        eprintln!("RLX_TEST_DEVICE unset/cpu — parity trivially holds; set metal/mlx");
        return;
    }
    let (w, cfg) = synth(2, 1);
    let hin = fill(cfg.batch * cfg.seq * cfg.hidden, 7);
    let a = run_on(Device::Cpu, &w, &cfg, &hin);
    let b = run_on(d, &w, &cfg, &hin);
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let am = a
        .iter()
        .cloned()
        .enumerate()
        .fold((0, f32::MIN), |m, (i, v)| if v > m.1 { (i, v) } else { m });
    let bm = b
        .iter()
        .cloned()
        .enumerate()
        .fold((0, f32::MIN), |m, (i, v)| if v > m.1 { (i, v) } else { m });
    eprintln!(
        "{d:?}: max|Δlogit| = {max:.3e}; argmax cpu={} dev={}",
        am.0, bm.0
    );
    assert!(max < 1e-2, "{d:?} diverges from CPU: max|Δ|={max:.3e}");
    assert_eq!(am.0, bm.0, "argmax differs cpu={} {d:?}={}", am.0, bm.0);
}
