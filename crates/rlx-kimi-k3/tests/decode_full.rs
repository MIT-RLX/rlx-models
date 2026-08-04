// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Full multi-layer **decode-vs-prefill** parity: run a 2-layer model (KDA layer +
//! MLA layer, each with AttnRes + a dense FFN) two ways —
//!   * PREFILL: the whole sequence at once via [`build_layer_pre_ffn`] + FFN;
//!   * DECODE: one token at a time via [`build_layer_decode_step`] + FFN, threading
//!     the cross-token KDA conv/scan state + MLA KV cache (AttnRes snapshots are
//!     per-position, so they reset each token).
//!
//! and assert the last token's hidden state matches. This verifies the decode
//! ORCHESTRATION (state threading across layers), on top of the per-op decode
//! parity in `kda_decode_runner` / `mla_decode`.

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};
use rlx_kimi_k3::flow::{
    AttnDecodeIn, AttnDecodeOut, AttnWeights, FfnWeights, FlowConfig, LayerWeights,
    build_layer_decode_step, build_layer_pre_ffn,
};
use rlx_kimi_k3::kda::{KdaDims, KdaWeights};
use rlx_kimi_k3::mla::{MlaDims, MlaWeights};
use rlx_kimi_k3::moe::{DenseMlpWeights, build_dense_mlp};
use rlx_runtime::Device;
use std::collections::HashMap;

fn dev() -> Device {
    match std::env::var("RLX_TEST_DEVICE").ok().as_deref() {
        Some("metal") | Some("mtl") => Device::Metal,
        Some("mlx") => Device::Mlx,
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

const HIDDEN: usize = 16;
const DINTER: usize = 24;

fn kda_dims(seq: usize) -> KdaDims {
    KdaDims {
        hidden: HIDDEN,
        num_heads: 2,
        head_dim: 8,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq,
    }
}
fn mla_dims(seq: usize) -> MlaDims {
    MlaDims {
        hidden: HIDDEN,
        num_heads: 2,
        q_lora_rank: 8,
        kv_lora_rank: 6,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 2,
        v_head_dim: 4,
        eps: 1e-5,
        batch: 1,
        seq,
    }
}

fn kda_w(d: KdaDims) -> KdaWeights {
    let (hidden, h, hd, proj, k) = (d.hidden, d.num_heads, d.head_dim, d.proj(), d.conv_kernel);
    KdaWeights {
        q_proj: fill(hidden * proj, 1),
        k_proj: fill(hidden * proj, 2),
        v_proj: fill(hidden * proj, 3),
        q_conv: fill(proj * k, 4),
        k_conv: fill(proj * k, 5),
        v_conv: fill(proj * k, 6),
        f_a: fill(hidden * hd, 7),
        f_b: fill(hd * proj, 8),
        dt_bias: fill(proj, 9),
        a_log: fill(hd, 10),
        b_proj: fill(hidden * h, 11),
        g_proj: fill(hidden * proj, 12),
        o_norm: vec![1.0; hd],
        o_proj: fill(proj * hidden, 13),
    }
}
fn mla_w(d: MlaDims) -> MlaWeights {
    let (hidden, h, ql, kvl, nope, rope, vd, qk) = (
        d.hidden,
        d.num_heads,
        d.q_lora_rank,
        d.kv_lora_rank,
        d.qk_nope_head_dim,
        d.qk_rope_head_dim,
        d.v_head_dim,
        d.qk(),
    );
    MlaWeights {
        q_a_proj: fill(hidden * ql, 21),
        q_a_layernorm: vec![1.0; ql],
        q_b_proj: fill(ql * h * qk, 22),
        kv_a_proj_with_mqa: fill(hidden * (kvl + rope), 23),
        kv_a_layernorm: vec![1.0; kvl],
        kv_b_proj: fill(kvl * h * (nope + vd), 24),
        g_proj: fill(hidden * h * vd, 25),
        o_proj: fill(h * vd * hidden, 26),
    }
}
fn dense(sd: u64) -> DenseMlpWeights {
    DenseMlpWeights {
        gate: fill(HIDDEN * DINTER, sd),
        up: fill(HIDDEN * DINTER, sd + 1),
        down: fill(DINTER * HIDDEN, sd + 2),
    }
}

fn layer_w(seq: usize, kda: bool) -> LayerWeights {
    let attn = if kda {
        AttnWeights::Kda(Box::new(kda_w(kda_dims(seq))))
    } else {
        AttnWeights::Mla(Box::new(mla_w(mla_dims(seq))))
    };
    LayerWeights {
        input_ln: vec![1.0; HIDDEN],
        post_ln: vec![1.0; HIDDEN],
        sa_res_norm: vec![1.0; HIDDEN],
        sa_res_proj: fill(HIDDEN, if kda { 40 } else { 50 }),
        mlp_res_norm: vec![1.0; HIDDEN],
        mlp_res_proj: fill(HIDDEN, if kda { 41 } else { 51 }),
        attn,
        ffn: FfnWeights::Dense(Box::new(dense(if kda { 60 } else { 70 }))),
    }
}

fn cfg(seq: usize) -> FlowConfig {
    FlowConfig {
        hidden: HIDDEN,
        vocab: 20,
        attn_res_block_size: 12,
        eps: 1e-5,
        kda: kda_dims(seq),
        mla: mla_dims(seq),
        moe: rlx_kimi_k3::moe::MoeDims {
            hidden: HIDDEN,
            latent: 12,
            moe_inter: 8,
            num_experts: 4,
            top_k: 2,
            num_shared: 1,
            routed_scaling: 1.0,
            eps: 1e-5,
            situ_beta: 4.0,
            situ_linear_beta: Some(25.0),
            batch: 1,
            seq,
        },
        dense_inter: DINTER,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch: 1,
        seq,
    }
}

/// One dense-FFN layer body (build_layer_pre_ffn → mn, stream; h = stream + dense).
fn prefill_layer(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    i: usize,
    h: HirNodeId,
    snaps: Vec<HirNodeId>,
    lw: &LayerWeights,
    c: &FlowConfig,
) -> (HirNodeId, Vec<HirNodeId>) {
    let (mn, stream, snaps_out) = build_layer_pre_ffn(g, params, i, h, snaps, lw, c).expect("pre");
    let FfnWeights::Dense(dw) = &lw.ffn else {
        unreachable!()
    };
    let ffn = build_dense_mlp(
        g,
        params,
        &format!("l{i}.mlp"),
        mn,
        dw,
        c.hidden,
        c.dense_inter,
        1,
        c.seq,
        c.situ_beta,
        c.situ_linear_beta,
    )
    .expect("ffn");
    (g.add(stream, ffn), snaps_out)
}

#[test]
fn full_layer_decode_matches_prefill() {
    let d = dev();
    let seq = 4;
    let lws: Vec<LayerWeights> = vec![layer_w(1, true), layer_w(1, false)]; // KDA, MLA
    let h_full = fill(seq * HIDDEN, 7);

    // ── PREFILL: whole sequence at once ──
    let c = cfg(seq);
    let mut hir = HirModule::new("prefill");
    let mut g = HirMut::new(&mut hir);
    let mut h = g.input("h", Shape::new(&[1, seq, HIDDEN], DType::F32));
    let mut params = HashMap::new();
    let mut snaps = Vec::new();
    for (i, lw) in lws.iter().enumerate() {
        let lwf = layer_w(seq, matches!(lw.attn, AttnWeights::Kda(_)));
        let (nh, ns) = prefill_layer(&mut g, &mut params, i, h, snaps, &lwf, &c);
        h = nh;
        snaps = ns;
    }
    g.set_outputs(vec![h]);
    let built = built_from_hir(hir, params).expect("prefill built");
    let mut compiled = compile_built(built, d).expect("prefill compile");
    let full = compiled.run(&[("h", h_full.as_slice())]).remove(0);
    let want_last = &full[(seq - 1) * HIDDEN..];

    // ── DECODE: one token at a time, threading cross-token attention state ──
    let c1 = cfg(1);
    let kd = kda_dims(1);
    let hq = mla_dims(1).num_heads * mla_dims(1).qk();
    // cross-token state: layer 0 KDA (conv q/k/v + scan), layer 1 MLA (k,v cache)
    let cs = (kd.conv_kernel - 1) * kd.proj();
    let (mut csq, mut csk, mut csv) = (vec![0f32; cs], vec![0f32; cs], vec![0f32; cs]);
    let mut scan = vec![0f32; kd.num_heads * kd.head_dim * kd.head_dim];
    let (mut mk, mut mv) = (Vec::<f32>::new(), Vec::<f32>::new());
    let mut got_last = Vec::new();

    for t in 0..seq {
        let s_past = t;
        let mut hir = HirModule::new("decode");
        let mut g = HirMut::new(&mut hir);
        let h_in = g.input("h", Shape::new(&[1, 1, HIDDEN], DType::F32));
        // state inputs
        let g_csq = g.input(
            "csq",
            Shape::new(&[1, kd.conv_kernel - 1, kd.proj()], DType::F32),
        );
        let g_csk = g.input(
            "csk",
            Shape::new(&[1, kd.conv_kernel - 1, kd.proj()], DType::F32),
        );
        let g_csv = g.input(
            "csv",
            Shape::new(&[1, kd.conv_kernel - 1, kd.proj()], DType::F32),
        );
        let g_scan = g.input(
            "scan",
            Shape::new(&[1, kd.num_heads, kd.head_dim, kd.head_dim], DType::F32),
        );
        let g_mk = g.input("mk", Shape::new(&[1, s_past, hq], DType::F32));
        let g_mv = g.input("mv", Shape::new(&[1, s_past, hq], DType::F32));
        let mut params = HashMap::new();

        let mut h = h_in;
        let mut snaps = Vec::new();
        let mut outs: Vec<(&str, HirNodeId)> = Vec::new();
        // layer 0: KDA
        {
            let lw = layer_w(1, true);
            let (mn, stream, ns, ao) = build_layer_decode_step(
                &mut g,
                &mut params,
                0,
                h,
                snaps,
                AttnDecodeIn::Kda {
                    csq: g_csq,
                    csk: g_csk,
                    csv: g_csv,
                    scan: g_scan,
                },
                &lw,
                &c1,
            )
            .expect("decode l0");
            let FfnWeights::Dense(dw) = &lw.ffn else {
                unreachable!()
            };
            let ffn = build_dense_mlp(
                &mut g,
                &mut params,
                "l0.mlp",
                mn,
                dw,
                HIDDEN,
                DINTER,
                1,
                1,
                c1.situ_beta,
                c1.situ_linear_beta,
            )
            .expect("ffn0");
            h = g.add(stream, ffn);
            snaps = ns;
            if let AttnDecodeOut::Kda {
                csq,
                csk,
                csv,
                scan,
            } = ao
            {
                outs.push(("ncsq", csq));
                outs.push(("ncsk", csk));
                outs.push(("ncsv", csv));
                outs.push(("nscan", scan));
            }
        }
        // layer 1: MLA
        {
            let lw = layer_w(1, false);
            let (mn, stream, _ns, ao) = build_layer_decode_step(
                &mut g,
                &mut params,
                1,
                h,
                snaps,
                AttnDecodeIn::Mla { ck: g_mk, cv: g_mv },
                &lw,
                &c1,
            )
            .expect("decode l1");
            let FfnWeights::Dense(dw) = &lw.ffn else {
                unreachable!()
            };
            let ffn = build_dense_mlp(
                &mut g,
                &mut params,
                "l1.mlp",
                mn,
                dw,
                HIDDEN,
                DINTER,
                1,
                1,
                c1.situ_beta,
                c1.situ_linear_beta,
            )
            .expect("ffn1");
            h = g.add(stream, ffn);
            if let AttnDecodeOut::Mla { k, v } = ao {
                outs.push(("nmk", k));
                outs.push(("nmv", v));
            }
        }
        let mut out_nodes = vec![h];
        for (_, n) in &outs {
            out_nodes.push(*n);
        }
        g.set_outputs(out_nodes);
        let built = built_from_hir(hir, params).expect("decode built");
        let mut comp = compile_built(built, d).expect("decode compile");
        let mut r = comp.run(&[
            ("h", &h_full[t * HIDDEN..(t + 1) * HIDDEN]),
            ("csq", csq.as_slice()),
            ("csk", csk.as_slice()),
            ("csv", csv.as_slice()),
            ("scan", scan.as_slice()),
            ("mk", mk.as_slice()),
            ("mv", mv.as_slice()),
        ]);
        got_last = r.remove(0);
        // outs order: ncsq, ncsk, ncsv, nscan, nmk, nmv
        csq = r.remove(0);
        csk = r.remove(0);
        csv = r.remove(0);
        scan = r.remove(0);
        mk = r.remove(0);
        mv = r.remove(0);
    }

    let worst = want_last
        .iter()
        .zip(&got_last)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("full decode vs prefill {d:?}: worst |Δ| = {worst:.3e}");
    assert!(
        worst < 1e-4,
        "full decode diverges from prefill: {worst:.3e}"
    );
}
