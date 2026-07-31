// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Real-weight sanity check: load Kimi-K3's **actual** layer-0 KDA weights from
//! shard-1 (`/Volumes/FOUR/kimi`, bf16 projections + f32 params) and run one KDA
//! block on CPU. Validates the loader, tensor layouts/transposes, the per-channel
//! gate on real `A_log`/`dt_bias`, and that the output is finite + sanely scaled.
//! Skips cleanly when the checkpoint isn't mounted.

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::kda::{KdaDims, KdaWeights, build_kda_layer};
use rlx_runtime::Device;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

const SHARD: &str = "/Volumes/FOUR/kimi/model-00001-of-000096.safetensors";
const P: &str = "language_model.model.layers.0.self_attn";

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn f32_of(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Load `{P}.{name}` (a bf16 `[out, in]` Linear weight) transposed to `[in, out]`.
fn linear_t(st: &SafeTensors, name: &str, out_dim: usize, in_dim: usize) -> Vec<f32> {
    let t = st.tensor(&format!("{P}.{name}")).expect("tensor");
    assert_eq!(t.shape(), &[out_dim, in_dim], "{name} shape");
    let flat = bf16_to_f32(t.data());
    let mut out = vec![0f32; in_dim * out_dim];
    for o in 0..out_dim {
        for i in 0..in_dim {
            out[i * out_dim + o] = flat[o * in_dim + i];
        }
    }
    out
}

fn raw_f32(st: &SafeTensors, name: &str) -> Vec<f32> {
    f32_of(st.tensor(&format!("{P}.{name}")).expect("tensor").data())
}

#[test]
fn real_layer0_kda_runs_finite() {
    if !Path::new(SHARD).exists() {
        eprintln!("skip: {SHARD} not mounted");
        return;
    }
    let buf = std::fs::read(SHARD).expect("read shard");
    let st = SafeTensors::deserialize(&buf).expect("deserialize");

    let (hidden, heads, hd) = (7168usize, 96usize, 128usize);
    let proj = heads * hd; // 12288

    let w = KdaWeights {
        q_proj: linear_t(&st, "q_proj.weight", proj, hidden),
        k_proj: linear_t(&st, "k_proj.weight", proj, hidden),
        v_proj: linear_t(&st, "v_proj.weight", proj, hidden),
        q_conv: raw_f32(&st, "q_conv1d.weight"), // [12288,1,4] → flat, used as [ch,1,k,1]
        k_conv: raw_f32(&st, "k_conv1d.weight"),
        v_conv: raw_f32(&st, "v_conv1d.weight"),
        f_a: linear_t(&st, "f_a_proj.weight", hd, hidden),
        f_b: linear_t(&st, "f_b_proj.weight", proj, hd),
        dt_bias: raw_f32(&st, "dt_bias"),
        a_log: raw_f32(&st, "A_log"), // [head_dim]
        b_proj: linear_t(&st, "b_proj.weight", heads, hidden),
        g_proj: linear_t(&st, "g_proj.weight", proj, hidden),
        o_norm: raw_f32(&st, "o_norm.weight"),
        o_proj: linear_t(&st, "o_proj.weight", hidden, proj),
    };
    drop(buf); // free the 2.3 GB shard buffer before compiling

    let d = KdaDims {
        hidden,
        num_heads: heads,
        head_dim: hd,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq: 4,
    };

    let mut hir = HirModule::new("real_kda");
    let mut g = HirMut::new(&mut hir);
    let h_in = g.input("h", Shape::new(&[1, d.seq, hidden], DType::F32));
    let mut params = HashMap::new();
    let out = build_kda_layer(&mut g, &mut params, "l0.self_attn", h_in, &w, d).expect("build");
    g.set_outputs(vec![out]);

    let built = built_from_hir(hir, params).expect("build model");
    let mut compiled = compile_built(built, Device::Cpu).expect("compile");

    // A small, realistically-scaled hidden state.
    let hin: Vec<f32> = (0..d.seq * hidden)
        .map(|i| ((i as f32 * 0.001).sin()) * 0.5)
        .collect();
    let y = compiled
        .run(&[("h", hin.as_slice())])
        .into_iter()
        .next()
        .expect("output");
    assert_eq!(y.len(), d.seq * hidden);
    let finite = y.iter().all(|v| v.is_finite());
    let maxabs = y.iter().fold(0f32, |m, v| m.max(v.abs()));
    eprintln!("real layer-0 KDA: finite={finite}, max|out|={maxabs:.4}");
    assert!(finite, "real KDA output must be finite");
    assert!(
        maxabs < 1e4,
        "real KDA output magnitude {maxabs} implausibly large"
    );
}
