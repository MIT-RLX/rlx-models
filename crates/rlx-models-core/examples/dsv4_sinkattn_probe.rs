// RLX — versatile ML compiler + runtime. GPLv3.
//! Validates the DeepSeek-V4 **sparse-window attention core**
//! ([`build_v4_sink_attention`], subsystem #3) — the dense-masked correctness
//! form of the reference `sparse_attn`: MQA latent attention (shared `kv` as
//! key+value), a per-head `attn_sink` logit in the softmax denominator, and an
//! additive window/compression mask — vs an inline dense reference. Synthetic
//! weights, causal mask (window ≥ seq), no checkpoint.
//!
//!   cargo run --release -p rlx-models-core --example dsv4_sinkattn_probe

use anyhow::Result;
use rlx_ir::{DType, Graph, Shape};
use rlx_models_core::standard_decoder::build_v4_sink_attention;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

fn rnd(seed: f64, i: usize) -> f32 {
    let x = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
    (x - x.floor()) as f32 - 0.5
}

fn main() -> Result<()> {
    // rows = queries (=seq), nk = keys (=seq, causal), nh heads, hd head_dim.
    let (rows, nh, hd) = (5usize, 3usize, 8usize);
    let nk = rows;
    let scale = (hd as f32).powf(-0.5);
    let q: Vec<f32> = (0..rows * nh * hd).map(|i| 0.4 * rnd(1.0, i)).collect();
    let kv: Vec<f32> = (0..nk * hd).map(|i| 0.4 * rnd(2.0, i)).collect();
    let sink: Vec<f32> = (0..nh).map(|i| 0.3 * rnd(3.0, i)).collect();
    // Causal additive mask [rows, nk]: 0 for k<=q, -1e30 otherwise.
    let neg = -1e30f32;
    let mut mask = vec![0f32; rows * nk];
    for qi in 0..rows {
        for ki in 0..nk {
            mask[qi * nk + ki] = if ki <= qi { 0.0 } else { neg };
        }
    }

    let mut g = Graph::new("dsv4_sinkattn");
    let qn = g.input("q", Shape::new(&[rows, nh, hd], DType::F32));
    let kvn = g.input("kv", Shape::new(&[nk, hd], DType::F32));
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let mn = g.param("mask", Shape::new(&[rows, nk], DType::F32));
    params.insert("mask".into(), mask.clone());
    let sn = g.param("sink", Shape::new(&[nh], DType::F32));
    params.insert("sink".into(), sink.clone());
    let out = build_v4_sink_attention(
        &mut g,
        &mut params,
        qn,
        kvn,
        mn,
        sn,
        scale,
        rows,
        nh,
        hd,
        nk,
        "t",
    );
    g.set_outputs(vec![out]);
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    let got = compiled
        .run(&[("q", q.as_slice()), ("kv", kv.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    // Reference: MQA masked attention with per-head sink.
    let mut refout = vec![0f32; rows * nh * hd];
    for qi in 0..rows {
        for h in 0..nh {
            let mut sc = vec![0f32; nk];
            for ki in 0..nk {
                let mut d = 0f32;
                for c in 0..hd {
                    d += q[(qi * nh + h) * hd + c] * kv[ki * hd + c];
                }
                sc[ki] = d * scale + mask[qi * nk + ki];
            }
            // softmax over [scores, sink] then drop sink
            let mut mx = sink[h];
            for &v in &sc {
                if v > mx {
                    mx = v;
                }
            }
            let mut den = (sink[h] - mx).exp();
            for v in sc.iter_mut() {
                *v = (*v - mx).exp();
                den += *v;
            }
            for c in 0..hd {
                let mut o = 0f32;
                for ki in 0..nk {
                    o += (sc[ki] / den) * kv[ki * hd + c];
                }
                refout[(qi * nh + h) * hd + c] = o;
            }
        }
    }

    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, b) in got.iter().zip(&refout) {
        dot += *a as f64 * *b as f64;
        na += *a as f64 * *a as f64;
        nb += *b as f64 * *b as f64;
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
    let maxerr = got
        .iter()
        .zip(&refout)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("── DeepSeek-V4 sparse-window attention (MQA + sink, dense-masked) vs reference ──");
    println!(
        "elements = {}  cosine = {cos:.8}  max|err| = {maxerr:.3e}",
        got.len()
    );
    if got.iter().all(|v| v.is_finite()) && cos > 0.999999 && maxerr < 1e-4 {
        println!("✅ V4 sparse-window attention core matches the dense reference");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "sink-attn mismatch: cos={cos:.8} maxerr={maxerr:.3e}"
        ))
    }
}
