// RLX — versatile ML compiler + runtime. GPLv3.
//! Validates the DeepSeek-V4 **grouped o-LoRA output projection**
//! ([`build_v4_o_lora`], subsystem #3) — per-group low-rank `einsum "bsgd,grd->
//! bsgr"` (`wo_a`) then `wo_b` up to `dim` — vs an inline reference of the
//! `Attention.forward` output block (deepseek-ai/DeepSeek-V4-Flash). Synthetic
//! weights, no checkpoint.
//!
//!   cargo run --release -p rlx-models-core --example dsv4_olora_probe

use anyhow::Result;
use rlx_ir::{DType, Graph, Shape};
use rlx_models_core::standard_decoder::build_v4_o_lora;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

fn rnd(seed: f64, i: usize) -> f32 {
    let x = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
    (x - x.floor()) as f32 - 0.5
}

fn main() -> Result<()> {
    let (rows, n_groups, o_lora, dpg, dim) = (3usize, 2usize, 4usize, 6usize, 8usize);
    let din = n_groups * dpg; // attention output width = n_heads*v_head_dim
    let inner = n_groups * o_lora;
    let o: Vec<f32> = (0..rows * din).map(|i| 0.5 * rnd(1.0, i)).collect();
    let wo_a: Vec<f32> = (0..n_groups * o_lora * dpg)
        .map(|i| 0.2 * rnd(2.0, i))
        .collect(); // [g, r, dpg]
    let wo_b: Vec<f32> = (0..dim * inner).map(|i| 0.2 * rnd(3.0, i)).collect(); // [dim, inner]
    // wo_b transposed → [inner, dim] for g.mm.
    let mut wo_b_t = vec![0f32; inner * dim];
    for o_ in 0..dim {
        for i in 0..inner {
            wo_b_t[i * dim + o_] = wo_b[o_ * inner + i];
        }
    }

    let mut g = Graph::new("dsv4_olora");
    let on = g.input("o", Shape::new(&[rows, din], DType::F32));
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let wan = g.param("wo_a", Shape::new(&[n_groups, o_lora, dpg], DType::F32));
    params.insert("wo_a".into(), wo_a.clone());
    let wbn = g.param("wo_b_t", Shape::new(&[inner, dim], DType::F32));
    params.insert("wo_b_t".into(), wo_b_t);
    let out = build_v4_o_lora(&mut g, on, wan, wbn, rows, n_groups, o_lora, dpg, dim);
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
        .run(&[("o", o.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    // Reference: proj[r_,g,rk] = Σ_d o[r_,g,d]*wo_a[g,rk,d]; out = flatten(proj) @ wo_b^T
    let mut refout = vec![0f32; rows * dim];
    for r_ in 0..rows {
        let mut inner_v = vec![0f32; inner];
        for grp in 0..n_groups {
            for rk in 0..o_lora {
                let mut s = 0f32;
                for d in 0..dpg {
                    s += o[r_ * din + grp * dpg + d] * wo_a[(grp * o_lora + rk) * dpg + d];
                }
                inner_v[grp * o_lora + rk] = s;
            }
        }
        for o_ in 0..dim {
            let mut s = 0f32;
            for i in 0..inner {
                s += inner_v[i] * wo_b[o_ * inner + i];
            }
            refout[r_ * dim + o_] = s;
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
    println!("── DeepSeek-V4 grouped o-LoRA output vs reference ──");
    println!(
        "elements = {}  cosine = {cos:.8}  max|err| = {maxerr:.3e}",
        got.len()
    );
    if got.iter().all(|v| v.is_finite()) && cos > 0.999999 && maxerr < 1e-4 {
        println!("✅ V4 grouped o-LoRA output projection matches the reference");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "o-LoRA mismatch: cos={cos:.8} maxerr={maxerr:.3e}"
        ))
    }
}
