// RLX — versatile ML compiler + runtime. GPLv3.
//! Validates the DeepSeek-V4 **KV Compressor** core gated-pooling
//! ([`build_kv_compressor_pool`], subsystem #4) — `Σ(kv · softmax(wgate+APE))`
//! per `ratio`-token window + RMSNorm — vs an inline port of the reference
//! `Compressor.forward` non-overlap prefill path (deepseek-ai/DeepSeek-V4-Flash).
//! FP4/Hadamard quant-sim (precision-only) and the compressed-tail RoPE are
//! omitted here (validated elsewhere). Synthetic weights, no checkpoint.
//!
//!   cargo run --release -p rlx-models-core --example dsv4_compressor_probe

use anyhow::Result;
use rlx_ir::{DType, Graph, Shape};
use rlx_models_core::standard_decoder::build_kv_compressor_pool;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

fn rnd(seed: f64, i: usize) -> f32 {
    let x = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
    (x - x.floor()) as f32 - 0.5
}

fn main() -> Result<()> {
    let (b, s, ratio, hd, eps) = (1usize, 8usize, 4usize, 8usize, 1e-6f32);
    let nwin = s / ratio;
    let rows = b * s;
    let kv: Vec<f32> = (0..rows * hd).map(|i| 0.5 * rnd(1.0, i)).collect();
    let score: Vec<f32> = (0..rows * hd).map(|i| 0.4 * rnd(2.0, i)).collect();
    let ape: Vec<f32> = (0..ratio * hd).map(|i| 0.3 * rnd(3.0, i)).collect();
    let norm_w: Vec<f32> = (0..hd).map(|i| 1.0 + 0.1 * rnd(4.0, i)).collect();

    let mut g = Graph::new("dsv4_comp");
    let kvn = g.input("kv", Shape::new(&[rows, hd], DType::F32));
    let scn = g.input("score", Shape::new(&[rows, hd], DType::F32));
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let apen = g.param("ape", Shape::new(&[ratio, hd], DType::F32));
    params.insert("ape".into(), ape.clone());
    let nwn = g.param("norm_w", Shape::new(&[hd], DType::F32));
    params.insert("norm_w".into(), norm_w.clone());
    let out = build_kv_compressor_pool(
        &mut g,
        &mut params,
        kvn,
        scn,
        apen,
        nwn,
        b,
        s,
        ratio,
        hd,
        eps,
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
        .run(&[("kv", kv.as_slice()), ("score", score.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    // Inline reference: per window w, softmax(score[w*ratio..]+ape) over ratio,
    // pooled = Σ kv·w, then RMSNorm.
    let mut refout = vec![0f32; b * nwin * hd];
    for wi in 0..nwin {
        // softmax over the ratio window, per channel.
        let mut sm = vec![0f32; ratio * hd];
        for c in 0..hd {
            let mut mx = f32::MIN;
            for r in 0..ratio {
                let v = score[(wi * ratio + r) * hd + c] + ape[r * hd + c];
                if v > mx {
                    mx = v;
                }
            }
            let mut den = 0f32;
            for r in 0..ratio {
                let v = (score[(wi * ratio + r) * hd + c] + ape[r * hd + c] - mx).exp();
                sm[r * hd + c] = v;
                den += v;
            }
            for r in 0..ratio {
                sm[r * hd + c] /= den;
            }
        }
        let mut pooled = vec![0f32; hd];
        for r in 0..ratio {
            for c in 0..hd {
                pooled[c] += kv[(wi * ratio + r) * hd + c] * sm[r * hd + c];
            }
        }
        // RMSNorm
        let ms: f32 = pooled.iter().map(|v| v * v).sum::<f32>() / hd as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for c in 0..hd {
            refout[wi * hd + c] = pooled[c] * inv * norm_w[c];
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
    println!("── DeepSeek-V4 KV Compressor (gated pooling) vs reference ──");
    println!(
        "elements = {}  cosine = {cos:.8}  max|err| = {maxerr:.3e}",
        got.len()
    );
    if got.iter().all(|v| v.is_finite()) && cos > 0.999999 && maxerr < 1e-4 {
        println!("✅ KV Compressor gated-pooling matches the reference");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "compressor mismatch: cos={cos:.8} maxerr={maxerr:.3e}"
        ))
    }
}
