// RLX — versatile ML compiler + runtime. GPLv3.
//! Numerical validation of **DeepSeek-V4 Hyper-Connections** — `build_hc_pre` →
//! (identity sublayer) → `build_hc_post` — against an inline reference of the
//! reference `hc_split_sinkhorn` + `hc_pre`/`hc_post` (deepseek-ai/DeepSeek-V4-
//! Flash `inference/{model.py,kernel.py}`). Proves the Sinkhorn-normalized
//! stream mixing (the novel residual scheme) on synthetic weights (no checkpoint).
//!
//!   cargo run --release -p rlx-models-core --example hc_probe

use anyhow::Result;
use rlx_ir::{DType, Graph, Shape};
use rlx_models_core::standard_decoder::{build_hc_post, build_hc_pre};
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

fn rnd(seed: f64, i: usize) -> f32 {
    let x = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
    (x - x.floor()) as f32 - 0.5
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn main() -> Result<()> {
    let (rows, hc, d, iters, eps) = (3usize, 4usize, 6usize, 20usize, 1e-6f32);
    let mix_hc = (2 + hc) * hc; // 24
    let hcd = hc * d; // 24

    let x: Vec<f32> = (0..rows * hc * d).map(|i| 0.5 * rnd(1.0, i)).collect();
    let hc_fn: Vec<f32> = (0..mix_hc * hcd).map(|i| 0.2 * rnd(2.0, i)).collect(); // [mix_hc, hcd]
    let scale: Vec<f32> = (0..3).map(|i| 0.5 + 0.2 * rnd(3.0, i)).collect();
    let base: Vec<f32> = (0..mix_hc).map(|i| 0.1 * rnd(4.0, i)).collect();
    // Transposed mix weight [hcd, mix_hc] for g.mm(x_flat[rows,hcd], .).
    let mut hc_fn_t = vec![0f32; hcd * mix_hc];
    for m in 0..mix_hc {
        for i in 0..hcd {
            hc_fn_t[i * mix_hc + m] = hc_fn[m * hcd + i];
        }
    }

    // ── graph: residual=x; (h,post,comb)=hc_pre(x); x_out=h (identity); y=hc_post ──
    let mut g = Graph::new("hc_probe");
    let xin = g.input("x", Shape::new(&[rows, hc, d], DType::F32));
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let fnn = g.param("hc_fn_t", Shape::new(&[hcd, mix_hc], DType::F32));
    params.insert("hc_fn_t".into(), hc_fn_t);
    let sc = g.param("hc_scale", Shape::new(&[3], DType::F32));
    params.insert("hc_scale".into(), scale.clone());
    let bs = g.param("hc_base", Shape::new(&[mix_hc], DType::F32));
    params.insert("hc_base".into(), base.clone());
    let (h_mixed, post_n, comb_n) = build_hc_pre(
        &mut g,
        &mut params,
        xin,
        fnn,
        sc,
        bs,
        rows,
        hc,
        d,
        eps,
        iters,
        "p",
    );
    let y = build_hc_post(&mut g, h_mixed, xin, post_n, comb_n, rows, hc, d);
    g.set_outputs(vec![y]);
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
    for (n, dd) in &params {
        compiled.set_param(n, dd);
    }
    let got = compiled
        .run(&[("x", x.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    // ── inline reference (hc_split_sinkhorn + hc_pre + hc_post) ──
    let mut refout = vec![0f32; rows * hc * d];
    for r in 0..rows {
        let xr = |hh: usize, dd: usize| x[(r * hc + hh) * d + dd];
        // rms over flattened hc*d
        let mut ms = 0f32;
        for i in 0..hcd {
            let v = x[r * hcd + i];
            ms += v * v;
        }
        let rsq = 1.0 / (ms / hcd as f32 + eps).sqrt();
        // mixes = (x_flat @ hc_fn^T) * rsq
        let mut mixes = vec![0f32; mix_hc];
        for m in 0..mix_hc {
            let mut s = 0f32;
            for i in 0..hcd {
                s += x[r * hcd + i] * hc_fn[m * hcd + i];
            }
            mixes[m] = s * rsq;
        }
        // sinkhorn
        let mut pre = vec![0f32; hc];
        let mut post = vec![0f32; hc];
        for j in 0..hc {
            pre[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps;
            post[j] = 2.0 * sigmoid(mixes[j + hc] * scale[1] + base[j + hc]);
        }
        let mut comb = vec![vec![0f32; hc]; hc];
        for j in 0..hc {
            for k in 0..hc {
                comb[j][k] = mixes[j * hc + k + 2 * hc] * scale[2] + base[j * hc + k + 2 * hc];
            }
        }
        // softmax over k + eps
        for row in comb.iter_mut() {
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let mut sm = 0f32;
            for v in row.iter_mut() {
                *v = (*v - mx).exp();
                sm += *v;
            }
            for v in row.iter_mut() {
                *v = *v / sm + eps;
            }
        }
        // / (sum over j + eps)
        for k in 0..hc {
            let cs: f32 = (0..hc).map(|j| comb[j][k]).sum();
            for row in comb.iter_mut() {
                row[k] /= cs + eps;
            }
        }
        for _ in 0..iters - 1 {
            for row in comb.iter_mut() {
                let rs: f32 = row.iter().sum();
                for v in row.iter_mut() {
                    *v /= rs + eps;
                }
            }
            for k in 0..hc {
                let cs: f32 = (0..hc).map(|j| comb[j][k]).sum();
                for row in comb.iter_mut() {
                    row[k] /= cs + eps;
                }
            }
        }
        // hc_pre reduce: h[dd] = Σ_hc pre·x
        let mut hmix = vec![0f32; d];
        for dd in 0..d {
            for hh in 0..hc {
                hmix[dd] += pre[hh] * xr(hh, dd);
            }
        }
        // hc_post (identity sublayer x_out=hmix): y[j,dd] = post[j]*hmix[dd] + Σ_k comb[j][k]*x[k,dd]
        for j in 0..hc {
            for dd in 0..d {
                let mut v = post[j] * hmix[dd];
                for k in 0..hc {
                    v += comb[j][k] * xr(k, dd);
                }
                refout[(r * hc + j) * d + dd] = v;
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
    let finite = got.iter().all(|v| v.is_finite());
    println!("── DeepSeek-V4 Hyper-Connections: rlx graph vs inline reference ──");
    println!("elements = {}  finite = {finite}", got.len());
    println!("cosine   = {cos:.8}");
    println!("max|err| = {maxerr:.3e}");
    if finite && cos > 0.999999 && maxerr < 1e-4 {
        println!("✅ Hyper-Connections (Sinkhorn stream-mixing) matches the reference");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "HC mismatch: cos={cos:.8} maxerr={maxerr:.3e}"
        ))
    }
}
