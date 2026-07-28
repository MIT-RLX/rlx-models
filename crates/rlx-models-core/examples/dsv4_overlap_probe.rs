// RLX — versatile ML compiler + runtime. GPLv3.
//! Validates the DeepSeek-V4 **overlapping KV Compressor**
//! ([`build_kv_compressor_overlap`], subsystem #4, the `ratio == 4` path) — the
//! `overlap=True` form of `Compressor.forward` (deepseek-ai/DeepSeek-V4-Flash):
//! each compressed window pools over `2*ratio` candidates (its own tokens' second
//! dim-half + the previous window's tokens' first dim-half, shifted, window-0
//! prev masked `-inf`), softmax over the `2*ratio` axis, RMSNorm. Compared vs an
//! inline port of `overlap_transform` + the prefill (`start_pos == 0`) path.
//! FP4/Hadamard (precision-sim / orthogonal-cancels) + compressed-tail RoPE
//! omitted (validated elsewhere). Synthetic weights, no checkpoint.
//!
//!   cargo run --release -p rlx-models-core --example dsv4_overlap_probe

use anyhow::Result;
use rlx_ir::{DType, Graph, Shape};
use rlx_models_core::standard_decoder::build_kv_compressor_overlap;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

fn rnd(seed: f64, i: usize) -> f32 {
    let x = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
    (x - x.floor()) as f32 - 0.5
}

fn main() -> Result<()> {
    // wkv/wgate outputs are coff*hd = 2*hd (already applied → passed as inputs).
    let (s, ratio, hd, eps) = (12usize, 4usize, 6usize, 1e-6f32);
    let nwin = s / ratio;
    let d2 = 2 * hd;
    let kv2: Vec<f32> = (0..s * d2).map(|i| 0.5 * rnd(1.0, i)).collect();
    let sc2: Vec<f32> = (0..s * d2).map(|i| 0.4 * rnd(2.0, i)).collect();
    let ape: Vec<f32> = (0..ratio * d2).map(|i| 0.3 * rnd(3.0, i)).collect();
    let norm_w: Vec<f32> = (0..hd).map(|i| 1.0 + 0.1 * rnd(4.0, i)).collect();

    let mut g = Graph::new("dsv4_overlap");
    let kvn = g.input("kv2", Shape::new(&[s, d2], DType::F32));
    let scn = g.input("score2", Shape::new(&[s, d2], DType::F32));
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let apen = g.param("ape", Shape::new(&[ratio, d2], DType::F32));
    params.insert("ape".into(), ape.clone());
    let nwn = g.param("norm_w", Shape::new(&[hd], DType::F32));
    params.insert("norm_w".into(), norm_w.clone());
    let out = build_kv_compressor_overlap(
        &mut g,
        &mut params,
        kvn,
        scn,
        apen,
        nwn,
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
        .run(&[("kv2", kv2.as_slice()), ("score2", sc2.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    // Inline reference — overlap_transform + prefill pooling.
    // sc4[w,j,:] = sc2[(w*ratio+j),:] + ape[j,:]  (over the full 2*hd dims).
    // candidate k in 0..2*ratio for window w:
    //   k <  ratio (prev): kv=kv2[((w-1)*ratio+k), 0:hd],  sc=sc4[w-1,k, 0:hd]  (w>0 else kv=0, sc=-inf)
    //   k >= ratio (curr): kv=kv2[(w*ratio+(k-ratio)), hd:2hd], sc=sc4[w,(k-ratio), hd:2hd]
    let neg = f32::NEG_INFINITY;
    let sc4 = |w: usize, j: usize, c: usize| sc2[(w * ratio + j) * d2 + c] + ape[j * d2 + c];
    let mut refout = vec![0f32; nwin * hd];
    for w in 0..nwin {
        for c in 0..hd {
            // gather 2*ratio (score, kv) candidates for channel c.
            let mut sc = vec![0f32; 2 * ratio];
            let mut kvc = vec![0f32; 2 * ratio];
            for k in 0..2 * ratio {
                if k < ratio {
                    let j = k;
                    if w > 0 {
                        sc[k] = sc4(w - 1, j, c); // first-half dims [0:hd]
                        kvc[k] = kv2[((w - 1) * ratio + j) * d2 + c];
                    } else {
                        sc[k] = neg;
                        kvc[k] = 0.0;
                    }
                } else {
                    let j = k - ratio;
                    sc[k] = sc4(w, j, hd + c); // second-half dims [hd:2hd]
                    kvc[k] = kv2[(w * ratio + j) * d2 + hd + c];
                }
            }
            // softmax over 2*ratio.
            let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut den = 0f32;
            let mut ex = vec![0f32; 2 * ratio];
            for k in 0..2 * ratio {
                ex[k] = if sc[k].is_finite() {
                    (sc[k] - mx).exp()
                } else {
                    0.0
                };
                den += ex[k];
            }
            let mut pooled = 0f32;
            for k in 0..2 * ratio {
                pooled += kvc[k] * (ex[k] / den);
            }
            refout[w * hd + c] = pooled;
        }
        // RMSNorm over hd.
        let ms: f32 = (0..hd)
            .map(|c| refout[w * hd + c] * refout[w * hd + c])
            .sum::<f32>()
            / hd as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for c in 0..hd {
            refout[w * hd + c] = refout[w * hd + c] * inv * norm_w[c];
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
    println!("── DeepSeek-V4 overlapping KV Compressor vs reference ──");
    println!(
        "elements = {}  cosine = {cos:.8}  max|err| = {maxerr:.3e}",
        got.len()
    );
    if got.iter().all(|v| v.is_finite()) && cos > 0.999999 && maxerr < 1e-4 {
        println!("✅ overlapping KV Compressor matches the reference");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "overlap compressor mismatch: cos={cos:.8} maxerr={maxerr:.3e}"
        ))
    }
}
