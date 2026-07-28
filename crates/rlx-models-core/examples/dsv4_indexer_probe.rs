// RLX — versatile ML compiler + runtime. GPLv3.
//! Validates the DeepSeek-V4 **Indexer** (subsystem #5): the learned scoring
//! ([`build_v4_indexer_score`]) that ranks compressed KV positions, and the
//! top-k dense gate ([`build_v4_topk_gate`]) that keeps each query's top-k — vs
//! an inline port of `Indexer.forward` (deepseek-ai/DeepSeek-V4-Flash). The
//! Hadamard `rotate_activation` is orthogonal so it cancels in the q·kv inner
//! product, and `fp4_act_quant` is precision-only — both omitted; RoPE is set to
//! identity (validated elsewhere) to isolate the novel scoring+selection.
//! Checks: (1) index_score cos-exact; (2) top-k KEEP-SET exactly matches the
//! reference selection. Synthetic weights, no checkpoint.
//!
//!   cargo run --release -p rlx-models-core --example dsv4_indexer_probe

use anyhow::Result;
use rlx_ir::{DType, Graph, Shape};
use rlx_models_core::standard_decoder::{build_v4_indexer_score, build_v4_topk_gate};
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

fn rnd(seed: f64, i: usize) -> f32 {
    let x = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
    (x - x.floor()) as f32 - 0.5
}

fn main() -> Result<()> {
    let (seq, nh, hd, rd, ncomp, ql, dim, k) = (
        6usize, 2usize, 4usize, 2usize, 6usize, 3usize, 5usize, 3usize,
    );
    let neg = -1e30f32;
    let qr: Vec<f32> = (0..seq * ql).map(|i| 0.5 * rnd(1.0, i)).collect();
    let x: Vec<f32> = (0..seq * dim).map(|i| 0.4 * rnd(2.0, i)).collect();
    let kvc: Vec<f32> = (0..ncomp * hd).map(|i| 0.5 * rnd(3.0, i)).collect();
    let wq_b: Vec<f32> = (0..ql * nh * hd).map(|i| 0.3 * rnd(4.0, i)).collect(); // [ql, nh*hd]
    let wproj: Vec<f32> = (0..dim * nh).map(|i| 0.3 * rnd(5.0, i)).collect(); // [dim, nh]
    // identity rope table [seq, hd/2]: cos=1, sin=0.
    let cos: Vec<f32> = vec![1.0; seq * (hd / 2)];
    let sin: Vec<f32> = vec![0.0; seq * (hd / 2)];
    // causal-on-compressed: allow t <= s (t is a compressed position), else -1e30.
    let mut causal = vec![0f32; seq * ncomp];
    for s in 0..seq {
        for t in 0..ncomp {
            causal[s * ncomp + t] = if t <= s { 0.0 } else { neg };
        }
    }

    let mut g = Graph::new("dsv4_indexer");
    let qrn = g.input("qr", Shape::new(&[seq, ql], DType::F32));
    let xn = g.input("x", Shape::new(&[seq, dim], DType::F32));
    let kvn = g.input("kvc", Shape::new(&[ncomp, hd], DType::F32));
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let wqb = g.param("wq_b", Shape::new(&[ql, nh * hd], DType::F32));
    params.insert("wq_b".into(), wq_b.clone());
    let wpj = g.param("wproj", Shape::new(&[dim, nh], DType::F32));
    params.insert("wproj".into(), wproj.clone());
    let cosn = g.param("cos", Shape::new(&[seq, hd / 2], DType::F32));
    params.insert("cos".into(), cos.clone());
    let sinn = g.param("sin", Shape::new(&[seq, hd / 2], DType::F32));
    params.insert("sin".into(), sin.clone());
    let caun = g.param("causal", Shape::new(&[seq, ncomp], DType::F32));
    params.insert("causal".into(), causal.clone());

    let score = build_v4_indexer_score(
        &mut g,
        &mut params,
        qrn,
        xn,
        kvn,
        wqb,
        wpj,
        cosn,
        sinn,
        seq,
        nh,
        hd,
        rd,
        ncomp,
        "t",
    );
    let gate = build_v4_topk_gate(&mut g, &mut params, score, caun, seq, ncomp, k, "t");
    g.set_outputs(vec![score, gate]);
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    let outs = compiled.run(&[
        ("qr", qr.as_slice()),
        ("x", x.as_slice()),
        ("kvc", kvc.as_slice()),
    ]);
    let got_score = &outs[0];
    let got_gate = &outs[1];

    // ── Reference index_score ──
    // q[s,h,d] = Σ_r qr[s,r]*wq_b[r, h*hd+d]  (rope = identity)
    // weights[s,h] = (Σ_dim x[s,dim]*wproj[dim,h]) * (hd^-0.5 * nh^-0.5)
    // score[s,t] = Σ_h relu(Σ_d q[s,h,d]*kvc[t,d]) * weights[s,h]
    let wscale = (hd as f32).powf(-0.5) * (nh as f32).powf(-0.5);
    let mut ref_score = vec![0f32; seq * ncomp];
    for s in 0..seq {
        let mut wts = vec![0f32; nh];
        for h in 0..nh {
            let mut acc = 0f32;
            for dd in 0..dim {
                acc += x[s * dim + dd] * wproj[dd * nh + h];
            }
            wts[h] = acc * wscale;
        }
        for t in 0..ncomp {
            let mut ssum = 0f32;
            for h in 0..nh {
                let mut raw = 0f32;
                for d in 0..hd {
                    let mut qv = 0f32;
                    for r in 0..ql {
                        qv += qr[s * ql + r] * wq_b[r * (nh * hd) + h * hd + d];
                    }
                    raw += qv * kvc[t * hd + d];
                }
                ssum += raw.max(0.0) * wts[h];
            }
            ref_score[s * ncomp + t] = ssum;
        }
    }
    // ── Reference top-k keep-set ──
    let mut ref_keep = vec![false; seq * ncomp];
    for s in 0..seq {
        // masked scores
        let mut ms: Vec<(usize, f32)> = (0..ncomp)
            .map(|t| (t, ref_score[s * ncomp + t] + causal[s * ncomp + t]))
            .collect();
        ms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let kk = k.min(ncomp);
        let thr = ms[kk - 1].1;
        for t in 0..ncomp {
            let sm = ref_score[s * ncomp + t] + causal[s * ncomp + t];
            ref_keep[s * ncomp + t] = sm >= thr && causal[s * ncomp + t] >= -1.0;
        }
    }

    // score cosine
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, b) in got_score.iter().zip(&ref_score) {
        dot += *a as f64 * *b as f64;
        na += *a as f64 * *a as f64;
        nb += *b as f64 * *b as f64;
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
    let score_err = got_score
        .iter()
        .zip(&ref_score)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    // gate keep-set: keep iff mask > -1.0
    let got_keep: Vec<bool> = got_gate.iter().map(|&v| v > -1.0).collect();
    let keep_match = got_keep
        .iter()
        .zip(&ref_keep)
        .filter(|(a, b)| a == b)
        .count();
    let total = seq * ncomp;

    println!("── DeepSeek-V4 Indexer: scoring + top-k gate vs reference ──");
    println!("index_score: cosine = {cos:.8}  max|err| = {score_err:.3e}");
    println!("top-k gate:  keep-set match = {keep_match}/{total}  (k={k}, ncomp={ncomp})");
    let ok = got_score.iter().all(|v| v.is_finite())
        && got_gate.iter().all(|v| v.is_finite())
        && cos > 0.999999
        && score_err < 1e-4
        && keep_match == total;
    if ok {
        println!("✅ Indexer scoring cos-exact AND top-k keep-set matches the reference");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "indexer mismatch: cos={cos:.8} err={score_err:.3e} keep={keep_match}/{total}"
        ))
    }
}
