// RLX — versatile ML compiler + runtime. GPLv3.
//! Validates the DeepSeek-V4 **`tid2eid` hash-routing** graph ops used on the
//! first `n_hash_layers` MoE layers (confirmed present as `gate.tid2eid` on
//! layers 0-2 of mlx-community/DeepSeek-V4-Flash-4bit): the expert indices come
//! from a per-token-id lookup `tid2eid[token_id]` (a Gather), and the routing
//! weights are the `sqrtsoftplus` scores gathered at those experts then
//! normalized — mirroring `Gate.forward(hash=True)` (deepseek-ai/DeepSeek-V4).
//! This is the ONLY delta vs the already-validated score-top-k MoE (the expert
//! compute is identical), so it isolates exactly that: index selection + weights.
//!
//!   cargo run --release -p rlx-models-core --example dsv4_hash_route_probe

use anyhow::Result;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

fn rnd(seed: f64, i: usize) -> f32 {
    let x = ((i as f64 + 1.0) * (seed + 1.3) * 12.9898).sin() * 43758.5453;
    (x - x.floor()) as f32
}

fn main() -> Result<()> {
    let (vocab, rows, n_exp, topk) = (7usize, 5usize, 8usize, 2usize);
    // tid2eid[vocab, topk]: expert ids per token id (stored f32, small ints).
    let tid2eid: Vec<f32> = (0..vocab * topk)
        .map(|i| ((i * 3 + 1) % n_exp) as f32)
        .collect();
    // token ids for the rows.
    let ids: Vec<u32> = (0..rows).map(|r| ((r * 2 + 1) % vocab) as u32).collect();
    // sqrtsoftplus scores [rows, n_exp] (already computed upstream).
    let scores: Vec<f32> = (0..rows * n_exp).map(|i| 0.1 + rnd(1.0, i)).collect();

    let mut g = Graph::new("dsv4_hash");
    let f = DType::F32;
    let idn = g.input("ids", Shape::new(&[rows], DType::I32));
    let scn = g.input("scores", Shape::new(&[rows, n_exp], f));
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let t2e = g.param("tid2eid", Shape::new(&[vocab, topk], f));
    params.insert("tid2eid".into(), tid2eid.clone());
    // Same ops as build_deepseek_moe_c's hash path:
    let sel = g.gather_(t2e, idn, 0); // [rows, topk] expert ids
    let top_idx = g.reshape_(sel, vec![rows as i64, topk as i64]);
    let top_w = g.add_node(
        Op::GatherElements { axis: 1 },
        vec![scn, top_idx],
        Shape::new(&[rows, topk], f),
    );
    let denom = g.sum(top_w, vec![1], true);
    let norm_w = g.div(top_w, denom);
    g.set_outputs(vec![top_idx, norm_w]);
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    let ids_f: Vec<f32> = ids.iter().map(|&x| x as f32).collect();
    let outs = compiled.run(&[("ids", ids_f.as_slice()), ("scores", scores.as_slice())]);
    let got_idx = &outs[0];
    let got_w = &outs[1];

    // Reference: idx[r,k] = tid2eid[ids[r], k]; w = scores[r, idx]; w /= Σ_k w.
    let mut ref_idx = vec![0f32; rows * topk];
    let mut ref_w = vec![0f32; rows * topk];
    for r in 0..rows {
        let tid = ids[r] as usize;
        let mut sum = 0f32;
        for k in 0..topk {
            let e = tid2eid[tid * topk + k] as usize;
            ref_idx[r * topk + k] = e as f32;
            ref_w[r * topk + k] = scores[r * n_exp + e];
            sum += ref_w[r * topk + k];
        }
        for k in 0..topk {
            ref_w[r * topk + k] /= sum;
        }
    }
    let idx_match = got_idx
        .iter()
        .zip(&ref_idx)
        .filter(|(a, b)| (**a - **b).abs() < 1e-4)
        .count();
    let werr = got_w
        .iter()
        .zip(&ref_w)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let total = rows * topk;
    println!("── DeepSeek-V4 tid2eid hash routing (gather + weight-select + norm) vs reference ──");
    println!("expert-index match = {idx_match}/{total}   weight max|err| = {werr:.3e}");
    if idx_match == total && werr < 1e-5 && got_w.iter().all(|v| v.is_finite()) {
        println!("✅ tid2eid hash routing selects the reference experts + weights");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "hash-route mismatch: idx {idx_match}/{total} werr {werr:.3e}"
        ))
    }
}
