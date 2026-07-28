// RLX — versatile ML compiler + runtime. GPLv3.
//! End-to-end spot check for the gpt-oss **MXFP4 packed MoE** builder: load the
//! real layer-0 MoE (router + 32 MXFP4 experts) from a downloaded
//! `mlx-community/gpt-oss-20b` checkpoint, build the MoE FFN graph via
//! [`build_gpt_oss_moe_ffn`], compile + run one token on CPU, and assert the
//! output is finite. Proves the loader (MXFP4 stacked experts), the
//! `Op::DequantGroupedMatMulMlx { MlxMxfp4 }` execution, the clamped-SwiGLU, and
//! the router/top-k/combine all work together on real weights.
//!
//!   cargo run --release -p rlx-models-core --example gpt_oss_moe_probe -- .mlx-test/gpt-oss-20b

use anyhow::{Result, anyhow};
use rlx_ir::{DType, Graph, Shape};
use rlx_models_core::standard_decoder::build_gpt_oss_moe_ffn;
use rlx_models_core::weight_loader::MlxLoader;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;
use std::path::Path;

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ".mlx-test/gpt-oss-20b".to_string());
    let dir = Path::new(&dir);
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let hidden = cfg["hidden_size"].as_u64().unwrap() as usize;
    let inter = cfg["intermediate_size"].as_u64().unwrap() as usize;
    let n_expert = cfg["num_local_experts"].as_u64().unwrap() as usize;
    let top_k = cfg["experts_per_token"].as_u64().unwrap() as usize;
    let limit = cfg["swiglu_limit"].as_f64().unwrap_or(7.0) as f32;
    eprintln!(
        "[probe] hidden={hidden} inter={inter} experts={n_expert} top_k={top_k} swiglu_limit={limit}"
    );

    let mut loader = MlxLoader::open(dir.to_str().ok_or_else(|| anyhow!("non-utf8"))?)?;
    let seq = 2usize;
    let mut g = Graph::new("gpt_oss_moe_probe");
    let x = g.input("x", Shape::new(&[1, seq, hidden], DType::F32));
    let mut packed: HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)> =
        HashMap::new();
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let out = build_gpt_oss_moe_ffn(
        &mut g,
        &mut params,
        &mut packed,
        &mut loader,
        "model.layers.0",
        x,
        1,
        seq,
        hidden,
        n_expert,
        top_k,
        inter,
        limit,
        1.702,
    )?;
    g.set_outputs(vec![out]);
    eprintln!(
        "[probe] graph built: {} params, {} packed tensors",
        params.len(),
        packed.len()
    );

    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    for (n, (b, _, _)) in &packed {
        compiled.set_param_typed(n, b, DType::U8);
    }

    // Deterministic pseudo-random input activation.
    let xdata: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i as f32 * 0.017).sin()) * 0.1)
        .collect();
    let res = compiled.run(&[("x", xdata.as_slice())]);
    let y = res.into_iter().next().ok_or_else(|| anyhow!("no output"))?;

    let finite = y.iter().all(|v| v.is_finite());
    let (mn, mx) = y
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let mean = y.iter().sum::<f32>() / y.len().max(1) as f32;
    println!("── gpt-oss MoE probe ──────────────────────────");
    println!("out len = {} (expect {})", y.len(), seq * hidden);
    println!("finite  = {finite}");
    println!("min/mean/max = {mn:.4} / {mean:.4} / {mx:.4}");
    if !finite || y.len() != seq * hidden {
        return Err(anyhow!(
            "gpt-oss MoE probe FAILED (finite={finite}, len={})",
            y.len()
        ));
    }
    println!("✅ gpt-oss MXFP4 packed MoE runs finite on real layer-0 weights");
    Ok(())
}
