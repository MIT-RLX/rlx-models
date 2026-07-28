// RLX — versatile ML compiler + runtime. GPLv3.
//! Validate the native **LFM2 / LFM2.5 ShortConv** prefill against the mlx-lm
//! oracle: build the hybrid ShortConv+attention graph via [`build_lfm2_prefill`],
//! run one prefill on CPU, and compare the last-token logits/argmax to
//! `oracle.json` + `oracle_prefill_last_logits.npy` (from `scripts/mlx_oracle_dump.py`).
//!
//!   cargo run --release -p rlx-models-core --example lfm2_prefill -- .mlx-test/lfm2-1.2b-4bit

use anyhow::{Context, Result, anyhow};
use rlx_ir::DType;
use rlx_models_core::standard_decoder::{Lfm2Spec, build_lfm2_prefill};
use rlx_models_core::weight_loader::MlxLoader;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;
use std::path::Path;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

fn read_npy_f32(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).with_context(|| format!("read {path:?}"))?;
    anyhow::ensure!(&b[..6] == b"\x93NUMPY", "{path:?}: not a .npy");
    let hlen = 10 + u16::from_le_bytes([b[8], b[9]]) as usize;
    Ok(b[hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn intermediate_size(c: &serde_json::Value) -> usize {
    let bff = c["block_ff_dim"].as_u64().unwrap() as usize;
    let mof = c["block_multiple_of"].as_u64().unwrap_or(256) as usize;
    let mut h = (2 * bff) / 3;
    if let Some(m) = c.get("block_ffn_dim_multiplier").and_then(|v| v.as_f64()) {
        h = (m * h as f64) as usize;
    }
    mof * h.div_ceil(mof)
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ".mlx-test/lfm2-1.2b-4bit".to_string());
    let dir = Path::new(&dir);
    let c: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let hidden = c["hidden_size"].as_u64().unwrap() as usize;
    let nh = c["num_attention_heads"].as_u64().unwrap() as usize;
    let spec = Lfm2Spec {
        vocab_size: c["vocab_size"].as_u64().unwrap() as usize,
        hidden_size: hidden,
        intermediate_size: intermediate_size(&c),
        num_hidden_layers: c["num_hidden_layers"].as_u64().unwrap() as usize,
        num_attention_heads: nh,
        num_key_value_heads: c["num_key_value_heads"].as_u64().unwrap() as usize,
        head_dim: hidden / nh,
        conv_dim: c
            .get("conv_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(hidden as u64) as usize,
        conv_kernel: c.get("conv_L_cache").and_then(|v| v.as_u64()).unwrap_or(3) as usize,
        full_attn_layers: c["full_attn_idxs"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default(),
        rope_theta: c
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(1_000_000.0),
        rms_norm_eps: c.get("norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32,
    };
    eprintln!(
        "[lfm2] hidden={} layers={} heads={}/{} head_dim={} inter={} conv_dim={} k={} attn_layers={:?}",
        spec.hidden_size,
        spec.num_hidden_layers,
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        spec.intermediate_size,
        spec.conv_dim,
        spec.conv_kernel,
        spec.full_attn_layers
    );

    let oracle: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("oracle.json")).context("need oracle.json")?,
    )?;
    let ids: Vec<u32> = oracle["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let oracle_argmax = oracle["prefill_argmax"].as_i64().unwrap();
    let seq = ids.len();

    let mut loader = MlxLoader::open(dir.to_str().unwrap())?;
    let mut packed: HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)> =
        HashMap::new();
    let (graph, params) = build_lfm2_prefill(&spec, &mut loader, seq, &mut packed)?;

    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    for (n, (b, _, _)) in &packed {
        compiled.set_param_typed(n, b, DType::U8);
    }
    let ids_f32: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
    let out = compiled.run(&[("input_ids", ids_f32.as_slice())]);
    let logits = out.into_iter().next().ok_or_else(|| anyhow!("no output"))?;

    let vocab = spec.vocab_size;
    let last = &logits[(seq - 1) * vocab..seq * vocab];
    let argmax = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as i64)
        .unwrap();

    let mut top: Vec<(usize, f32)> = last.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\n── LFM2 ShortConv prefill vs mlx-lm oracle ──────────");
    println!("rlx argmax    = {argmax}");
    println!("oracle argmax = {oracle_argmax}   (\" Paris\")");
    println!("finite        = {}", last.iter().all(|v| v.is_finite()));
    println!("rlx top5      = {:?}", &top[..5]);
    if let Ok(oracle_logits) = read_npy_f32(&dir.join("oracle_prefill_last_logits.npy")) {
        if oracle_logits.len() == last.len() {
            println!("cosine        = {:.6}", cosine(last, &oracle_logits));
        }
    }
    if argmax == oracle_argmax {
        println!("✅ LFM2 ShortConv prefill MATCHES the mlx-lm oracle");
        Ok(())
    } else {
        Err(anyhow!("argmax {argmax} != oracle {oracle_argmax}"))
    }
}
