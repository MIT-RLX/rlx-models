// RLX — versatile ML compiler + runtime. GPLv3.
//! Validate the native **gpt-oss** prefill (attention-with-sinks + MXFP4 MoE)
//! against the mlx-lm oracle. Builds the full 24-layer graph via
//! [`build_gpt_oss_prefill`], runs one prefill on CPU, compares the last-token
//! argmax/logits to `oracle.json` + `oracle_prefill_last_logits.npy`.
//!
//!   cargo run --release -p rlx-models-core --example gpt_oss_prefill -- .mlx-test/gpt-oss-20b

use anyhow::{Context, Result, anyhow};
use rlx_ir::DType;
use rlx_models_core::RopeScaling;
use rlx_models_core::standard_decoder::{GptOssSpec, build_gpt_oss_prefill};
use rlx_models_core::weight_loader::MlxLoader;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;
use std::path::Path;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-12)
}

fn read_npy_f32(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path)?;
    anyhow::ensure!(&b[..6] == b"\x93NUMPY", "not npy");
    let hlen = 10 + u16::from_le_bytes([b[8], b[9]]) as usize;
    Ok(b[hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ".mlx-test/gpt-oss-20b".to_string());
    let dir = Path::new(&dir);
    let c: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let u = |k: &str| c[k].as_u64().unwrap() as usize;
    let rs = &c["rope_scaling"];
    let spec = GptOssSpec {
        vocab_size: u("vocab_size"),
        hidden_size: u("hidden_size"),
        num_hidden_layers: u("num_hidden_layers"),
        num_attention_heads: u("num_attention_heads"),
        num_key_value_heads: u("num_key_value_heads"),
        head_dim: c["head_dim"].as_u64().unwrap() as usize,
        num_experts: c["num_local_experts"].as_u64().unwrap() as usize,
        experts_per_token: c["experts_per_token"].as_u64().unwrap() as usize,
        moe_inter: u("intermediate_size"),
        swiglu_limit: c["swiglu_limit"].as_f64().unwrap_or(7.0) as f32,
        rope_theta: c["rope_theta"].as_f64().unwrap(),
        rope_scaling: RopeScaling::Yarn {
            factor: rs["factor"].as_f64().unwrap_or(32.0),
            original_max_position_embeddings: rs["original_max_position_embeddings"]
                .as_f64()
                .unwrap_or(4096.0),
            beta_fast: rs["beta_fast"].as_f64().unwrap_or(32.0),
            beta_slow: rs["beta_slow"].as_f64().unwrap_or(1.0),
            attention_factor: rs.get("attention_factor").and_then(|v| v.as_f64()),
        },
        rms_norm_eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
    };
    eprintln!(
        "[gpt-oss] hidden={} layers={} heads={}/{} head_dim={} experts={} top_k={} inter={} limit={}",
        spec.hidden_size,
        spec.num_hidden_layers,
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        spec.num_experts,
        spec.experts_per_token,
        spec.moe_inter,
        spec.swiglu_limit
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
    let t0 = std::time::Instant::now();
    let (graph, params) = build_gpt_oss_prefill(&spec, &mut loader, seq, &mut packed)?;
    eprintln!(
        "[gpt-oss] graph built in {:.1?} ({} params, {} packed)",
        t0.elapsed(),
        params.len(),
        packed.len()
    );

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
    let t1 = std::time::Instant::now();
    let out = compiled.run(&[("input_ids", ids_f32.as_slice())]);
    eprintln!("[gpt-oss] prefill ran in {:.1?}", t1.elapsed());
    let logits = out.into_iter().next().ok_or_else(|| anyhow!("no output"))?;

    let vocab = spec.vocab_size;
    let last = &logits[(seq - 1) * vocab..seq * vocab];
    let argmax = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as i64)
        .unwrap();
    println!("\n── gpt-oss prefill vs mlx-lm oracle ─────────────────");
    println!("rlx argmax    = {argmax}");
    println!("oracle argmax = {oracle_argmax}   (\" Paris\")");
    println!("finite        = {}", last.iter().all(|v| v.is_finite()));
    if let Ok(ol) = read_npy_f32(&dir.join("oracle_prefill_last_logits.npy")) {
        if ol.len() == last.len() {
            println!("cosine        = {:.6}", cosine(last, &ol));
        }
    }
    if argmax == oracle_argmax {
        println!("✅ gpt-oss prefill MATCHES the mlx-lm oracle");
        Ok(())
    } else {
        Err(anyhow!("argmax {argmax} != oracle {oracle_argmax}"))
    }
}
