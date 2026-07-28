// RLX — versatile ML compiler + runtime. GPLv3.
//! Validate the native **GLM-4.5 (glm4_moe)** prefill (GQA + partial-RoPE +
//! deepseek-style fine-grained MoE) against the mlx-lm oracle.
//!
//!   cargo run --release -p rlx-models-core --example glm4moe_prefill -- .mlx-test/glm45-air-2bit

use anyhow::{Context, Result, anyhow};
use rlx_ir::DType;
use rlx_models_core::standard_decoder::{Glm4MoeSpec, build_glm4moe_prefill};
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
fn read_npy_f32(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p)?;
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
        .unwrap_or_else(|| ".mlx-test/glm45-air-2bit".to_string());
    let dir = Path::new(&dir);
    let c: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let u = |k: &str| c[k].as_u64().unwrap() as usize;
    let spec = Glm4MoeSpec {
        vocab_size: u("vocab_size"),
        hidden_size: u("hidden_size"),
        num_hidden_layers: u("num_hidden_layers"),
        num_attention_heads: u("num_attention_heads"),
        num_key_value_heads: u("num_key_value_heads"),
        head_dim: c["head_dim"]
            .as_u64()
            .unwrap_or((u("hidden_size") / u("num_attention_heads")) as u64)
            as usize,
        partial_rotary_factor: c
            .get("partial_rotary_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5),
        intermediate_size: u("intermediate_size"),
        moe_intermediate_size: u("moe_intermediate_size"),
        n_routed_experts: u("n_routed_experts"),
        num_experts_per_tok: u("num_experts_per_tok"),
        n_shared_experts: c["n_shared_experts"].as_u64().unwrap_or(1) as usize,
        first_k_dense_replace: c["first_k_dense_replace"].as_u64().unwrap_or(1) as usize,
        routed_scaling_factor: c["routed_scaling_factor"].as_f64().unwrap_or(1.0) as f32,
        norm_topk_prob: c["norm_topk_prob"].as_bool().unwrap_or(true),
        rope_theta: c["rope_theta"].as_f64().unwrap_or(1_000_000.0),
        // mlx glm4_moe uses nn.RoPE(traditional=False) = NeoX (rotate-half).
        rope_neox: std::env::var("RLX_GLM_GPTJ").is_err(),
        rms_norm_eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
    };
    eprintln!(
        "[glm4moe] layers={} hidden={} heads={}/{} head_dim={} prot={} experts={} top_k={} shared={} moe_inter={} first_dense={}",
        spec.num_hidden_layers,
        spec.hidden_size,
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        spec.partial_rotary_factor,
        spec.n_routed_experts,
        spec.num_experts_per_tok,
        spec.n_shared_experts,
        spec.moe_intermediate_size,
        spec.first_k_dense_replace
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
    let (graph, params) = build_glm4moe_prefill(&spec, &mut loader, seq, &mut packed)?;
    eprintln!(
        "[glm4moe] graph built in {:.1?} ({} params, {} packed)",
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
    eprintln!("[glm4moe] prefill ran in {:.1?}", t1.elapsed());
    let logits = out.into_iter().next().ok_or_else(|| anyhow!("no output"))?;

    let vocab = spec.vocab_size;
    let last = &logits[(seq - 1) * vocab..seq * vocab];
    let argmax = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as i64)
        .unwrap();
    println!("\n── GLM-4.5 MoE prefill vs mlx-lm oracle ─────────────");
    println!("rlx argmax    = {argmax}");
    println!("oracle argmax = {oracle_argmax}");
    println!("finite        = {}", last.iter().all(|v| v.is_finite()));
    if let Ok(ol) = read_npy_f32(&dir.join("oracle_prefill_last_logits.npy")) {
        if ol.len() == last.len() {
            println!("cosine        = {:.6}", cosine(last, &ol));
        }
    }
    if argmax == oracle_argmax {
        println!("✅ GLM-4.5 MoE prefill MATCHES the mlx-lm oracle");
        Ok(())
    } else {
        Err(anyhow!("argmax {argmax} != oracle {oracle_argmax}"))
    }
}
