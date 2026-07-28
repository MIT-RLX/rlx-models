// RLX — versatile ML compiler + runtime. GPLv3.
//! Validate the native **DeepSeek MLA + fine-grained MoE** prefill against the
//! mlx-lm oracle (DeepSeek-V2-Lite / Moonlight-16B / any deepseek_v2|v3 dir).
//! Builds via [`build_deepseek_prefill`] (packed), runs one prefill on CPU,
//! compares last-token argmax/logits to oracle.json + the .npy.
//!
//!   cargo run --release -p rlx-models-core --example deepseek_prefill -- .mlx-test/moonlight-16b-4bit

use anyhow::{Context, Result, anyhow};
use rlx_ir::DType;
use rlx_models_core::RopeScaling;
use rlx_models_core::standard_decoder::{DeepseekSpec, build_deepseek_prefill};
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
        .unwrap_or_else(|| ".mlx-test/moonlight-16b-4bit".to_string());
    let dir = Path::new(&dir);
    let c: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let u = |k: &str| c[k].as_u64().unwrap() as usize;
    // DeepSeek YaRN: scale the rope inv_freq + fold mscale² into the attention scale.
    let qk = u("qk_nope_head_dim") + u("qk_rope_head_dim");
    let no_yarn = std::env::var("RLX_DS_NOYARN").is_ok();
    let rs = c.get("rope_scaling").filter(|_| !no_yarn);
    let (rope_scaling, attn_score_scale) = match rs
        .filter(|v| v.get("type").and_then(|t| t.as_str()) == Some("yarn"))
    {
        Some(rs) => {
            let getf = |k: &str, d: f64| rs.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
            let factor = getf("factor", 1.0);
            let mscale_all = getf("mscale_all_dim", getf("mscale", 1.0));
            let yarn = RopeScaling::Yarn {
                factor,
                original_max_position_embeddings: getf("original_max_position_embeddings", 4096.0),
                beta_fast: getf("beta_fast", 32.0),
                beta_slow: getf("beta_slow", 1.0),
                attention_factor: Some(1.0), // cos/sin unscaled (deepseek folds mscale into attn)
            };
            let m = if factor > 1.0 {
                0.1 * mscale_all * factor.ln() + 1.0
            } else {
                1.0
            };
            (yarn, Some(((qk as f64).powf(-0.5) * m * m) as f32))
        }
        None => (RopeScaling::None, None),
    };
    let spec = DeepseekSpec {
        vocab_size: u("vocab_size"),
        hidden_size: u("hidden_size"),
        num_hidden_layers: u("num_hidden_layers"),
        num_attention_heads: u("num_attention_heads"),
        // q_lora_rank: 0/absent → direct q_proj (V2-Lite); >0 → q-LoRA (V3/Kimi-K2).
        q_lora_rank: c.get("q_lora_rank").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        absorbed_mla: false, // standard deepseek checkpoints store kv_b_proj
        kv_lora_rank: u("kv_lora_rank"),
        qk_nope_head_dim: u("qk_nope_head_dim"),
        qk_rope_head_dim: u("qk_rope_head_dim"),
        v_head_dim: u("v_head_dim"),
        intermediate_size: u("intermediate_size"),
        moe_intermediate_size: u("moe_intermediate_size"),
        n_routed_experts: u("n_routed_experts"),
        num_experts_per_tok: u("num_experts_per_tok"),
        n_shared_experts: u("n_shared_experts"),
        first_k_dense_replace: c["first_k_dense_replace"].as_u64().unwrap_or(1) as usize,
        routed_scaling_factor: c["routed_scaling_factor"].as_f64().unwrap_or(1.0) as f32,
        norm_topk_prob: c["norm_topk_prob"].as_bool().unwrap_or(false),
        sigmoid_gate: c.get("scoring_func").and_then(|v| v.as_str()) == Some("sigmoid"),
        sqrtsoftplus_gate: false,
        swiglu_limit: 0.0,
        rope_theta: c["rope_theta"].as_f64().unwrap_or(10000.0),
        rope_scaling,
        attn_score_scale,
        rope_neox: std::env::var("RLX_DS_NEOX").is_ok(),
        rms_norm_eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
    };
    eprintln!(
        "[deepseek] {} layers={} heads={} kv_lora={} qk={}+{} v={} experts={} top_k={} shared={} moe_inter={} sigmoid={} first_dense={}",
        c["model_type"].as_str().unwrap_or("?"),
        spec.num_hidden_layers,
        spec.num_attention_heads,
        spec.kv_lora_rank,
        spec.qk_nope_head_dim,
        spec.qk_rope_head_dim,
        spec.v_head_dim,
        spec.n_routed_experts,
        spec.num_experts_per_tok,
        spec.n_shared_experts,
        spec.moe_intermediate_size,
        spec.sigmoid_gate,
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
    let (graph, params) = build_deepseek_prefill(&spec, &mut loader, seq, &mut packed)?;
    eprintln!(
        "[deepseek] graph built in {:.1?} ({} params, {} packed)",
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
    eprintln!("[deepseek] prefill ran in {:.1?}", t1.elapsed());
    let logits = out.into_iter().next().ok_or_else(|| anyhow!("no output"))?;

    let vocab = spec.vocab_size;
    let last = &logits[(seq - 1) * vocab..seq * vocab];
    let argmax = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as i64)
        .unwrap();
    println!("\n── DeepSeek MLA+MoE prefill vs mlx-lm oracle ────────");
    println!("rlx argmax    = {argmax}");
    println!("oracle argmax = {oracle_argmax}");
    println!("finite        = {}", last.iter().all(|v| v.is_finite()));
    if let Ok(ol) = read_npy_f32(&dir.join("oracle_prefill_last_logits.npy")) {
        if ol.len() == last.len() {
            println!("cosine        = {:.6}", cosine(last, &ol));
        }
    }
    if argmax == oracle_argmax {
        println!("✅ DeepSeek MLA+MoE prefill MATCHES the mlx-lm oracle");
        Ok(())
    } else {
        Err(anyhow!("argmax {argmax} != oracle {oracle_argmax}"))
    }
}
