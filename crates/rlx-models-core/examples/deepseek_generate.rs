// RLX — versatile ML compiler + runtime. GPLv3.
//! **Multi-token** oracle validation of the DeepSeek MLA+MoE pipeline (beyond the
//! single-token prefill check). Teacher-forces the mlx-lm greedy continuation:
//! runs ONE prefill on `[prompt ++ greedy_gen[:-1]]` and asserts the argmax at
//! every generation position equals mlx-lm's next token — i.e. rlx would greedy-
//! generate the exact same sequence. Cheap (one forward) and validates the full
//! multi-token generation, not just the first token.
//!
//!   cargo run --release -p rlx-models-core --example deepseek_generate -- .mlx-test/dsv2-lite-4bit

use anyhow::{Context, Result, anyhow};
use rlx_ir::DType;
use rlx_models_core::standard_decoder::{DeepseekSpec, RopeScaling, build_deepseek_prefill};
use rlx_models_core::weight_loader::MlxLoader;
use rlx_runtime::{Device, Session};
use std::collections::HashMap;
use std::path::Path;

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ".mlx-test/dsv2-lite-4bit".to_string());
    let dir = Path::new(&dir);
    let c: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let u = |k: &str| c[k].as_u64().unwrap() as usize;
    let qk = u("qk_nope_head_dim") + u("qk_rope_head_dim");
    let rs = c.get("rope_scaling");
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
                attention_factor: Some(1.0),
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
        q_lora_rank: c.get("q_lora_rank").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        absorbed_mla: false,
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
        rope_neox: false,
        rms_norm_eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
    };

    let oracle: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("oracle.json")).context("need oracle.json")?,
    )?;
    let prompt: Vec<u32> = oracle["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let gen_ids: Vec<u32> = oracle["greedy_gen_ids"]
        .as_array()
        .context("need greedy_gen_ids")?
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    anyhow::ensure!(!gen_ids.is_empty(), "empty greedy_gen_ids");

    // Teacher-forced input: prompt ++ gen_ids[:-1]. Position (plen-1+t) predicts gen_ids[t].
    let plen = prompt.len();
    let mut input: Vec<u32> = prompt.clone();
    input.extend_from_slice(&gen_ids[..gen_ids.len() - 1]);
    let seq = input.len();

    let mut loader = MlxLoader::open(dir.to_str().unwrap())?;
    let mut packed: HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)> =
        HashMap::new();
    let (graph, params) = build_deepseek_prefill(&spec, &mut loader, seq, &mut packed)?;
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
    let ids_f32: Vec<f32> = input.iter().map(|&t| t as f32).collect();
    let logits = compiled
        .run(&[("input_ids", ids_f32.as_slice())])
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no output"))?;
    let vocab = spec.vocab_size;

    let argmax_at = |pos: usize| -> i64 {
        logits[pos * vocab..(pos + 1) * vocab]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as i64)
            .unwrap()
    };
    println!("── DeepSeek multi-token greedy vs mlx-lm oracle ─────");
    println!("prompt ids   : {prompt:?}");
    println!(
        "oracle greedy: {gen_ids:?}  ({:?})",
        oracle["greedy_text"].as_str().unwrap_or("")
    );
    let mut rlx_gen: Vec<i64> = Vec::new();
    let mut matches = 0usize;
    for (t, &want) in gen_ids.iter().enumerate() {
        let got = argmax_at(plen - 1 + t);
        rlx_gen.push(got);
        if got == want as i64 {
            matches += 1;
        }
    }
    println!("rlx greedy   : {rlx_gen:?}");
    println!("matched {matches}/{} generation steps", gen_ids.len());
    if matches == gen_ids.len() {
        println!(
            "✅ rlx DeepSeek greedy-generates the EXACT mlx-lm continuation (multi-token oracle match)"
        );
        Ok(())
    } else {
        Err(anyhow!(
            "multi-token mismatch: {matches}/{} matched",
            gen_ids.len()
        ))
    }
}
