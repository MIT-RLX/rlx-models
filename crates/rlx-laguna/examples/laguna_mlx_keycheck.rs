// RLX — versatile ML compiler + runtime. GPLv3.
//! Offline validation of the Laguna mlx-dir loader WITHOUT downloading weights:
//! parse the real `config.json`, print the dense/MoE layer pattern, then verify
//! every tensor `mlx_load` will request is present in the checkpoint's
//! `model.safetensors.index.json`. Proves the loader is correct-by-construction
//! (config parse + exact tensor-name coverage) before any 18.8GB download.
//!
//!   cargo run -p rlx-laguna --example laguna_mlx_keycheck -- <dir-with-config+index>

use anyhow::{Context, Result};
use rlx_laguna::config::LagunaConfig;
use std::collections::HashSet;

const LM: &str = "language_model.model";
const HEAD: &str = "language_model.lm_head.weight";

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .context("usage: laguna_mlx_keycheck <dir>")?;
    let cfg = LagunaConfig::from_json_path(format!("{dir}/config.json"))?;
    println!(
        "cfg: type={} layers={} hidden={} heads={}/{} head_dim={} experts={} top_k={} moe_inter={} shared_inter={} tie={}",
        cfg.model_type,
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.moe_intermediate_size,
        cfg.shared_expert_intermediate_size,
        cfg.tie_word_embeddings,
    );
    let pattern: String = (0..cfg.num_hidden_layers)
        .map(|i| if cfg.is_dense_mlp(i) { '-' } else { 'E' })
        .collect();
    println!("dense/MoE pattern (- dense, E moe): {pattern}");

    // Available tensors from the index (strip .scales/.biases quant siblings).
    let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(format!(
        "{dir}/model.safetensors.index.json"
    ))?)?;
    let avail: HashSet<String> = idx["weight_map"]
        .as_object()
        .context("no weight_map")?
        .keys()
        .filter(|k| !k.ends_with(".scales") && !k.ends_with(".biases"))
        .cloned()
        .collect();

    // Reconstruct exactly the keys mlx_load::load_mlx_weights requests.
    let mut want: Vec<String> = Vec::new();
    want.push(format!("{LM}.embed_tokens.weight"));
    want.push(format!("{LM}.norm.weight"));
    if !cfg.tie_word_embeddings {
        want.push(HEAD.to_string());
    }
    for il in 0..cfg.num_hidden_layers {
        let p = format!("{LM}.layers.{il}");
        let sa = format!("{p}.self_attn");
        want.push(format!("{p}.input_layernorm.weight"));
        want.push(format!("{p}.post_attention_layernorm.weight"));
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            want.push(format!("{sa}.{proj}.weight"));
        }
        // Optional attn extras: only require if the model declares them (checked
        // by presence, so not asserted here) — but report if missing everywhere.
        let mp = format!("{p}.mlp");
        if cfg.is_dense_mlp(il) {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                want.push(format!("{mp}.{proj}.weight"));
            }
        } else {
            want.push(format!("{mp}.gate.proj.weight"));
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                want.push(format!("{mp}.switch_mlp.{proj}.weight"));
                want.push(format!("{mp}.shared_expert.{proj}.weight"));
            }
        }
    }

    let missing: Vec<&String> = want.iter().filter(|k| !avail.contains(*k)).collect();
    // Optional tensors (present in some Laguna variants): report coverage.
    let optional_present = |suffix: &str| -> usize {
        (0..cfg.num_hidden_layers)
            .filter(|il| avail.contains(&format!("{LM}.layers.{il}.{suffix}")))
            .count()
    };
    println!(
        "optional per-layer: q_norm={} k_norm={} g_proj={} moe_bias={}",
        optional_present("self_attn.q_norm.weight"),
        optional_present("self_attn.k_norm.weight"),
        optional_present("self_attn.g_proj.weight"),
        optional_present("mlp.gate.e_score_correction_bias"),
    );
    println!(
        "required keys: {} requested, {} in checkpoint",
        want.len(),
        want.len() - missing.len()
    );
    if missing.is_empty() {
        println!("✅ every required tensor mlx_load requests is present in the checkpoint index");
        Ok(())
    } else {
        for m in missing.iter().take(20) {
            println!("  MISSING: {m}");
        }
        Err(anyhow::anyhow!(
            "{} required tensors missing from index",
            missing.len()
        ))
    }
}
