// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Gemma numerical + functional parity: RLX vs PyTorch-aligned reference and optional HF weights.
//!
//! PyTorch / transformers reference (optional):
//!   RLX_GEMMA_WEIGHTS=... RLX_GEMMA_CONFIG=... \
//!   cargo test -p rlx-models --features parity-pytorch gemma_pytorch --release -- --nocapture

#![cfg(feature = "parity-pytorch")]

mod compile_support;

use anyhow::{Context, Result};
use rlx_core::weight_map::WeightMap;
use rlx_gemma::{GemmaArch, GemmaConfig, GemmaRopeMap, build_gemma_graph_sized_last_logits};
use rlx_runtime::Device;
use std::path::PathBuf;

const COSINE_MIN: f32 = 0.9995;
const COSINE_DIST_MAX: f64 = 5e-4;

fn weights_path() -> Option<String> {
    rlx_ir::env::var("RLX_GEMMA_WEIGHTS")
}

fn config_path() -> Option<String> {
    rlx_ir::env::var("RLX_GEMMA_CONFIG")
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / na.sqrt() / nb.sqrt()) as f32
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    (1.0 - cosine_similarity(a, b) as f64).max(0.0)
}

fn max_mean_abs_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut max = 0f32;
    let mut sum = 0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        sum += d as f64;
        if d > max {
            max = d;
        }
    }
    (max, (sum / a.len() as f64) as f32)
}

fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn gemma_cfg_from_json(path: &str) -> Result<GemmaConfig> {
    let data = std::fs::read_to_string(path)?;
    let v: serde_json::Value = serde_json::from_str(&data)?;
    let head_dim = v
        .get("head_dim")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or_else(|| {
            let h = v["hidden_size"].as_u64().unwrap() as usize;
            let heads = v["num_attention_heads"].as_u64().unwrap() as usize;
            h / heads
        });
    Ok(GemmaConfig {
        arch: GemmaArch::Gemma,
        vocab_size: v["vocab_size"].as_u64().unwrap() as usize,
        hidden_size: v["hidden_size"].as_u64().unwrap() as usize,
        intermediate_size: v["intermediate_size"].as_u64().unwrap() as usize,
        num_hidden_layers: v["num_hidden_layers"].as_u64().unwrap() as usize,
        num_attention_heads: v["num_attention_heads"].as_u64().unwrap() as usize,
        num_key_value_heads: v["num_key_value_heads"].as_u64().unwrap() as usize,
        max_position_embeddings: v["max_position_embeddings"].as_u64().unwrap() as usize,
        rms_norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6),
        rope_theta: v["rope_theta"].as_f64().unwrap_or(10_000.0),
        tie_word_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(true),
        attention_bias: v["attention_bias"].as_bool().unwrap_or(false),
        head_dim: Some(head_dim),
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        sliding_window: None,
        query_pre_attn_scalar: None,
        effective_num_layers: None,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        expert_weights_scale: 1.0,
        layer_types: Vec::new(),
        rope_parameters: GemmaRopeMap::default(),
        global_head_dim: None,
        num_global_key_value_heads: None,
        attention_k_eq_v: false,
        use_bidirectional_attention: None,
        hidden_size_per_layer_input: 0,
        vocab_size_per_layer_input: 0,
        num_kv_shared_layers: 0,
        use_double_wide_mlp: false,
        enable_moe_block: false,
        eog_token_ids: Vec::new(),
        activation_sparsity_pattern: Vec::new(),
        altup_num_inputs: 0,
        altup_active_idx: 0,
        altup_coef_clip: None,
        altup_correct_scale: false,
        laurel_rank: 0,
        rope_local_base_freq: 10_000.0,
    })
}

fn run_rlx_last_logits(
    cfg: &GemmaConfig,
    wm: &mut WeightMap,
    batch: usize,
    seq: usize,
    ids: &[u32],
) -> Result<Vec<f32>> {
    let profile = if cfg.arch == GemmaArch::Gemma2 {
        rlx_flow::CompileProfile::gemma_prefill()
    } else {
        rlx_flow::CompileProfile::llama32_prefill()
    };
    run_rlx_last_logits_profile(cfg, wm, batch, seq, ids, &profile)
}

fn run_rlx_last_logits_profile(
    cfg: &GemmaConfig,
    wm: &mut WeightMap,
    batch: usize,
    seq: usize,
    ids: &[u32],
    profile: &rlx_flow::CompileProfile,
) -> Result<Vec<f32>> {
    let (graph, params) = build_gemma_graph_sized_last_logits(cfg, wm, batch, seq, false)?;
    let mut compiled = compile_support::compile_with_profile(Device::Cpu, graph, params, profile);
    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let outs = compiled.run(&[("input_ids", ids_f32.as_slice())]);
    Ok(outs.into_iter().next().unwrap_or_default())
}

fn reference_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/gemma_parity_reference.py")
}

#[test]
fn gemma_pytorch_reference_last_logits() -> Result<()> {
    use std::path::Path;
    use std::process::Command;

    let (weights, config) = match (weights_path(), config_path()) {
        (Some(w), Some(c)) => (w, c),
        _ => {
            eprintln!("skip gemma_pytorch: set RLX_GEMMA_WEIGHTS + RLX_GEMMA_CONFIG");
            return Ok(());
        }
    };
    if !Path::new(&weights).exists() {
        eprintln!("skip gemma_pytorch: weights not found");
        return Ok(());
    }

    let script = reference_script();
    if !script.is_file() {
        anyhow::bail!("missing reference script at {}", script.display());
    }

    let out = Command::new("python3")
        .arg(&script)
        .arg(&weights)
        .arg(&config)
        .output()
        .context("running gemma_parity_reference.py")?;
    if !out.status.success() {
        anyhow::bail!("reference failed: {}", String::from_utf8_lossy(&out.stderr));
    }

    let line = std::str::from_utf8(&out.stdout)?
        .lines()
        .last()
        .context("empty reference stdout")?;
    let v: serde_json::Value = serde_json::from_str(line)?;
    let pt_logits: Vec<f32> = v["logits"]
        .as_array()
        .context("logits array")?
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let ids: Vec<u32> = v["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let seq = ids.len();

    let rlx_cfg = gemma_cfg_from_json(&config)?;
    let mut wm = WeightMap::from_file(&weights)?;
    let rlx_logits = run_rlx_last_logits(&rlx_cfg, &mut wm, 1, seq, &ids)?;

    let (max_d, mean_d) = max_mean_abs_diff(&rlx_logits, &pt_logits);
    let cos = cosine_similarity(&rlx_logits, &pt_logits);
    let cos_dist = cosine_distance(&rlx_logits, &pt_logits);
    eprintln!(
        "pytorch reference: max_abs={max_d:.6} mean_abs={mean_d:.6} cosine={cos:.8} cos_dist={cos_dist:.8}"
    );
    assert!(cos >= COSINE_MIN, "pytorch cosine {cos:.8}");
    assert!(
        cos_dist <= COSINE_DIST_MAX * 10.0,
        "pytorch cos_dist {cos_dist:.8}"
    );
    assert_eq!(argmax(&rlx_logits), argmax(&pt_logits), "top-1 vs pytorch");
    Ok(())
}
