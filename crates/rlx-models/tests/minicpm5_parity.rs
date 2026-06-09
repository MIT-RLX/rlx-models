// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// MiniCPM5 numerical parity: RLX (llama32 graph) vs HuggingFace PyTorch.
//
// ```sh
// just fetch-minicpm5
//
// RLX_MINICPM5_WEIGHTS=/path/model-00000-of-00001.safetensors \
// RLX_MINICPM5_CONFIG=/path/config.json \
//   cargo test -p rlx-models --test minicpm5_parity --features parity-pytorch --release \
//     minicpm5_pytorch -- --nocapture
// ```

#![cfg(feature = "parity-pytorch")]

mod compile_support;

use anyhow::{Context, Result};
use rlx_models::minicpm5::llama_config_from_hf;
use rlx_models::weight_map::WeightMap;
use rlx_models::{Llama32Config, build_llama32_graph_sized_last_logits};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::process::Command;

const COSINE_MIN: f32 = 0.9990;
const COSINE_DIST_MAX: f64 = 1e-3;
const LOGIT_MAX_ABS: f32 = 0.15;
const LOGIT_MEAN_ABS: f32 = 0.02;

fn weights_path() -> Option<String> {
    rlx_ir::env::var("RLX_MINICPM5_WEIGHTS")
}

fn config_path() -> Option<String> {
    rlx_ir::env::var("RLX_MINICPM5_CONFIG")
}

fn reference_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/minicpm5_parity_reference.py")
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

fn run_rlx_last_logits(
    cfg: &Llama32Config,
    weights: &str,
    batch: usize,
    seq: usize,
    ids: &[u32],
) -> Result<Vec<f32>> {
    let mut wm = WeightMap::from_file(weights)?;
    let (graph, params) = build_llama32_graph_sized_last_logits(cfg, &mut wm, batch, seq, false)?;
    let mut compiled = compile_support::compile_llama32_prefill(Device::Cpu, graph, params);
    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let last_idx = vec![(seq - 1) as f32];
    let outs = compiled.run(&[
        ("input_ids", ids_f32.as_slice()),
        ("last_token_idx", last_idx.as_slice()),
    ]);
    Ok(outs[0].to_vec())
}

fn run_pytorch_reference(weights: &str, config: &str) -> Result<(Vec<f32>, Vec<u32>)> {
    let script = reference_script();
    if !script.is_file() {
        anyhow::bail!("missing {}", script.display());
    }
    let out = Command::new("python3")
        .arg(&script)
        .arg(weights)
        .arg(config)
        .output()
        .context("minicpm5_parity_reference.py")?;
    if !out.status.success() {
        anyhow::bail!(
            "python reference failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let line = std::str::from_utf8(&out.stdout)?
        .lines()
        .last()
        .context("empty reference stdout")?;
    let v: serde_json::Value = serde_json::from_str(line)?;
    let pt_logits: Vec<f32> = v["logits"]
        .as_array()
        .context("logits")?
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let ids: Vec<u32> = v["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    Ok((pt_logits, ids))
}

#[test]
fn minicpm5_pytorch_last_logits() -> Result<()> {
    let (weights, config) = match (weights_path(), config_path()) {
        (Some(w), Some(c)) => (w, c),
        _ => {
            eprintln!("skip: set RLX_MINICPM5_WEIGHTS + RLX_MINICPM5_CONFIG");
            return Ok(());
        }
    };
    if !Path::new(&weights).exists() {
        eprintln!("skip: weights not found at {weights}");
        return Ok(());
    }

    let (pt_logits, ids) = run_pytorch_reference(&weights, &config)?;
    let seq = ids.len();
    let rlx_cfg = llama_config_from_hf(Path::new(&weights))?;
    let rlx_logits = run_rlx_last_logits(&rlx_cfg, &weights, 1, seq, &ids)?;

    assert_eq!(rlx_logits.len(), pt_logits.len(), "vocab size mismatch");

    let (max_d, mean_d) = max_mean_abs_diff(&rlx_logits, &pt_logits);
    let cos = cosine_similarity(&rlx_logits, &pt_logits);
    let cos_dist = cosine_distance(&rlx_logits, &pt_logits);
    let rlx_top = argmax(&rlx_logits);
    let pt_top = argmax(&pt_logits);

    eprintln!(
        "MiniCPM5 parity (L={seq}): max_abs={max_d:.6} mean_abs={mean_d:.6} \
         cosine={cos:.8} cos_dist={cos_dist:.8} top1 rlx={rlx_top} pt={pt_top}"
    );

    assert!(cos >= COSINE_MIN, "cosine {cos:.8} < {COSINE_MIN}");
    assert!(
        cos_dist <= COSINE_DIST_MAX,
        "cosine distance {cos_dist:.8} > {COSINE_DIST_MAX}"
    );
    assert!(max_d <= LOGIT_MAX_ABS, "max_abs {max_d} > {LOGIT_MAX_ABS}");
    assert!(
        mean_d <= LOGIT_MEAN_ABS,
        "mean_abs {mean_d} > {LOGIT_MEAN_ABS}"
    );
    assert_eq!(rlx_top, pt_top, "top-1 token mismatch");
    Ok(())
}

#[test]
fn minicpm5_config_matches_hf_card() -> Result<()> {
    let (weights, _config) = match (weights_path(), config_path()) {
        (Some(w), Some(c)) => (w, c),
        _ => {
            eprintln!("skip: set RLX_MINICPM5_WEIGHTS + RLX_MINICPM5_CONFIG");
            return Ok(());
        }
    };
    let cfg = llama_config_from_hf(Path::new(&weights))?;
    assert_eq!(cfg.hidden_size, 1536);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.vocab_size, 130_560);
    assert!((cfg.rope_theta - 5_000_000.0).abs() < 1.0);
    eprintln!(
        "MiniCPM5-1B config ok: hidden={} layers={} vocab={} rope_theta={}",
        cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size, cfg.rope_theta
    );
    Ok(())
}
