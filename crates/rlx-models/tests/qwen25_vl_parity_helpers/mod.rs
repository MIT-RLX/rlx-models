// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug)]
pub struct ReferenceDump {
    pub seq_len: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub vision_start_idx: usize,
    pub n_vision_tokens: usize,
    pub resized_w: Option<usize>,
    pub resized_h: Option<usize>,
    pub grid_h: Option<usize>,
    pub grid_w: Option<usize>,
    pub vision_proj_dim: Option<usize>,
    pub vision_embeddings: Option<Vec<f32>>,
    pub vision_mu_scores: Option<Vec<f32>>,
    pub vision_token_entropy: Option<Vec<f32>>,
    pub vision_dynamics: Option<Vec<Vec<f32>>>,
    pub aif_mask_ratio: Option<f32>,
    pub aif_s0: Option<f32>,
    pub aif_blocked_keys: Option<Vec<usize>>,
    pub aif_dynamics_mode: Option<String>,
    pub input_ids: Vec<u32>,
    pub logits: Vec<f32>,
    pub hidden: Vec<f32>,
}

pub fn run_docker_reference(image: &Path, out_dir: &Path) -> Result<ReferenceDump> {
    run_reference_with_env(image, out_dir, true, &[])
}

pub fn run_reference(image: &Path, out_dir: &Path, use_docker: bool) -> Result<ReferenceDump> {
    run_reference_with_env(image, out_dir, use_docker, &[])
}

pub fn run_reference_with_env(
    image: &Path,
    out_dir: &Path,
    use_docker: bool,
    extra_env: &[(&str, &str)],
) -> Result<ReferenceDump> {
    let helper_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/qwen25_vl_parity_helpers");
    let status = if use_docker {
        Command::new("bash")
            .arg(helper_dir.join("run-ref.sh"))
            .env("RLX_QWEN25_VL_IMAGE", image)
            .env("RLX_QWEN25_VL_OUT_DIR", out_dir)
            .envs(forward_hf_env())
            .envs(extra_env.iter().copied())
            .status()
            .with_context(|| "run docker reference")?
    } else {
        Command::new(
            std::env::var("RLX_QWEN25_VL_PYTHON").unwrap_or_else(|_| "python3".into()),
        )
        .arg(helper_dir.join("dump_reference.py"))
        .env("RLX_QWEN25_VL_IMAGE", image)
        .env("RLX_QWEN25_VL_OUT_DIR", out_dir)
        .envs(forward_hf_env())
        .envs(extra_env.iter().copied())
        .status()
        .with_context(|| "run python reference")?
    };
    anyhow::ensure!(status.success(), "reference dump failed");
    load_reference_dump(out_dir)
}

pub fn load_reference_dump(out_dir: &Path) -> Result<ReferenceDump> {
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out_dir.join("meta.json"))?)?;
    let logits = read_npy_f32(&out_dir.join("logits_last.npy"))?;
    let hidden = read_npy_f32(&out_dir.join("hidden_last.npy"))?;
    let input_ids = read_npy_u32(&out_dir.join("input_ids.npy"))?;
    let vision_embeddings = out_dir
        .join("vision_embeddings.npy")
        .exists()
        .then(|| read_npy_f32(&out_dir.join("vision_embeddings.npy")))
        .transpose()?;
    let vision_mu_scores = out_dir
        .join("vision_mu_scores.npy")
        .exists()
        .then(|| read_npy_f32(&out_dir.join("vision_mu_scores.npy")))
        .transpose()?;
    let vision_token_entropy = out_dir
        .join("vision_token_entropy.npy")
        .exists()
        .then(|| read_npy_f32(&out_dir.join("vision_token_entropy.npy")))
        .transpose()?;
    let vision_dynamics = load_vision_dynamics(out_dir, &meta)?;
    let aif_mask_ratio = meta
        .get("aif_mask_ratio")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);
    let aif_s0 = meta
        .get("aif_s0")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);
    let aif_blocked_keys = meta
        .get("aif_blocked_keys")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        });
    Ok(ReferenceDump {
        seq_len: meta["seq_len"].as_u64().unwrap_or(0) as usize,
        vocab_size: meta["vocab_size"].as_u64().unwrap_or(0) as usize,
        hidden_size: meta["hidden_size"].as_u64().unwrap_or(0) as usize,
        vision_start_idx: meta["vision_start_idx"].as_u64().unwrap_or(0) as usize,
        n_vision_tokens: meta["n_vision_tokens"].as_u64().unwrap_or(0) as usize,
        resized_w: meta["resized_w"].as_u64().map(|v| v as usize),
        resized_h: meta["resized_h"].as_u64().map(|v| v as usize),
        grid_h: meta
            .get("image_grid_thw")
            .and_then(|g| g.get("h"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        grid_w: meta
            .get("image_grid_thw")
            .and_then(|g| g.get("w"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        vision_proj_dim: meta["vision_proj_dim"].as_u64().map(|v| v as usize),
        vision_embeddings,
        vision_mu_scores,
        vision_token_entropy,
        vision_dynamics,
        aif_mask_ratio,
        aif_s0,
        aif_blocked_keys,
        aif_dynamics_mode: meta
            .get("aif_dynamics_mode")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        input_ids,
        logits,
        hidden,
    })
}

fn load_vision_dynamics(
    out_dir: &Path,
    meta: &serde_json::Value,
) -> Result<Option<Vec<Vec<f32>>>> {
    let path = out_dir.join("vision_dynamics.npy");
    if !path.exists() {
        return Ok(None);
    }
    let flat = read_npy_f32(&path)?;
    let n_vision = meta["n_vision_tokens"].as_u64().unwrap_or(0) as usize;
    let n_layers = meta
        .get("aif_n_layers")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or_else(|| {
            if n_vision > 0 && flat.len().is_multiple_of(n_vision) {
                flat.len() / n_vision
            } else {
                0
            }
        });
    if n_vision == 0 || n_layers == 0 || flat.len() != n_vision * n_layers {
        return Ok(None);
    }
    Ok(Some(
        flat.chunks(n_layers)
            .map(|row| row.to_vec())
            .collect(),
    ))
}

fn forward_hf_env() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for name in [
        "RLX_QWEN25_VL_HF_DIR",
        "RLX_QWEN25_VL_DOWNLOAD",
        "RLX_QWEN25_VL_DEVICE",
        "RLX_QWEN25_VL_PROMPT",
        "RLX_QWEN25_VL_IMAGE_TAG",
        "RLX_AIF_DYNAMICS",
        "HF_TOKEN",
        "HF_HOME",
    ] {
        if let Ok(v) = std::env::var(name) {
            out.push((name.to_string(), v));
        }
    }
    out
}

fn read_npy_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    parse_npy_f32(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn read_npy_u32(path: &Path) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path)?;
    parse_npy_u32(&bytes).with_context(|| format!("parse {}", path.display()))
}

/// Minimal little-endian `.npy` f32 loader (C-order, no pickle).
fn parse_npy_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    let data = npy_payload(bytes)?;
    anyhow::ensure!(data.len().is_multiple_of(4), "npy f32 payload size");
    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn parse_npy_u32(bytes: &[u8]) -> Result<Vec<u32>> {
    let header = npy_header(bytes)?;
    let data = npy_payload(bytes)?;
    if header.contains("'int64'") || header.contains("<i8") {
        anyhow::ensure!(data.len().is_multiple_of(8), "npy i64 payload size");
        Ok(data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32)
            .collect())
    } else {
        anyhow::ensure!(data.len().is_multiple_of(4), "npy u32 payload size");
        Ok(data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

fn npy_header(bytes: &[u8]) -> Result<String> {
    anyhow::ensure!(bytes.starts_with(b"\x93NUMPY"), "not npy");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    Ok(String::from_utf8_lossy(&bytes[10..10 + header_len]).into_owned())
}

fn npy_payload(bytes: &[u8]) -> Result<&[u8]> {
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    Ok(&bytes[10 + header_len..])
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut max = 0f32;
    let mut idx = 0;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > max {
            max = d;
            idx = i;
        }
    }
    (max, idx)
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let av = x as f64;
        let bv = y as f64;
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (1.0 - dot / denom).max(0.0)
    }
}

pub fn top1_match(a: &[f32], b: &[f32]) -> bool {
    let ia = argmax(a);
    let ib = argmax(b);
    ia == ib
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
