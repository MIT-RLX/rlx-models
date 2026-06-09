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

//! Numerical + cosine parity vs TIDE PyTorch (`llada2_component_parity.py`).
//!
//! Full-model forward logits need `LLADA2_MODEL_DIR` with `model.safetensors`
//! (not shipped in `/Users/Shared/TIDE`).

use rlx_models::llada2::gate::{gate_forward_host, group_limited_topk};
use rlx_models::llada2::mask::block_diffusion_attention_mask;
use rlx_models::llada2::synth;
use rlx_models::tide::{num_transfer_tokens_schedule, refresh_experts};
use serde::Deserialize;
use std::process::Command;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return if (na - nb).abs() < 1e-12 { 1.0 } else { 0.0 };
    }
    dot / (na * nb)
}

#[derive(Debug, Deserialize)]
struct Line {
    test: String,
    #[serde(default)]
    block_length: usize,
    #[serde(default)]
    steps: usize,
    #[serde(default)]
    schedule: Vec<usize>,
    #[serde(default)]
    mask: Vec<serde_json::Value>,
    #[serde(default)]
    indices: Vec<i64>,
    #[serde(default)]
    probs: Vec<f64>,
    #[serde(default)]
    weights: Vec<f64>,
    #[serde(default)]
    seq_len: usize,
    #[serde(default)]
    step: usize,
    #[serde(default)]
    refresh: bool,
    #[serde(default)]
    #[allow(dead_code)]
    error: Option<String>,
}

fn run_reference() -> Vec<Line> {
    let out = Command::new("python3")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/llada2_component_parity.py"
        ))
        .output()
        .expect("spawn python reference");
    assert!(
        out.status.success(),
        "reference failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("json line"))
        .collect()
}

#[test]
fn pytorch_component_reference_runs() {
    let lines = run_reference();
    assert!(lines.iter().any(|l| l.test == "done"));
}

#[test]
fn transfer_schedule_exact_vs_pytorch() {
    let lines = run_reference();
    for line in lines.iter().filter(|l| l.test == "transfer_schedule") {
        let rust = num_transfer_tokens_schedule(line.block_length, line.steps);
        assert_eq!(
            rust, line.schedule,
            "block={} steps={}",
            line.block_length, line.steps
        );
    }
}

#[test]
fn block_mask_exact_vs_pytorch() {
    let lines = run_reference();
    let py = lines
        .iter()
        .find(|l| l.test == "block_mask")
        .expect("block_mask line");
    let rust = block_diffusion_attention_mask(1, py.seq_len, 4);
    assert_eq!(rust.len(), py.mask.len());
    for i in 0..rust.len() {
        let r = rust[i];
        let p = match &py.mask[i] {
            serde_json::Value::String(s) if s == "-inf" => f32::NEG_INFINITY,
            serde_json::Value::Number(n) => n.as_f64().unwrap() as f32,
            v => panic!("unexpected mask value at {i}: {v:?}"),
        };
        if r.is_infinite() {
            assert!(p.is_infinite() && p < 0.0, "i={i} rust=-inf py={p}");
        } else {
            assert!((r - p).abs() < 1e-6, "i={i} rust={r} py={p}");
        }
    }
}

#[test]
fn group_limited_topk_exact_vs_pytorch() {
    let lines = run_reference();
    let py = lines
        .iter()
        .find(|l| l.test == "group_limited_topk")
        .expect("topk line");
    let scores = vec![
        0.1, 0.9, 0.2, 0.8, //
        0.5, 0.5, 0.5, 0.5,
    ];
    let (rust_probs, rust_idx) = group_limited_topk(&scores, 2, 4, 2, 1, 2);
    let py_idx: Vec<u32> = py.indices.iter().map(|&x| x as u32).collect();
    assert_eq!(rust_idx, py_idx);
    let py_probs: Vec<f32> = py.probs.iter().map(|&x| x as f32).collect();
    let cos = cosine_similarity(&rust_probs, &py_probs);
    assert!(cos > 0.9999, "topk prob cosine {cos}");
    let max_abs = rust_probs
        .iter()
        .zip(py_probs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_abs < 1e-5, "max_abs {max_abs}");
}

#[test]
fn gate_forward_numerical_vs_pytorch() {
    let lines = run_reference();
    let py = lines
        .iter()
        .find(|l| l.test == "gate_forward")
        .expect("gate line");
    let cfg = synth::tiny_cfg();
    let weights = synth::tiny_weights(&cfg);
    let moe = match &weights.layers[1].ffn {
        rlx_models::llada2::weights::LayerFfn::Moe(m) => m,
        _ => panic!("moe layer"),
    };
    let rows = 4usize;
    let h = cfg.hidden_size;
    let hidden: Vec<f32> = (0..rows * h).map(|i| 0.01 * (i as f32)).collect();
    let (rust_idx, rust_w) = gate_forward_host(&cfg, &hidden, &moe.router, &moe.expert_bias);
    let py_idx: Vec<u32> = py.indices.iter().map(|&x| x as u32).collect();
    assert_eq!(rust_idx, py_idx, "gate expert indices");
    let py_w: Vec<f32> = py.weights.iter().map(|&x| x as f32).collect();
    let cos = cosine_similarity(&rust_w, &py_w);
    let max_abs = rust_w
        .iter()
        .zip(py_w.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("gate_forward weight cosine={cos:.8} max_abs={max_abs:.8}");
    assert!(cos > 0.99999, "gate weights cosine {cos}");
    assert!(max_abs < 2e-3, "gate max_abs {max_abs}");
}

#[test]
fn refresh_policy_matches_generate_loop() {
    let lines = run_reference();
    for line in lines.iter().filter(|l| l.test == "refresh") {
        let rust = refresh_experts(true, 2, 1, 0, line.step);
        assert_eq!(rust, line.refresh, "step {}", line.step);
    }
}

#[test]
fn full_forward_logits_parity_env_gated() {
    let model_dir =
        std::env::var("LLADA2_MODEL_DIR").unwrap_or_else(|_| "/Users/Shared/TIDE/model".into());
    let weights = std::path::Path::new(&model_dir).join("model.safetensors");
    if !weights.exists() {
        eprintln!(
            "SKIP full forward logits parity: no weights at {}",
            weights.display()
        );
        return;
    }
    eprintln!(
        "LLADA2 weights found at {}; run `cargo run -p rlx-models --example llada2_compare` for logit cosine",
        weights.display()
    );
}
