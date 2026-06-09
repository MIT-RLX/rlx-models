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

//! Full forward + generate e2e parity (env-gated on `LLADA2_MODEL_DIR`).

use rlx_models::llada2::{GenerateConfig, LLaDA2Runner, load_llada2_partial};
use rlx_models::tide::num_transfer_tokens_schedule;
use rlx_runtime::Device;
use serde::Deserialize;
use std::path::Path;
use std::process::Command;

fn llada2_weights_dir() -> Option<String> {
    let p = std::env::var("LLADA2_MODEL_DIR").ok()?;
    let dir = Path::new(&p);
    if dir.join("model.safetensors").is_file() {
        return Some(p);
    }
    if dir.join("config.json").is_file()
        && std::fs::read_dir(dir).ok()?.any(|e| {
            e.ok()
                .map(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
                .unwrap_or(false)
        })
    {
        return Some(p);
    }
    None
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 1.0;
    }
    dot / (na * nb)
}

#[derive(Debug, Deserialize)]
struct ReferenceForward {
    #[allow(dead_code)]
    test: String,
    #[allow(dead_code)]
    seq_len: usize,
    #[allow(dead_code)]
    vocab_size: usize,
    logits: Vec<Vec<f32>>,
}

#[test]
fn full_forward_logit_cosine_vs_pytorch() {
    let model_dir = match llada2_weights_dir() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: set LLADA2_MODEL_DIR with HF safetensors for e2e forward parity");
            return;
        }
    };

    let seq_len = 8usize;
    let block_length = 4usize;
    let max_layers = std::env::var("LLADA2_E2E_MAX_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    eprintln!("e2e: loading {max_layers} layers from {model_dir}");
    let (cfg, weights) =
        load_llada2_partial(std::path::Path::new(&model_dir), max_layers).expect("load");
    let mut runner = LLaDA2Runner::builder()
        .inline_weights(cfg.clone(), weights)
        .device(Device::Cpu)
        .batch_seq(1, seq_len)
        .build()
        .expect("runner");

    let prompt = [1u32, 2, 3];
    let mut ids = vec![cfg.mask_token_id as f32; seq_len];
    let pos: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    for (i, &t) in prompt.iter().enumerate() {
        ids[i] = t as f32;
    }
    let mask = rlx_models::llada2::block_diffusion_attention_mask(1, seq_len, block_length);
    let mut full = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for r in 0..seq_len {
        for c in 0..seq_len {
            full[r * seq_len + c] = mask[r * seq_len + c];
        }
    }
    let rlx = runner.forward_logits(&ids, &pos, &full).expect("forward");

    let tide_code =
        std::env::var("TIDE_MODEL_CODE").unwrap_or_else(|_| "/Users/Shared/TIDE/model".into());
    let out = Command::new("python3")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/llada2_full_parity_reference.py"
        ))
        .arg("--model-dir")
        .arg(&model_dir)
        .arg("--seq-len")
        .arg(seq_len.to_string())
        .arg("--block-length")
        .arg(block_length.to_string())
        .arg("--max-layers")
        .arg(max_layers.to_string())
        .env("TIDE_MODEL_CODE", &tide_code)
        .env("LLADA2_E2E_MAX_LAYERS", max_layers.to_string())
        .output()
        .expect("reference");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let reference: ReferenceForward = serde_json::from_slice(&out.stdout).expect("json");

    let vocab = cfg.vocab_size;
    let mut min_cos = 1.0f32;
    for p in 0..seq_len {
        let base = p * vocab;
        let cos = cosine(&rlx[base..base + vocab], &reference.logits[p]);
        min_cos = min_cos.min(cos);
        eprintln!("e2e forward pos={p} cosine={cos:.6}");
    }
    assert!(min_cos > 0.99, "min forward cosine {min_cos}");
}

#[test]
fn full_generate_runs_with_offload_on_weights() {
    let model_dir = match llada2_weights_dir() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: LLADA2_MODEL_DIR + weights required for generate e2e");
            return;
        }
    };

    let max_layers = std::env::var("LLADA2_E2E_MAX_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let (cfg, weights) =
        load_llada2_partial(std::path::Path::new(&model_dir), max_layers).expect("load");
    let max_seq = 64usize;
    let mut runner = LLaDA2Runner::builder()
        .inline_weights(cfg.clone(), weights)
        .device(Device::Cpu)
        .batch_seq(1, max_seq)
        .enable_predictive_expert_offload(8)
        .jump_steps(2)
        .moe_collect_stats(true)
        .build()
        .expect("runner");

    let gen_cfg = GenerateConfig {
        block_length: 32,
        steps: 4,
        gen_length: 32,
        collect_stats: true,
        predictive_offload_enabled: true,
        jump_steps: 2,
        ..GenerateConfig::from_model(&cfg)
    };
    assert_eq!(num_transfer_tokens_schedule(32, 4), vec![8, 8, 8, 8]);
    let (tokens, stats) = runner.generate(&gen_cfg, &[1, 2, 3]).expect("generate");
    eprintln!(
        "generate e2e: {} tokens, {} stat steps, offload enabled={}",
        tokens.len(),
        stats.len(),
        runner.predictive_offload_enabled()
    );
    assert!(!tokens.is_empty());
}
