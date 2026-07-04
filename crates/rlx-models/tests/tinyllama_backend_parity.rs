// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// CPU vs every available backend on a TinyLlama-1.1B-shaped synthetic llama32 graph.
//
//   cargo test -p rlx-models --test tinyllama_backend_parity --features all-backends,tinyllama --release

#![allow(dead_code)]

mod compile_support;

use rlx_models::weight_map::WeightMap;
use rlx_models::{Llama32Config, build_llama32_graph_sized_last_logits};
use rlx_runtime::Device;
use rlx_tinyllama::tinyllama_1_1b_preset;
use std::collections::HashMap;

fn tiny_cfg() -> Llama32Config {
    let mut c = tinyllama_1_1b_preset();
    c.vocab_size = 64;
    c.hidden_size = 32;
    c.intermediate_size = 96;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.max_position_embeddings = 128;
    c.head_dim = Some(8);
    c.rope_theta = 10_000.0;
    c
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
}

fn synthetic_weights(cfg: &Llama32Config) -> WeightMap {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    let int_dim = cfg.intermediate_size;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert(
        "model.embed_tokens.weight".into(),
        (ramp(cfg.vocab_size * h, 0.001), vec![cfg.vocab_size, h]),
    );
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        t.insert(
            format!("{lp}.input_layernorm.weight"),
            (vec![1.0; h], vec![h]),
        );
        t.insert(
            format!("{lp}.post_attention_layernorm.weight"),
            (vec![1.0; h], vec![h]),
        );
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (ramp(q_dim * h, 0.01), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (ramp(kv_dim * h, 0.01), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (ramp(kv_dim * h, 0.01), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (ramp(h * q_dim, 0.01), vec![h, q_dim]),
        );
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (ramp(int_dim * h, 0.01), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (ramp(int_dim * h, 0.01), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (ramp(h * int_dim, 0.01), vec![h, int_dim]),
        );
    }
    t.insert("model.norm.weight".into(), (vec![1.0; h], vec![h]));
    t.insert(
        "lm_head.weight".into(),
        (ramp(cfg.vocab_size * h, 0.001), vec![cfg.vocab_size, h]),
    );
    WeightMap::from_tensors(t)
}

fn run_last_logits(device: Device) -> Vec<f32> {
    let cfg = tiny_cfg();
    let mut wm = synthetic_weights(&cfg);
    let (graph, params) =
        build_llama32_graph_sized_last_logits(&cfg, &mut wm, 1, 4, false).expect("build");
    let mut compiled = compile_support::compile_llama32_prefill(device, graph, params);
    let ids = vec![1.0f32, 2.0, 3.0, 4.0];
    let outs = compiled.run(&[("input_ids", &ids), ("last_token_idx", &[3.0f32])]);
    outs[0].to_vec()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
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

fn assert_backend_matches_cpu(name: &str, device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip tinyllama {name}: {device:?} not available");
        return;
    }
    let cpu = run_last_logits(Device::Cpu);
    let other = run_last_logits(device);
    let c = cosine(&cpu, &other);
    eprintln!("tinyllama cpu vs {name} cosine={c:.8}");
    assert!(c > 0.99, "tinyllama cpu vs {name} cosine {c}");
}

#[test]
fn cpu_reference_logits_finite() {
    let logits = run_last_logits(Device::Cpu);
    assert_eq!(logits.len(), tiny_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[cfg(feature = "metal")]
#[test]
fn metal_matches_cpu() {
    assert_backend_matches_cpu("metal", Device::Metal);
}

#[cfg(feature = "mlx")]
#[test]
fn mlx_matches_cpu() {
    assert_backend_matches_cpu("mlx", Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matches_cpu() {
    assert_backend_matches_cpu("cuda", Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_matches_cpu() {
    assert_backend_matches_cpu("rocm", Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_matches_cpu() {
    assert_backend_matches_cpu("wgpu", Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_matches_cpu() {
    assert_backend_matches_cpu("vulkan", Device::Vulkan);
}
