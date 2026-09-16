// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// CPU vs available backends on a HY-MT-shaped (Qwen3 + QK-norm) synthetic graph.
//
//   cargo test -p rlx-models --test hy_mt_backend_parity --features all-backends,hy-mt --release

#![allow(dead_code)]

mod compile_support;
mod qwen3_common;

use rlx_hy_mt::hy_mt_1_8b_preset;
use rlx_models::qwen3::{Qwen3Config, build_qwen3_graph_sized_last_logits};
use rlx_models::weight_map::WeightMap;
use rlx_runtime::Device;
use std::collections::HashMap;

fn tiny_hy_mt_cfg() -> Qwen3Config {
    let mut c = hy_mt_1_8b_preset();
    c.vocab_size = 64;
    c.hidden_size = 32;
    c.intermediate_size = 64;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.head_dim = 8;
    c.max_position_embeddings = 128;
    c.rope_theta = 10_000.0;
    c
}

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| 0.001 + scale * (i as f32) * 0.01).collect()
}

fn synthetic_weights(cfg: &Qwen3Config) -> WeightMap {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    let int_dim = cfg.intermediate_size;
    let dh = cfg.head_dim;
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
        // HY-MT HF names — WeightMap aliases also accept q_norm.
        t.insert(
            format!("{lp}.self_attn.query_layernorm.weight"),
            (vec![1.0; dh], vec![dh]),
        );
        t.insert(
            format!("{lp}.self_attn.key_layernorm.weight"),
            (vec![1.0; dh], vec![dh]),
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
    WeightMap::from_tensors(t)
}

fn run_last_logits(device: Device) -> Vec<f32> {
    let cfg = tiny_hy_mt_cfg();
    let mut wm = synthetic_weights(&cfg);
    let (graph, params) =
        build_qwen3_graph_sized_last_logits(&cfg, &mut wm, 1, 4, true).expect("build");
    let mut compiled = compile_support::compile_qwen3_prefill(device, graph, params);
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
        eprintln!("skip hy-mt {name}: {device:?} not available");
        return;
    }
    let cpu = run_last_logits(Device::Cpu);
    let other = run_last_logits(device);
    let c = cosine(&cpu, &other);
    eprintln!("hy-mt cpu vs {name} cosine={c:.8}");
    assert!(c > 0.99, "hy-mt cpu vs {name} cosine {c}");
}

#[test]
fn cpu_reference_logits_finite() {
    let logits = run_last_logits(Device::Cpu);
    assert_eq!(logits.len(), tiny_hy_mt_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn hunyuan_norm_aliases_resolve() {
    let cfg = tiny_hy_mt_cfg();
    let mut wm = synthetic_weights(&cfg);
    let (q, shape) = wm
        .take("model.layers.0.self_attn.q_norm.weight")
        .expect("alias query_layernorm → q_norm");
    assert_eq!(shape, vec![cfg.head_dim]);
    assert_eq!(q.len(), cfg.head_dim);
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

#[cfg(feature = "gpu")]
#[test]
fn wgpu_matches_cpu() {
    assert_backend_matches_cpu("wgpu", Device::Gpu);
}
