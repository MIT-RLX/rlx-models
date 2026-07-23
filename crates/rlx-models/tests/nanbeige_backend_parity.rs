// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Looped-Transformer (Nanbeige) synthetic parity: CPU vs every available RLX backend.
// Exercises `num_loops = 2` prefill + decode paths shared with `rlx-llama32`.
//
//   cargo test -p rlx-models --test nanbeige_backend_parity --features all-backends,nanbeige,llama32 --release -- --nocapture
//   just features=all-backends test-nanbeige-backends

#![allow(dead_code)]

mod compile_support;

use rlx_nanbeige::nanbeige42_3b_preset;
use rlx_models::weight_map::WeightMap;
use rlx_models::{
    Llama32Config, build_llama32_decode_graph_sized_ext, build_llama32_graph_sized_last_logits,
};
use rlx_runtime::Device;
use std::collections::HashMap;

fn tiny_looped_cfg() -> Llama32Config {
    let mut c = nanbeige42_3b_preset();
    c.vocab_size = 64;
    c.hidden_size = 32;
    c.intermediate_size = 96;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.max_position_embeddings = 128;
    c.head_dim = Some(8);
    c.num_loops = 2;
    c.skip_loop_final_norm = false;
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
    for i in 0..cfg.physical_layers() {
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

fn run_prefill_logits(device: Device) -> Vec<f32> {
    let cfg = tiny_looped_cfg();
    assert_eq!(cfg.kv_layers(), 4);
    let mut wm = synthetic_weights(&cfg);
    let (graph, params) =
        build_llama32_graph_sized_last_logits(&cfg, &mut wm, 1, 4, false).expect("prefill build");
    let mut compiled = compile_support::compile_llama32_prefill(device, graph, params);
    let ids = vec![1.0f32, 2.0, 3.0, 4.0];
    let outs = compiled.run(&[("input_ids", &ids), ("last_token_idx", &[3.0f32])]);
    outs[0].to_vec()
}

fn run_decode_logits(device: Device) -> Vec<f32> {
    let cfg = tiny_looped_cfg();
    let past_seq = 4usize;
    let mut wm = synthetic_weights(&cfg);
    let (graph, params) =
        build_llama32_decode_graph_sized_ext(&cfg, &mut wm, 1, past_seq, false).expect("decode build");
    let mut compiled = compile_support::compile_llama32_decode(device, graph, params);

    let kv_dim = cfg.kv_proj_dim();
    let zeros = vec![0.0f32; past_seq * kv_dim];
    let ids = vec![7.0f32];
    let mut named: Vec<(String, Vec<f32>)> = Vec::with_capacity(1 + 2 * cfg.kv_layers());
    named.push(("input_ids".into(), ids));
    for i in 0..cfg.kv_layers() {
        named.push((format!("past_k_{i}"), zeros.clone()));
        named.push((format!("past_v_{i}"), zeros.clone()));
    }
    let refs: Vec<(&str, &[f32])> = named
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    let outs = compiled.run(&refs);
    outs[0].to_vec()
}

fn assert_prefill_matches_cpu(name: &str, device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip nanbeige prefill {name}: {device:?} not available");
        return;
    }
    rlx_llama32::validate_device(&tiny_looped_cfg(), device, false)
        .unwrap_or_else(|e| panic!("nanbeige validate {name}: {e:#}"));
    let cpu = run_prefill_logits(Device::Cpu);
    let other = run_prefill_logits(device);
    let c = cosine(&cpu, &other);
    eprintln!("nanbeige looped prefill cpu vs {name} cosine={c:.8}");
    assert!(c > 0.99, "nanbeige looped prefill cpu vs {name} cosine {c}");
}

fn assert_decode_matches_cpu(name: &str, device: Device) {
    if !rlx_runtime::is_available(device) {
        eprintln!("skip nanbeige decode {name}: {device:?} not available");
        return;
    }
    let cpu = run_decode_logits(Device::Cpu);
    let other = run_decode_logits(device);
    let c = cosine(&cpu, &other);
    eprintln!("nanbeige looped decode cpu vs {name} cosine={c:.8}");
    assert!(c > 0.99, "nanbeige looped decode cpu vs {name} cosine {c}");
}

#[test]
fn cpu_looped_prefill_logits_finite() {
    let cfg = tiny_looped_cfg();
    assert_eq!(cfg.num_loops, 2);
    assert_eq!(cfg.kv_layers(), cfg.physical_layers() * 2);
    let logits = run_prefill_logits(Device::Cpu);
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn cpu_looped_decode_logits_finite() {
    let logits = run_decode_logits(Device::Cpu);
    assert_eq!(logits.len(), tiny_looped_cfg().vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn all_standard_devices_validate() {
    let cfg = tiny_looped_cfg();
    for &dev in rlx_llama32::STANDARD_DEVICES {
        rlx_llama32::validate_device(&cfg, dev, false).unwrap();
        rlx_llama32::validate_device(&cfg, dev, true).unwrap();
    }
}

#[cfg(feature = "metal")]
#[test]
fn metal_prefill_matches_cpu() {
    assert_prefill_matches_cpu("metal", Device::Metal);
}

#[cfg(feature = "metal")]
#[test]
fn metal_decode_matches_cpu() {
    assert_decode_matches_cpu("metal", Device::Metal);
}

#[cfg(feature = "mlx")]
#[test]
fn mlx_prefill_matches_cpu() {
    assert_prefill_matches_cpu("mlx", Device::Mlx);
}

#[cfg(feature = "mlx")]
#[test]
fn mlx_decode_matches_cpu() {
    assert_decode_matches_cpu("mlx", Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_prefill_matches_cpu() {
    assert_prefill_matches_cpu("cuda", Device::Cuda);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_decode_matches_cpu() {
    assert_decode_matches_cpu("cuda", Device::Cuda);
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_prefill_matches_cpu() {
    assert_prefill_matches_cpu("rocm", Device::Rocm);
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_decode_matches_cpu() {
    assert_decode_matches_cpu("rocm", Device::Rocm);
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_prefill_matches_cpu() {
    assert_prefill_matches_cpu("wgpu", Device::Gpu);
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_decode_matches_cpu() {
    assert_decode_matches_cpu("wgpu", Device::Gpu);
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_prefill_matches_cpu() {
    assert_prefill_matches_cpu("vulkan", Device::Vulkan);
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_decode_matches_cpu() {
    assert_decode_matches_cpu("vulkan", Device::Vulkan);
}

#[cfg(feature = "coreml")]
#[test]
fn coreml_ane_validates() {
    let cfg = tiny_looped_cfg();
    rlx_llama32::validate_device(&cfg, Device::Ane, false).unwrap();
}
