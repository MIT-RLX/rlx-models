// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Wall-clock prefill report across backends (synthetic MiniCPM5-shaped graph).
//
//   cargo test -p rlx-models --test minicpm5_bench_report --features all-backends --release \
//     minicpm5_bench_report -- --nocapture

mod compile_support;

use rlx_minicpm5::minicpm5_1b_preset;
use rlx_models::weight_map::WeightMap;
use rlx_models::{Llama32Config, build_llama32_graph_sized_last_logits};
use rlx_runtime::Device;
use std::collections::HashMap;
use std::time::Instant;

fn tiny_cfg() -> Llama32Config {
    let mut c = minicpm5_1b_preset();
    c.vocab_size = 128;
    c.hidden_size = 64;
    c.intermediate_size = 192;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.max_position_embeddings = 256;
    c.head_dim = Some(16);
    c
}

fn synthetic_weights(cfg: &Llama32Config) -> WeightMap {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    let int_dim = cfg.intermediate_size;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let z = |n: usize| vec![0.01f32; n];
    t.insert(
        "model.embed_tokens.weight".into(),
        (z(cfg.vocab_size * h), vec![cfg.vocab_size, h]),
    );
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            t.insert(format!("{lp}.{name}"), (z(h), vec![h]));
        }
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (z(q_dim * h), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (z(kv_dim * h), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (z(kv_dim * h), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (z(h * q_dim), vec![h, q_dim]),
        );
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (z(int_dim * h), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (z(int_dim * h), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (z(h * int_dim), vec![h, int_dim]),
        );
    }
    t.insert("model.norm.weight".into(), (z(h), vec![h]));
    t.insert(
        "lm_head.weight".into(),
        (z(cfg.vocab_size * h), vec![cfg.vocab_size, h]),
    );
    WeightMap::from_tensors(t)
}

fn time_prefill(device: Device, seq: usize, warmup: usize, iters: usize) -> Option<f64> {
    if device != Device::Cpu && !rlx_runtime::is_available(device) {
        return None;
    }
    let cfg = tiny_cfg();
    let mut wm = synthetic_weights(&cfg);
    let (graph, params) =
        build_llama32_graph_sized_last_logits(&cfg, &mut wm, 1, seq, false).ok()?;
    let mut compiled = compile_support::compile_llama32_prefill(device, graph, params);
    let ids: Vec<f32> = (0..seq).map(|i| (i + 1) as f32).collect();
    let last = vec![(seq - 1) as f32];
    for _ in 0..warmup {
        let _ = compiled.run(&[("input_ids", &ids), ("last_token_idx", &last)]);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = compiled.run(&[("input_ids", &ids), ("last_token_idx", &last)]);
    }
    Some(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
}

fn bench_device(name: &str, device: Device) {
    if device != Device::Cpu && !rlx_runtime::is_available(device) {
        eprintln!("  {name}: skip (unavailable)");
        return;
    }
    for seq in [8usize, 32, 64] {
        if let Some(ms) = time_prefill(device, seq, 1, 5) {
            eprintln!("  {name} L={seq}: {ms:.3} ms/prefill");
        }
    }
}

#[test]
fn minicpm5_bench_report() {
    eprintln!("\n=== MiniCPM5 synthetic prefill bench (ms) ===");
    bench_device("cpu", Device::Cpu);
    #[cfg(feature = "metal")]
    bench_device("metal", Device::Metal);
    #[cfg(feature = "mlx")]
    bench_device("mlx", Device::Mlx);
    #[cfg(feature = "cuda")]
    bench_device("cuda", Device::Cuda);
    #[cfg(feature = "rocm")]
    bench_device("rocm", Device::Rocm);
    #[cfg(feature = "gpu")]
    bench_device("wgpu", Device::Gpu);
    #[cfg(feature = "vulkan")]
    bench_device("vulkan", Device::Vulkan);
}
