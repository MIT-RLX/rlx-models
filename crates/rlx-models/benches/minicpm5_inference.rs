// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Criterion bench: MiniCPM5-shaped llama32 prefill (CPU + optional backends).
//!
//! ```bash
//! cargo bench -p rlx-models --bench minicpm5_inference --release
//! just bench-minicpm5-all-backends
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rlx_flow::CompileProfile;
use rlx_ir::logical_kernel::KernelDispatchConfig;
use rlx_minicpm5::minicpm5_1b_preset;
use rlx_models::flow_bridge::compile_options_from_profile;
use rlx_models::weight_map::WeightMap;
use rlx_models::{Llama32Config, build_llama32_graph_sized_last_logits};
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;
use std::hint::black_box;

fn bench_cfg() -> Llama32Config {
    let mut c = minicpm5_1b_preset();
    c.vocab_size = 512;
    c.hidden_size = 256;
    c.intermediate_size = 768;
    c.num_hidden_layers = 4;
    c.num_attention_heads = 8;
    c.num_key_value_heads = 2;
    c.max_position_embeddings = 512;
    c.head_dim = Some(32);
    c.rope_theta = 5_000_000.0;
    c
}

fn pat(n: usize, salt: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(salt)) >> 8;
            (x as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

fn synthetic_weights(cfg: &Llama32Config) -> WeightMap {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    let int_dim = cfg.intermediate_size;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert(
        "model.embed_tokens.weight".into(),
        (pat(cfg.vocab_size * h, 1), vec![cfg.vocab_size, h]),
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
            (pat(q_dim * h, 10 + i as u32), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (pat(kv_dim * h, 20 + i as u32), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (pat(kv_dim * h, 30 + i as u32), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (pat(h * q_dim, 40 + i as u32), vec![h, q_dim]),
        );
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (pat(int_dim * h, 50 + i as u32), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (pat(int_dim * h, 60 + i as u32), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (pat(h * int_dim, 70 + i as u32), vec![h, int_dim]),
        );
    }
    t.insert("model.norm.weight".into(), (vec![1.0; h], vec![h]));
    t.insert(
        "lm_head.weight".into(),
        (pat(cfg.vocab_size * h, 99), vec![cfg.vocab_size, h]),
    );
    WeightMap::from_tensors(t)
}

fn compile_prefill(device: Device, seq: usize) -> CompiledGraph {
    let cfg = bench_cfg();
    let mut wm = synthetic_weights(&cfg);
    let (graph, params) =
        build_llama32_graph_sized_last_logits(&cfg, &mut wm, 1, seq, false).unwrap();
    let profile = CompileProfile::llama32_prefill();
    let opts = compile_options_from_profile(&profile, device, KernelDispatchConfig::default());
    let mut compiled = Session::new(device).compile_with(graph, &opts);
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    compiled
}

fn bench_prefill_on(c: &mut Criterion, device: Device, label: &'static str) {
    if device != Device::Cpu && !rlx_runtime::is_available(device) {
        return;
    }
    let mut group = c.benchmark_group(format!("minicpm5_prefill/{label}"));
    for seq in [8usize, 32, 64] {
        group.throughput(Throughput::Elements(seq as u64));
        group.bench_with_input(BenchmarkId::new(label, seq), &seq, |b, &seq| {
            let mut compiled = compile_prefill(device, seq);
            let ids: Vec<f32> = (0..seq).map(|i| (i + 1) as f32).collect();
            let last = vec![(seq - 1) as f32];
            b.iter(|| {
                let outs = compiled.run(&[
                    ("input_ids", black_box(ids.as_slice())),
                    ("last_token_idx", black_box(last.as_slice())),
                ]);
                black_box(&outs[0]);
            });
        });
    }
    group.finish();
}

fn bench_prefill(c: &mut Criterion) {
    bench_prefill_on(c, Device::Cpu, "cpu");
    #[cfg(feature = "metal")]
    bench_prefill_on(c, Device::Metal, "metal");
    #[cfg(feature = "mlx")]
    bench_prefill_on(c, Device::Mlx, "mlx");
    #[cfg(feature = "cuda")]
    bench_prefill_on(c, Device::Cuda, "cuda");
    #[cfg(feature = "rocm")]
    bench_prefill_on(c, Device::Rocm, "rocm");
    #[cfg(feature = "gpu")]
    bench_prefill_on(c, Device::Gpu, "wgpu");
    #[cfg(feature = "vulkan")]
    bench_prefill_on(c, Device::Vulkan, "vulkan");
}

criterion_group!(benches, bench_prefill);
criterion_main!(benches);
