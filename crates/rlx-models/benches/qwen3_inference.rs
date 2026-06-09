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

//! Criterion bench: Qwen3 prefill + decode on CPU.
//!
//! Three benchmark groups:
//!
//!   1. **prefill** — single forward pass over `[1, S]` tokens for
//!      S ∈ {8, 32, 64, 128}. Dominant work is the per-layer attention
//!      whose cost grows ~S² in the SDPA inner loop plus ~S in MLP.
//!
//!   2. **decode_step** — one cached decode step over a `past_seq`
//!      of P ∈ {0, 16, 64, 128} tokens, producing one new token's
//!      logits. Dominant work scales ~P (linear in cache size, vs
//!      naive's ~P² recompute).
//!
//!   3. **generate_8** — end-to-end: prefill an 8-token prompt then
//!      generate 8 more tokens, naive (recompute each step) vs cached
//!      (KV-cache decode each step). The naive/cached ratio shows the
//!      KV-cache speedup directly.
//!
//! Uses a "small-realistic" synthetic config — large enough that the
//! kernels do meaningful work (hidden=512, 4 layers, GQA 8→4 KV
//! heads), small enough to bench in seconds. The relative shapes
//! (prefill scales quadratically with seq, decode scales linearly
//! with cache) hold at any model size.
//!
//! ```bash
//! cargo bench -p rlx-models --bench qwen3_inference
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rlx_models::qwen3::{Qwen3Config, Qwen3Generator, SampleOpts};
use rlx_models::weight_map::WeightMap;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::hint::black_box;

fn bench_cfg() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 1024,
        hidden_size: 512,
        intermediate_size: 1024,
        num_hidden_layers: 4,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        head_dim: 64,
        max_position_embeddings: 256,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        attention_bias: false,
        qk_norm: true,
        sliding_window: None,
        max_window_layers: usize::MAX,
        use_sliding_window: false,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

fn synthetic_weights(cfg: &Qwen3Config) -> WeightMap {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    let int_dim = cfg.intermediate_size;
    let dh = cfg.head_dim;
    let pat = |n: usize, salt: u32| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(salt)) >> 8;
                (x as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    };
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert(
        "model.embed_tokens.weight".into(),
        (pat(cfg.vocab_size * h, 1), vec![cfg.vocab_size, h]),
    );
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        t.insert(
            format!("{lp}.input_layernorm.weight"),
            (pat(h, 100 + i as u32), vec![h]),
        );
        t.insert(
            format!("{lp}.post_attention_layernorm.weight"),
            (pat(h, 200 + i as u32), vec![h]),
        );
        t.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            (pat(q_dim * h, 300 + i as u32), vec![q_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.k_proj.weight"),
            (pat(kv_dim * h, 400 + i as u32), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.v_proj.weight"),
            (pat(kv_dim * h, 500 + i as u32), vec![kv_dim, h]),
        );
        t.insert(
            format!("{lp}.self_attn.o_proj.weight"),
            (pat(h * q_dim, 600 + i as u32), vec![h, q_dim]),
        );
        t.insert(
            format!("{lp}.self_attn.q_norm.weight"),
            (pat(dh, 700 + i as u32), vec![dh]),
        );
        t.insert(
            format!("{lp}.self_attn.k_norm.weight"),
            (pat(dh, 800 + i as u32), vec![dh]),
        );
        t.insert(
            format!("{lp}.mlp.gate_proj.weight"),
            (pat(int_dim * h, 900 + i as u32), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.up_proj.weight"),
            (pat(int_dim * h, 1000 + i as u32), vec![int_dim, h]),
        );
        t.insert(
            format!("{lp}.mlp.down_proj.weight"),
            (pat(h * int_dim, 1100 + i as u32), vec![h, int_dim]),
        );
    }
    t.insert("model.norm.weight".into(), (pat(h, 2000), vec![h]));
    t.insert(
        "lm_head.weight".into(),
        (pat(cfg.vocab_size * h, 3000), vec![cfg.vocab_size, h]),
    );
    WeightMap::from_tensors(t)
}

fn make_generator(cfg: &Qwen3Config) -> Qwen3Generator {
    let mut wm = synthetic_weights(cfg);
    Qwen3Generator::from_loader(cfg.clone(), &mut wm, Device::Cpu).unwrap()
}

fn bench_prefill(c: &mut Criterion) {
    let cfg = bench_cfg();
    let mut g = c.benchmark_group("qwen3/prefill");
    for &seq in &[8usize, 32, 64, 128] {
        g.throughput(Throughput::Elements(seq as u64));
        g.bench_with_input(BenchmarkId::from_parameter(seq), &seq, |b, &seq| {
            let prompt: Vec<u32> = (1..=seq as u32).collect();
            let mut gn = make_generator(&cfg);
            b.iter(|| {
                gn.prefill(&prompt);
                // One naive step IS a single prefill pass over `seq` tokens.
                let t = gn.step(SampleOpts::greedy()).unwrap();
                black_box(t);
            });
        });
    }
    g.finish();
}

fn bench_decode_step(c: &mut Criterion) {
    let cfg = bench_cfg();
    let mut g = c.benchmark_group("qwen3/decode_step");
    for &past in &[0usize, 16, 64, 128] {
        g.throughput(Throughput::Elements(1));
        g.bench_with_input(BenchmarkId::from_parameter(past), &past, |b, &past| {
            // Build a prompt of length `past + 1` so the first
            // step_cached call seeds the cache and produces a
            // generator at past_seq == past + 1, then each
            // subsequent bench iteration runs ONE decode step.
            let prompt: Vec<u32> = (1..=(past + 1) as u32).collect();
            let mut gn = make_generator(&cfg);
            gn.prefill(&prompt);
            let _seed = gn.step_cached(SampleOpts::greedy()).unwrap();
            b.iter(|| {
                let t = gn.step_cached(SampleOpts::greedy()).unwrap();
                black_box(t);
            });
        });
    }
    g.finish();
}

fn bench_generate_naive_vs_cached(c: &mut Criterion) {
    let cfg = bench_cfg();
    let prompt: Vec<u32> = (1..=8u32).collect();
    let new_tokens = 8usize;

    let mut g = c.benchmark_group("qwen3/generate_8tok");
    g.throughput(Throughput::Elements(new_tokens as u64));

    g.bench_function("naive", |b| {
        let mut gn = make_generator(&cfg);
        b.iter(|| {
            gn.prefill(&prompt);
            let toks = gn.generate(new_tokens, SampleOpts::greedy()).unwrap();
            black_box(toks);
        });
    });

    g.bench_function("cached", |b| {
        let mut gn = make_generator(&cfg);
        b.iter(|| {
            gn.prefill(&prompt);
            let toks = gn
                .generate_cached(new_tokens, SampleOpts::greedy())
                .unwrap();
            black_box(toks);
        });
    });

    // Same workload but with both compile caches enabled. The first
    // iteration warms each bucket (cold compile); subsequent
    // iterations within the same bucket hit cache. Across 8 generated
    // tokens with `max_past=64` buckets [1..2, 2..3, 3..5, 5..9, …],
    // a fresh generation crosses ~4 bucket boundaries → 4 cold
    // compiles vs 8 in the uncached path. Across the bench's repeat
    // iterations, all buckets are hot, so steady-state cost is near
    // pure compute.
    g.bench_function("cached+compile-cache", |b| {
        let mut gn = make_generator(&cfg)
            .with_prefill_cache(/*capacity*/ 4)
            .with_decode_cache(/*max_past*/ 64);
        b.iter(|| {
            gn.prefill(&prompt);
            let toks = gn
                .generate_cached(new_tokens, SampleOpts::greedy())
                .unwrap();
            black_box(toks);
        });
    });

    g.finish();
}

fn bench_decode_step_cached_compile(c: &mut Criterion) {
    let cfg = bench_cfg();
    let mut g = c.benchmark_group("qwen3/decode_step_compile_cached");
    for &past in &[0usize, 16, 64] {
        g.throughput(Throughput::Elements(1));
        g.bench_with_input(BenchmarkId::from_parameter(past), &past, |b, &past| {
            let prompt: Vec<u32> = (1..=(past + 1) as u32).collect();
            let mut gn = make_generator(&cfg).with_decode_cache(128);
            gn.prefill(&prompt);
            // Two seed calls so we're past the very first compile by
            // the time b.iter runs; subsequent iters within the same
            // bucket are pure compute.
            let _ = gn.step_cached(SampleOpts::greedy()).unwrap();
            let _ = gn.step_cached(SampleOpts::greedy()).unwrap();
            b.iter(|| {
                let t = gn.step_cached(SampleOpts::greedy()).unwrap();
                black_box(t);
            });
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_prefill,
    bench_decode_step,
    bench_decode_step_cached_compile,
    bench_generate_naive_vs_cached
);
criterion_main!(benches);
