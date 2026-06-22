// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// Kernel-level benchmarks for the native Kyutai TTS primitives at the
// dimensions used by the published `kyutai/tts-1.6b-en_fr` checkpoint:
//
//   backbone:  d_model = 2048, num_heads = 16, head_dim = 128, ff = 8448
//   depformer: d_model = 1024, num_heads = 16, head_dim = 64,  ff = 3072
//   codebook:  card    = 2048, low_rank = 128
//   speaker:   d_kv    = 2048 (after speaker projection)
//
// All workloads are single-step (`t = 1`) — that's the streaming shape during
// generation.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ndarray::{Array, Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlx_kyutai_tts::{
    CrossAttention, LogitsProcessor, LowRankEmbedding, apply_rope_vec, linear, rms_norm,
    rope_tables, sin_pos_embed, softmax_last_dim, swiglu_mlp,
};

const BACKBONE_DIM: usize = 2048;
const BACKBONE_HEADS: usize = 16;
const BACKBONE_HEAD_DIM: usize = BACKBONE_DIM / BACKBONE_HEADS;
const BACKBONE_FF: usize = 8448; // 2048 × 4.125
const DEPFORMER_DIM: usize = 1024;
const DEPFORMER_FF: usize = 3072;
const CARD: usize = 2048;
const LOW_RANK: usize = 128;

fn rand_mat(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array::from_shape_fn((rows, cols), |_| rng.r#gen::<f32>() - 0.5)
}

fn rand_vec(n: usize, seed: u64) -> Array1<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array::from_shape_fn(n, |_| rng.r#gen::<f32>() - 0.5)
}

fn bench_linear(c: &mut Criterion) {
    let mut g = c.benchmark_group("linear");
    for &(t, d_in, d_out, label) in &[
        (1usize, BACKBONE_DIM, BACKBONE_DIM, "self_attn_qkv_step"),
        (1, BACKBONE_DIM, 3 * BACKBONE_DIM, "qkv_pack_step"),
        (1, BACKBONE_DIM, 2 * BACKBONE_FF, "swiglu_in_step"),
        (1, BACKBONE_FF, BACKBONE_DIM, "swiglu_out_step"),
        (1, DEPFORMER_DIM, CARD, "depformer_logits_step"),
    ] {
        let x = rand_mat(t, d_in, 1);
        let w = rand_mat(d_out, d_in, 2);
        g.throughput(Throughput::Elements((d_in * d_out) as u64));
        g.bench_function(label, |b| {
            b.iter(|| {
                let y = linear(black_box(x.view()), black_box(&w));
                black_box(y);
            })
        });
    }
    g.finish();
}

fn bench_rms_norm(c: &mut Criterion) {
    let mut g = c.benchmark_group("rms_norm");
    for &(t, d, label) in &[
        (1usize, BACKBONE_DIM, "backbone_step"),
        (1, DEPFORMER_DIM, "depformer_step"),
    ] {
        let x = rand_mat(t, d, 3);
        let alpha = rand_vec(d, 4);
        g.bench_function(label, |b| {
            b.iter(|| {
                let y = rms_norm(black_box(x.view()), black_box(&alpha));
                black_box(y);
            })
        });
    }
    g.finish();
}

fn bench_swiglu(c: &mut Criterion) {
    let mut g = c.benchmark_group("swiglu_mlp");
    for &(d, ff, label) in &[
        (BACKBONE_DIM, BACKBONE_FF, "backbone_step"),
        (DEPFORMER_DIM, DEPFORMER_FF, "depformer_step"),
    ] {
        let x = rand_mat(1, d, 5);
        let lin_in = rand_mat(2 * ff, d, 6);
        let lin_out = rand_mat(d, ff, 7);
        g.bench_function(label, |b| {
            b.iter(|| {
                let y = swiglu_mlp(black_box(x.view()), black_box(&lin_in), black_box(&lin_out));
                black_box(y);
            })
        });
    }
    g.finish();
}

fn bench_rope(c: &mut Criterion) {
    let mut g = c.benchmark_group("rope");
    g.bench_function("backbone_table_500_positions", |b| {
        let positions: Vec<usize> = (0..500).collect();
        b.iter(|| {
            let (cos, sin) = rope_tables(
                black_box(BACKBONE_HEAD_DIM),
                black_box(10_000),
                black_box(&positions),
            );
            black_box((cos, sin));
        })
    });
    g.bench_function("apply_to_one_head_vec", |b| {
        let (cos, sin) = rope_tables(BACKBONE_HEAD_DIM, 10_000, &[123]);
        let mut q = vec![0.5f32; BACKBONE_HEAD_DIM];
        let mut k = vec![0.5f32; BACKBONE_HEAD_DIM];
        b.iter(|| {
            apply_rope_vec(black_box(&mut q), black_box(&mut k), cos.row(0), sin.row(0));
        })
    });
    g.finish();
}

fn bench_low_rank_embedding(c: &mut Criterion) {
    let mut g = c.benchmark_group("low_rank_embedding");
    let a = rand_mat(CARD, LOW_RANK, 8);
    let b_mat = rand_mat(LOW_RANK, BACKBONE_DIM, 9);
    let lre = LowRankEmbedding::new(a, b_mat);
    g.bench_function("forward_one_backbone_dim", |b| {
        b.iter(|| {
            let y = lre.forward_one(black_box(7));
            black_box(y);
        })
    });

    // Compare against a dense embedding (materialised) — what we'd otherwise
    // store directly. This measures the FLOP / cache-locality tradeoff.
    let dense = lre.materialise();
    g.bench_function("forward_one_dense_lookup", |b| {
        b.iter(|| {
            let row = black_box(&dense).row(black_box(7usize)).to_owned();
            black_box(row);
        })
    });
    g.finish();
}

fn bench_cross_attention_step(c: &mut Criterion) {
    let mut g = c.benchmark_group("cross_attention");
    let d = BACKBONE_DIM;
    let xa = CrossAttention {
        d_model: d,
        num_heads: BACKBONE_HEADS,
        head_dim: BACKBONE_HEAD_DIM,
        w_q: rand_mat(d, d, 10),
        w_k: rand_mat(d, d, 11),
        w_v: rand_mat(d, d, 12),
        w_o: rand_mat(d, d, 13),
        pos_emb: true,
        pos_emb_scale: 1.0,
        pos_max_period: 10_000.0,
    };
    for &t_kv in &[1usize, 4, 16, 64] {
        let ctx = rand_mat(t_kv, d, 14);
        let kv = xa.prepare_kv(&ctx).unwrap();
        let h = rand_vec(d, 15);
        g.bench_with_input(BenchmarkId::new("forward_step", t_kv), &t_kv, |b, _| {
            b.iter(|| {
                let y = xa.forward_step(black_box(&h), black_box(&kv)).unwrap();
                black_box(y);
            })
        });
    }
    g.finish();
}

fn bench_softmax_card(c: &mut Criterion) {
    let mut g = c.benchmark_group("softmax");
    let logits = rand_mat(1, CARD, 16);
    g.throughput(Throughput::Elements(CARD as u64));
    g.bench_function("over_card_2048", |b| {
        b.iter(|| {
            let y = softmax_last_dim(black_box(&logits));
            black_box(y);
        })
    });
    g.finish();
}

fn bench_sampling(c: &mut Criterion) {
    let mut g = c.benchmark_group("sampling");
    let logits = rand_vec(CARD, 17);
    g.bench_function("top_k_256_temp_0_6", |b| {
        let mut lp = LogitsProcessor::new(0.6, 256, 42);
        b.iter(|| {
            let id = lp.sample(black_box(&logits));
            black_box(id);
        })
    });
    g.bench_function("argmax_temp_0", |b| {
        let mut lp = LogitsProcessor::new(0.0, 0, 0);
        b.iter(|| {
            let id = lp.sample(black_box(&logits));
            black_box(id);
        })
    });
    g.finish();
}

fn bench_sin_pos_embed(c: &mut Criterion) {
    c.bench_function("sin_pos_embed_500x2048", |b| {
        b.iter(|| {
            let pe = sin_pos_embed(black_box(500), black_box(BACKBONE_DIM), black_box(10_000.0));
            black_box(pe);
        })
    });
}

criterion_group!(
    benches,
    bench_linear,
    bench_rms_norm,
    bench_swiglu,
    bench_rope,
    bench_low_rank_embedding,
    bench_cross_attention_step,
    bench_softmax_card,
    bench_sampling,
    bench_sin_pos_embed,
);
criterion_main!(benches);
