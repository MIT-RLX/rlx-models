// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// One-step backbone benchmark — a full Helium-style transformer step at the
// dimensions used by `kyutai/tts-1.6b-en_fr`:
//
//   16 layers × {RMSNorm + RoPE causal self-attn + RMSNorm + cross-attn + RMSNorm + SwiGLU}
//   d_model = 2048, num_heads = 16, head_dim = 128, ff = 8448, ctx = 500
//
// Reports wall-clock for one generation step (t=1) with a warm KV cache, both
// with and without the cross-attention block enabled.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::{Array, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlx_kyutai_tts::{
    AttnWeights, BackboneConfig, CrossAttention, LayerWeights, StreamingTransformer,
    config::PositionalEmbedding,
};

const D: usize = 2048;
const H: usize = 16;
const HD: usize = D / H;
const FF: usize = 8448;
const CTX: usize = 500;
const NUM_LAYERS: usize = 16;
const SPEAKER_CTX_FRAMES: usize = 16;

fn rand_mat(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array::from_shape_fn((rows, cols), |_| (rng.r#gen::<f32>() - 0.5) * 0.02)
}

fn rand_vec(n: usize, seed: u64) -> ndarray::Array1<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array::from_shape_fn(n, |_| (rng.r#gen::<f32>() - 0.5) * 0.02)
}

fn make_layer(seed_base: u64, with_cross: bool) -> LayerWeights {
    let cross = if with_cross {
        Some(CrossAttention {
            d_model: D,
            num_heads: H,
            head_dim: HD,
            w_q: rand_mat(D, D, seed_base + 100),
            w_k: rand_mat(D, D, seed_base + 101),
            w_v: rand_mat(D, D, seed_base + 102),
            w_o: rand_mat(D, D, seed_base + 103),
            pos_emb: true,
            pos_emb_scale: 1.0,
            pos_max_period: 10_000.0,
        })
    } else {
        None
    };
    LayerWeights {
        norm1_alpha: rand_vec(D, seed_base + 1),
        norm2_alpha: rand_vec(D, seed_base + 2),
        attn: AttnWeights {
            in_proj: rand_mat(3 * D, D, seed_base + 3),
            out_proj: rand_mat(D, D, seed_base + 4),
            num_heads: H,
            head_dim: HD,
        },
        gate_in: rand_mat(2 * FF, D, seed_base + 5),
        gate_out: rand_mat(D, FF, seed_base + 6),
        cross_attn: cross,
        norm_cross_alpha: if with_cross {
            Some(rand_vec(D, seed_base + 7))
        } else {
            None
        },
    }
}

fn make_backbone(with_cross: bool) -> StreamingTransformer {
    let cfg = BackboneConfig {
        d_model: D,
        num_heads: H,
        num_layers: NUM_LAYERS,
        dim_feedforward: FF,
        causal: true,
        context: CTX,
        max_period: 10_000,
        positional_embedding: PositionalEmbedding::Rope,
        cross_attention: with_cross,
    };
    let layers = (0..NUM_LAYERS as u64)
        .map(|i| make_layer(i * 1000, with_cross))
        .collect();
    StreamingTransformer::new(cfg, layers).unwrap()
}

fn warm_up(t: &mut StreamingTransformer, x: &Array2<f32>, steps: usize) {
    for _ in 0..steps {
        let _ = t.forward(x, None).unwrap();
    }
}

fn bench_backbone_step_no_cross(c: &mut Criterion) {
    let mut t = make_backbone(false);
    let x = rand_mat(1, D, 42);
    warm_up(&mut t, &x, 10);

    c.bench_function("backbone_step/16L_d2048_h16_no_cross", |b| {
        b.iter(|| {
            let y = t.forward(black_box(&x), None).unwrap();
            black_box(y);
        });
    });
}

fn bench_backbone_step_with_cross(c: &mut Criterion) {
    let mut t = make_backbone(true);
    let x = rand_mat(1, D, 42);
    let speaker_ctx = rand_mat(SPEAKER_CTX_FRAMES, D, 99);

    // Prepare K/V cache from the first layer's cross-attn weights — we reuse
    // it across every layer for the benchmark (in practice each layer has its
    // own KV cache; this still measures the dominant cost of `forward_step`).
    let kv = {
        // grab the cross-attn out of the (cloned) layer 0 to prep KV — we
        // can't directly reach inside the transformer's private layers, so
        // re-instantiate one CrossAttention with the same weights as the
        // backbone uses for layer 0.
        let xa_layer0 = make_layer(0, true);
        let xa = xa_layer0.cross_attn.unwrap();
        xa.prepare_kv(&speaker_ctx).unwrap()
    };

    warm_up(&mut t, &x, 5);

    c.bench_function("backbone_step/16L_d2048_h16_cross_16frames", |b| {
        b.iter(|| {
            let y = t.forward(black_box(&x), Some(black_box(&kv))).unwrap();
            black_box(y);
        });
    });
}

fn bench_backbone_step_short_context(c: &mut Criterion) {
    // Cold KV cache (first step) — measures the floor cost when context is empty.
    c.bench_function("backbone_step/16L_d2048_h16_cold_kv", |b| {
        b.iter_batched(
            || (make_backbone(false), rand_mat(1, D, 7)),
            |(mut t, x)| {
                let y = t.forward(black_box(&x), None).unwrap();
                black_box(y);
            },
            criterion::BatchSize::PerIteration,
        );
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20).warm_up_time(std::time::Duration::from_secs(2));
    targets =
        bench_backbone_step_no_cross,
        bench_backbone_step_with_cross,
        bench_backbone_step_short_context,
}
criterion_main!(benches);
