#![allow(dead_code)]

//! Shared synthetic cross-backend checks for Kyutai TTS RLX graphs.

use rlx_kyutai_tts::config::{KyutaiTtsConfig, PositionalEmbedding};
use rlx_kyutai_tts::rlx_lm::{TtsDims, temporal_decode_bucketed_rlx};
use rlx_runtime::{Device, is_available};
use std::collections::HashMap;

pub const BACKENDS: &[(Device, &str)] = &[
    (Device::Cpu, "CPU"),
    (Device::Metal, "Metal"),
    (Device::Mlx, "MLX"),
    (Device::Cuda, "CUDA"),
    (Device::Rocm, "ROCm"),
    (Device::Gpu, "wgpu/Gpu"),
    (Device::Vulkan, "Vulkan"),
    (Device::Ane, "CoreML/ANE"),
];

pub fn fill(map: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str, shape: &[usize]) {
    let n: usize = shape.iter().product();
    let seed: u32 = key
        .bytes()
        .fold(2166136261u32, |h, b| (h ^ b as u32).wrapping_mul(16777619));
    let mut s = seed | 1;
    let mut data = Vec::with_capacity(n);
    let is_norm = key.ends_with(".alpha");
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let u = (s >> 8) as f32 / (1u32 << 24) as f32;
        data.push(if is_norm {
            0.9 + 0.2 * u
        } else {
            (u - 0.5) * 0.04
        });
    }
    map.insert(key.to_string(), (data, shape.to_vec()));
}

pub fn tiny_cfg() -> KyutaiTtsConfig {
    let mut cfg = KyutaiTtsConfig::v1_6b_en_fr();
    cfg.dim = 64;
    cfg.num_heads = 4;
    cfg.num_layers = 2;
    cfg.hidden_scale = 4.0;
    cfg.text_card = 10;
    cfg.context = 8;
    cfg.positional_embedding = PositionalEmbedding::Rope;
    cfg
}

pub fn synthetic_weights(cfg: &KyutaiTtsConfig) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let d = cfg.dim;
    let h = (cfg.dim as f32 * cfg.hidden_scale / 2.0).round() as usize;
    let vocab = cfg.text_card;
    let mut w = HashMap::new();
    fill(&mut w, "text_linear.weight", &[vocab, d]);
    fill(&mut w, "out_norm.alpha", &[1, 1, d]);
    for li in 0..cfg.num_layers {
        let p = format!("transformer.layers.{li}");
        fill(&mut w, &format!("{p}.norm1.alpha"), &[1, 1, d]);
        fill(&mut w, &format!("{p}.norm2.alpha"), &[1, 1, d]);
        fill(
            &mut w,
            &format!("{p}.self_attn.in_proj_weight"),
            &[3 * d, d],
        );
        fill(&mut w, &format!("{p}.self_attn.out_proj.weight"), &[d, d]);
        fill(&mut w, &format!("{p}.gating.linear_in.weight"), &[2 * h, d]);
        fill(&mut w, &format!("{p}.gating.linear_out.weight"), &[d, h]);
        fill(
            &mut w,
            &format!("{p}.cross_attention.in_proj_weight"),
            &[3 * d, d],
        );
        fill(
            &mut w,
            &format!("{p}.cross_attention.out_proj.weight"),
            &[d, d],
        );
        fill(&mut w, &format!("{p}.norm_cross.weight"), &[d]);
        fill(&mut w, &format!("{p}.norm_cross.bias"), &[d]);
    }
    w
}

pub fn cosine(a: &[f32], b: &[f32]) -> (f64, f32) {
    let (mut dot, mut na, mut nb, mut maxd) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
        maxd = maxd.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), maxd)
}

pub fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(
            (0, f32::MIN),
            |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) },
        )
        .0
}

struct Fixture {
    dims: TtsDims,
    weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
    emb: Vec<f32>,
    cross_ctx: Vec<f32>,
    upper: usize,
}

fn fixture() -> Fixture {
    let cfg = tiny_cfg();
    let d = cfg.dim;
    let mut emb = vec![0.0f32; d];
    for (i, v) in emb.iter_mut().enumerate() {
        *v = 0.01 * (i as f32 + 1.0);
    }
    Fixture {
        dims: TtsDims::from_cfg(&cfg, 1),
        weights: synthetic_weights(&cfg),
        cross_ctx: vec![0.02f32; d],
        emb,
        upper: cfg.context,
    }
}

fn assert_logits_match_cpu(label: &str, cpu: &[f32], other: &[f32]) {
    assert!(
        other.iter().all(|x| x.is_finite()),
        "{label}: non-finite logits"
    );
    let (cos, maxd) = cosine(cpu, other);
    eprintln!(
        "{label}: cosine={cos:.6} max|Δ|={maxd:.2e} argmax {} vs cpu {}",
        argmax(other),
        argmax(cpu)
    );
    assert!(
        cos > 0.999,
        "{label}: logits diverge from CPU (cosine {cos})"
    );
    assert_eq!(argmax(other), argmax(cpu), "{label}: argmax mismatch");
}

/// First bucketed decode step (`past_seq = 0`).
pub fn temporal_decode_step0_on_device(device: Device, label: &str) {
    if device != Device::Cpu && !is_available(device) {
        eprintln!("{label}: skipped (not available / feature off)");
        return;
    }
    let f = fixture();
    let cpu = temporal_decode_bucketed_rlx(
        &f.dims,
        &f.weights,
        &f.emb,
        &f.cross_ctx,
        &[],
        0,
        f.upper,
        Device::Cpu,
    )
    .expect("cpu reference")
    .0;

    if device == Device::Cpu {
        assert_logits_match_cpu(label, &cpu, &cpu);
        return;
    }

    let logits = temporal_decode_bucketed_rlx(
        &f.dims,
        &f.weights,
        &f.emb,
        &f.cross_ctx,
        &[],
        0,
        f.upper,
        device,
    )
    .unwrap_or_else(|e| panic!("{label}: run failed: {e:#}"))
    .0;
    assert_logits_match_cpu(label, &cpu, &logits);
}

/// Second decode step with KV from a CPU warm-up (`past_seq = 1`).
pub fn temporal_decode_step1_on_device(device: Device, label: &str) {
    if device != Device::Cpu && !is_available(device) {
        eprintln!("{label}: skipped (not available / feature off)");
        return;
    }
    let f = fixture();
    let (_, _, kv0) = temporal_decode_bucketed_rlx(
        &f.dims,
        &f.weights,
        &f.emb,
        &f.cross_ctx,
        &[],
        0,
        f.upper,
        Device::Cpu,
    )
    .expect("cpu warm-up step 0");

    let emb2: Vec<f32> = f.emb.iter().map(|v| v * 1.1 + 0.001).collect();
    let cpu = temporal_decode_bucketed_rlx(
        &f.dims,
        &f.weights,
        &emb2,
        &f.cross_ctx,
        &kv0,
        1,
        f.upper,
        Device::Cpu,
    )
    .expect("cpu reference step 1")
    .0;

    if device == Device::Cpu {
        assert_logits_match_cpu(label, &cpu, &cpu);
        return;
    }

    let logits = temporal_decode_bucketed_rlx(
        &f.dims,
        &f.weights,
        &emb2,
        &f.cross_ctx,
        &kv0,
        1,
        f.upper,
        device,
    )
    .unwrap_or_else(|e| panic!("{label}: run failed: {e:#}"))
    .0;
    assert_logits_match_cpu(label, &cpu, &logits);
}
