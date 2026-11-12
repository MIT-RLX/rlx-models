//! Compiled text graph vs the eager reference (which is itself checked against
//! transformers on the HF tiny fixture in `hf_tiny_parity.rs`).
//!
//! The synthetic weights in `synth` are tiny ramps with unit norms, which leave
//! routing and attention nearly uniform. These tests replace every tensor with
//! seeded noise of a realistic scale so a wrong head, expert or mask changes
//! the logits.

use rlx_inkling::eager::{TextWeights, forward_logits};
use rlx_inkling::{InklingTextConfig, forward_logits_compiled, synthetic_text_weights, tiny_cfg};
use rlx_runtime::Device;

fn noise(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((((s >> 33) as f32) / (u32::MAX as f32 / 2.0)) - 1.0) * scale
        })
        .collect()
}

/// Same keys and sizes as `synthetic_text_weights`, seeded noise values.
fn noisy_weights(cfg: &InklingTextConfig, seed: u64) -> TextWeights {
    let mut w = synthetic_text_weights(cfg);
    let mut keys: Vec<String> = w.tensors.keys().cloned().collect();
    keys.sort();
    for (i, k) in keys.iter().enumerate() {
        let n = w.tensors[k].len();
        let is_norm = k.ends_with("norm");
        let is_scale = k.ends_with("global_scale");
        let v = if is_scale {
            vec![1.0]
        } else if is_norm {
            noise(n, seed + i as u64, 0.2)
                .iter()
                .map(|x| 1.0 + x)
                .collect()
        } else if k.ends_with("gate_bias") {
            noise(n, seed + i as u64, 0.3)
        } else {
            noise(n, seed + i as u64, 0.5)
        };
        w.tensors.insert(k.clone(), v);
    }
    w
}

/// 4 layers: dense/sliding, MoE/sliding, MoE/global, MoE/sliding. Window 4 and
/// relative extent 6, so a 12-token prompt crosses both.
fn cfg() -> InklingTextConfig {
    let mut c = tiny_cfg();
    c.num_hidden_layers = 4;
    c.dense_mlp_idx = 1;
    c.local_layer_ids = vec![0, 1, 3];
    c.layer_types = (0..4)
        .map(|i| {
            if c.local_layer_ids.contains(&i) {
                rlx_inkling::AttnLayerType::Sliding
            } else {
                rlx_inkling::AttnLayerType::Global
            }
        })
        .collect();
    c.mlp_layer_types = (0..4)
        .map(|i| {
            if i < 1 {
                rlx_inkling::MlpLayerType::Dense
            } else {
                rlx_inkling::MlpLayerType::Sparse
            }
        })
        .collect();
    c.sliding_window_size = 4;
    c.rel_extent = 6;
    c.n_routed_experts = 6;
    c.num_experts_per_tok = 3;
    c.n_shared_experts = 2;
    c.route_scale = 2.5;
    c
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .fold(0.0, |m, (x, y)| m.max((x - y).abs() / scale))
}

fn check(c: &InklingTextConfig, ids: &[u32], seed: u64) -> f32 {
    let w = noisy_weights(c, seed);
    let want = forward_logits(c, &w, ids).unwrap();
    let got = forward_logits_compiled(c, &w, ids, Device::Cpu).unwrap();
    assert_eq!(got.len(), c.output_vocab_size());
    assert!(got.iter().all(|v| v.is_finite()));
    max_rel(&got, &want)
}

const IDS: [u32; 12] = [3, 17, 5, 5, 29, 1, 8, 22, 13, 0, 7, 19];

#[test]
fn graph_matches_eager() {
    let c = cfg();
    for seed in [1, 2, 3] {
        let d = check(&c, &IDS, seed);
        assert!(d < 2e-4, "seed {seed}: max relative diff {d}");
    }
}

#[test]
fn graph_matches_eager_with_log_scaling_mup_and_padded_vocab() {
    let mut c = cfg();
    // Positions >= 3 get tau > 1 on the global layer, so the term is exercised.
    c.log_scaling_n_floor = Some(3);
    c.log_scaling_alpha = 0.7;
    c.logits_mup_width_multiplier = 4.0;
    c.unpadded_vocab_size = Some(29);
    assert_eq!(c.output_vocab_size(), 29);
    for seed in [4, 5] {
        let d = check(&c, &IDS, seed);
        assert!(d < 2e-4, "seed {seed}: max relative diff {d}");
    }
}

#[test]
fn graph_handles_prompt_shorter_than_window() {
    let c = cfg();
    let d = check(&c, &IDS[..3], 6);
    assert!(d < 2e-4, "max relative diff {d}");
}
