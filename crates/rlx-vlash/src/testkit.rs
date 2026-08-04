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

//! Test scaffolding: a tiny [`VlashConfig`] and a synthetic [`WeightMap`]
//! covering every canonical key the vision + denoise graphs load. Lets the
//! graph plumbing (shapes, RoPE, joint attention, adaRMS, flow output) be
//! exercised on CPU without the multi-GB `lerobot/pi0_base` checkpoint.

use std::collections::HashMap;

use rlx_core::weight_map::WeightMap;

use crate::config::{GemmaConfig, VisionConfig, VlashConfig, VlashVariant};

/// A miniature config with the same *topology* as π₀ / π₀.₅ but tiny dims.
pub fn tiny_config(variant: VlashVariant) -> VlashConfig {
    let adarms = variant == VlashVariant::Pi05;
    VlashConfig {
        variant,
        vision: VisionConfig {
            width: 16,
            layers: 2,
            heads: 2,
            head_dim: 8,
            intermediate: 32,
            patch_size: 14,
            image_size: 28,
            projection_dim: 16,
            ln_eps: 1e-6,
        },
        vlm: GemmaConfig {
            hidden: 16,
            layers: 2,
            heads: 2,
            head_dim: 8,
            num_kv_heads: 1,
            intermediate: 32,
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
            use_adarms: false,
        },
        expert: GemmaConfig {
            hidden: 12,
            layers: 2,
            heads: 2,
            head_dim: 8,
            num_kv_heads: 1,
            intermediate: 24,
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
            use_adarms: adarms,
        },
        max_state_dim: 5,
        max_action_dim: 4,
        chunk_size: 3,
        n_action_steps: 3,
        num_inference_steps: 2,
        min_period: 4e-3,
        max_period: 4.0,
        tokenizer_max_length: 8,
        image_size: 28,
        state_cond: adarms,
    }
}

/// Deterministic small filler in roughly `[-0.06, 0.06]`.
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + seed * 13) % 13) as f32) * 0.01 - 0.06)
        .collect()
}

/// Synthesize a [`WeightMap`] with every canonical key the graphs need.
pub fn synth_weights(cfg: &VlashConfig) -> WeightMap {
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1usize;
    let mut put =
        |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            seed += 1;
            t.insert(key.to_string(), (fill(n, seed), shape));
        };

    let v = &cfg.vision;
    // Vision embeddings.
    put(
        &mut t,
        "vision.embeddings.patch_embedding.weight",
        vec![v.width, 3, v.patch_size, v.patch_size],
    );
    put(
        &mut t,
        "vision.embeddings.patch_embedding.bias",
        vec![v.width],
    );
    put(
        &mut t,
        "vision.embeddings.position_embedding.weight",
        vec![v.num_patches(), v.width],
    );
    // Vision encoder layers.
    for i in 0..v.layers {
        let p = format!("vision.encoder.layers.{i}");
        for ln in ["layer_norm1", "layer_norm2"] {
            put(&mut t, &format!("{p}.{ln}.weight"), vec![v.width]);
            put(&mut t, &format!("{p}.{ln}.bias"), vec![v.width]);
        }
        for pr in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            put(
                &mut t,
                &format!("{p}.self_attn.{pr}.weight"),
                vec![v.width, v.width],
            );
            put(&mut t, &format!("{p}.self_attn.{pr}.bias"), vec![v.width]);
        }
        put(
            &mut t,
            &format!("{p}.mlp.fc1.weight"),
            vec![v.intermediate, v.width],
        );
        put(&mut t, &format!("{p}.mlp.fc1.bias"), vec![v.intermediate]);
        put(
            &mut t,
            &format!("{p}.mlp.fc2.weight"),
            vec![v.width, v.intermediate],
        );
        put(&mut t, &format!("{p}.mlp.fc2.bias"), vec![v.width]);
    }
    put(&mut t, "vision.post_layernorm.weight", vec![v.width]);
    put(&mut t, "vision.post_layernorm.bias", vec![v.width]);
    put(
        &mut t,
        "vision.projector.weight",
        vec![v.projection_dim, v.width],
    );
    put(&mut t, "vision.projector.bias", vec![v.projection_dim]);

    // Gemma stacks.
    let gemma_layer =
        |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
         put: &mut dyn FnMut(&mut HashMap<String, (Vec<f32>, Vec<usize>)>, &str, Vec<usize>),
         prefix: &str,
         g: &GemmaConfig| {
            for i in 0..g.layers {
                let p = format!("{prefix}.layers.{i}");
                // norms (standard or adaRMS)
                for ln in ["input_layernorm", "post_attention_layernorm"] {
                    if g.use_adarms {
                        put(
                            t,
                            &format!("{p}.{ln}.dense.weight"),
                            vec![3 * g.hidden, g.hidden],
                        );
                        put(t, &format!("{p}.{ln}.dense.bias"), vec![3 * g.hidden]);
                    } else {
                        put(t, &format!("{p}.{ln}.weight"), vec![g.hidden]);
                    }
                }
                put(
                    t,
                    &format!("{p}.self_attn.q_proj.weight"),
                    vec![g.attn_dim(), g.hidden],
                );
                put(
                    t,
                    &format!("{p}.self_attn.k_proj.weight"),
                    vec![g.kv_dim(), g.hidden],
                );
                put(
                    t,
                    &format!("{p}.self_attn.v_proj.weight"),
                    vec![g.kv_dim(), g.hidden],
                );
                put(
                    t,
                    &format!("{p}.self_attn.o_proj.weight"),
                    vec![g.hidden, g.attn_dim()],
                );
                put(
                    t,
                    &format!("{p}.mlp.gate_proj.weight"),
                    vec![g.intermediate, g.hidden],
                );
                put(
                    t,
                    &format!("{p}.mlp.up_proj.weight"),
                    vec![g.intermediate, g.hidden],
                );
                put(
                    t,
                    &format!("{p}.mlp.down_proj.weight"),
                    vec![g.hidden, g.intermediate],
                );
            }
            // final norm
            if g.use_adarms {
                put(
                    t,
                    &format!("{prefix}.norm.dense.weight"),
                    vec![3 * g.hidden, g.hidden],
                );
                put(t, &format!("{prefix}.norm.dense.bias"), vec![3 * g.hidden]);
            } else {
                put(t, &format!("{prefix}.norm.weight"), vec![g.hidden]);
            }
        };
    gemma_layer(&mut t, &mut put, "vlm", &cfg.vlm);
    gemma_layer(&mut t, &mut put, "expert", &cfg.expert);

    // Suffix embedder + action head.
    let d = cfg.expert.hidden;
    put(
        &mut t,
        "suffix.action_in_proj.weight",
        vec![d, cfg.max_action_dim],
    );
    put(&mut t, "suffix.action_in_proj.bias", vec![d]);
    match cfg.variant {
        VlashVariant::Pi0 => {
            put(
                &mut t,
                "suffix.state_proj.weight",
                vec![d, cfg.max_state_dim],
            );
            put(&mut t, "suffix.state_proj.bias", vec![d]);
            put(&mut t, "suffix.action_time_mlp_in.weight", vec![d, 2 * d]);
            put(&mut t, "suffix.action_time_mlp_in.bias", vec![d]);
            put(&mut t, "suffix.action_time_mlp_out.weight", vec![d, d]);
            put(&mut t, "suffix.action_time_mlp_out.bias", vec![d]);
        }
        VlashVariant::Pi05 => {
            put(&mut t, "suffix.time_mlp_in.weight", vec![d, d]);
            put(&mut t, "suffix.time_mlp_in.bias", vec![d]);
            put(&mut t, "suffix.time_mlp_out.weight", vec![d, d]);
            put(&mut t, "suffix.time_mlp_out.bias", vec![d]);
            if cfg.state_cond {
                put(
                    &mut t,
                    "suffix.state_proj.weight",
                    vec![d, cfg.max_state_dim],
                );
                put(&mut t, "suffix.state_proj.bias", vec![d]);
                put(&mut t, "suffix.state_mlp_in.weight", vec![d, d]);
                put(&mut t, "suffix.state_mlp_in.bias", vec![d]);
                put(&mut t, "suffix.state_mlp_out.weight", vec![d, d]);
                put(&mut t, "suffix.state_mlp_out.bias", vec![d]);
            }
        }
    }
    put(
        &mut t,
        "action_out_proj.weight",
        vec![cfg.max_action_dim, d],
    );
    put(&mut t, "action_out_proj.bias", vec![cfg.max_action_dim]);

    WeightMap::from_tensors(t)
}
