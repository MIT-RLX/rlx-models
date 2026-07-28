// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke tests for the MiniMax-M3 vision tower and multimodal projector: tiny
//! synthetic-weight graphs that compile on CPU and run finite.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_minimax::m3::config::M3VisionConfig;
use rlx_minimax::m3::{build_m3_projector_flow, build_m3_vision_flow, vision_rope_tables};
use rlx_runtime::Device;
use std::collections::HashMap;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.1
        })
        .collect()
}

// Tiny vision config: hidden 16, 4 heads (head_dim 4), 2 layers, inter 24,
// patch 2, temporal 1, 3 channels, merge 2, text hidden 12, proj hidden 8.
fn tiny_vcfg() -> M3VisionConfig {
    serde_json::from_str(
        r#"{
            "hidden_size": 16, "num_attention_heads": 4, "num_hidden_layers": 2,
            "intermediate_size": 24, "patch_size": 2, "temporal_patch_size": 1,
            "num_channels": 3, "layer_norm_eps": 1e-5, "rope_theta": 10000.0,
            "spatial_merge_size": 2, "projection_dim": 12, "projector_hidden_size": 8
        }"#,
    )
    .expect("parse tiny vision config")
}

fn vision_weights(cfg: &M3VisionConfig) -> WeightMap {
    let embed = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let patch_dim = cfg.patch_dim();
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    let mut put =
        |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            seed += 3;
            t.insert(k, (fill(n, seed), shape));
        };
    let vm = "vision_tower.vision_model";
    put(
        &mut t,
        format!("{vm}.embeddings.patch_embedding.weight"),
        vec![embed, patch_dim],
    );
    put(&mut t, format!("{vm}.pre_layrnorm.weight"), vec![embed]);
    put(&mut t, format!("{vm}.pre_layrnorm.bias"), vec![embed]);
    for l in 0..cfg.num_hidden_layers {
        let lp = format!("{vm}.encoder.layers.{l}");
        put(&mut t, format!("{lp}.layer_norm1.weight"), vec![embed]);
        put(&mut t, format!("{lp}.layer_norm1.bias"), vec![embed]);
        put(&mut t, format!("{lp}.layer_norm2.weight"), vec![embed]);
        put(&mut t, format!("{lp}.layer_norm2.bias"), vec![embed]);
        for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            put(
                &mut t,
                format!("{lp}.self_attn.{p}.weight"),
                vec![embed, embed],
            );
            put(&mut t, format!("{lp}.self_attn.{p}.bias"), vec![embed]);
        }
        put(&mut t, format!("{lp}.mlp.fc1.weight"), vec![inter, embed]);
        put(&mut t, format!("{lp}.mlp.fc1.bias"), vec![inter]);
        put(&mut t, format!("{lp}.mlp.fc2.weight"), vec![embed, inter]);
        put(&mut t, format!("{lp}.mlp.fc2.bias"), vec![embed]);
    }
    WeightMap::from_tensors(t)
}

fn projector_weights(cfg: &M3VisionConfig) -> WeightMap {
    let embed = cfg.hidden_size;
    let ph = cfg.projector_hidden_size;
    let text = cfg.projection_dim;
    let merge2 = cfg.spatial_merge_size * cfg.spatial_merge_size;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    let mut put = |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: &str, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        seed += 3;
        t.insert(k.to_string(), (fill(n, seed), shape));
    };
    put(
        &mut t,
        "multi_modal_projector.linear_1.weight",
        vec![ph, embed],
    );
    put(&mut t, "multi_modal_projector.linear_1.bias", vec![ph]);
    put(
        &mut t,
        "multi_modal_projector.linear_2.weight",
        vec![ph, ph],
    );
    put(&mut t, "multi_modal_projector.linear_2.bias", vec![ph]);
    put(
        &mut t,
        "patch_merge_mlp.linear_1.weight",
        vec![ph, ph * merge2],
    );
    put(&mut t, "patch_merge_mlp.linear_1.bias", vec![ph]);
    put(&mut t, "patch_merge_mlp.linear_2.weight", vec![text, ph]);
    put(&mut t, "patch_merge_mlp.linear_2.bias", vec![text]);
    WeightMap::from_tensors(t)
}

#[test]
fn m3_vision_tower_compiles_and_runs() {
    let cfg = tiny_vcfg();
    // grid 1x4x4 = 16 patches.
    let (gt, gh, gw) = (1usize, 4usize, 4usize);
    let np = gt * gh * gw;
    let mut wm = vision_weights(&cfg);
    let built = build_m3_vision_flow(&cfg, &mut wm, np).expect("build vision flow");
    let mut compiled = compile_built(built, Device::Cpu).expect("compile vision flow");

    let px = fill(np * cfg.patch_dim(), 5);
    let (cos, sin) = vision_rope_tables(gt, gh, gw, cfg.axis_dim(), cfg.rope_theta);
    let out = compiled
        .run(&[
            ("pixel_values", px.as_slice()),
            ("vcos", cos.as_slice()),
            ("vsin", sin.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("vision forward returned output");
    assert_eq!(out.len(), np * cfg.hidden_size);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "vision hidden must be finite"
    );
}

#[test]
fn m3_projector_compiles_and_runs() {
    let cfg = tiny_vcfg();
    let np = 16usize; // divisible by merge²=4
    let mut wm = projector_weights(&cfg);
    let built = build_m3_projector_flow(&cfg, &mut wm, np).expect("build projector flow");
    let mut compiled = compile_built(built, Device::Cpu).expect("compile projector flow");

    let vh = fill(np * cfg.hidden_size, 7);
    let out = compiled
        .run(&[("vision_hidden", vh.as_slice())])
        .into_iter()
        .next()
        .expect("projector forward returned output");
    let np_out = np / (cfg.spatial_merge_size * cfg.spatial_merge_size);
    assert_eq!(out.len(), np_out * cfg.projection_dim);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "image features must be finite"
    );
}
