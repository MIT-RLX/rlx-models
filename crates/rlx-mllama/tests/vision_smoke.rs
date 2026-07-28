// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: build the mllama vision + projector flow with a tiny synthetic
//! config and random weights, compile on CPU, and run it. Verifies the graph
//! (tap interleave + concat + projector) actually compiles and produces a
//! finite `[1, seq, text_hidden]` tensor — catches shape-inference bugs without
//! the real 20 GB checkpoint.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_mllama::config::MllamaVisionConfig;
use rlx_mllama::vision::build_vision_flow;
use rlx_runtime::Device;
use std::collections::HashMap;

/// Deterministic small pseudo-random fill (no external rng dep).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / (u32::MAX as f32); // [0,1)
            (u - 0.5) * 0.05
        })
        .collect()
}

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn tiny_config() -> MllamaVisionConfig {
    MllamaVisionConfig {
        hidden_size: 32,
        num_hidden_layers: 3,
        num_global_layers: 2,
        num_attention_heads: 4,
        intermediate_size: 64,
        vision_output_dim: 96, // 32 * (1 + 2 taps)
        image_size: 28,
        patch_size: 14, // -> (28/14)^2 + 1 = 5 patches/tile
        max_num_tiles: 2,
        norm_eps: 1e-5,
        num_channels: 3,
        intermediate_layers_indices: vec![0, 1],
        supported_aspect_ratios: vec![vec![1, 1], vec![1, 2]],
    }
}

fn layer_weights(
    t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    cfg: &MllamaVisionConfig,
    gated: bool,
    mut seed: u64,
) {
    let w = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let mut next =
        |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, key: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            seed = seed.wrapping_add(101);
            t.insert(key, (fill(n, seed), shape));
        };
    for name in ["input_layernorm", "post_attention_layernorm"] {
        next(t, format!("{prefix}.{name}.weight"), vec![w]);
        next(t, format!("{prefix}.{name}.bias"), vec![w]);
    }
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        next(t, format!("{prefix}.self_attn.{name}.weight"), vec![w, w]);
    }
    next(t, format!("{prefix}.mlp.fc1.weight"), vec![inter, w]);
    next(t, format!("{prefix}.mlp.fc1.bias"), vec![inter]);
    next(t, format!("{prefix}.mlp.fc2.weight"), vec![w, inter]);
    next(t, format!("{prefix}.mlp.fc2.bias"), vec![w]);
    if gated {
        next(t, format!("{prefix}.gate_attn"), vec![1]);
        next(t, format!("{prefix}.gate_ffn"), vec![1]);
    }
}

fn synth_weights(cfg: &MllamaVisionConfig, text_hidden: usize) -> WeightMap {
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let w = cfg.hidden_size;
    for name in ["layernorm_pre", "layernorm_post"] {
        t.insert(format!("vision_model.{name}.weight"), (fill(w, 1), vec![w]));
        t.insert(format!("vision_model.{name}.bias"), (fill(w, 2), vec![w]));
    }
    for l in 0..cfg.num_hidden_layers {
        layer_weights(
            &mut t,
            &format!("vision_model.transformer.layers.{l}"),
            cfg,
            false,
            1000 + l as u64 * 97,
        );
    }
    for l in 0..cfg.num_global_layers {
        layer_weights(
            &mut t,
            &format!("vision_model.global_transformer.layers.{l}"),
            cfg,
            true,
            5000 + l as u64 * 97,
        );
    }
    let cw = cfg.concat_width();
    t.insert(
        "multi_modal_projector.weight".into(),
        (fill(text_hidden * cw, 9), vec![text_hidden, cw]),
    );
    t.insert(
        "multi_modal_projector.bias".into(),
        (fill(text_hidden, 10), vec![text_hidden]),
    );
    WeightMap::from_tensors(t)
}

#[test]
fn vision_encoder_compiles_and_runs() {
    let cfg = tiny_config();
    let text_hidden = 16usize;
    let num_tiles = 2usize;
    let seq = num_tiles * cfg.num_patches();
    assert_eq!(seq, 10);

    let mut wm = synth_weights(&cfg, text_hidden);
    let built =
        build_vision_flow(&cfg, &mut wm, text_hidden, num_tiles).expect("build vision flow");
    let mut compiled = compile_built(built, dev()).expect("compile vision flow");

    let hidden = fill(seq * cfg.hidden_size, 42);
    let post_tile = fill(seq * cfg.hidden_size, 43);
    let out = compiled
        .run(&[
            ("hidden", hidden.as_slice()),
            ("post_tile", post_tile.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("vision forward returned an output");

    assert_eq!(
        out.len(),
        seq * text_hidden,
        "cross_states shape [1,{seq},{text_hidden}]"
    );
    assert!(
        out.iter().all(|v| v.is_finite()),
        "cross_states must be finite"
    );
}
