// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! End-to-end VL smoke test: encode a tiny image through the vision tower +
//! projector, splice the image feature into the prompt's token embeddings at the
//! placeholder position, prefill the text decoder, and greedily generate — all
//! finite on CPU.

use rlx_minimax::m3::config::{M3VisionConfig, MiniMaxM3Config};
use rlx_minimax::m3::{ImageInput, M3ImagePreprocessor, MiniMaxM3VlRunner};
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

fn text_cfg() -> MiniMaxM3Config {
    MiniMaxM3Config::from_text_config_json(
        r#"{
            "vocab_size": 20, "hidden_size": 16, "num_hidden_layers": 3,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
            "rotary_dim": 2, "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
            "dense_intermediate_size": 24, "intermediate_size": 8, "shared_intermediate_size": 8,
            "num_local_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
            "routed_scaling_factor": 2.0, "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
            "image_token_index": 0,
            "moe_layer_freq": [0, 1, 1],
            "sparse_attention_config": {
                "sparse_num_index_heads": 2, "sparse_index_dim": 4, "sparse_block_size": 2,
                "sparse_topk_blocks": 2, "sparse_local_block": 1,
                "sparse_attention_freq": [0, 1, 1]
            }
        }"#,
    )
    .expect("text cfg")
}

fn vision_cfg() -> M3VisionConfig {
    // hidden 16 == text hidden so projected image features match; head_dim 4 → axis_dim 0
    // (degenerate no-rope, exercised by the existing vision smoke test).
    serde_json::from_str(
        r#"{
            "hidden_size": 16, "num_attention_heads": 4, "num_hidden_layers": 2,
            "intermediate_size": 24, "patch_size": 2, "temporal_patch_size": 1,
            "num_channels": 3, "layer_norm_eps": 1e-5, "rope_theta": 10000.0,
            "spatial_merge_size": 2, "projection_dim": 16, "projector_hidden_size": 8
        }"#,
    )
    .expect("vision cfg")
}

fn combined_snapshot(
    tc: &MiniMaxM3Config,
    vc: &M3VisionConfig,
) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let h = tc.hidden_size;
    let hd = tc.head_dim();
    let nh = tc.num_attention_heads;
    let kv = tc.num_key_value_heads;
    let idd = tc.sparse.index_head_dim;
    let idh = tc.sparse.index_n_heads;
    let mi = tc.moe_intermediate_size;
    let si = tc.shared_inter();
    let di = tc.dense_intermediate_size;
    let e = tc.num_local_experts;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    let mut put =
        |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            seed += 3;
            t.insert(k, (fill(n, seed), shape));
        };

    // --- Text weights (stacked experts) ---
    put(
        &mut t,
        "model.embed_tokens.weight".into(),
        vec![tc.vocab_size, h],
    );
    for i in 0..tc.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        put(&mut t, format!("{lp}.input_layernorm.weight"), vec![h]);
        put(
            &mut t,
            format!("{lp}.post_attention_layernorm.weight"),
            vec![h],
        );
        let sa = format!("{lp}.self_attn");
        put(&mut t, format!("{sa}.q_proj.weight"), vec![nh * hd, h]);
        put(&mut t, format!("{sa}.k_proj.weight"), vec![kv * hd, h]);
        put(&mut t, format!("{sa}.v_proj.weight"), vec![kv * hd, h]);
        put(&mut t, format!("{sa}.o_proj.weight"), vec![h, nh * hd]);
        put(&mut t, format!("{sa}.q_norm.weight"), vec![hd]);
        put(&mut t, format!("{sa}.k_norm.weight"), vec![hd]);
        if tc.is_sparse_layer(i) {
            put(
                &mut t,
                format!("{sa}.index_q_proj.weight"),
                vec![idh * idd, h],
            );
            put(&mut t, format!("{sa}.index_k_proj.weight"), vec![idd, h]);
            put(&mut t, format!("{sa}.index_q_norm.weight"), vec![idd]);
            put(&mut t, format!("{sa}.index_k_norm.weight"), vec![idd]);
        }
        if tc.is_moe_layer(i) {
            let mp = format!("{lp}.block_sparse_moe");
            put(&mut t, format!("{mp}.gate.weight"), vec![e, h]);
            put(&mut t, format!("{mp}.e_score_correction_bias"), vec![e]);
            put(
                &mut t,
                format!("{mp}.experts.gate_up_proj"),
                vec![e, 2 * mi, h],
            );
            put(&mut t, format!("{mp}.experts.down_proj"), vec![e, h, mi]);
            put(
                &mut t,
                format!("{mp}.shared_experts.gate_proj.weight"),
                vec![si, h],
            );
            put(
                &mut t,
                format!("{mp}.shared_experts.up_proj.weight"),
                vec![si, h],
            );
            put(
                &mut t,
                format!("{mp}.shared_experts.down_proj.weight"),
                vec![h, si],
            );
        } else {
            let mp = format!("{lp}.mlp");
            put(&mut t, format!("{mp}.gate_proj.weight"), vec![di, h]);
            put(&mut t, format!("{mp}.up_proj.weight"), vec![di, h]);
            put(&mut t, format!("{mp}.down_proj.weight"), vec![h, di]);
        }
    }
    put(&mut t, "model.norm.weight".into(), vec![h]);
    put(&mut t, "lm_head.weight".into(), vec![tc.vocab_size, h]);

    // --- Vision tower weights ---
    let ve = vc.hidden_size;
    let vi = vc.intermediate_size;
    let pd = vc.patch_dim();
    let vm = "vision_tower.vision_model";
    put(
        &mut t,
        format!("{vm}.embeddings.patch_embedding.weight"),
        vec![ve, pd],
    );
    put(&mut t, format!("{vm}.pre_layrnorm.weight"), vec![ve]);
    put(&mut t, format!("{vm}.pre_layrnorm.bias"), vec![ve]);
    for l in 0..vc.num_hidden_layers {
        let lp = format!("{vm}.encoder.layers.{l}");
        put(&mut t, format!("{lp}.layer_norm1.weight"), vec![ve]);
        put(&mut t, format!("{lp}.layer_norm1.bias"), vec![ve]);
        put(&mut t, format!("{lp}.layer_norm2.weight"), vec![ve]);
        put(&mut t, format!("{lp}.layer_norm2.bias"), vec![ve]);
        for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            put(&mut t, format!("{lp}.self_attn.{p}.weight"), vec![ve, ve]);
            put(&mut t, format!("{lp}.self_attn.{p}.bias"), vec![ve]);
        }
        put(&mut t, format!("{lp}.mlp.fc1.weight"), vec![vi, ve]);
        put(&mut t, format!("{lp}.mlp.fc1.bias"), vec![vi]);
        put(&mut t, format!("{lp}.mlp.fc2.weight"), vec![ve, vi]);
        put(&mut t, format!("{lp}.mlp.fc2.bias"), vec![ve]);
    }

    // --- Projector weights ---
    let ph = vc.projector_hidden_size;
    let text = vc.projection_dim;
    let merge2 = vc.spatial_merge_size * vc.spatial_merge_size;
    put(
        &mut t,
        "multi_modal_projector.linear_1.weight".into(),
        vec![ph, ve],
    );
    put(
        &mut t,
        "multi_modal_projector.linear_1.bias".into(),
        vec![ph],
    );
    put(
        &mut t,
        "multi_modal_projector.linear_2.weight".into(),
        vec![ph, ph],
    );
    put(
        &mut t,
        "multi_modal_projector.linear_2.bias".into(),
        vec![ph],
    );
    put(
        &mut t,
        "patch_merge_mlp.linear_1.weight".into(),
        vec![ph, ph * merge2],
    );
    put(&mut t, "patch_merge_mlp.linear_1.bias".into(), vec![ph]);
    put(
        &mut t,
        "patch_merge_mlp.linear_2.weight".into(),
        vec![text, ph],
    );
    put(&mut t, "patch_merge_mlp.linear_2.bias".into(), vec![text]);

    t
}

#[test]
fn m3_vl_runner_prefills_and_generates() {
    let tc = text_cfg();
    let vc = vision_cfg();
    let snap = combined_snapshot(&tc, &vc);
    let mut runner = MiniMaxM3VlRunner::from_snapshot(tc.clone(), vc.clone(), snap, Device::Cpu);

    // 1×2×2 = 4 patches → merge² = 4 → 1 image token.
    let img = ImageInput {
        pixel_values: fill(4 * vc.patch_dim(), 42),
        grid_t: 1,
        grid_h: 2,
        grid_w: 2,
    };
    let feats = runner.encode_image(&img).expect("encode image");
    assert_eq!(
        feats.len(),
        tc.hidden_size,
        "one image token of width hidden"
    );
    assert!(feats.iter().all(|v| v.is_finite()));

    // Prompt: [1, <image=0>, 3, 4] with the image at index 1.
    let prompt = [1u32, 0, 3, 4];
    let positions = [1usize];
    let logits = runner
        .prefill_multimodal(&prompt, &positions, &img)
        .expect("prefill multimodal");
    assert_eq!(logits.len(), tc.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()), "vl logits finite");

    // Greedy multimodal generation.
    let mut seen = Vec::new();
    let out = runner
        .generate_multimodal(&prompt, &positions, &img, 3, |t| {
            seen.push(t);
            true
        })
        .expect("generate multimodal");
    assert_eq!(out.len(), 3);
    assert_eq!(out, seen);
    assert!(out.iter().all(|&t| (t as usize) < tc.vocab_size));
}

#[test]
fn m3_image_preprocess_feeds_vision_encoder() {
    let vc = vision_cfg();
    let pre = M3ImagePreprocessor::from_vision_config(&vc, 64);
    // 4×4 RGB, patch 2 → 2×2 grid → 4 patches.
    let rgb: Vec<u8> = (0..4 * 4 * 3).map(|i| ((i * 7) % 256) as u8).collect();
    let img = pre.preprocess_rgb_u8(&rgb, 4, 4).expect("preprocess");
    assert_eq!((img.grid_t, img.grid_h, img.grid_w), (1, 2, 2));
    assert_eq!(img.num_patches(), 4);
    assert_eq!(img.pixel_values.len(), 4 * vc.patch_dim());
    assert!(img.pixel_values.iter().all(|v| v.is_finite()));

    let tc = text_cfg();
    let snap = combined_snapshot(&tc, &vc);
    let mut runner = MiniMaxM3VlRunner::from_snapshot(tc, vc.clone(), snap, Device::Cpu);
    let feats = runner
        .encode_image(&img)
        .expect("encode preprocessed image");
    assert_eq!(feats.len(), vc.projection_dim); // np_out = 4/merge²=1 token
    assert!(feats.iter().all(|v| v.is_finite()));
}
