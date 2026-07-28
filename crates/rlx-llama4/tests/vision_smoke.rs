// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: build the Llama-4 vision tower + pixel-shuffle adapter +
//! projector with self-consistent tiny synthetic dims, compile on CPU, run,
//! and check finite `[1, n_out, text_hidden]` image features.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_llama4::config::Llama4VisionConfig;
use rlx_llama4::rope::build_vision_rope_tables;
use rlx_llama4::vision::build_llama4_vision_flow;
use rlx_runtime::Device;
use std::collections::HashMap;

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.05
        })
        .collect()
}

fn cfg() -> Llama4VisionConfig {
    // hidden 16, heads 4 (head_dim 4), intermediate = hidden*4 = 64 (== pixel-shuffle channels),
    // image 28 / patch 14 → grid 2, num_patches 5; ratio 0.5 → n_out 1, c_shuf 64.
    serde_json::from_str(
        r#"{"hidden_size":16,"num_hidden_layers":2,"num_attention_heads":4,"intermediate_size":64,
            "vision_output_dim":8,"image_size":28,"patch_size":14,"pixel_shuffle_ratio":0.5,
            "projector_input_dim":8,"projector_output_dim":8}"#,
    )
    .unwrap()
}

fn weights(c: &Llama4VisionConfig, text_hidden: usize) -> WeightMap {
    let h = c.hidden_size;
    let inter = c.intermediate_size;
    let qd = c.num_attention_heads * c.head_dim();
    let pin = c.projector_input_dim;
    let pout = c.projector_output_dim;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    let mut put =
        |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            seed += 3;
            t.insert(k, (fill(n, seed), shape));
        };
    for nm in ["layernorm_pre", "layernorm_post"] {
        put(&mut t, format!("vision_model.{nm}.weight"), vec![h]);
        put(&mut t, format!("vision_model.{nm}.bias"), vec![h]);
    }
    for i in 0..c.num_hidden_layers {
        let lp = format!("vision_model.model.layers.{i}");
        for nm in ["input_layernorm", "post_attention_layernorm"] {
            put(&mut t, format!("{lp}.{nm}.weight"), vec![h]);
            put(&mut t, format!("{lp}.{nm}.bias"), vec![h]);
        }
        for nm in ["q_proj", "k_proj", "v_proj"] {
            put(&mut t, format!("{lp}.self_attn.{nm}.weight"), vec![qd, h]);
            put(&mut t, format!("{lp}.self_attn.{nm}.bias"), vec![qd]);
        }
        put(&mut t, format!("{lp}.self_attn.o_proj.weight"), vec![h, qd]);
        put(&mut t, format!("{lp}.self_attn.o_proj.bias"), vec![h]);
        put(&mut t, format!("{lp}.mlp.fc1.weight"), vec![inter, h]);
        put(&mut t, format!("{lp}.mlp.fc1.bias"), vec![inter]);
        put(&mut t, format!("{lp}.mlp.fc2.weight"), vec![h, inter]);
        put(&mut t, format!("{lp}.mlp.fc2.bias"), vec![h]);
    }
    // adapter MLP2 (no bias): fc1 [pin, c_shuf=inter], fc2 [pout, pout]
    put(
        &mut t,
        "vision_model.vision_adapter.mlp.fc1.weight".into(),
        vec![pin, inter],
    );
    put(
        &mut t,
        "vision_model.vision_adapter.mlp.fc2.weight".into(),
        vec![pout, pout],
    );
    put(
        &mut t,
        "multi_modal_projector.linear_1.weight".into(),
        vec![text_hidden, c.vision_output_dim],
    );
    WeightMap::from_tensors(t)
}

#[test]
fn vision_flow_compiles_and_runs() {
    let c = cfg();
    let text_hidden = 8usize;
    let np = c.num_patches();
    let half = c.head_dim() / 2;
    assert_eq!(np, 5);

    let mut wm = weights(&c, text_hidden);
    let built = build_llama4_vision_flow(&c, &mut wm, text_hidden).expect("build vision flow");
    let mut compiled = compile_built(built, dev()).expect("compile vision flow");

    let hidden = fill(np * c.hidden_size, 77);
    let (cos, sin) = build_vision_rope_tables(
        c.image_size,
        c.patch_size,
        c.hidden_size,
        c.num_attention_heads,
        10000.0,
    );
    assert_eq!(cos.len(), np * half);
    let out = compiled
        .run(&[
            ("hidden", hidden.as_slice()),
            ("v_rope_cos", cos.as_slice()),
            ("v_rope_sin", sin.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("vision forward returned an output");
    // n_out = (grid * ratio)^2 = 1
    assert_eq!(
        out.len(),
        text_hidden,
        "image features [1, 1, {text_hidden}]"
    );
    assert!(
        out.iter().all(|v| v.is_finite()),
        "image features must be finite"
    );
}
