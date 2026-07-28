// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: full DeepSeek-V3 text prefill graph (dense first layer + MLA/MoE
//! layers) with tiny synthetic weights, compile on CPU, run; finite logits.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_deepseek::config::DeepseekV3Config;
use rlx_deepseek::flow::build_deepseek_text_flow;
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
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.1
        })
        .collect()
}

fn weights(cfg: &DeepseekV3Config) -> WeightMap {
    let h = cfg.hidden_size;
    let hh = cfg.num_attention_heads;
    let qk = cfg.qk_head_dim();
    let ql = cfg.q_lora_rank.unwrap();
    let kvl = cfg.kv_lora_rank;
    let rope = cfg.qk_rope_head_dim;
    let vd = cfg.v_head_dim;
    let nope = cfg.qk_nope_head_dim;
    let mi = cfg.moe_intermediate_size;
    let e = cfg.n_routed_experts;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1;
    let mut put =
        |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            seed += 3;
            t.insert(k, (fill(n, seed), shape));
        };
    put(
        &mut t,
        "model.embed_tokens.weight".into(),
        vec![cfg.vocab_size, h],
    );
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        put(&mut t, format!("{lp}.input_layernorm.weight"), vec![h]);
        put(
            &mut t,
            format!("{lp}.post_attention_layernorm.weight"),
            vec![h],
        );
        let sa = format!("{lp}.self_attn");
        put(&mut t, format!("{sa}.q_a_proj.weight"), vec![ql, h]);
        put(&mut t, format!("{sa}.q_a_layernorm.weight"), vec![ql]);
        put(&mut t, format!("{sa}.q_b_proj.weight"), vec![hh * qk, ql]);
        put(
            &mut t,
            format!("{sa}.kv_a_proj_with_mqa.weight"),
            vec![kvl + rope, h],
        );
        put(&mut t, format!("{sa}.kv_a_layernorm.weight"), vec![kvl]);
        put(
            &mut t,
            format!("{sa}.kv_b_proj.weight"),
            vec![hh * (nope + vd), kvl],
        );
        put(&mut t, format!("{sa}.o_proj.weight"), vec![h, hh * vd]);
        let mlp = format!("{lp}.mlp");
        if cfg.is_moe_layer(i) {
            put(&mut t, format!("{mlp}.gate.weight"), vec![e, h]);
            put(
                &mut t,
                format!("{mlp}.gate.e_score_correction_bias"),
                vec![e],
            );
            put(
                &mut t,
                format!("{mlp}.experts.gate_up_proj"),
                vec![e, 2 * mi, h],
            );
            put(&mut t, format!("{mlp}.experts.down_proj"), vec![e, h, mi]);
            put(
                &mut t,
                format!("{mlp}.shared_experts.gate_proj.weight"),
                vec![mi, h],
            );
            put(
                &mut t,
                format!("{mlp}.shared_experts.up_proj.weight"),
                vec![mi, h],
            );
            put(
                &mut t,
                format!("{mlp}.shared_experts.down_proj.weight"),
                vec![h, mi],
            );
        } else {
            let di = cfg.intermediate_size;
            put(&mut t, format!("{mlp}.gate_proj.weight"), vec![di, h]);
            put(&mut t, format!("{mlp}.up_proj.weight"), vec![di, h]);
            put(&mut t, format!("{mlp}.down_proj.weight"), vec![h, di]);
        }
    }
    put(&mut t, "model.norm.weight".into(), vec![h]);
    put(&mut t, "lm_head.weight".into(), vec![cfg.vocab_size, h]);
    WeightMap::from_tensors(t)
}

#[test]
fn deepseek_text_flow_compiles_and_runs() {
    let cfg: DeepseekV3Config = serde_json::from_str(
        r#"{"vocab_size":20,"hidden_size":16,"intermediate_size":16,"moe_intermediate_size":8,
            "num_hidden_layers":3,"num_attention_heads":2,"num_key_value_heads":2,"n_shared_experts":1,
            "n_routed_experts":4,"kv_lora_rank":6,"q_lora_rank":8,"qk_rope_head_dim":2,"v_head_dim":4,
            "qk_nope_head_dim":4,"n_group":2,"topk_group":1,"num_experts_per_tok":2,
            "first_k_dense_replace":1,"rms_norm_eps":1e-6}"#,
    )
    .unwrap();
    // layer 0 dense, layers 1,2 MoE
    assert!(!cfg.is_moe_layer(0) && cfg.is_moe_layer(1));

    let seq = 3usize;
    let half = cfg.qk_rope_head_dim / 2;
    let mut wm = weights(&cfg);
    let built = build_deepseek_text_flow(&cfg, &mut wm, seq, true).expect("build deepseek flow");
    let mut compiled = compile_built(built, dev()).expect("compile deepseek flow");

    let ids: Vec<f32> = vec![1.0, 5.0, 3.0];
    let cos = fill(seq * half, 7);
    let sin = fill(seq * half, 8);
    let out = compiled
        .run(&[
            ("input_ids", ids.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("deepseek forward returned output");
    assert_eq!(out.len(), seq * cfg.vocab_size);
    assert!(out.iter().all(|v| v.is_finite()), "logits must be finite");
}
