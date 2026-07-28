// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: build the full Llama-4 text prefill graph (mixed RoPE/NoPE
//! layers, all-MoE) with tiny synthetic weights, compile on CPU, run, and
//! check finite `[1, seq, vocab]` logits.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_llama4::config::Llama4TextConfig;
use rlx_llama4::flow::build_llama4_text_flow;
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

fn weights(cfg: &Llama4TextConfig) -> WeightMap {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim();
    let q_dim = cfg.num_attention_heads * hd;
    let kv_dim = cfg.num_key_value_heads * hd;
    let inter = cfg.intermediate_size;
    let e = cfg.num_local_experts;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    let mut put =
        |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            seed += 5;
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
        put(
            &mut t,
            format!("{lp}.self_attn.q_proj.weight"),
            vec![q_dim, h],
        );
        put(
            &mut t,
            format!("{lp}.self_attn.k_proj.weight"),
            vec![kv_dim, h],
        );
        put(
            &mut t,
            format!("{lp}.self_attn.v_proj.weight"),
            vec![kv_dim, h],
        );
        put(
            &mut t,
            format!("{lp}.self_attn.o_proj.weight"),
            vec![h, q_dim],
        );
        let ff = format!("{lp}.feed_forward");
        put(&mut t, format!("{ff}.router.weight"), vec![e, h]);
        put(
            &mut t,
            format!("{ff}.experts.gate_up_proj"),
            vec![e, h, 2 * inter],
        );
        put(&mut t, format!("{ff}.experts.down_proj"), vec![e, inter, h]);
        put(
            &mut t,
            format!("{ff}.shared_expert.gate_proj.weight"),
            vec![inter, h],
        );
        put(
            &mut t,
            format!("{ff}.shared_expert.up_proj.weight"),
            vec![inter, h],
        );
        put(
            &mut t,
            format!("{ff}.shared_expert.down_proj.weight"),
            vec![h, inter],
        );
    }
    put(&mut t, "model.norm.weight".into(), vec![h]);
    put(&mut t, "lm_head.weight".into(), vec![cfg.vocab_size, h]);
    WeightMap::from_tensors(t)
}

#[test]
fn text_flow_compiles_and_runs() {
    let cfg: Llama4TextConfig = serde_json::from_str(
        r#"{"vocab_size":20,"hidden_size":8,"intermediate_size":16,"intermediate_size_mlp":16,
            "num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":2,
            "num_local_experts":4,"num_experts_per_tok":1,"no_rope_layer_interval":2,
            "interleave_moe_layer_step":1}"#,
    )
    .unwrap();
    // layer 0,2 = RoPE; layer 1 = NoPE; all MoE.
    assert!(cfg.uses_rope(0) && !cfg.uses_rope(1) && cfg.uses_rope(2));
    assert_eq!(cfg.moe_layers_vec().len(), 3);

    let seq = 4usize;
    let half = cfg.head_dim() / 2;
    let mut wm = weights(&cfg);
    let built = build_llama4_text_flow(&cfg, &mut wm, seq, true, false).expect("build text flow");
    let mut compiled = compile_built(built, dev()).expect("compile text flow");

    let ids: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0];
    let cos: Vec<f32> = (0..seq * half).map(|k| ((k / half) as f32).cos()).collect();
    let sin: Vec<f32> = (0..seq * half).map(|k| ((k / half) as f32).sin()).collect();
    let out = compiled
        .run(&[
            ("input_ids", ids.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("text forward returned output");
    assert_eq!(
        out.len(),
        seq * cfg.vocab_size,
        "logits [1,{seq},{}]",
        cfg.vocab_size
    );
    assert!(out.iter().all(|v| v.is_finite()), "logits must be finite");
}
