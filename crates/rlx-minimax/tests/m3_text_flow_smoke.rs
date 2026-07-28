// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke tests for the MiniMax-M3 text decoder: config parsing, and a tiny
//! synthetic-weight prefill graph (dense+full-attn layer 0, MoE+MSA-sparse
//! layers 1-2) that compiles on CPU (or `RLX_TEST_DEVICE`) and runs finite.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_minimax::m3::config::MiniMaxM3Config;
use rlx_minimax::m3::{build_m3_text_flow, rope_tables};
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

// Tiny config: hidden 16, 3 layers, GQA 4/2, head_dim 4, partial rope n_rot 2,
// 4 experts (top-2) + 1 shared, index dim 4, block 2, top-2 blocks, local 1.
// layer 0 = dense + full attention; layers 1,2 = MoE + MSA sparse.
fn tiny_cfg() -> MiniMaxM3Config {
    MiniMaxM3Config::from_text_config_json(
        r#"{
            "vocab_size": 20, "hidden_size": 16, "num_hidden_layers": 3,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
            "rotary_dim": 2, "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
            "dense_intermediate_size": 24, "intermediate_size": 8, "shared_intermediate_size": 8,
            "num_local_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
            "routed_scaling_factor": 2.0, "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
            "tie_word_embeddings": false,
            "moe_layer_freq": [0, 1, 1],
            "sparse_attention_config": {
                "sparse_num_index_heads": 2, "sparse_index_dim": 4, "sparse_block_size": 2,
                "sparse_topk_blocks": 2, "sparse_local_block": 1,
                "sparse_attention_freq": [0, 1, 1]
            }
        }"#,
    )
    .expect("parse tiny m3 config")
}

fn weights(cfg: &MiniMaxM3Config) -> WeightMap {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim();
    let nh = cfg.num_attention_heads;
    let kv = cfg.num_key_value_heads;
    let idh = cfg.sparse.index_n_heads;
    let idd = cfg.sparse.index_head_dim;
    let mi = cfg.moe_intermediate_size;
    let si = cfg.shared_inter();
    let di = cfg.dense_intermediate_size;
    let e = cfg.num_local_experts;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
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
        put(&mut t, format!("{sa}.q_proj.weight"), vec![nh * hd, h]);
        put(&mut t, format!("{sa}.k_proj.weight"), vec![kv * hd, h]);
        put(&mut t, format!("{sa}.v_proj.weight"), vec![kv * hd, h]);
        put(&mut t, format!("{sa}.o_proj.weight"), vec![h, nh * hd]);
        put(&mut t, format!("{sa}.q_norm.weight"), vec![hd]);
        put(&mut t, format!("{sa}.k_norm.weight"), vec![hd]);
        if cfg.is_sparse_layer(i) {
            put(
                &mut t,
                format!("{sa}.index_q_proj.weight"),
                vec![idh * idd, h],
            );
            put(&mut t, format!("{sa}.index_k_proj.weight"), vec![idd, h]);
            put(&mut t, format!("{sa}.index_q_norm.weight"), vec![idd]);
            put(&mut t, format!("{sa}.index_k_norm.weight"), vec![idd]);
        }
        if cfg.is_moe_layer(i) {
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
    put(&mut t, "lm_head.weight".into(), vec![cfg.vocab_size, h]);
    WeightMap::from_tensors(t)
}

#[test]
fn m3_config_derives_layer_types() {
    let cfg = tiny_cfg();
    assert_eq!(cfg.head_dim(), 4);
    assert_eq!(cfg.n_rot(), 2);
    assert_eq!(cfg.kv_groups(), 2);
    // layer 0 dense+full; layers 1,2 MoE+sparse.
    assert!(!cfg.is_moe_layer(0) && cfg.is_moe_layer(1) && cfg.is_moe_layer(2));
    assert!(!cfg.is_sparse_layer(0) && cfg.is_sparse_layer(1) && cfg.is_sparse_layer(2));
}

#[test]
fn m3_text_flow_compiles_and_runs() {
    let cfg = tiny_cfg();
    let seq = 5usize;
    let mut wm = weights(&cfg);
    let built = build_m3_text_flow(&cfg, &mut wm, seq, true).expect("build m3 flow");
    let mut compiled = compile_built(built, dev()).expect("compile m3 flow");

    let ids: Vec<f32> = (0..seq).map(|i| (i % cfg.vocab_size) as f32).collect();
    let (cos, sin) = rope_tables(seq, cfg.n_rot(), cfg.rope_theta);
    let out = compiled
        .run(&[
            ("input_ids", ids.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("m3 forward returned output");
    assert_eq!(out.len(), seq * cfg.vocab_size);
    assert!(out.iter().all(|v| v.is_finite()), "logits must be finite");
}
