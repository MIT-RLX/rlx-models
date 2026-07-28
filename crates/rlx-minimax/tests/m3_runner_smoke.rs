// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test for the MiniMax-M3 prefill runner + LmRunner: a tiny synthetic
//! snapshot drives predict_logits (finite, vocab-length) and the default
//! re-prefill greedy generate loop. Also checks per-expert weight stacking.

use rlx_cli::LmRunner;
use rlx_minimax::m3::MiniMaxM3Runner;
use rlx_minimax::m3::config::MiniMaxM3Config;
use rlx_minimax::m3::weights::normalize_snapshot;
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

fn tiny_cfg() -> MiniMaxM3Config {
    MiniMaxM3Config::from_text_config_json(
        r#"{
            "vocab_size": 20, "hidden_size": 16, "num_hidden_layers": 3,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
            "rotary_dim": 2, "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
            "dense_intermediate_size": 24, "intermediate_size": 8, "shared_intermediate_size": 8,
            "num_local_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
            "routed_scaling_factor": 2.0, "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
            "moe_layer_freq": [0, 1, 1],
            "sparse_attention_config": {
                "sparse_num_index_heads": 2, "sparse_index_dim": 4, "sparse_block_size": 2,
                "sparse_topk_blocks": 2, "sparse_local_block": 1,
                "sparse_attention_freq": [0, 1, 1]
            }
        }"#,
    )
    .expect("parse tiny cfg")
}

/// Build a snapshot with PER-EXPERT tensors so `normalize_snapshot` exercises
/// its stacking path (w1/w3 → gate_up, w2 → down).
fn per_expert_snapshot(cfg: &MiniMaxM3Config) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim();
    let nh = cfg.num_attention_heads;
    let kv = cfg.num_key_value_heads;
    let idd = cfg.sparse.index_head_dim;
    let idh = cfg.sparse.index_n_heads;
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
            for j in 0..e {
                put(&mut t, format!("{mp}.experts.{j}.w1.weight"), vec![mi, h]);
                put(&mut t, format!("{mp}.experts.{j}.w3.weight"), vec![mi, h]);
                put(&mut t, format!("{mp}.experts.{j}.w2.weight"), vec![h, mi]);
            }
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
    t
}

#[test]
fn m3_runner_predicts_and_generates() {
    let cfg = tiny_cfg();
    let raw = per_expert_snapshot(&cfg);
    // Stacking: per-expert w1/w3/w2 collapse to gate_up_proj / down_proj.
    let snap = normalize_snapshot(&cfg, raw).expect("normalize snapshot");
    assert!(snap.contains_key("model.layers.1.block_sparse_moe.experts.gate_up_proj"));
    assert!(!snap.contains_key("model.layers.1.block_sparse_moe.experts.0.w1.weight"));
    let (gu, gu_shape) = &snap["model.layers.1.block_sparse_moe.experts.gate_up_proj"];
    assert_eq!(gu_shape, &vec![4, 2 * 8, 16]);
    assert_eq!(gu.len(), 4 * 2 * 8 * 16);

    let mut runner = MiniMaxM3Runner::from_snapshot(cfg.clone(), snap, Device::Cpu);
    assert_eq!(runner.vocab_size(), 20);
    assert_eq!(runner.family(), "minimax-m3");

    let prompt = [1u32, 5, 3, 8];
    let logits = runner.predict_logits(&prompt).expect("predict");
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()), "logits finite");

    // Default re-prefill greedy generate (grows context, recompiles per length).
    let mut got = Vec::new();
    let out = runner
        .generate(&prompt, 3, &mut |t| {
            got.push(t);
            true
        })
        .expect("generate");
    assert_eq!(out.len(), 3);
    assert_eq!(out, got);
    assert!(out.iter().all(|&t| (t as usize) < cfg.vocab_size));
}
