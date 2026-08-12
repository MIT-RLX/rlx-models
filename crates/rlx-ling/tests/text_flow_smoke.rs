// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Full Ling 3.0 prefill graph on tiny synthetic weights: the hybrid KDA/MLA
//! stack with a dense first layer and MoE elsewhere, compiled and run.
//!
//! The config mirrors the published Ling-3.0-tiny shape relationships (scaled
//! down) so the layer interleave, the MLA head split and the grouped router are
//! all exercised, and per-expert weights go through the stacking preprocessor.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_ling::config::AttnKind;
use rlx_ling::{LingConfig, build_ling_text_flow, prepare_checkpoint};
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

/// Scaled-down Ling-3.0-tiny: 8 layers, `layer_group_size` 4 ⇒ MLA at 3 and 7.
fn tiny_config() -> LingConfig {
    LingConfig::from_json_str(
        r#"{"vocab_size":32,"hidden_size":16,"intermediate_size":24,"num_hidden_layers":8,
            "num_attention_heads":2,"head_dim":8,"rms_norm_eps":1e-6,"rope_theta":600000.0,
            "num_experts":8,"num_experts_per_tok":2,"num_shared_experts":1,
            "moe_intermediate_size":8,"moe_shared_expert_intermediate_size":8,
            "n_group":2,"topk_group":1,"routed_scaling_factor":2.5,"first_k_dense_replace":1,
            "q_lora_rank":12,"kv_lora_rank":10,"qk_nope_head_dim":8,"qk_rope_head_dim":4,
            "v_head_dim":8,"rope_interleave":true,
            "gated_attention_proj_granularity_type":"head_wise",
            "layer_group_size":4,"short_conv_kernel_size":4,"no_kda_lora":true,
            "kda_safe_gate":true,"kda_lower_bound":-5.0,"tie_word_embeddings":false}"#,
    )
    .expect("parse tiny config")
}

fn weights(cfg: &LingConfig) -> WeightMap {
    let h = cfg.hidden_size;
    let hh = cfg.num_attention_heads;
    let proj = cfg.kda_proj_dim();
    let hd = cfg.head_dim;
    let qk = cfg.qk_head_dim();
    let ql = cfg.q_lora_rank.unwrap();
    let kvl = cfg.kv_lora_rank;
    let rope = cfg.qk_rope_head_dim;
    let nope = cfg.qk_nope_head_dim;
    let vd = cfg.v_head_dim;
    let mi = cfg.moe_intermediate_size;
    let si = cfg.shared_intermediate_size();
    let e = cfg.num_experts;

    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    let mut put =
        |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            seed += 3;
            t.insert(k, (fill(n, seed), shape));
        };

    put(&mut t, rlx_ling::EMBED_KEY.into(), vec![cfg.vocab_size, h]);
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        put(&mut t, format!("{lp}.input_layernorm.weight"), vec![h]);
        put(
            &mut t,
            format!("{lp}.post_attention_layernorm.weight"),
            vec![h],
        );
        let at = format!("{lp}.attention");
        match cfg.attn_kind(i) {
            AttnKind::Mla => {
                put(&mut t, format!("{at}.q_a_proj.weight"), vec![ql, h]);
                put(&mut t, format!("{at}.q_a_layernorm.weight"), vec![ql]);
                put(&mut t, format!("{at}.q_b_proj.weight"), vec![hh * qk, ql]);
                put(
                    &mut t,
                    format!("{at}.kv_a_proj_with_mqa.weight"),
                    vec![kvl + rope, h],
                );
                put(&mut t, format!("{at}.kv_a_layernorm.weight"), vec![kvl]);
                put(
                    &mut t,
                    format!("{at}.kv_b_proj.weight"),
                    vec![hh * (nope + vd), kvl],
                );
                put(&mut t, format!("{at}.g_proj.weight"), vec![hh, h]);
                put(&mut t, format!("{at}.dense.weight"), vec![h, hh * vd]);
            }
            AttnKind::Kda => {
                for p in ["q_proj", "k_proj", "v_proj", "f_proj", "g_proj"] {
                    put(&mut t, format!("{at}.{p}.weight"), vec![proj, h]);
                }
                for c in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                    put(
                        &mut t,
                        format!("{at}.{c}.weight"),
                        vec![proj, 1, cfg.short_conv_kernel_size],
                    );
                }
                put(&mut t, format!("{at}.b_proj.weight"), vec![hh, h]);
                put(&mut t, format!("{at}.A_log"), vec![hh]);
                put(&mut t, format!("{at}.dt_bias"), vec![proj]);
                put(&mut t, format!("{at}.o_norm.weight"), vec![hd]);
                put(&mut t, format!("{at}.o_proj.weight"), vec![h, proj]);
            }
        }
        let mlp = format!("{lp}.mlp");
        if cfg.is_moe_layer(i) {
            put(&mut t, format!("{mlp}.gate.weight"), vec![e, h]);
            put(&mut t, format!("{mlp}.gate.expert_bias"), vec![e]);
            for ei in 0..e {
                let b = format!("{mlp}.experts.{ei}");
                put(&mut t, format!("{b}.gate_proj.weight"), vec![mi, h]);
                put(&mut t, format!("{b}.up_proj.weight"), vec![mi, h]);
                put(&mut t, format!("{b}.down_proj.weight"), vec![h, mi]);
            }
            put(
                &mut t,
                format!("{mlp}.shared_experts.gate_proj.weight"),
                vec![si, h],
            );
            put(
                &mut t,
                format!("{mlp}.shared_experts.up_proj.weight"),
                vec![si, h],
            );
            put(
                &mut t,
                format!("{mlp}.shared_experts.down_proj.weight"),
                vec![h, si],
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

fn run(cfg: &LingConfig, seq: usize) -> Vec<f32> {
    let mut wm = weights(cfg);
    prepare_checkpoint(cfg, &mut wm).expect("prepare checkpoint");
    let built = build_ling_text_flow(cfg, &mut wm, seq, true).expect("build ling flow");
    let mut compiled = compile_built(built, dev()).expect("compile ling flow");

    let (cos, sin) = cfg.rope_tables(seq);
    let ids: Vec<f32> = (0..seq)
        .map(|i| ((i * 7) % cfg.vocab_size) as f32)
        .collect();
    compiled
        .run(&[
            ("input_ids", ids.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("ling forward returned output")
}

#[test]
fn ling_text_flow_compiles_and_runs() {
    let cfg = tiny_config();
    // The hybrid interleave is what makes this model unusual — assert it holds
    // before spending a compile on it.
    assert_eq!(cfg.attn_kind(0), AttnKind::Kda);
    assert_eq!(cfg.attn_kind(3), AttnKind::Mla);
    assert_eq!(cfg.attn_kind(7), AttnKind::Mla);
    assert!(!cfg.is_moe_layer(0) && cfg.is_moe_layer(1));

    let seq = 5usize;
    let out = run(&cfg, seq);
    assert_eq!(out.len(), seq * cfg.vocab_size);
    assert!(out.iter().all(|v| v.is_finite()), "logits must be finite");
    assert!(
        out.iter().any(|v| v.abs() > 1e-9),
        "logits must not be all-zero"
    );
}

/// KDA is a causal recurrence and MLA is causally masked, so extending the
/// prompt must not disturb logits already produced for earlier positions.
/// This is the cheapest end-to-end check that the short conv's left-pad, the
/// delta-net scan and the attention mask all run in the same time direction.
#[test]
fn ling_prefill_is_causal() {
    let cfg = tiny_config();
    let v = cfg.vocab_size;
    let short = run(&cfg, 4);
    let long = run(&cfg, 6);
    for pos in 0..4 {
        for c in 0..v {
            let (a, b) = (short[pos * v + c], long[pos * v + c]);
            assert!(
                (a - b).abs() <= 2e-4 * a.abs().max(1.0),
                "position {pos} channel {c} changed when the prompt grew: {a} vs {b}"
            );
        }
    }
}

/// Without a lower bound the KDA gate takes the softplus branch. Both branches
/// must build and produce finite logits.
#[test]
fn ling_plain_softplus_gate_runs() {
    let mut cfg = tiny_config();
    cfg.kda_lower_bound = None;
    cfg.kda_safe_gate = false;
    let out = run(&cfg, 4);
    assert!(out.iter().all(|v| v.is_finite()));
}

/// `gated_attention_proj_granularity_type: null` drops `g_proj` from MLA.
#[test]
fn ling_ungated_mla_runs() {
    let mut cfg = tiny_config();
    cfg.gated_attention_proj_granularity_type = None;
    let mut wm = weights(&cfg); // no g_proj emitted for MLA layers now
    prepare_checkpoint(&cfg, &mut wm).unwrap();
    let built = build_ling_text_flow(&cfg, &mut wm, 4, true).expect("build ungated");
    let mut compiled = compile_built(built, dev()).expect("compile ungated");
    let (cos, sin) = cfg.rope_tables(4);
    let ids = vec![1.0f32, 2.0, 3.0, 4.0];
    let out = compiled
        .run(&[
            ("input_ids", ids.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ])
        .into_iter()
        .next()
        .unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
}
