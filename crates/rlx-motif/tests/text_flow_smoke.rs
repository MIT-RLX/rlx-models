// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Full Motif-3 prefill graph on tiny synthetic weights: MHC hyper-connections
//! wrapping GDLA attention (global / sliding-window interleave) and a PolyNorm
//! FFN that is dense on the first layers and MoE after.
//!
//! The config keeps every shape *relationship* of the published Motif-3 —
//! `num_key_value_heads == num_noise_heads`, `v_head_dim < head_dim`,
//! `qk_nope + qk_rope == head_dim`, `n_dense_first_layers` before the MoE, the
//! `sliding_window_period` interleave, YaRN past `original_seq_len` — at a size
//! that compiles in a test.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_motif::{
    LayerAttn, MotifConfig, build_motif_text_flow, drop_mtp_layers, prepare_checkpoint,
};
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
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.4
        })
        .collect()
}

/// Scaled-down Motif-3: 4 layers — global attention on 0/2, sliding on 1/3,
/// dense MLP on 0/1 and MoE on 2/3.
fn tiny_config() -> MotifConfig {
    MotifConfig::from_json_str(
        r#"{"vocab_size":32,"hidden_size":12,"intermediate_size":20,"num_hidden_layers":4,
            "num_attention_heads":6,"num_key_value_heads":2,"num_noise_heads":2,
            "head_dim":8,"qk_rope_head_dim":4,"v_head_dim":6,
            "q_lora_rank":5,"kv_lora_rank":7,
            "hidden_act":"poly_norm","attention_cls":"gdla","diff_v2":true,
            "elementwise_attn_output_gate":true,
            "rms_norm_eps":1e-5,"rope_theta":10000.0,"swa_rope_theta":10000.0,
            "max_position_embeddings":64,"original_seq_len":4,"rope_factor":8.0,"mscale":1.0,
            "rope_scaling":{"rope_type":"yarn","factor":8.0,
                            "original_max_position_embeddings":4,
                            "beta_fast":32.0,"beta_slow":1.0,"rope_theta":10000.0},
            "use_sliding_window":true,"sliding_window":2,
            "sliding_window_pattern":"interleave","sliding_window_period":2,
            "num_experts":8,"experts_top_k":2,"num_shared_experts":1,
            "moe_intermediate_size":6,"interleave_moe_layer_step":1,
            "n_dense_first_layers":2,"score_func":"sigmoid","route_norm":true,
            "route_scale":2.0,"load_balance_coeff":0.0001,
            "mhc_enabled":true,"mhc_expansion_rate":3,"mhc_sinkhorn_iters":20,
            "polynorm_output_scale":0.5,"polynorm_bias_clamp":0.5,"hidden_clamp":1000000.0,
            "tie_word_embeddings":false,"num_nextn_predict_layers":1}"#,
    )
    .expect("parse tiny config")
}

fn weights(cfg: &MotifConfig) -> WeightMap {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim();
    let heads = cfg.num_attention_heads;
    let kv = cfg.num_key_value_heads;
    let nope = cfg.qk_nope_head_dim();
    let rope = cfg.qk_rope_head_dim();
    let vd = cfg.v_head_dim();
    let sig = cfg.n_signal_heads();
    let ql = cfg.q_lora_rank;
    let kvl = cfg.kv_lora_rank;
    let e = cfg.num_experts;
    let mi = cfg.moe_intermediate_size();
    let ex = cfg.mhc_expansion_rate;

    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    let mut put =
        |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            seed += 3;
            t.insert(k, (fill(n, seed), shape));
        };

    put(&mut t, rlx_motif::EMBED_KEY.into(), vec![cfg.vocab_size, h]);
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        put(&mut t, format!("{lp}.input_layernorm.weight"), vec![h]);
        put(
            &mut t,
            format!("{lp}.post_attention_layernorm.weight"),
            vec![h],
        );

        let at = format!("{lp}.self_attn");
        put(&mut t, format!("{at}.wq_a.weight"), vec![ql, h]);
        put(&mut t, format!("{at}.q_norm.weight"), vec![ql]);
        put(&mut t, format!("{at}.wq_b.weight"), vec![heads * hd, ql]);
        put(&mut t, format!("{at}.wq_b_gate.weight"), vec![sig * vd, ql]);
        put(&mut t, format!("{at}.wkv_a.weight"), vec![kvl + rope, h]);
        put(&mut t, format!("{at}.kv_norm.weight"), vec![kvl]);
        put(
            &mut t,
            format!("{at}.wkv_b.weight"),
            vec![kv * (nope + vd), kvl],
        );
        put(&mut t, format!("{at}.lambda_proj.weight"), vec![sig, h]);
        put(&mut t, format!("{at}.wo.weight"), vec![h, sig * vd]);

        if cfg.mhc_enabled {
            for m in ["mhc_attn", "mhc_ffn"] {
                let mp = format!("{lp}.{m}");
                put(&mut t, format!("{mp}.rms_norm.weight"), vec![ex * h]);
                put(&mut t, format!("{mp}.proj_pre.weight"), vec![ex, ex * h]);
                put(&mut t, format!("{mp}.proj_post.weight"), vec![ex, ex * h]);
                put(
                    &mut t,
                    format!("{mp}.proj_res.weight"),
                    vec![ex * ex, ex * h],
                );
                put(&mut t, format!("{mp}.bias_pre"), vec![ex]);
                put(&mut t, format!("{mp}.bias_post"), vec![ex]);
                put(&mut t, format!("{mp}.bias_res"), vec![ex, ex]);
                for a in ["alpha_pre", "alpha_post", "alpha_res"] {
                    put(&mut t, format!("{mp}.{a}"), vec![1]);
                }
            }
        }

        if cfg.is_moe_layer(i) {
            let mp = format!("{lp}.moe");
            put(&mut t, format!("{mp}.router.gate.weight"), vec![e, h]);
            put(&mut t, format!("{mp}.expert_bias"), vec![e]);
            put(
                &mut t,
                format!("{mp}.experts.gate_up_proj"),
                vec![e, 2 * mi, h],
            );
            put(&mut t, format!("{mp}.experts.down_proj"), vec![e, h, mi]);
            put(&mut t, format!("{mp}.experts.act_fn.weight"), vec![e, 3]);
            put(&mut t, format!("{mp}.experts.act_fn.bias"), vec![e, 1]);
            for p in ["gate_proj", "up_proj"] {
                put(
                    &mut t,
                    format!("{mp}.shared_experts.{p}.weight"),
                    vec![mi, h],
                );
            }
            put(
                &mut t,
                format!("{mp}.shared_experts.down_proj.weight"),
                vec![h, mi],
            );
            put(
                &mut t,
                format!("{mp}.shared_experts.act_fn.weight"),
                vec![3],
            );
            put(&mut t, format!("{mp}.shared_experts.act_fn.bias"), vec![1]);
        } else {
            let mp = format!("{lp}.mlp");
            let di = cfg.intermediate_size;
            put(&mut t, format!("{mp}.gate_proj.weight"), vec![di, h]);
            put(&mut t, format!("{mp}.up_proj.weight"), vec![di, h]);
            put(&mut t, format!("{mp}.down_proj.weight"), vec![h, di]);
            put(&mut t, format!("{mp}.act_fn.weight"), vec![3]);
            put(&mut t, format!("{mp}.act_fn.bias"), vec![1]);
        }
    }
    // The unused MTP head the checkpoint ships.
    put(
        &mut t,
        "model.mtp_layers.0.input_proj.weight".into(),
        vec![h, 2 * h],
    );
    put(&mut t, "model.norm.weight".into(), vec![h]);
    put(&mut t, "lm_head.weight".into(), vec![cfg.vocab_size, h]);
    WeightMap::from_tensors(t)
}

fn run(cfg: &MotifConfig, seq: usize) -> Vec<f32> {
    let mut wm = weights(cfg);
    assert_eq!(drop_mtp_layers(&mut wm), 1, "MTP block should be dropped");
    prepare_checkpoint(cfg, &mut wm).expect("prepare checkpoint");
    let built = build_motif_text_flow(cfg, &mut wm, seq, true).expect("build motif flow");
    let mut compiled = compile_built(built, dev()).expect("compile motif flow");

    let (cos, sin) = cfg.rope_tables(seq);
    let (swa_cos, swa_sin) = cfg.swa_rope_tables(seq);
    let ids: Vec<f32> = (0..seq)
        .map(|i| ((i * 7) % cfg.vocab_size) as f32)
        .collect();
    let mut inputs: Vec<(&str, &[f32])> = vec![
        ("input_ids", ids.as_slice()),
        ("rope_cos", cos.as_slice()),
        ("rope_sin", sin.as_slice()),
    ];
    if cfg.has_sliding_layers() {
        inputs.push(("swa_rope_cos", swa_cos.as_slice()));
        inputs.push(("swa_rope_sin", swa_sin.as_slice()));
    }
    compiled
        .run(&inputs)
        .into_iter()
        .next()
        .expect("motif forward returned output")
}

#[test]
fn motif_text_flow_compiles_and_runs() {
    let cfg = tiny_config();
    // The two interleaves are what make this model unusual — assert them before
    // spending a compile.
    assert_eq!(cfg.layer_attn(0), LayerAttn::Global);
    assert_eq!(cfg.layer_attn(1), LayerAttn::Sliding(2));
    assert!(!cfg.is_moe_layer(1) && cfg.is_moe_layer(2));

    let seq = 5usize;
    let out = run(&cfg, seq);
    assert_eq!(out.len(), seq * cfg.vocab_size);
    assert!(out.iter().all(|v| v.is_finite()), "logits must be finite");
    assert!(
        out.iter().any(|v| v.abs() > 1e-9),
        "logits must not be all-zero"
    );
}

/// Every mixing path in the model is causal — attention is masked, MHC and the
/// MoE router are per token — so extending the prompt must not move logits that
/// were already produced. This is the cheapest end-to-end check that no stage
/// leaks information backwards (a wrong Sinkhorn axis or a mis-shaped mask would
/// show up here).
#[test]
fn motif_prefill_is_causal() {
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

/// `mhc_enabled = false` falls back to `MotifDecoderLayer`'s ordinary pre-norm
/// residual stack — no expansion, no gates, no Sinkhorn.
#[test]
fn motif_without_mhc_runs() {
    let mut cfg = tiny_config();
    cfg.mhc_enabled = false;
    let out = run(&cfg, 4);
    assert_eq!(out.len(), 4 * cfg.vocab_size);
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(out.iter().any(|v| v.abs() > 1e-9));
}

/// All-global (`use_sliding_window = false`) needs no SWA tables at all.
#[test]
fn motif_without_sliding_window_runs() {
    let mut cfg = tiny_config();
    cfg.use_sliding_window = false;
    assert!(!cfg.has_sliding_layers());
    let out = run(&cfg, 4);
    assert!(out.iter().all(|v| v.is_finite()));
}

/// A dense-only configuration (no experts) must still build: the FFN is then the
/// PolyNorm `MotifMLP` on every layer.
#[test]
fn motif_dense_only_runs() {
    let mut cfg = tiny_config();
    cfg.num_experts = 0;
    cfg.interleave_moe_layer_step = 0;
    assert!((0..cfg.num_hidden_layers).all(|i| !cfg.is_moe_layer(i)));
    let out = run(&cfg, 4);
    assert!(out.iter().all(|v| v.is_finite()));
}
