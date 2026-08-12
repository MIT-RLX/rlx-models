// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: build a tiny Llama-3.2 text graph with one cross-attention layer
//! inserted via `Llama32Flow::layer()` + `cross_attn_stage`, compile on CPU, and
//! run it. Verifies the self-attention / cross-attention layer mix, GQA
//! repeat_kv, and the vision-KV synthesized constant all compile and produce a
//! finite `[1, seq, hidden]` hidden state — without the real checkpoint.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_ir::{DType, Shape};
use rlx_llama32::{Llama32Config, Llama32Flow};
use rlx_mllama::cross_attn::{CROSS_STATES_INPUT, CrossAttnDims, cross_attn_stage};
use std::collections::HashMap;

fn dev() -> rlx_runtime::Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(rlx_runtime::Device::Cpu)
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

fn tiny_cfg(layers: usize) -> Llama32Config {
    Llama32Config {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: layers,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 16,
        rms_norm_eps: 1e-5,
        rope_theta: 500_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        attention_bias: false,
        head_dim: None,
        rope_scaling: None,
        embedding_scale: None,
        residual_scale: None,
        attention_scale: None,
        logit_scale: None,
        num_loops: 1,
        skip_loop_final_norm: false,
        rope_style: rlx_ir::RopeStyle::NeoX,
        gguf_arch: None,
        rope_dim: None,
        sliding_window: None,
        sliding_window_pattern: None,
        final_logit_softcap: None,
    }
}

fn text_weights(cfg: &Llama32Config, cross: &[usize], kv_seq: usize) -> WeightMap {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_proj_dim();
    let kv_dim = cfg.kv_proj_dim();
    let hd = cfg.head_dim();
    let int_dim = cfg.intermediate_size;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1u64;
    let mut put =
        |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            seed += 7;
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
            format!("{lp}.mlp.gate_proj.weight"),
            vec![int_dim, h],
        );
        put(&mut t, format!("{lp}.mlp.up_proj.weight"), vec![int_dim, h]);
        put(
            &mut t,
            format!("{lp}.mlp.down_proj.weight"),
            vec![h, int_dim],
        );
        if cross.contains(&i) {
            put(
                &mut t,
                format!("{lp}.cross_attn.q_proj.weight"),
                vec![q_dim, h],
            );
            put(
                &mut t,
                format!("{lp}.cross_attn.k_proj.weight"),
                vec![kv_dim, h],
            );
            put(
                &mut t,
                format!("{lp}.cross_attn.v_proj.weight"),
                vec![kv_dim, h],
            );
            put(
                &mut t,
                format!("{lp}.cross_attn.o_proj.weight"),
                vec![h, q_dim],
            );
            put(&mut t, format!("{lp}.cross_attn.q_norm.weight"), vec![hd]);
            put(&mut t, format!("{lp}.cross_attn.k_norm.weight"), vec![hd]);
            put(&mut t, format!("{lp}.cross_attn_attn_gate"), vec![1]);
            put(&mut t, format!("{lp}.cross_attn_mlp_gate"), vec![1]);
        } else {
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
        }
    }
    put(&mut t, "model.norm.weight".into(), vec![h]);
    let _ = kv_seq;
    WeightMap::from_tensors(t)
}

#[test]
fn text_graph_with_cross_attn_compiles_and_runs() {
    let layers = 3usize;
    let cfg = tiny_cfg(layers);
    let cross_layers = vec![1usize];
    let seq = 4usize;
    let kv_seq = 5usize;
    let d = CrossAttnDims {
        hidden: cfg.hidden_size,
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim(),
        eps: cfg.rms_norm_eps as f32,
        text_seq: seq,
        kv_seq,
    };
    let cross_states = fill(kv_seq * cfg.hidden_size, 99);
    let mut wm = text_weights(&cfg, &cross_layers, kv_seq);

    let hook_cross = cross_layers.clone();
    let kv_shape = Shape::new(&[1, kv_seq, cfg.hidden_size], DType::F32);
    let built = Llama32Flow::new(&cfg)
        .prefill()
        .batch(1)
        .seq(seq)
        .hidden_only()
        .layer(move |ctx| {
            if hook_cross.contains(&ctx.index()) {
                cross_attn_stage(ctx.weight_index(), d)
            } else {
                ctx.default_stage()
            }
        })
        .patch_flow(move |flow| flow.input(CROSS_STATES_INPUT, kv_shape.clone()))
        .build(&mut wm)
        .expect("build mllama text prefill flow");

    let mut compiled = compile_built(built, dev()).expect("compile text flow");
    let ids: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0];
    let out = compiled
        .run(&[
            ("input_ids", ids.as_slice()),
            (CROSS_STATES_INPUT, cross_states.as_slice()),
        ]) // device via dev()
        .into_iter()
        .next()
        .expect("text forward returned an output");

    assert_eq!(
        out.len(),
        seq * cfg.hidden_size,
        "hidden [1,{seq},{}]",
        cfg.hidden_size
    );
    assert!(out.iter().all(|v| v.is_finite()), "hidden must be finite");
}
