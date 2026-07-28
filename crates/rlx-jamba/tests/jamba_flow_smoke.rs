// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: full Jamba text prefill graph with interleaved Mamba-1 and
//! attention layers (+ dense FFN), tiny synthetic weights, compile on CPU, run;
//! finite `[1, seq, vocab]` logits.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_jamba::attention::JambaAttnDims;
use rlx_jamba::flow::{JambaFlowDims, build_jamba_text_flow};
use rlx_jamba::mamba::MambaDims;
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

fn weights(d: &JambaFlowDims) -> WeightMap {
    let h = d.hidden;
    let m = &d.mamba;
    let a = &d.attn;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1;
    let mut put =
        |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            seed += 3;
            t.insert(k, (fill(n, seed), shape));
        };
    put(&mut t, "model.embed_tokens.weight".into(), vec![d.vocab, h]);
    for (i, &is_attn) in d.layer_is_attention.iter().enumerate() {
        let lp = format!("model.layers.{i}");
        put(&mut t, format!("{lp}.input_layernorm.weight"), vec![h]);
        put(&mut t, format!("{lp}.pre_ff_layernorm.weight"), vec![h]);
        if is_attn {
            let sa = format!("{lp}.self_attn");
            put(
                &mut t,
                format!("{sa}.q_proj.weight"),
                vec![a.num_heads * a.head_dim, h],
            );
            put(
                &mut t,
                format!("{sa}.k_proj.weight"),
                vec![a.num_kv_heads * a.head_dim, h],
            );
            put(
                &mut t,
                format!("{sa}.v_proj.weight"),
                vec![a.num_kv_heads * a.head_dim, h],
            );
            put(
                &mut t,
                format!("{sa}.o_proj.weight"),
                vec![h, a.num_heads * a.head_dim],
            );
        } else {
            let mp = format!("{lp}.mamba");
            put(
                &mut t,
                format!("{mp}.in_proj.weight"),
                vec![2 * m.d_inner, h],
            );
            put(
                &mut t,
                format!("{mp}.conv1d.weight"),
                vec![m.d_inner, 1, m.d_conv],
            );
            put(&mut t, format!("{mp}.conv1d.bias"), vec![m.d_inner]);
            put(
                &mut t,
                format!("{mp}.x_proj.weight"),
                vec![m.dt_rank + 2 * m.state, m.d_inner],
            );
            put(
                &mut t,
                format!("{mp}.dt_proj.weight"),
                vec![m.d_inner, m.dt_rank],
            );
            put(&mut t, format!("{mp}.dt_proj.bias"), vec![m.d_inner]);
            put(&mut t, format!("{mp}.dt_layernorm.weight"), vec![m.dt_rank]);
            put(&mut t, format!("{mp}.b_layernorm.weight"), vec![m.state]);
            put(&mut t, format!("{mp}.c_layernorm.weight"), vec![m.state]);
            put(&mut t, format!("{mp}.A_log"), vec![m.d_inner, m.state]);
            put(&mut t, format!("{mp}.D"), vec![m.d_inner]);
            put(&mut t, format!("{mp}.out_proj.weight"), vec![h, m.d_inner]);
        }
        let ff = format!("{lp}.feed_forward");
        put(
            &mut t,
            format!("{ff}.gate_proj.weight"),
            vec![d.ffn_inter, h],
        );
        put(&mut t, format!("{ff}.up_proj.weight"), vec![d.ffn_inter, h]);
        put(
            &mut t,
            format!("{ff}.down_proj.weight"),
            vec![h, d.ffn_inter],
        );
    }
    put(&mut t, "model.norm.weight".into(), vec![h]);
    put(&mut t, "lm_head.weight".into(), vec![d.vocab, h]);
    WeightMap::from_tensors(t)
}

#[test]
fn jamba_flow_compiles_and_runs() {
    let seq = 5usize;
    let d = JambaFlowDims {
        hidden: 8,
        vocab: 20,
        eps: 1e-5,
        seq,
        ffn_inter: 16,
        tie_word_embeddings: false,
        mamba: MambaDims {
            hidden: 8,
            d_inner: 16,
            dt_rank: 2,
            state: 4,
            d_conv: 4,
            eps: 1e-5,
            seq,
        },
        attn: JambaAttnDims {
            hidden: 8,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            seq,
        },
        layer_is_attention: vec![false, true, false], // mamba, attention, mamba
    };
    let mut wm = weights(&d);
    let built = build_jamba_text_flow(&d, &mut wm, true).expect("build jamba flow");
    let mut compiled = compile_built(built, dev()).expect("compile jamba flow");

    let ids: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    let out = compiled
        .run(&[("input_ids", ids.as_slice())])
        .into_iter()
        .next()
        .expect("jamba forward returned output");
    assert_eq!(out.len(), seq * d.vocab);
    assert!(out.iter().all(|v| v.is_finite()), "logits must be finite");
}
