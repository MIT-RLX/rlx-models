// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: build DeepSeek-V3 MLA with tiny synthetic weights, compile on
//! CPU, run. Exercises the low-rank Q/KV LoRA path, nope/rope split, decoupled
//! RoPE, value zero-pad to qk_head_dim + output slice, and o_proj.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_deepseek::mla::{MlaDims, emit_mla_attention};
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};
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

#[test]
fn mla_compiles_and_runs() {
    let d = MlaDims {
        hidden: 16,
        num_heads: 2,
        q_lora_rank: 8,
        kv_lora_rank: 6,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 2,
        v_head_dim: 4,
        eps: 1e-6,
        seq: 3,
        score_scale: (6f32).powf(-0.5),
    };
    let h = d.hidden;
    let qk = d.qk_nope_head_dim + d.qk_rope_head_dim; // 6
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1;
    let mut put = |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: &str, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        seed += 5;
        t.insert(k.into(), (fill(n, seed), shape));
    };
    put(&mut t, "att.q_a_proj.weight", vec![d.q_lora_rank, h]);
    put(&mut t, "att.q_a_layernorm.weight", vec![d.q_lora_rank]);
    put(
        &mut t,
        "att.q_b_proj.weight",
        vec![d.num_heads * qk, d.q_lora_rank],
    );
    put(
        &mut t,
        "att.kv_a_proj_with_mqa.weight",
        vec![d.kv_lora_rank + d.qk_rope_head_dim, h],
    );
    put(&mut t, "att.kv_a_layernorm.weight", vec![d.kv_lora_rank]);
    put(
        &mut t,
        "att.kv_b_proj.weight",
        vec![
            d.num_heads * (d.qk_nope_head_dim + d.v_head_dim),
            d.kv_lora_rank,
        ],
    );
    put(
        &mut t,
        "att.o_proj.weight",
        vec![h, d.num_heads * d.v_head_dim],
    );
    let mut wm = WeightMap::from_tensors(t);

    let f = DType::F32;
    let half = d.qk_rope_head_dim / 2; // 1
    let flow = ModelFlow::new("mla")
        .with_profile(CompileProfile::llama32_prefill())
        .input("hidden", Shape::new(&[1, d.seq, h], f))
        .input("rope_cos", Shape::new(&[d.seq, half], f))
        .input("rope_sin", Shape::new(&[d.seq, half], f))
        .plugin_named("mla", move |emit, _prev| {
            let hid = emit.flow_input("hidden")?.hir_id();
            let out = emit_mla_attention(emit, "att", hid, d)?;
            Ok(Some(emit.wrap(out, Shape::new(&[1, d.seq, h], f))))
        });
    let built = flow
        .output("out")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build mla");
    let mut compiled = compile_built(built, dev()).expect("compile mla");

    let hidden = fill(d.seq * h, 42);
    let cos = fill(d.seq * half, 7);
    let sin = fill(d.seq * half, 8);
    let out = compiled
        .run(&[
            ("hidden", hidden.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("mla forward returned output");
    assert_eq!(out.len(), d.seq * h);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "mla output must be finite"
    );
}
