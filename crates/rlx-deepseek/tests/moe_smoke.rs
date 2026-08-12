// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: build DeepSeek-V3 fine-grained MoE (group-limited router +
//! GroupedMatMul experts + shared expert) with tiny synthetic weights, compile
//! on CPU, run.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_deepseek::moe::{DeepseekMoeDims, emit_deepseek_moe};
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
fn moe_compiles_and_runs() {
    let d = DeepseekMoeDims {
        hidden: 8,
        moe_inter: 16,
        n_routed: 8,
        top_k: 2,
        n_group: 2,
        topk_group: 1,
        routed_scaling: 1.0,
        shared_inter: 16,
        seq: 3,
        experts_pretransposed: false,
        mxfp4_group: None,
    };
    let (h, inter, e) = (d.hidden, d.moe_inter, d.n_routed);
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1;
    let mut put = |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: &str, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        seed += 5;
        t.insert(k.into(), (fill(n, seed), shape));
    };
    put(&mut t, "ff.gate.weight", vec![e, h]);
    put(&mut t, "ff.gate.e_score_correction_bias", vec![e]);
    put(&mut t, "ff.experts.gate_up_proj", vec![e, 2 * inter, h]); // [E,N,K]
    put(&mut t, "ff.experts.down_proj", vec![e, h, inter]); // [E,N,K]
    put(&mut t, "ff.shared_experts.gate_proj.weight", vec![inter, h]);
    put(&mut t, "ff.shared_experts.up_proj.weight", vec![inter, h]);
    put(&mut t, "ff.shared_experts.down_proj.weight", vec![h, inter]);
    let mut wm = WeightMap::from_tensors(t);

    let f = DType::F32;
    let flow = ModelFlow::new("deepseek_moe")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[1, d.seq, h], f))
        .plugin_named("moe", move |emit, _prev| {
            let hid = emit.flow_input("hidden")?.hir_id();
            let out = emit_deepseek_moe(emit, "ff", hid, d)?;
            Ok(Some(emit.wrap(out, Shape::new(&[1, d.seq, h], f))))
        });
    let built = flow
        .output("out")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build moe");
    let mut compiled = compile_built(built, dev()).expect("compile moe");

    let hidden = fill(d.seq * h, 42);
    let out = compiled
        .run(&[("hidden", hidden.as_slice())])
        .into_iter()
        .next()
        .expect("moe forward returned output");
    assert_eq!(out.len(), d.seq * h);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "moe output must be finite"
    );
}
