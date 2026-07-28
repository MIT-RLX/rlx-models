// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: build the Llama-4 top-1 MoE FFN with tiny synthetic weights,
//! compile on CPU, run. Exercises the router → TopK → sigmoid input-scaling →
//! GroupedMatMul(gate_up) → SwiGLU → GroupedMatMul(down) + shared-expert path.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};
use rlx_llama4::moe::emit_moe_ffn;
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
fn moe_ffn_compiles_and_runs() {
    let (h, inter, e, seq, top_k) = (8usize, 16usize, 4usize, 3usize, 1usize);
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert("ff.router.weight".into(), (fill(e * h, 1), vec![e, h]));
    t.insert(
        "ff.experts.gate_up_proj".into(),
        (fill(e * h * 2 * inter, 2), vec![e, h, 2 * inter]),
    );
    t.insert(
        "ff.experts.down_proj".into(),
        (fill(e * inter * h, 3), vec![e, inter, h]),
    );
    t.insert(
        "ff.shared_expert.gate_proj.weight".into(),
        (fill(inter * h, 4), vec![inter, h]),
    );
    t.insert(
        "ff.shared_expert.up_proj.weight".into(),
        (fill(inter * h, 5), vec![inter, h]),
    );
    t.insert(
        "ff.shared_expert.down_proj.weight".into(),
        (fill(h * inter, 6), vec![h, inter]),
    );
    let mut wm = WeightMap::from_tensors(t);

    let f = DType::F32;
    let flow = ModelFlow::new("llama4_moe")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[1, seq, h], f))
        .plugin_named("moe", move |emit, _prev| {
            let hid = emit.flow_input("hidden")?.hir_id();
            let out = emit_moe_ffn(emit, "ff", hid, seq, h, inter, top_k)?;
            Ok(Some(emit.wrap(out, Shape::new(&[1, seq, h], f))))
        });
    let built = flow
        .output("out")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build moe flow");
    let mut compiled = compile_built(built, dev()).expect("compile moe flow");

    let hidden = fill(seq * h, 42);
    let out = compiled
        .run(&[("hidden", hidden.as_slice())])
        .into_iter()
        .next()
        .expect("moe forward returned an output");
    assert_eq!(out.len(), seq * h);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "moe output must be finite"
    );
}
