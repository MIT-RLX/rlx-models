// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: the full Jamba Mamba-1 mixer (in_proj → causal conv1d → silu →
//! x_proj → dt/B/C RMSNorm → dt_proj → selective_scan + D → gate → out_proj)
//! with tiny synthetic weights, compile on CPU, run; finite `[1,s,hidden]`.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};
use rlx_jamba::mamba::{MambaDims, emit_mamba1_block};
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
fn mamba1_block_compiles_and_runs() {
    let d = MambaDims {
        hidden: 8,
        d_inner: 16,
        dt_rank: 2,
        state: 4,
        d_conv: 4,
        eps: 1e-5,
        seq: 5,
    };
    let (h, di, st, dc, dr) = (d.hidden, d.d_inner, d.state, d.d_conv, d.dt_rank);
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut seed = 1;
    let mut put = |t: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: &str, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        seed += 5;
        t.insert(k.into(), (fill(n, seed), shape));
    };
    put(&mut t, "mamba.in_proj.weight", vec![2 * di, h]);
    put(&mut t, "mamba.conv1d.weight", vec![di, 1, dc]);
    put(&mut t, "mamba.conv1d.bias", vec![di]);
    put(&mut t, "mamba.x_proj.weight", vec![dr + 2 * st, di]);
    put(&mut t, "mamba.dt_proj.weight", vec![di, dr]);
    put(&mut t, "mamba.dt_proj.bias", vec![di]);
    put(&mut t, "mamba.dt_layernorm.weight", vec![dr]);
    put(&mut t, "mamba.b_layernorm.weight", vec![st]);
    put(&mut t, "mamba.c_layernorm.weight", vec![st]);
    put(&mut t, "mamba.A_log", vec![di, st]);
    put(&mut t, "mamba.D", vec![di]);
    put(&mut t, "mamba.out_proj.weight", vec![h, di]);
    let mut wm = WeightMap::from_tensors(t);

    let f = DType::F32;
    let flow = ModelFlow::new("mamba")
        .with_profile(CompileProfile::encoder())
        .input("hidden", Shape::new(&[1, d.seq, h], f))
        .plugin_named("mamba", move |emit, _prev| {
            let hid = emit.flow_input("hidden")?.hir_id();
            let out = emit_mamba1_block(emit, "mamba", hid, d)?;
            Ok(Some(emit.wrap(out, Shape::new(&[1, d.seq, h], f))))
        });
    let built = flow
        .output("out")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build mamba block");
    let mut compiled = compile_built(built, dev()).expect("compile mamba block");

    let hidden = fill(d.seq * h, 42);
    let out = compiled
        .run(&[("hidden", hidden.as_slice())])
        .into_iter()
        .next()
        .expect("mamba block returned output");
    assert_eq!(out.len(), d.seq * h);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "mamba output must be finite"
    );
}
