// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Smoke test: run a Mamba-1 selective scan (`Op::SelectiveScan`) inside a flow
//! via rlx-ssm's `MambaScanStage`, with the `ssm.dt_raw`/`ssm.b`/`ssm.c` named
//! handles fed from flow inputs. Validates the SSM primitive path — the core of
//! every Mamba/hybrid model — on a device.

use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::{DType, Shape};
use rlx_runtime::Device;
use rlx_ssm::{MambaScanStage, MambaScanWeightKeys, register_ir_ops};
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
fn selective_scan_runs_in_flow() {
    register_ir_ops();
    let (d_inner, state, seq) = (8usize, 4usize, 5usize);
    let f = DType::F32;

    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert(
        "blk0.A_log".into(),
        (fill(d_inner * state, 1), vec![d_inner, state]),
    );
    t.insert("blk0.D".into(), (fill(d_inner, 2), vec![d_inner]));
    let mut wm = WeightMap::from_tensors(t);

    let flow = ModelFlow::new("scan")
        .with_profile(CompileProfile::encoder())
        .input("x", Shape::new(&[1, seq, d_inner], f))
        .input("dt_raw", Shape::new(&[1, seq, d_inner], f))
        .input("b", Shape::new(&[1, seq, state], f))
        .input("c", Shape::new(&[1, seq, state], f))
        .plugin_named("setup", |emit, _prev| {
            let x = emit.flow_input("x")?;
            let dt = emit.flow_input("dt_raw")?.hir_id();
            let b = emit.flow_input("b")?.hir_id();
            let c = emit.flow_input("c")?.hir_id();
            emit.set_named("ssm.dt_raw", dt);
            emit.set_named("ssm.b", b);
            emit.set_named("ssm.c", c);
            Ok(Some(x))
        });
    let stage = MambaScanStage::new(
        MambaScanWeightKeys::hf("blk0"),
        state,
        Shape::new(&[1, seq, d_inner], f),
    );
    let flow = flow.plugin_named("scan", stage.plugin());

    let built = flow
        .output("y")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .expect("build scan flow");
    let mut compiled = compile_built(built, dev()).expect("compile scan flow");

    let x = fill(seq * d_inner, 10);
    let dt = fill(seq * d_inner, 11);
    let b = fill(seq * state, 12);
    let c = fill(seq * state, 13);
    let out = compiled
        .run(&[
            ("x", x.as_slice()),
            ("dt_raw", dt.as_slice()),
            ("b", b.as_slice()),
            ("c", c.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("scan returned output");
    assert_eq!(out.len(), seq * d_inner);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "scan output must be finite"
    );
}
