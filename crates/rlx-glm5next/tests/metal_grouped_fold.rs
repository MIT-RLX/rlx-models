// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Metal folds `Transpose(bank, [0,2,1]) -> GroupedMatMul` into a `[E, N, K]`
//! kernel, and must get the same answer for it.
//!
//! `Op::GroupedMatMul` wants the bank as `[E, K, N]` while GGUF stores it as
//! `[E, N, K]`, so a dense MoE layer transposes every bank on every forward. On
//! an M4 Pro that copy is 5.5 ms per bank against 0.52 ms for the eight grouped
//! matmuls it feeds — the copy costs an order of magnitude more than the maths,
//! because at decode each matmul touches only one expert slab while the
//! transpose rewrites the whole bank.
//!
//! Correctness is the whole risk here: the folded kernel reads a different
//! layout with a different threadgroup mapping, and getting either wrong
//! produces finite, plausible numbers. Checked against the definition, per
//! expert, on real hardware.

mod common;
use common::{dev, fill};
use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

const E: usize = 6;
const O: usize = 128;
const I: usize = 64;
const M: usize = 1;

/// GroupedMatMul over Transpose(bank) — foldable — optionally with a second
/// reader that forces materialization.
fn run(dev: Device, extra: bool, bank: &[f32], x: &[f32], ids: &[f32]) -> Vec<f32> {
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert("bank".into(), (bank.to_vec(), vec![E, O, I]));
    let mut wm = WeightMap::from_tensors(t);
    let out = Shape::new(&[M, O], DType::F32);
    let built = ModelFlow::new("g")
        .with_profile(CompileProfile::llama32_prefill())
        .input("x", Shape::new(&[M, I], DType::F32))
        .input("idx", Shape::new(&[M], DType::F32))
        .plugin_named("g", move |emit, _p| {
            let b = emit.load_param("bank", false)?;
            let xi = emit.flow_input("x")?.hir_id();
            let ii = emit.flow_input("idx")?.hir_id();
            let mut gb = HirMut::new(emit.hir());
            let bt = gb.transpose_(b, vec![0, 2, 1]);
            let y = gb.grouped_matmul(xi, bt, ii);
            let y = if extra {
                let s = gb.add(bt, bt);
                let _ = s;
                y
            } else {
                y
            };
            Ok(Some(emit.wrap(y, out.clone())))
        })
        .output("y")
        .build_with(&mut WeightMapSource(&mut wm), None)
        .unwrap();
    compile_built(built, dev)
        .unwrap()
        .run(&[("x", x), ("idx", ids)])
        .into_iter()
        .next()
        .unwrap()
}

fn reference(bank: &[f32], x: &[f32], e: usize) -> Vec<f32> {
    (0..O)
        .map(|j| {
            (0..I)
                .map(|k| x[k] as f64 * bank[e * O * I + j * I + k] as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

#[test]
fn metal_folded_grouped_matmul_matches() {
    let bank = fill(E * O * I, 3);
    let x = fill(I, 9);
    for e in 0..E {
        let ids = vec![e as f32];
        let want = reference(&bank, &x, e);
        let cpu = run(Device::Cpu, false, &bank, &x, &ids);
        let err = |v: &[f32]| {
            v.iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs() / b.abs().max(1e-3))
                .fold(0f32, f32::max)
        };
        assert!(err(&cpu) < 1e-4, "cpu expert {e} err {}", err(&cpu));
        #[cfg(feature = "metal")]
        {
            let mtl = run(Device::Metal, false, &bank, &x, &ids);
            assert!(
                err(&mtl) < 1e-4,
                "metal expert {e} rel err {} — the folded _bt kernel reads the bank wrong",
                err(&mtl)
            );
        }
    }
    let _ = dev();
}
