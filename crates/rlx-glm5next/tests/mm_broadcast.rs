//! Regression test for batched-matmul broadcasting, both ways round.
//!
//! `rlx_ir::shape::matmul_shape` broadcasts a rank-2 operand across the other
//! operand's batch, so `[M,K] @ [B,K,N]` and `[B,M,K] @ [K,N]` are both legal
//! and both produce `[B,M,N]`. rlx-cpu only ever wired up the second one: the
//! first fell through to the 2-D flatten path, which emits a *single* `Sgemm`
//! against the rhs's first matrix and leaves every output batch after the first
//! holding whatever was already in the arena. A silent wrong answer, not a
//! crash.
//!
//! `rlx_glm5next::mla` expands the MLA latent to per-head keys and values, which
//! is exactly a rank-2 × rank-3 product — it hit this, and only the
//! prefill-vs-decode equivalence test caught it (decode happened to spell the
//! same maths the other way round and was right). The emitter now keeps the
//! batched operand on the left regardless, and this test pins the primitive so a
//! regression shows up here rather than as a wrong logit 40 layers later.
use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

const B: usize = 3; // batch (heads)
const M: usize = 2;
const K: usize = 4;
const N: usize = 5;

fn seqv(n: usize, off: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.37 + off).sin()).collect()
}

/// lhs rank-2 broadcast across a rank-3 rhs batch: [M,K] @ [B,K,N] -> [B,M,N]
#[test]
fn lhs_rank2_broadcast() {
    let a = seqv(M * K, 0.1);
    let bw = seqv(B * K * N, 1.3);
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert("w".into(), (bw.clone(), vec![B, K, N]));
    let mut wm = WeightMap::from_tensors(t);

    let flow = ModelFlow::new("p")
        .with_profile(CompileProfile::llama32_prefill())
        .input("a", Shape::new(&[M, K], DType::F32))
        .plugin_named("s", move |emit, _p| {
            let a = emit.flow_input("a")?.hir_id();
            let w = emit.load_param("w", false)?;
            let mut gb = HirMut::new(emit.hir());
            let o = gb.mm(a, w);
            Ok(Some(emit.wrap(o, Shape::new(&[B, M, N], DType::F32))))
        })
        .output("o");
    let built = flow
        .build_with(&mut WeightMapSource(&mut wm), None)
        .unwrap();
    let mut cp = compile_built(built, Device::Cpu).unwrap();
    let got = cp.run(&[("a", a.as_slice())]).pop().unwrap();

    let mut want = vec![0f32; B * M * N];
    for b in 0..B {
        for m in 0..M {
            for n in 0..N {
                let mut acc = 0f32;
                for k in 0..K {
                    acc += a[m * K + k] * bw[b * K * N + k * N + n];
                }
                want[b * M * N + m * N + n] = acc;
            }
        }
    }
    let worst = got
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    println!("lhs rank2 broadcast: max|delta| = {worst:.6}");
    println!("  got[..6]  = {:?}", &got[..6.min(got.len())]);
    println!("  want[..6] = {:?}", &want[..6]);
    assert!(worst < 1e-5, "rank-2 lhs broadcast is wrong");
}

/// rhs rank-2 broadcast across a rank-3 lhs batch: [B,M,K] @ [K,N] -> [B,M,N]
#[test]
fn rhs_rank2_broadcast() {
    let a = seqv(B * M * K, 0.1);
    let bw = seqv(K * N, 1.3);
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert("w".into(), (bw.clone(), vec![K, N]));
    let mut wm = WeightMap::from_tensors(t);

    let flow = ModelFlow::new("p")
        .with_profile(CompileProfile::llama32_prefill())
        .input("a", Shape::new(&[B, M, K], DType::F32))
        .plugin_named("s", move |emit, _p| {
            let a = emit.flow_input("a")?.hir_id();
            let w = emit.load_param("w", false)?;
            let mut gb = HirMut::new(emit.hir());
            let o = gb.mm(a, w);
            Ok(Some(emit.wrap(o, Shape::new(&[B, M, N], DType::F32))))
        })
        .output("o");
    let built = flow
        .build_with(&mut WeightMapSource(&mut wm), None)
        .unwrap();
    let mut cp = compile_built(built, Device::Cpu).unwrap();
    let got = cp.run(&[("a", a.as_slice())]).pop().unwrap();

    let mut want = vec![0f32; B * M * N];
    for b in 0..B {
        for m in 0..M {
            for n in 0..N {
                let mut acc = 0f32;
                for k in 0..K {
                    acc += a[b * M * K + m * K + k] * bw[k * N + n];
                }
                want[b * M * N + m * N + n] = acc;
            }
        }
    }
    let worst = got
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    println!("rhs rank2 broadcast: max|delta| = {worst:.6}");
    assert!(worst < 1e-5, "rank-2 rhs broadcast is wrong");
}
