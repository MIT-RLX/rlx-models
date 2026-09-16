//! `ScatterElements` / `GatherElements` with **narrower indices than data**.
//!
//! ONNX lets `indices` be smaller than `data` along any axis; a flat position in
//! `indices` then decomposes by the *indices'* strides, not the data's.
//! rlx-cpu's `ScatterElements` was not given the indices' shape at all, so it
//! guessed with the data's axis stride and silently wrote to the wrong rows —
//! correct only when the two shapes happened to agree. `GatherElements` was
//! always right, which is what made the asymmetry easy to miss.
//!
//! The DSA indexer scatters `[seq, select_k]` selections into a `[seq, seq]`
//! visibility mask, i.e. exactly this case, and the result was a mask that let
//! queries attend to their own future. Fixed upstream by threading
//! `indices_shape` into the thunk; pinned here because the failure mode is a
//! plausible-looking wrong answer rather than a crash.
use rlx_core::flow_util::{WeightMapSource, compile_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, ModelFlow};
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Op, ScatterNdReduction};
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

const R: usize = 5; // rows
const C: usize = 7; // data cols
const K: usize = 3; // index cols  (K < C — the case the indexer uses)

#[test]
fn scatter_elements_narrow_indices() {
    // data [R,C] zeros; indices [R,K]; updates [R,K] = 1.0
    let idx: Vec<f32> = (0..R)
        .flat_map(|r| (0..K).map(move |j| ((r + j) % C) as f32))
        .collect();
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert("idx".into(), (idx.clone(), vec![R, K]));
    t.insert("upd".into(), (vec![1.0; R * K], vec![R, K]));
    t.insert("zero".into(), (vec![0.0; R * C], vec![R, C]));
    let mut wm = WeightMap::from_tensors(t);

    let os = Shape::new(&[R, C], DType::F32);
    let flow = ModelFlow::new("p")
        .with_profile(CompileProfile::llama32_prefill())
        .input("dummy", Shape::new(&[1, 1], DType::F32))
        .plugin_named("s", move |emit, _p| {
            let z = emit.load_param("zero", false)?;
            let i = emit.load_param("idx", false)?;
            let u = emit.load_param("upd", false)?;
            let mut gb = HirMut::new(emit.hir());
            let o = gb.add_node(
                Op::ScatterElements {
                    axis: 1,
                    reduction: ScatterNdReduction::Max,
                },
                vec![z, i, u],
                Shape::new(&[R, C], DType::F32),
            );
            Ok(Some(emit.wrap(o, os.clone())))
        })
        .output("out");
    let built = flow
        .build_with(&mut WeightMapSource(&mut wm), None)
        .unwrap();
    let mut cp = compile_built(built, Device::Cpu).unwrap();
    let got = cp.run(&[("dummy", &[0.0f32][..])]).pop().unwrap();

    let mut want = vec![0f32; R * C];
    for r in 0..R {
        for j in 0..K {
            want[r * C + (idx[r * K + j] as usize)] = 1.0;
        }
    }

    println!("got:");
    for r in 0..R {
        println!("  {:?}", &got[r * C..(r + 1) * C]);
    }
    println!("want:");
    for r in 0..R {
        println!("  {:?}", &want[r * C..(r + 1) * C]);
    }
    assert_eq!(got, want, "ScatterElements axis=1 with K<C indices");
}

#[test]
fn gather_elements_narrow_indices() {
    // data [R,C]; indices [R,K] -> out [R,K]
    let data: Vec<f32> = (0..R * C).map(|i| i as f32).collect();
    let idx: Vec<f32> = (0..R)
        .flat_map(|r| (0..K).map(move |j| ((r + j) % C) as f32))
        .collect();
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    t.insert("data".into(), (data.clone(), vec![R, C]));
    t.insert("idx".into(), (idx.clone(), vec![R, K]));
    let mut wm = WeightMap::from_tensors(t);

    let os = Shape::new(&[R, K], DType::F32);
    let flow = ModelFlow::new("p")
        .with_profile(CompileProfile::llama32_prefill())
        .input("dummy", Shape::new(&[1, 1], DType::F32))
        .plugin_named("s", move |emit, _p| {
            let d = emit.load_param("data", false)?;
            let i = emit.load_param("idx", false)?;
            let mut gb = HirMut::new(emit.hir());
            let o = gb.add_node(
                Op::GatherElements { axis: 1 },
                vec![d, i],
                Shape::new(&[R, K], DType::F32),
            );
            Ok(Some(emit.wrap(o, os.clone())))
        })
        .output("out");
    let built = flow
        .build_with(&mut WeightMapSource(&mut wm), None)
        .unwrap();
    let mut cp = compile_built(built, Device::Cpu).unwrap();
    let got = cp.run(&[("dummy", &[0.0f32][..])]).pop().unwrap();

    let mut want = vec![0f32; R * K];
    for r in 0..R {
        for j in 0..K {
            want[r * K + j] = data[r * C + (idx[r * K + j] as usize)];
        }
    }
    println!("gather got:  {got:?}");
    println!("gather want: {want:?}");
    assert_eq!(got, want, "GatherElements axis=1 with K<C indices");
}
