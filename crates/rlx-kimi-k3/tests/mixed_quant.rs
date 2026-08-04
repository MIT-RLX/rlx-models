//! Regression test for the `RLX_KIMI_QUANT=mixed` per-projection precision policy
//! — the measured 4-bit fix (int8 on the recurrence-amplified `o_proj`, int4 on the
//! rest). Two parts in ONE test (env is process-global; keep it single-threaded):
//!   1) policy logic — `resolve_quant(name)` picks int8 for `o_proj`, int4 else,
//!      and honors `RLX_KIMI_MIXED_HI`.
//!   2) real-weight ladder — on a REAL KDA layer the full-layer output error obeys
//!      int8 < mixed < int4 (mixed strictly beats uniform int4). Skips if unmounted.

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Philox4x32, Shape};
use rlx_kimi_k3::common::{WeightQuant, resolve_quant};
use rlx_kimi_k3::kda::{KdaDims, KdaWeights, build_kda_layer};
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::Path;

const CKPT: &str = "/Volumes/FOUR/kimi";

fn dims(seq: usize) -> KdaDims {
    KdaDims {
        hidden: 7168,
        num_heads: 96,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq,
    }
}

fn run(w: &KdaWeights, d: KdaDims, x: &[f32], mode: &str) -> Vec<f32> {
    unsafe { std::env::set_var("RLX_KIMI_QUANT", mode) };
    let mut hir = HirModule::new("kda");
    let mut g = HirMut::new(&mut hir);
    let hin = g.input("h", Shape::new(&[1, d.seq, d.hidden], DType::F32));
    let mut p = HashMap::new();
    let out = build_kda_layer(&mut g, &mut p, "kda", hin, w, d).unwrap();
    g.set_outputs(vec![out]);
    let mut c = compile_built(built_from_hir(hir, p).unwrap(), Device::Cpu).unwrap();
    let y = c.run(&[("h", x)]).remove(0);
    unsafe { std::env::remove_var("RLX_KIMI_QUANT") };
    y
}

fn rel(base: &[f32], q: &[f32]) -> f32 {
    let (mut sd, mut sb) = (0f64, 0f64);
    for (b, v) in base.iter().zip(q) {
        let e = (*b - *v) as f64;
        sd += e * e;
        sb += (*b as f64) * (*b as f64);
    }
    (sd / sb.max(1e-30)).sqrt() as f32
}

#[test]
fn mixed_policy_and_real_ladder() {
    // ---- part 1: policy logic (no checkpoint needed) ----
    unsafe { std::env::set_var("RLX_KIMI_QUANT", "mixed") };
    unsafe { std::env::remove_var("RLX_KIMI_MIXED_HI") };
    assert_eq!(
        resolve_quant("o_proj"),
        WeightQuant::Int8Ch,
        "mixed must keep o_proj int8"
    );
    assert_eq!(
        resolve_quant("kda.o_proj"),
        WeightQuant::Int8Ch,
        "substring match"
    );
    assert_eq!(
        resolve_quant("q_proj"),
        WeightQuant::Int4G64,
        "non-sensitive → int4"
    );
    assert_eq!(resolve_quant("v_proj"), WeightQuant::Int4G64);
    // configurable high-precision set.
    unsafe { std::env::set_var("RLX_KIMI_MIXED_HI", "o_proj,v_proj") };
    assert_eq!(
        resolve_quant("v_proj"),
        WeightQuant::Int8Ch,
        "RLX_KIMI_MIXED_HI adds v_proj"
    );
    assert_eq!(resolve_quant("q_proj"), WeightQuant::Int4G64);
    unsafe { std::env::remove_var("RLX_KIMI_MIXED_HI") };
    unsafe { std::env::remove_var("RLX_KIMI_QUANT") };

    // ---- part 2: real-weight precision ladder int8 < mixed < int4 ----
    if !Path::new(CKPT).join("config.json").exists() {
        eprintln!("skip real-weight ladder: {CKPT} not mounted");
        return;
    }
    let d = dims(8);
    let mut ck = CheckpointLoader::open(CKPT).unwrap();
    let w = ck
        .load_kda("language_model.model.layers.0", d)
        .expect("load kda 0");
    let mut rng = Philox4x32::new(0x11CED);
    let mut x = vec![0f32; d.seq * d.hidden];
    rng.fill_normal(&mut x);

    let base = run(&w, d, &x, "off");
    assert!(base.iter().all(|v| v.is_finite()));
    let r8 = rel(&base, &run(&w, d, &x, "int8"));
    let rm = rel(&base, &run(&w, d, &x, "mixed"));
    let r4 = rel(&base, &run(&w, d, &x, "int4"));
    eprintln!("KDA L0 full-layer relL2:  int8 {r8:.3e} < mixed {rm:.3e} < int4 {r4:.3e}");
    assert!(
        rm < r4,
        "mixed ({rm:.3e}) must beat uniform int4 ({r4:.3e})"
    );
    assert!(
        r8 <= rm,
        "int8 ({r8:.3e}) should be at least as good as mixed ({rm:.3e})"
    );
}
