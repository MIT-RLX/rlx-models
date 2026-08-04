//! MLA `v_head_dim` (asymmetric-V attention) must be numerically identical to
//! the old V-zero-pad path: padding V's rope columns with zeros contributes
//! exactly 0 to score·V, and those columns are sliced off — so the two paths
//! compute the same `v_head_dim`-wide output. `RLX_MLA_VDIM` toggles the path.
use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::mla::{MlaDims, MlaWeights, build_mla_layer};
use rlx_runtime::Device;
use std::collections::HashMap;

fn fill(n: usize, s: u64) -> Vec<f32> {
    let mut x = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((x >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.1
        })
        .collect()
}

fn dims() -> MlaDims {
    MlaDims {
        hidden: 64,
        num_heads: 4,
        q_lora_rank: 16,
        kv_lora_rank: 16,
        qk_nope_head_dim: 8,
        qk_rope_head_dim: 4,
        v_head_dim: 8,
        eps: 1e-5,
        batch: 1,
        seq: 5,
    }
}

fn run(vdim: bool) -> Vec<f32> {
    match vdim {
        true => unsafe { std::env::set_var("RLX_MLA_VDIM", "1") },
        false => unsafe { std::env::set_var("RLX_MLA_VDIM", "0") },
    }
    let d = dims();
    let nh = d.num_heads;
    let qk = d.qk_nope_head_dim + d.qk_rope_head_dim;
    let w = MlaWeights {
        q_a_proj: fill(d.hidden * d.q_lora_rank, 1),
        q_a_layernorm: vec![1.0; d.q_lora_rank],
        q_b_proj: fill(d.q_lora_rank * nh * qk, 2),
        kv_a_proj_with_mqa: fill(d.hidden * (d.kv_lora_rank + d.qk_rope_head_dim), 3),
        kv_a_layernorm: vec![1.0; d.kv_lora_rank],
        kv_b_proj: fill(d.kv_lora_rank * nh * (d.qk_nope_head_dim + d.v_head_dim), 4),
        g_proj: fill(d.hidden * nh * d.v_head_dim, 5),
        o_proj: fill(nh * d.v_head_dim * d.hidden, 6),
    };
    let mut hir = HirModule::new("mla");
    let mut g = HirMut::new(&mut hir);
    let hin = g.input("h", Shape::new(&[1, d.seq, d.hidden], DType::F32));
    let mut p = HashMap::new();
    let out = build_mla_layer(&mut g, &mut p, "mla", hin, &w, d).unwrap();
    g.set_outputs(vec![out]);
    let mut c = compile_built(built_from_hir(hir, p).unwrap(), Device::Cpu).unwrap();
    c.run(&[("h", fill(d.seq * d.hidden, 100).as_slice())])
        .remove(0)
}

#[test]
fn mla_vdim_matches_pad() {
    let vdim = run(true);
    let pad = run(false);
    unsafe { std::env::remove_var("RLX_MLA_VDIM") };
    assert_eq!(vdim.len(), pad.len(), "output length differs");
    assert!(vdim.iter().all(|v| v.is_finite()));
    let max_abs = vdim
        .iter()
        .zip(&pad)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    // Same ops, same order → should be bit-identical (padding adds exact 0).
    assert!(
        max_abs < 1e-6,
        "MLA vdim vs pad diverged: max|Δ|={max_abs:.3e}"
    );
    eprintln!("MLA vdim-vs-pad max|Δ|={max_abs:.3e}");
}
