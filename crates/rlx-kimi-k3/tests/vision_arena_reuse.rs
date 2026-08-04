//! Regression for the MoonViT vision tower at real head dims (hidden=1024,
//! qkv_hidden=nh*dh=1536): it must stay finite and layout-independent under
//! arena slot reuse. The CPU auto-fusion used to fold the vision attention into
//! a `FusedAttnBlock` that hardcoded the BERT invariant `hidden == nh*dh`,
//! over-reading/over-writing when they differ (NaN under reuse, segfault under a
//! different layout). Fusion is now gated to `hidden == hs` (benchmarking showed
//! the unfused BLAS path is faster for wide-head attention anyway), so the tower
//! runs the unfused path — this guards against the corruption regressing.
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::vision::{VisionBlockWeights, VisionDims, VisionWeights, build_vision};
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

fn dims() -> VisionDims {
    VisionDims {
        hidden: 1024,
        qkv_hidden: 1536,
        num_heads: 12,
        head_dim: 128,
        inter: 4096,
        merge: 2,
        text_hidden: 7168,
        proj_mid: 4096,
        eps: 1e-5,
        grid_h: 8,
        grid_w: 8,
    }
}

fn weights(d: &VisionDims) -> VisionWeights {
    let (hid, qh) = (d.hidden, d.qkv_hidden);
    let blocks: Vec<VisionBlockWeights> = (0..27)
        .map(|i| {
            let sd = 100 + i as u64 * 50;
            VisionBlockWeights {
                norm0: vec![1.0; hid],
                wqkv: fill(hid * 3 * qh, sd + 1),
                wo: fill(qh * hid, sd + 2),
                norm1: vec![1.0; hid],
                fc0: fill(hid * d.inter, sd + 3),
                fc1: fill(d.inter * hid, sd + 4),
            }
        })
        .collect();
    VisionWeights {
        blocks,
        final_norm: vec![1.0; hid],
        proj0: fill(d.merge_in() * d.proj_mid, 700),
        proj2: fill(d.proj_mid * d.text_hidden, 701),
        post_norm: vec![1.0; d.text_hidden],
    }
}

/// Build + compile + run the tower on CPU, returning the projected tokens.
fn run_tower() -> (Vec<f32>, usize) {
    use rlx_core::flow_util::{built_from_hir, compile_built};
    use rlx_runtime::Device;
    let d = dims();
    let w = weights(&d);
    let (l, hid, hd) = (d.seq_len(), d.hidden, d.head_dim);
    let mut hir = HirModule::new("vision");
    let mut g = HirMut::new(&mut hir);
    let hh = g.input("hidden", Shape::new(&[1, l, hid], DType::F32));
    let cos = g.input("cos", Shape::new(&[l, hd / 2], DType::F32));
    let sin = g.input("sin", Shape::new(&[l, hd / 2], DType::F32));
    let mut p = HashMap::new();
    let out = build_vision(&mut g, &mut p, hh, cos, sin, &w, d).unwrap();
    g.set_outputs(vec![out]);
    let mut c = compile_built(built_from_hir(hir, p).unwrap(), Device::Cpu).unwrap();
    let y = c
        .run(&[
            ("hidden", fill(l * hid, 1).as_slice()),
            ("cos", fill(l * (hd / 2), 2).as_slice()),
            ("sin", fill(l * (hd / 2), 3).as_slice()),
        ])
        .remove(0);
    let n_merged = (d.grid_h / d.merge) * (d.grid_w / d.merge);
    (y, n_merged * d.text_hidden)
}

#[test]
fn vision_tower_finite_under_arena_reuse() {
    let (y, expect_len) = run_tower();
    assert_eq!(y.len(), expect_len, "wrong output length");
    assert!(
        y.iter().all(|v| v.is_finite()),
        "vision output has non-finite values under arena reuse (fused-attn mis-fire regression)"
    );
}
