// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! **#2 go/no-go:** prove Kimi's raw MXFP4 expert bytes drive the fused GPU
//! `Op::DequantGroupedMatMulMlx{MlxMxfp4}` bit-for-bit vs the current CPU path
//! (dequant to f32 → grouped matmul). Layout: `codes` = `weight_packed` raw (U8,
//! 2 E2M1 nibbles/byte), `scales` = each E8M0 byte → `bf16(e<<7)` = 2^(e-127),
//! `biases` = zeros (BF16); weights fed via `CompiledGraph::set_param_typed`.
//! If green, the paged-MoE experts can skip the CPU dequant entirely (opt #2).

use half::bf16;
use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::op::Op;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

fn dev() -> Device {
    match std::env::var("RLX_TEST_DEVICE").ok().as_deref() {
        Some("metal") | Some("mtl") => Device::Metal,
        Some("mlx") => Device::Mlx,
        _ => Device::Cpu,
    }
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.15
        })
        .collect()
}
fn bytes(n: usize, seed: u64, lo: u8, hi: u8) -> Vec<u8> {
    let span = (hi - lo) as u64 + 1;
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            lo + ((s >> 33) % span) as u8
        })
        .collect()
}

const VAL: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Reference: dequant one `[out, in]` expert (codes+E8M0 scales) → f32 `[out*in]`.
fn dequant(codes: &[u8], scales: &[u8], out: usize, k: usize, gs: usize) -> Vec<f32> {
    let (pcols, scols) = (k / 2, k / gs);
    let mut w = vec![0f32; out * k];
    for r in 0..out {
        let prow = &codes[r * pcols..(r + 1) * pcols];
        let srow = &scales[r * scols..(r + 1) * scols];
        for c in 0..k {
            let byte = prow[c / 2];
            let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let s = 2f32.powi(srow[c / gs] as i32 - 127);
            w[r * k + c] = VAL[nib as usize] * s;
        }
    }
    w
}

#[test]
fn mxfp4_grouped_matmul_matches_f32() {
    let d = dev();
    let (e, out, k, m, gs) = (3usize, 8usize, 64usize, 4usize, 32usize);
    let ng = k / gs;

    let codes = bytes(e * out * (k / 2), 1, 0, 255); // 2 E2M1 nibbles/byte
    let scales = bytes(e * out * ng, 2, 118, 132); // E8M0 bytes near 2^0
    let x = fill(m * k, 3);
    let eidx: Vec<f32> = (0..m).map(|i| (i % e) as f32).collect();

    // ── f32 reference: dequant each expert, grouped matmul y[m,:] = x[m] @ W_e^T ──
    let mut want = vec![0f32; m * out];
    for row in 0..m {
        let ei = eidx[row] as usize;
        let w = dequant(
            &codes[ei * out * (k / 2)..(ei + 1) * out * (k / 2)],
            &scales[ei * out * ng..(ei + 1) * out * ng],
            out,
            k,
            gs,
        );
        for o in 0..out {
            let mut acc = 0f32;
            for c in 0..k {
                acc += x[row * k + c] * w[o * k + c];
            }
            want[row * out + o] = acc;
        }
    }

    // ── op path: scales E8M0→bf16(e<<7); biases zero; feed via set_param_typed ──
    let scales_bf16: Vec<u8> = scales
        .iter()
        .flat_map(|&s| bf16::from_bits((s as u16) << 7).to_le_bytes())
        .collect();
    let biases_bf16: Vec<u8> = vec![0u8; e * out * ng * 2];

    let scheme = QuantScheme::MlxMxfp4 {
        group_size: gs as u32,
    };
    let mut hir = HirModule::new("gmm");
    let mut g = HirMut::new(&mut hir);
    let x_id = g.input("x", Shape::new(&[m, k], DType::F32));
    let idx_id = g.input("eidx", Shape::new(&[m], DType::F32));
    let c_id = g.param("codes", Shape::new(&[e * out * (k / 2)], DType::U8));
    let s_id = g.param("scales", Shape::new(&[e, out, ng], DType::BF16));
    let b_id = g.param("biases", Shape::new(&[e, out, ng], DType::BF16));
    let y = g.add_node(
        Op::DequantGroupedMatMulMlx { scheme },
        vec![x_id, c_id, s_id, b_id, idx_id],
        Shape::new(&[m, out], DType::F32),
    );
    g.set_outputs(vec![y]);
    let built = built_from_hir(hir, HashMap::new()).expect("built");
    let mut compiled = compile_built(built, d).expect("compile");
    compiled.set_param_typed("codes", &codes, DType::U8);
    compiled.set_param_typed("scales", &scales_bf16, DType::BF16);
    compiled.set_param_typed("biases", &biases_bf16, DType::BF16);
    let got = compiled
        .run(&[("x", x.as_slice()), ("eidx", eidx.as_slice())])
        .remove(0);

    let worst = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("MXFP4 grouped matmul (op vs f32) {d:?}: worst |Δ| = {worst:.3e}");
    assert!(
        worst < 1e-3,
        "op diverges from f32 grouped matmul: {worst:.3e}"
    );
}
