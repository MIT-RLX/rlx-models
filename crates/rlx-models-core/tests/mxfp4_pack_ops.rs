// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! **Gate for [`rlx_models_core::mxfp4_pack`]**: prove the encoder's byte layout
//! is the one rlx's MXFP4 *kernels* actually read.
//!
//! The unit tests in the module only check the encoder against its own
//! `dequantize`, so a consistent misreading of the layout (swapped nibble order,
//! transposed scale grid, wrong per-expert stride) would pass both. These tests
//! close that loop by feeding the packed bytes to the real ops —
//! `Op::DequantMatMul` (dense) and `Op::DequantGroupedMatMulMlx` (MoE) — and
//! comparing against an f32 matmul of the dequantized weight.
//!
//! Tolerance is f32-accumulation only (~1e-4 relative): both sides multiply the
//! *same* dequantized values, so quantization error cancels exactly. A layout
//! bug does not produce a small error — it produces garbage.
//!
//! `RLX_TEST_DEVICE=metal|mlx|cuda|…` runs the same checks on a backend.

use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::op::Op;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_models_core::flow_util::{built_from_hir, compile_built};
use rlx_models_core::mxfp4_pack::{GROUP_SIZE, dequantize, quantize_rows};
use rlx_runtime::Device;
use std::collections::HashMap;

fn dev() -> Device {
    match std::env::var("RLX_TEST_DEVICE").ok().as_deref() {
        Some("metal") | Some("mtl") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("cuda") => Device::Cuda,
        Some("rocm") => Device::Rocm,
        Some("wgpu") | Some("gpu") => Device::Gpu,
        _ => Device::Cpu,
    }
}

/// Weight-like values: a mix of scales so the per-group exponent search is
/// actually exercised (a uniform magnitude would make every group pick the same
/// scale and hide an indexing bug in the scale grid).
fn weights(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((s >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0;
            // Per-group magnitude sweeps 2^-6 … 2^0 across the tensor.
            u * f32::exp2(-(((i / GROUP_SIZE) % 7) as f32))
        })
        .collect()
}

fn acts(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((s >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0) * 0.5
        })
        .collect()
}

/// `out[m,n] = x[m,k] @ w[n,k]ᵀ` in f64 (reference accumulation).
fn matmul_bt(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f64;
            for p in 0..k {
                acc += (x[r * k + p] as f64) * (w[c * k + p] as f64);
            }
            out[r * n + c] = acc as f32;
        }
    }
    out
}

fn report(tag: &str, want: &[f32], got: &[f32], scale: f32) {
    let worst = want
        .iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let rel = worst / scale.max(1e-9);
    eprintln!(
        "{tag} on {:?}: worst |Δ| = {worst:.3e} (rel {rel:.3e})",
        dev()
    );
    assert!(
        rel < 2e-4,
        "{tag}: relative error {rel:.3e} — layout mismatch, not f32 noise"
    );
}

#[test]
fn dense_matmul_reads_the_packed_layout() {
    let (m, k, n) = (4usize, 256usize, 32usize);
    let w = weights(n * k, 11);
    let x = acts(m * k, 12);

    let q = quantize_rows(&w, n, k, GROUP_SIZE);
    let w_deq = dequantize(&q);
    let want = matmul_bt(&x, &w_deq, m, k, n);
    let ng = k / GROUP_SIZE;

    let scheme = QuantScheme::MlxMxfp4 {
        group_size: GROUP_SIZE as u32,
    };
    let mut hir = HirModule::new("mxfp4_dense");
    let mut g = HirMut::new(&mut hir);
    let x_id = g.input("x", Shape::new(&[m, k], DType::F32));
    let c_id = g.param("codes", Shape::new(&[q.codes.len()], DType::U8));
    // Dense convention: scales are the RAW E8M0 bytes (see mxfp4_pack docs).
    let s_id = g.param("scales", Shape::new(&[n, ng], DType::U8));
    let b_id = g.param("biases", Shape::new(&[n, ng], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_id, c_id, s_id, b_id],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let built = built_from_hir(hir, HashMap::new()).expect("built");
    let mut compiled = compile_built(built, dev()).expect("compile");
    compiled.set_param_typed("codes", &q.codes, DType::U8);
    compiled.set_param_typed("scales", q.scales_e8m0(), DType::U8);
    compiled.set_param_typed("biases", &q.zero_biases_u8(), DType::U8);
    let got = compiled.run(&[("x", x.as_slice())]).remove(0);

    report("dense DequantMatMul{MlxMxfp4}", &want, &got, 4.0);
}

#[test]
fn grouped_matmul_reads_the_packed_layout() {
    let (e, m, k, n) = (3usize, 6usize, 128usize, 16usize);
    let w = weights(e * n * k, 21);
    let x = acts(m * k, 22);
    let eidx: Vec<f32> = (0..m).map(|i| ((i * 2) % e) as f32).collect();

    // The bank is `[E, n, k]`; flattening to `E*n` rows is exactly how both the
    // packer and the kernel index it (expert e, row j → slab row e*n + j).
    let q = quantize_rows(&w, e * n, k, GROUP_SIZE);
    let w_deq = dequantize(&q);
    let ng = k / GROUP_SIZE;

    let mut want = vec![0f32; m * n];
    for r in 0..m {
        let ei = eidx[r] as usize;
        let we = &w_deq[ei * n * k..(ei + 1) * n * k];
        let row = matmul_bt(&x[r * k..(r + 1) * k], we, 1, k, n);
        want[r * n..(r + 1) * n].copy_from_slice(&row);
    }

    let scheme = QuantScheme::MlxMxfp4 {
        group_size: GROUP_SIZE as u32,
    };
    let mut hir = HirModule::new("mxfp4_grouped");
    let mut g = HirMut::new(&mut hir);
    let x_id = g.input("x", Shape::new(&[m, k], DType::F32));
    let idx_id = g.input("eidx", Shape::new(&[m], DType::F32));
    let c_id = g.param("codes", Shape::new(&[q.codes.len()], DType::U8));
    // Grouped convention: scales are the DECODED float, as bf16.
    let s_id = g.param("scales", Shape::new(&[e, n, ng], DType::BF16));
    let b_id = g.param("biases", Shape::new(&[e, n, ng], DType::BF16));
    let y = g.add_node(
        Op::DequantGroupedMatMulMlx { scheme },
        vec![x_id, c_id, s_id, b_id, idx_id],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let built = built_from_hir(hir, HashMap::new()).expect("built");
    let mut compiled = compile_built(built, dev()).expect("compile");
    compiled.set_param_typed("codes", &q.codes, DType::U8);
    compiled.set_param_typed("scales", &q.scales_bf16(), DType::BF16);
    compiled.set_param_typed("biases", &q.zero_biases_bf16(), DType::BF16);
    let got = compiled
        .run(&[("x", x.as_slice()), ("eidx", eidx.as_slice())])
        .remove(0);

    report(
        "grouped DequantGroupedMatMulMlx{MlxMxfp4}",
        &want,
        &got,
        4.0,
    );
}

/// End-to-end accuracy claim: against the ORIGINAL f32 weight (not the
/// dequantized one), MXFP4 must still track the exact matmul closely. This is
/// the number that predicts model quality; the two tests above only prove the
/// plumbing.
#[test]
fn quantized_matmul_tracks_the_f32_matmul() {
    let (m, k, n) = (8usize, 512usize, 64usize);
    let w = weights(n * k, 31);
    let x = acts(m * k, 32);
    let exact = matmul_bt(&x, &w, m, k, n);
    let approx = matmul_bt(
        &x,
        &dequantize(&quantize_rows(&w, n, k, GROUP_SIZE)),
        m,
        k,
        n,
    );

    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, b) in exact.iter().zip(&approx) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    eprintln!("MXFP4 matmul vs f32 matmul: cosine = {cos:.6}");
    assert!(cos > 0.999, "MXFP4 matmul cosine {cos:.6} too low");
}
