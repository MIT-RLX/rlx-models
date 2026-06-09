// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Q4_0 / Q8_0 fused matmul: CPU reference and optional Metal vs CPU parity.
//!
//! ```text
//! cargo test -p rlx-models --test gguf_legacy_quant_matmul --release
//!
//! GGUF_LEGACY_METAL_PARITY=1 cargo test -p rlx-models --test gguf_legacy_quant_matmul metal --release --features metal
//! ```

mod compile_support;

use rlx_cpu::gguf_matmul::gguf_matmul_bt;
use rlx_flow::CompileProfile;
use rlx_gguf::QK4_0;
use rlx_ir::hir::FusionPolicy;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, HirModule, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / na.sqrt() / nb.sqrt()) as f32
}

fn synthetic_q4_0_packed(k: usize, n: usize) -> Vec<u8> {
    let d = half::f16::from_f32(0.5);
    let blocks = (k * n) / QK4_0;
    let mut packed = Vec::with_capacity(blocks * 18);
    for b in 0..blocks {
        packed.extend_from_slice(&d.to_le_bytes());
        for j in 0..QK4_0 / 2 {
            let nib = ((b + j) as u8 & 0x0F) | ((((b + j + 1) as u8) & 0x0F) << 4);
            packed.push(nib);
        }
    }
    packed
}

fn reference_matmul(x: &[f32], packed: &[u8], m: usize, k: usize, n: usize) -> Vec<f32> {
    let w = rlx_gguf::dequant_q4_0(packed, k * n).expect("dequant_q4_0");
    let mut out = vec![0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[row * k + i] * w[col * k + i];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn compile_q4_0_graph(
    device: Device,
    packed: &[u8],
    m: usize,
    k: usize,
    n: usize,
) -> rlx_runtime::CompiledGraph {
    let mut hir = HirModule::new("gguf_q4_0").with_fusion_policy(FusionPolicy::Direct);
    let x = hir.input("x", Shape::new(&[m, k], DType::F32));
    let w = hir.param("w_q", Shape::new(&[packed.len()], DType::U8));
    let y = hir.dequant_matmul(
        x,
        w,
        None,
        None,
        QuantScheme::GgufQ4_0,
        Shape::new(&[m, n], DType::F32),
    );
    hir.outputs = vec![y];
    let graph = hir.lower_to_mir().expect("lower").into_graph();
    let mut compiled = compile_support::compile_with_profile(
        device,
        graph,
        HashMap::new(),
        &CompileProfile::encoder(),
    );
    compiled.set_param_typed("w_q", packed, DType::U8);
    compiled
}

fn run_compiled(
    device: Device,
    packed: &[u8],
    x: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut compiled = compile_q4_0_graph(device, packed, m, k, n);
    compiled.run(&[("x", x)])[0].clone()
}

#[test]
fn fused_q4_0_matches_reference() {
    let k = 32;
    let n = 4;
    let m = 2;
    let packed = synthetic_q4_0_packed(k, n);
    let x: Vec<f32> = (0..m * k).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    let mut fused = vec![0f32; m * n];
    gguf_matmul_bt(&x, &packed, &mut fused, m, k, n, QuantScheme::GgufQ4_0);
    let expected = reference_matmul(&x, &packed, m, k, n);
    for i in 0..fused.len() {
        assert!(
            (fused[i] - expected[i]).abs() < 1e-4,
            "i={i}: fused={} ref={}",
            fused[i],
            expected[i]
        );
    }
}

#[test]
fn compiled_cpu_q4_0_matches_fused() {
    let k = 32;
    let n = 2;
    let m = 1;
    let packed = synthetic_q4_0_packed(k, n);
    let x: Vec<f32> = (0..m * k).map(|i| 0.02 * i as f32).collect();
    let mut fused = vec![0f32; m * n];
    gguf_matmul_bt(&x, &packed, &mut fused, m, k, n, QuantScheme::GgufQ4_0);
    let compiled = run_compiled(Device::Cpu, &packed, &x, m, k, n);
    let c = cosine(&fused, &compiled);
    eprintln!("q4_0 compiled vs fused cosine={c:.6}");
    assert!(c > 0.9999, "compiled cpu vs fused cosine {c}");
}

#[cfg(feature = "metal")]
#[test]
fn metal_q4_0_matches_cpu_compiled() {
    if std::env::var("GGUF_LEGACY_METAL_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip metal_q4_0: set GGUF_LEGACY_METAL_PARITY=1");
        return;
    }
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip metal_q4_0: Metal not available");
        return;
    }
    let k = 32;
    let n = 2;
    let m = 1;
    let packed = synthetic_q4_0_packed(k, n);
    let x: Vec<f32> = (0..m * k).map(|i| 0.02 * i as f32).collect();
    let cpu = run_compiled(Device::Cpu, &packed, &x, m, k, n);
    let metal = run_compiled(Device::Metal, &packed, &x, m, k, n);
    let c = cosine(&cpu, &metal);
    eprintln!("q4_0 metal vs cpu compiled cosine={c:.6}");
    assert!(c > 0.999, "metal vs cpu cosine {c}");
}
