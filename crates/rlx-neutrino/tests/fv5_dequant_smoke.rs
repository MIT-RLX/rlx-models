// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Device-parametrized smoke test for Neutrino-8B's FV5 / FV5B ternary
//! `Op::DequantMatMul` — the two custom ggml types now have on-device dequant
//! kernels. Packs a tiny synthetic weight (FV5 has no float quantizer; packs
//! are produced offline), runs `x @ dequant(w)^T` on `RLX_TEST_DEVICE`, and
//! checks the result is finite and matches the CPU reference.
//!
//! Set `RLX_TEST_DEVICE=metal|mlx|gpu|coreml|cuda|vulkan` (default CPU) and
//! build the matching cargo feature to exercise a backend.

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn dev() -> Device {
    match std::env::var("RLX_TEST_DEVICE").ok().as_deref() {
        Some("metal") | Some("mtl") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("gpu") | Some("wgpu") => Device::Gpu,
        Some("coreml") | Some("ane") => Device::Ane,
        Some("cuda") => Device::Cuda,
        Some("vulkan") | Some("vk") => Device::Vulkan,
        _ => Device::Cpu,
    }
}

/// Pack one FV5 block (104 bytes) from 256 five-value codes in {-2,-1,0,1,2}.
fn pack_fv5_block(codes: &[i8], s_lo: f32, s_hi: f32) -> Vec<u8> {
    let mut b = vec![0u8; 104];
    b[0..4].copy_from_slice(&s_lo.to_le_bytes());
    b[4..8].copy_from_slice(&s_hi.to_le_bytes());
    for (j, &c) in codes.iter().enumerate() {
        let (byte, bit) = (j / 8, 1u8 << (j % 8));
        let (p, ng, hi) = match c {
            1 => (true, false, false),
            2 => (true, false, true),
            -1 => (false, true, false),
            -2 => (false, true, true),
            _ => (false, false, false),
        };
        if p {
            b[8 + byte] |= bit;
        }
        if ng {
            b[40 + byte] |= bit;
        }
        if hi {
            b[72 + byte] |= bit;
        }
    }
    b
}

/// Pack one FV5B block (260 bytes): one f32 scale + 256 int8 codes.
fn pack_fv5b_block(qs: &[i8], s: f32) -> Vec<u8> {
    let mut b = vec![0u8; 260];
    b[0..4].copy_from_slice(&s.to_le_bytes());
    for (i, &q) in qs.iter().enumerate() {
        b[4 + i] = q as u8;
    }
    b
}

/// Run `x[m,k] @ dequant(w)[n,k]^T` on `device` and on CPU; return
/// (device_output, cpu_output). `None` if the device isn't available.
fn run(scheme: QuantScheme, packed: &[u8], m: usize, k: usize, n: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    let device = dev();
    if device != Device::Cpu && !rlx_runtime::is_available(device) {
        eprintln!("skip: {device:?} unavailable");
        return None;
    }
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let mut g = Graph::new("neutrino_fv5_dq_smoke");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let go = |d: Device| -> Vec<f32> {
        let mut c = Session::new(d).compile(g.clone());
        c.set_param_typed("w", packed, DType::U8);
        c.run(&[("x", x.as_slice())]).remove(0)
    };
    Some((go(device), go(Device::Cpu)))
}

fn check(name: &str, out: Option<(Vec<f32>, Vec<f32>)>, elems: usize) {
    let Some((dev_out, cpu_out)) = out else { return };
    assert_eq!(dev_out.len(), elems, "{name}: output length");
    assert!(
        dev_out.iter().all(|v| v.is_finite()),
        "{name}: output must be finite on {:?}",
        dev()
    );
    let max_abs = cpu_out
        .iter()
        .zip(&dev_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("neutrino {name} on {:?}: max_abs={max_abs:.6e}", dev());
    assert!(max_abs <= 1e-3, "{name}: {:?} diverges from CPU: {max_abs}", dev());
}

#[test]
fn fv5_and_fv5b_dequant_matmul_smoke() {
    let (m, k, n) = (4usize, 256usize, 8usize); // k a multiple of 256 → 1 block/row

    // FV5 (ggml 43) — the transformer linears.
    let mut fv5 = Vec::new();
    for row in 0..n {
        let codes: [i8; 256] = std::array::from_fn(|j| match (j + row) % 5 {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => -1,
            _ => -2,
        });
        fv5.extend_from_slice(&pack_fv5_block(&codes, 0.05, 0.2));
    }
    check("FV5", run(QuantScheme::GgufFV5, &fv5, m, k, n), m * n);

    // FV5B (ggml 44) — the int8 embed / lm_head rows.
    let mut fv5b = Vec::new();
    for row in 0..n {
        let qs: [i8; 256] = std::array::from_fn(|i| ((i as i32 * 7 + row as i32) % 251 - 125) as i8);
        fv5b.extend_from_slice(&pack_fv5b_block(&qs, 0.03));
    }
    check("FV5B", run(QuantScheme::GgufFV5B, &fv5b, m, k, n), m * n);
}
