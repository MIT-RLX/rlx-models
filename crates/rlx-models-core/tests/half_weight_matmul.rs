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

//! `f32 activations @ f16/bf16 weight` on every compiled-in backend — the
//! contract `WeightStorage::{F16, Bf16}` stands on: weights stored at 2 B,
//! math in f32, result equal to the f32 matmul over the *rounded* weights.

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn devices() -> Vec<Device> {
    // `mut` only matters with a backend feature on.
    #[allow(unused_mut)]
    let mut d = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    d.push(Device::Metal);
    #[cfg(feature = "mlx")]
    d.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    d.push(Device::Gpu);
    #[cfg(feature = "cuda")]
    d.push(Device::Cuda);
    d
}

/// How the weight reaches the backend: raw half bytes (`set_param_typed`),
/// or f32 values the backend narrows at bind time (`set_param` — what
/// `rlx_flow`'s `load_param_typed` relies on).
#[derive(Clone, Copy, Debug)]
enum Upload {
    Typed,
    F32,
}

#[allow(clippy::too_many_arguments)]
fn run(
    device: Device,
    x: &[f32],
    w: &[f32],
    w_bytes: &[u8],
    dtype: DType,
    upload: Upload,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut g = Graph::new("half_weight_mm");
    let xi = g.input("x", Shape::new(&[1, m, k], DType::F32));
    let wn = g.param("w", Shape::new(&[k, n], dtype));
    let y = g.mm(xi, wn);
    let y = g.relu(y); // keep the matmul off the graph boundary
    g.set_outputs(vec![y]);
    let mut c = Session::new(device).compile(g);
    match upload {
        Upload::Typed => c.set_param_typed("w", w_bytes, dtype),
        Upload::F32 => c.set_param("w", w),
    }
    c.run(&[("x", x)]).into_iter().next().expect("output")
}

#[test]
fn half_weights_match_f32_math_on_every_backend() {
    check(5, 96, 40);
}

/// A real projection's shape: backends route big matmuls to other kernels
/// (MPS on Metal) than the small case.
///
/// wgpu sends an F16-typed matmul of this size to its cooperative f16 GEMM,
/// which rounds the ACTIVATIONS to f16 too (~0.16 off here) — a precision
/// policy, not storage; `RLX_WGPU_MATMUL_F32_ONLY` keeps the math f32, which
/// is the contract this test checks. BF16 weights are exact either way.
#[test]
fn half_weights_match_f32_math_at_model_shapes() {
    rlx_ir::env::set("RLX_WGPU_MATMUL_F32_ONLY", "1");
    check(64, 2048, 6144);
}

/// Decode shapes (M = 1..4): Metal reads BF16 weights in-kernel here
/// (`gemv_bf16w_splitk8[_n64]`, `gemv_bf16w_splitk`, `sgemm_bf16w_small_m`)
/// instead of unpacking them, picking the kernel by N — so cover both
/// column tiles (N/256 below and above 64 threadgroups), an N that is even but
/// not a multiple of 8, and the multi-row kernel.
#[test]
fn half_weights_match_f32_math_at_decode_shapes() {
    for (m, k, n) in [(1, 256, 512), (1, 384, 17_408), (1, 96, 200), (3, 160, 96)] {
        check(m, k, n);
    }
}

fn check(m: usize, k: usize, n: usize) {
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 37.0)
        .collect();
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i * 53 % 97) as f32 - 48.0) / 61.0)
        .collect();
    let mut failures = Vec::new();
    for (dtype, round, bytes) in [
        (
            DType::F16,
            (|v: f32| half::f16::from_f32(v).to_f32()) as fn(f32) -> f32,
            w.iter()
                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                .collect::<Vec<u8>>(),
        ),
        (
            DType::BF16,
            |v: f32| half::bf16::from_f32(v).to_f32(),
            w.iter()
                .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
                .collect(),
        ),
    ] {
        let wr: Vec<f32> = w.iter().map(|&v| round(v)).collect();
        let mut want = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let s: f64 = (0..k)
                    .map(|t| x[i * k + t] as f64 * wr[t * n + j] as f64)
                    .sum();
                want[i * n + j] = (s as f32).max(0.0);
            }
        }
        for dev in devices() {
            for upload in [Upload::Typed, Upload::F32] {
                let got = run(dev, &x, &w, &bytes, dtype, upload, m, k, n);
                let err = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!("{dev:?} {dtype:?} {upload:?}: max |d| {err:.2e}");
                if err.is_nan() || err >= 1e-3 * (k as f32 / 96.0).sqrt() {
                    failures.push(format!("{dev:?} {dtype:?} {upload:?}: {err:.2e}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "half-weight matmul wrong on: {failures:?}"
    );
}

/// Two half-weight matmuls reading one input — SwiGLU's gate/up, QKV. Fusion
/// merges such siblings into one matmul over `Concat(weights)`; for BF16 that
/// Concat was misread on Metal and wgpu, and wgpu's packed-BF16 side buffer
/// overlapped its second weight with the first. Each case is compared with
/// the CPU (exact) result.
#[test]
fn half_weight_siblings_match_on_every_backend() {
    use rlx_ir::NodeId;
    type Build = fn(&mut Graph, NodeId, NodeId, NodeId) -> NodeId;
    let cases: [(&str, Build); 3] = [
        ("x@a * x@b", |g, x, a, b| {
            let p = g.mm(x, a);
            let q = g.mm(x, b);
            g.mul(p, q)
        }),
        ("silu(x@a) * x@b", |g, x, a, b| {
            let p = g.mm(x, a);
            let s = g.silu(p);
            let q = g.mm(x, b);
            g.mul(s, q)
        }),
        ("x@a + x@b", |g, x, a, b| {
            let p = g.mm(x, a);
            let q = g.mm(x, b);
            g.add(p, q)
        }),
    ];
    let (m, k, n) = (6usize, 64usize, 48usize);
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 37.0)
        .collect();
    let w1: Vec<f32> = (0..k * n)
        .map(|i| ((i * 53 % 97) as f32 - 48.0) / 61.0)
        .collect();
    let w2: Vec<f32> = (0..k * n)
        .map(|i| ((i * 31 % 89) as f32 - 44.0) / 57.0)
        .collect();
    let mut failures = Vec::new();
    for dtype in [DType::BF16, DType::F16] {
        let enc = |v: &[f32]| -> Vec<u8> {
            match dtype {
                DType::BF16 => v
                    .iter()
                    .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
                    .collect(),
                _ => v
                    .iter()
                    .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                    .collect(),
            }
        };
        for (name, build) in cases {
            let mut outs = Vec::new();
            for dev in devices() {
                let mut g = Graph::new("half_siblings");
                let xi = g.input("x", Shape::new(&[1, m, k], DType::F32));
                let a = g.param("w1", Shape::new(&[k, n], dtype));
                let b = g.param("w2", Shape::new(&[k, n], dtype));
                let y = build(&mut g, xi, a, b);
                g.set_outputs(vec![y]);
                let mut c = Session::new(dev).compile(g);
                c.set_param_typed("w1", &enc(&w1), dtype);
                c.set_param_typed("w2", &enc(&w2), dtype);
                outs.push((dev, c.run(&[("x", &x)]).into_iter().next().unwrap()));
            }
            let reference = outs[0].1.clone();
            for (dev, o) in &outs {
                let err = o
                    .iter()
                    .zip(&reference)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!("{dtype:?} {name:18} {dev:?}: {err:.2e}");
                if err.is_nan() || err >= 1e-3 {
                    failures.push(format!("{dtype:?} {name} {dev:?}: {err:.2e}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "sibling half-weight matmuls wrong on {failures:?}"
    );
}
