// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Compare the speech LM's two heaviest op shapes on a device against an exact
//! host reference (Q2_0) or CPU (f32):
//! `Q2_0` `DequantMatMul` at the projection shapes and the tied LM head
//! (`[1, 2048] × [2048, 158241]`, one 1.3 GB weight).
//!
//! ```text
//! cargo run -p rlx-kittentts2 --release --features gpu --example backend_probe -- gpu
//! ```

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn run(device: Device, g: Graph, params: &[(&str, &[u8], DType)], x: &[f32]) -> Vec<f32> {
    let mut c = Session::new(device).compile(g);
    for (n, b, d) in params {
        c.set_param_typed(n, b, *d);
    }
    let out = c.run(&[("x", x)]).remove(0);
    if std::env::var_os("KT2_PROBE_TIME").is_some() && device != Device::Cpu {
        let t = std::time::Instant::now();
        for _ in 0..20 {
            c.run(&[("x", x)]);
        }
        eprintln!(
            "    {:.2} ms/run on {device:?}",
            t.elapsed().as_secs_f64() * 1e3 / 20.0
        );
    }
    out
}

fn report(what: &str, a: &[f32], b: &[f32]) {
    let d = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let s = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    println!(
        "{what:>32}: max|Δ| {d:.3e} of {s:.3e}{}",
        if d > 1e-2 * s { "   ← MISMATCH" } else { "" }
    );
}

fn main() {
    let dev = rlx_runtime::parse_device(&std::env::args().nth(1).unwrap_or("gpu".into())).unwrap();
    for (m, k, n) in [
        (1, 2048, 6144),
        (1, 2048, 49152),
        (297, 2048, 6144),
        (1, 6144, 2048),
        (297, 6144, 2048),
    ] {
        let w: Vec<f32> = (0..n * k)
            .map(|i| [-1.0f32, 0.0, 1.0][(i * 7 + i / 128) % 3] * 0.01 * (1 + (i / 128) % 5) as f32)
            .collect();
        let packed = rlx_gguf::q2_dequant::quantize_q2_0(&w).unwrap();
        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.37).sin()).collect();
        let build = || {
            let mut g = Graph::new("q2");
            let xi = g.input("x", Shape::new(&[m, k], DType::F32));
            let wi = g.param("w", Shape::new(&[packed.len()], DType::U8));
            let y = g.add_node(
                Op::DequantMatMul {
                    scheme: QuantScheme::GgufQ2_0,
                },
                vec![xi, wi],
                Shape::new(&[m, n], DType::F32),
            );
            g.set_outputs(vec![y]);
            g
        };
        let p = [("w", packed.as_slice(), DType::U8)];
        // Exact reference: dequantize, f64 dot. (rlx-cpu's m = 1 Q2_0 GEMV
        // quantizes activations to int8, so CPU is not the yardstick here.)
        let wd = rlx_gguf::q2_dequant::dequant_q2_0(&packed, n * k).unwrap();
        let exact: Vec<f32> = (0..m * n)
            .map(|i| {
                let (r, c) = (i / n, i % n);
                (0..k)
                    .map(|j| x[r * k + j] as f64 * wd[c * k + j] as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        report(
            &format!("Q2_0 m={m} k={k} n={n}"),
            &run(dev, build(), &p, &x),
            &exact,
        );
    }
    // Param round trip: y = w + 0 over the whole tensor; where does it break?
    for mib in [512usize, 1024, 1100, 1300] {
        let len = mib << 18; // f32 elements
        let w: Vec<f32> = (0..len).map(|i| (i % 1_000_003) as f32).collect();
        let wb: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let build = || {
            let mut g = Graph::new("copy");
            let xi = g.input("x", Shape::new(&[len], DType::F32));
            let wi = g.param("w", Shape::new(&[len], DType::F32));
            let y = g.add_node(
                Op::Binary(rlx_ir::op::BinaryOp::Add),
                vec![xi, wi],
                Shape::new(&[len], DType::F32),
            );
            g.set_outputs(vec![y]);
            g
        };
        let zeros = vec![0f32; len];
        let got = run(dev, build(), &[("w", wb.as_slice(), DType::F32)], &zeros);
        let first_bad = got.iter().zip(&w).position(|(a, b)| a != b);
        println!(
            "{:>32}: first mismatch {}",
            format!("param round trip {mib} MiB"),
            first_bad.map_or("none".into(), |i| format!(
                "at element {i} (byte {})",
                4 * i
            ))
        );
    }
    for n in [32_768usize, 32_769, 32_770, 158_240, 158_241] {
        let k = 2048;
        let w: Vec<f32> = (0..k * n)
            .map(|i| ((i % 9973) as f32 * 1e-4) - 0.5)
            .collect();
        let wb: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.11).cos()).collect();
        let build = || {
            let mut g = Graph::new("head");
            let xi = g.input("x", Shape::new(&[1, k], DType::F32));
            let wi = g.param("w", Shape::new(&[k, n], DType::F32));
            let y = g.add_node(Op::MatMul, vec![xi, wi], Shape::new(&[1, n], DType::F32));
            g.set_outputs(vec![y]);
            g
        };
        let p = [("w", wb.as_slice(), DType::F32)];
        report(
            &format!("f32 head n={n}"),
            &run(dev, build(), &p, &x),
            &run(Device::Cpu, build(), &p, &x),
        );
    }
}
