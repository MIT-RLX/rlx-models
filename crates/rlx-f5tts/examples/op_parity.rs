//! CPU vs Metal vs wgpu parity for F5-hot ops (MatMul already bit-exact).
use std::f32::consts::PI;

use rlx_ir::infer::GraphExt;
use rlx_ir::op::AdaNormKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, GraphDevices, is_available};

fn stats(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let n = a.len().min(b.len());
    let mut d = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut mae = 0f64;
    let mut mx = 0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        d += x * y;
        na += x * x;
        nb += y * y;
        let e = (x - y).abs();
        mae += e;
        mx = mx.max(e);
    }
    (d / (na.sqrt() * nb.sqrt() + 1e-12), mae / n as f64, mx)
}

fn fill(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
            (x * PI).sin() * 0.5
        })
        .collect()
}

fn report(
    label: &str,
    cpu: &[f32],
    name: &str,
    dev: Device,
    runner: &mut GraphDevices,
    feeds: &[(&str, &[f32])],
) {
    if !is_available(dev) {
        println!("{label} {name}: skip");
        return;
    }
    let out = runner.run(dev, feeds).expect(name);
    let (cos, mae, mx) = stats(cpu, &out[0]);
    println!("{label} {name}: cos={cos:.8} mae={mae:.4e} max={mx:.4e}");
}

fn main() {
    // Softmax [32, 580] — F5 attention scores
    {
        let rows = 32usize;
        let cols = 580usize;
        let x = fill(rows * cols, 3);
        let mut g = Graph::new("sm");
        let xi = g.input("x", Shape::new(&[rows, cols], DType::F32));
        let y = g.sm(xi, -1);
        g.set_outputs(vec![y]);
        let mut runner = GraphDevices::new(g);
        let cpu = runner.run(Device::Cpu, &[("x", &x)]).unwrap();
        println!("=== Softmax [{rows},{cols}] ===");
        report(
            "softmax",
            &cpu[0],
            "metal",
            Device::Metal,
            &mut runner,
            &[("x", &x)],
        );
        report(
            "softmax",
            &cpu[0],
            "wgpu",
            Device::Gpu,
            &mut runner,
            &[("x", &x)],
        );
    }

    // AdaLayerNorm [2,580,1024]
    {
        let (b, s, d) = (2usize, 580usize, 1024usize);
        let x = fill(b * s * d, 5);
        let scale = fill(b * d, 6);
        let shift = fill(b * d, 7);
        let mut g = Graph::new("adaln");
        let xi = g.input("x", Shape::new(&[b, s, d], DType::F32));
        let sc = g.input("scale", Shape::new(&[b, 1, d], DType::F32));
        let sh = g.input("shift", Shape::new(&[b, 1, d], DType::F32));
        let y = g.ada_layer_norm(xi, sc, sh, AdaNormKind::LayerNorm, 1e-5);
        g.set_outputs(vec![y]);
        let mut runner = GraphDevices::new(g);
        let feeds = [
            ("x", x.as_slice()),
            ("scale", scale.as_slice()),
            ("shift", shift.as_slice()),
        ];
        let cpu = runner.run(Device::Cpu, &feeds).unwrap();
        println!("=== AdaLN [{b},{s},{d}] ===");
        report(
            "adaln",
            &cpu[0],
            "metal",
            Device::Metal,
            &mut runner,
            &feeds,
        );
        report("adaln", &cpu[0], "wgpu", Device::Gpu, &mut runner, &feeds);
    }

    // GatedResidual
    {
        let (b, s, d) = (2usize, 580usize, 1024usize);
        let x = fill(b * s * d, 8);
        let yv = fill(b * s * d, 9);
        let gate = fill(b * d, 10);
        let mut g = Graph::new("gr");
        let xi = g.input("x", Shape::new(&[b, s, d], DType::F32));
        let yi = g.input("y", Shape::new(&[b, s, d], DType::F32));
        let gi = g.input("g", Shape::new(&[b, 1, d], DType::F32));
        let out = g.gated_residual(xi, yi, gi);
        g.set_outputs(vec![out]);
        let mut runner = GraphDevices::new(g);
        let feeds = [
            ("x", x.as_slice()),
            ("y", yv.as_slice()),
            ("g", gate.as_slice()),
        ];
        let cpu = runner.run(Device::Cpu, &feeds).unwrap();
        println!("=== GatedResidual [{b},{s},{d}] ===");
        report("gr", &cpu[0], "metal", Device::Metal, &mut runner, &feeds);
        report("gr", &cpu[0], "wgpu", Device::Gpu, &mut runner, &feeds);
    }

    // Softmax large inner (stress Kahan path): [16, 2048]
    {
        let rows = 16usize;
        let cols = 2048usize;
        let x = fill(rows * cols, 12);
        let mut g = Graph::new("sm2");
        let xi = g.input("x", Shape::new(&[rows, cols], DType::F32));
        let y = g.sm(xi, -1);
        g.set_outputs(vec![y]);
        let mut runner = GraphDevices::new(g);
        let cpu = runner.run(Device::Cpu, &[("x", &x)]).unwrap();
        println!("=== Softmax [{rows},{cols}] ===");
        report(
            "softmax2",
            &cpu[0],
            "metal",
            Device::Metal,
            &mut runner,
            &[("x", &x)],
        );
        report(
            "softmax2",
            &cpu[0],
            "wgpu",
            Device::Gpu,
            &mut runner,
            &[("x", &x)],
        );
    }
}
