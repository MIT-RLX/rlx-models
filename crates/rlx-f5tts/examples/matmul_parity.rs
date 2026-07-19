//! Isolated MatMul parity: CPU vs Metal vs wgpu for F5-ish shapes.
use std::f32::consts::PI;

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
            (x * PI).sin() * 0.1
        })
        .collect()
}

fn run(m: usize, k: usize, n: usize, batch: usize) {
    let a_ne = batch * m * k;
    let b_ne = k * n;
    let a = fill(a_ne, 1);
    let w = fill(b_ne, 2);
    let mut g = Graph::new("mm");
    let a_shape = if batch > 1 {
        Shape::new(&[batch, m, k], DType::F32)
    } else {
        Shape::new(&[m, k], DType::F32)
    };
    let out_shape = if batch > 1 {
        Shape::new(&[batch, m, n], DType::F32)
    } else {
        Shape::new(&[m, n], DType::F32)
    };
    let ai = g.input("a", a_shape);
    let bi = g.param("b", Shape::new(&[k, n], DType::F32));
    let c = g.matmul(ai, bi, out_shape);
    g.set_outputs(vec![c]);
    let mut runner = GraphDevices::new(g);
    runner.set_param("b", &w);
    let cpu = runner.run(Device::Cpu, &[("a", &a)]).expect("cpu");
    for (name, dev) in [("metal", Device::Metal), ("wgpu", Device::Gpu)] {
        if !is_available(dev) {
            println!("{name}: skip");
            continue;
        }
        let out = runner.run(dev, &[("a", &a)]).expect(name);
        let (cos, mae, mx) = stats(&cpu[0], &out[0]);
        println!("{name} [{batch},{m},{k}]@[{k},{n}]: cos={cos:.8} mae={mae:.4e} max={mx:.4e}");
    }
}

fn main() {
    run(580, 1024, 1024, 2);
    run(580, 1024, 2048, 2);
    run(1160, 64, 64, 16);
    run(64, 1024, 1024, 1);
}
