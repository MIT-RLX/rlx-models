use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, GraphDevices, is_available};
use std::f32::consts::PI;

fn stats(a: &[f32], b: &[f32]) -> (f64, f64) {
    let n = a.len().min(b.len());
    let mut d = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut mae = 0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        d += x * y;
        na += x * x;
        nb += y * y;
        mae += (x - y).abs();
    }
    (d / (na.sqrt() * nb.sqrt() + 1e-12), mae / n as f64)
}
fn fill(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
            (x * PI).sin() * 0.1
        })
        .collect()
}
fn main() {
    // Gather along axis 0: table [612,1024], indices [580]
    let rows = 612usize;
    let dim = 1024usize;
    let nidx = 580usize;
    let table = fill(rows * dim, 1);
    let idx: Vec<f32> = (0..nidx).map(|i| (i % rows) as f32).collect();
    let mut g = Graph::new("gath");
    let t = g.input("t", Shape::new(&[rows, dim], DType::F32));
    let i = g.input("i", Shape::new(&[nidx], DType::F32));
    let y = g.gather(t, i, 0, Shape::new(&[nidx, dim], DType::F32));
    g.set_outputs(vec![y]);
    let mut runner = GraphDevices::new(g);
    let feeds = [("t", table.as_slice()), ("i", idx.as_slice())];
    let cpu = runner.run(Device::Cpu, &feeds).unwrap();
    for (name, dev) in [("metal", Device::Metal), ("wgpu", Device::Gpu)] {
        if !is_available(dev) {
            continue;
        }
        let out = runner.run(dev, &feeds).unwrap();
        let (cos, mae) = stats(&cpu[0], &out[0]);
        println!("gather {name}: cos={cos:.8} mae={mae:.4e}");
    }

    // Transpose [2,16,580,64] -> [2,16,64,580]
    let n = 2 * 16 * 580 * 64;
    let x = fill(n, 2);
    let mut g = Graph::new("tr");
    let xi = g.input("x", Shape::new(&[2, 16, 580, 64], DType::F32));
    let y = g.transpose_(xi, vec![0, 1, 3, 2]);
    g.set_outputs(vec![y]);
    let mut runner = GraphDevices::new(g);
    let cpu = runner.run(Device::Cpu, &[("x", &x)]).unwrap();
    for (name, dev) in [("metal", Device::Metal), ("wgpu", Device::Gpu)] {
        if !is_available(dev) {
            continue;
        }
        let out = runner.run(dev, &[("x", &x)]).unwrap();
        let (cos, mae) = stats(&cpu[0], &out[0]);
        println!("transpose {name}: cos={cos:.8} mae={mae:.4e}");
    }
}
