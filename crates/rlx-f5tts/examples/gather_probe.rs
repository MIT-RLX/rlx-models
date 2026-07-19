use rlx_runtime::{DType, Device, Graph, Op, Session, Shape};
fn to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn stats(a: &[f32], b: &[f32]) -> (f64, f64) {
    let n = a.len().min(b.len());
    let mut d = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut mx = 0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        d += x * y;
        na += x * x;
        nb += y * y;
        mx = mx.max((x - y).abs());
    }
    (d / (na.sqrt() * nb.sqrt() + 1e-12), mx)
}
fn make(dev: Device, table: &[f32], idx: &[i64], vocab: usize, dim: usize, n: usize) -> Vec<f32> {
    let mut g = Graph::new("g");
    let tab = g.param("t", Shape::new(&[vocab, dim], DType::F32));
    let ix = g.input("i", Shape::new(&[1, n], DType::I64));
    let y = g.add_node(
        Op::Gather { axis: 0 },
        vec![tab, ix],
        Shape::new(&[1, n, dim], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut s = Session::new(dev).compile(g);
    s.set_param("t", table);
    let ib: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
    to_f32(&s.run_typed(&[("i", ib.as_slice(), DType::I64)])[0].0)
}
fn main() {
    let vocab = 100usize;
    let dim = 8;
    let n = 16;
    let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
    let idx: Vec<i64> = (0..n).map(|i| ((i * 3) % vocab) as i64).collect();
    let cpu = make(Device::Cpu, &table, &idx, vocab, dim, n);
    println!("cpu first row {:?}", &cpu[..dim]);
    for (dev, name) in [
        (Device::Metal, "metal"),
        (Device::Mlx, "mlx"),
        (Device::Gpu, "wgpu"),
    ] {
        match std::panic::catch_unwind(|| make(dev, &table, &idx, vocab, dim, n)) {
            Ok(g) => {
                let (c, m) = stats(&cpu, &g);
                println!("{name}: cos={c:.6} max={m:.3e} first={:?}", &g[..dim]);
            }
            Err(_) => println!("{name}: PANIC"),
        }
    }
}
