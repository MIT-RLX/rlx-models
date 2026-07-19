//! Faithful-shape ScatterNd probe matching F5_Transformer's rope scatter:
//! data=[2,141,64] F32, idx=[2,141,32,3] I64, updates=[2,141,32] F32.
use rlx_runtime::op::ScatterNdReduction;
use rlx_runtime::{DType, Device, Graph, Op, Session, Shape};

fn to_f32(b: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::I64 => b
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        _ => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    }
}
fn stats(a: &[f32], b: &[f32]) -> (f64, f64) {
    let n = a.len().min(b.len());
    let (mut d, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f64);
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

fn main() {
    let b = 2usize;
    let s = 141usize;
    let d64 = 64usize;
    let k = 32usize;
    let data: Vec<f32> = vec![0.0; b * s * d64]; // "empty" buffer, matches qk_rotated_empty (zeros)
    let mut idx: Vec<i64> = Vec::with_capacity(b * s * k * 3);
    // Mimic rotate-half scatter: for each (batch,pos), write k=32 values at
    // column offsets [0..32) or [32..64) alternately across chained calls.
    for bi in 0..b {
        for si in 0..s {
            for ki in 0..k {
                idx.push(bi as i64);
                idx.push(si as i64);
                idx.push(ki as i64);
            }
        }
    }
    let upd: Vec<f32> = (0..b * s * k)
        .map(|i| ((i % 97) as f32) * 0.01 - 0.5)
        .collect();

    let mut g = Graph::new("scatter_probe2");
    let din = g.input("data", Shape::new(&[b, s, d64], DType::F32));
    let iin = g.input("idx", Shape::new(&[b, s, k, 3], DType::I64));
    let uin = g.input("upd", Shape::new(&[b, s, k], DType::F32));
    let y = g.add_node(
        Op::ScatterNd {
            reduction: ScatterNdReduction::None,
        },
        vec![din, iin, uin],
        Shape::new(&[b, s, d64], DType::F32),
    );
    g.set_outputs(vec![y]);

    let data_b: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let idx_b: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
    let upd_b: Vec<u8> = upd.iter().flat_map(|v| v.to_le_bytes()).collect();
    let inputs: [(&str, &[u8], DType); 3] = [
        ("data", &data_b, DType::F32),
        ("idx", &idx_b, DType::I64),
        ("upd", &upd_b, DType::F32),
    ];
    let cpu_out = Session::new(Device::Cpu)
        .compile(g.clone())
        .run_typed(&inputs);
    let cpu_v = to_f32(&cpu_out[0].0, cpu_out[0].1);
    let metal_out = Session::new(Device::Metal)
        .compile(g.clone())
        .run_typed(&inputs);
    let metal_v = to_f32(&metal_out[0].0, metal_out[0].1);
    let (cos, mx) = stats(&cpu_v, &metal_v);
    println!(
        "single ScatterNd metal: cos={cos:.8} maxdiff={mx:.4e} n={}",
        cpu_v.len()
    );
    let gpu_out = Session::new(Device::Gpu).compile(g).run_typed(&inputs);
    let gpu_v = to_f32(&gpu_out[0].0, gpu_out[0].1);
    let (cosg, mxg) = stats(&cpu_v, &gpu_v);
    println!(
        "single ScatterNd wgpu:  cos={cosg:.8} maxdiff={mxg:.4e} n={}",
        gpu_v.len()
    );

    // Now chain 88 ScatterNd calls (each takes the previous output as `data`),
    // matching F5_Transformer's rotate-half chain pattern.
    let mut g2 = Graph::new("scatter_chain");
    let din2 = g2.input("data", Shape::new(&[b, s, d64], DType::F32));
    let mut cur = din2;
    let mut idx_ins = vec![];
    let mut upd_ins = vec![];
    for i in 0..88 {
        let iname: &'static str = Box::leak(format!("idx{i}").into_boxed_str());
        let uname: &'static str = Box::leak(format!("upd{i}").into_boxed_str());
        let ii = g2.input(iname, Shape::new(&[b, s, k, 3], DType::I64));
        let uu = g2.input(uname, Shape::new(&[b, s, k], DType::F32));
        idx_ins.push((iname, ii));
        upd_ins.push((uname, uu));
        cur = g2.add_node(
            Op::ScatterNd {
                reduction: ScatterNdReduction::None,
            },
            vec![cur, ii, uu],
            Shape::new(&[b, s, d64], DType::F32),
        );
    }
    g2.set_outputs(vec![cur]);
    let mut inputs2: Vec<(&str, Vec<u8>, DType)> = vec![("data", data_b.clone(), DType::F32)];
    for i in 0..88 {
        let upd_i: Vec<f32> = (0..b * s * k)
            .map(|j| (((i * 97 + j) % 97) as f32) * 0.01 - 0.5)
            .collect();
        let upd_ib: Vec<u8> = upd_i.iter().flat_map(|v| v.to_le_bytes()).collect();
        inputs2.push((idx_ins[i].0, idx_b.clone(), DType::I64));
        inputs2.push((upd_ins[i].0, upd_ib, DType::F32));
    }
    let inputs2_ref: Vec<(&str, &[u8], DType)> = inputs2
        .iter()
        .map(|(n, b, d)| (*n, b.as_slice(), *d))
        .collect();
    let cpu2 = Session::new(Device::Cpu)
        .compile(g2.clone())
        .run_typed(&inputs2_ref);
    let cpu2_v = to_f32(&cpu2[0].0, cpu2[0].1);
    let metal2 = Session::new(Device::Metal)
        .compile(g2.clone())
        .run_typed(&inputs2_ref);
    let metal2_v = to_f32(&metal2[0].0, metal2[0].1);
    let (cos2, mx2) = stats(&cpu2_v, &metal2_v);
    println!("chained 88x ScatterNd metal: cos={cos2:.8} maxdiff={mx2:.4e}");
    let gpu2 = Session::new(Device::Gpu)
        .compile(g2)
        .run_typed(&inputs2_ref);
    let gpu2_v = to_f32(&gpu2[0].0, gpu2[0].1);
    let (cos2g, mx2g) = stats(&cpu2_v, &gpu2_v);
    println!("chained 88x ScatterNd wgpu:  cos={cos2g:.8} maxdiff={mx2g:.4e}");
}
