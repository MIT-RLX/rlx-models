use rlx_runtime::op::CmpOp;
use rlx_runtime::{DType, Device, Graph, Op, Session, Shape};
fn to_f32(b: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::Bool => {
            // Metal widens bool to f32; CPU may be 1-byte or f32
            if b.len() == 580 {
                b.iter().map(|&x| x as f32).collect()
            } else {
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            }
        }
        _ => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    }
}
fn main() {
    let n = 580usize;
    let lhs: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let thr = 290f32;
    for (dev, name) in [(Device::Cpu, "cpu"), (Device::Metal, "metal")] {
        let mut g = Graph::new("c");
        let a = g.input("a", Shape::new(&[1, n], DType::F32));
        let b = g.param("t", Shape::new(&[], DType::F32)); // scalar
        let y = g.add_node(
            Op::Compare(CmpOp::Lt),
            vec![a, b],
            Shape::new(&[1, n], DType::Bool),
        );
        g.set_outputs(vec![y]);
        let mut s = Session::new(dev).compile(g);
        s.set_param("t", &[thr]);
        let ab: Vec<u8> = lhs.iter().flat_map(|v| v.to_le_bytes()).collect();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let out = s.run_typed(&[("a", ab.as_slice(), DType::F32)]);
            to_f32(&out[0].0, out[0].1)
        })) {
            Ok(v) => {
                let trues = v.iter().filter(|&&x| x != 0.0).count();
                let at289 = v.get(289).copied();
                let at290 = v.get(290).copied();
                println!(
                    "{name}: n={} trues={trues}/{} v[289]={at289:?} v[290]={at290:?}",
                    v.len(),
                    n
                );
            }
            Err(_) => println!("{name}: PANIC"),
        }
    }
}
