// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! YuNet's ops on the CPU and on Metal: the largest difference, to find a backend that
//! disagrees. Each op alone (small graphs), `RESIZE_SWEEP=1` the 2× upsample at several shapes,
//! `YUNET_WEIGHTS=W.safetensors` the whole network layer by layer ([`TAPS`]) and output by
//! output (how the MPSGraph lowering's shifted nearest upsample was found).
//!
//!   cargo run --release -p rlx-yunet --features metal --example device_ops
//!
//! [`TAPS`]: rlx_yunet::model::TAPS

use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use rlx_runtime::{Device, Session};

fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

type Build = fn(&mut Graph, NodeId) -> (NodeId, Vec<(&'static str, Vec<usize>)>);

fn check(name: &str, dims: &[usize], build: Build) {
    let mut outs = Vec::new();
    for dev in [Device::Cpu, Device::Metal] {
        let mut g = Graph::new(name);
        let x = g.input("x", Shape::new(dims, DType::F32));
        let (y, params) = build(&mut g, x);
        g.set_outputs(vec![y]);
        let mut c = Session::new(dev).compile(g);
        for (i, (p, shape)) in params.iter().enumerate() {
            c.set_param(p, &noise(shape.iter().product(), 7 + i as u32));
        }
        let input = noise(dims.iter().product(), 3);
        outs.push(c.run(&[("x", &input)]).remove(0));
    }
    let worst = outs[0]
        .iter()
        .zip(&outs[1])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "{name:28} {:>8} values  max |cpu - metal| = {worst:.2e}{}",
        outs[0].len(),
        if worst > 1e-3 { "   <-- differs" } else { "" }
    );
}

/// `ResizeNearest2x` alone at several shapes.
fn resize_sweep() {
    for dims in [
        [1, 8, 16, 24],
        [1, 64, 6, 10],
        [1, 64, 12, 20],
        [1, 1, 6, 10],
        [1, 2, 3, 5],
        [1, 8, 6, 10],
        [1, 64, 16, 24],
        [1, 3, 4, 4],
        [1, 3, 6, 6],
    ] {
        let mut outs = Vec::new();
        for dev in [Device::Cpu, Device::Metal] {
            let mut g = Graph::new("resize");
            let x = g.input("x", Shape::new(&dims, DType::F32));
            let y = g.add_node(
                rlx_ir::op::Op::ResizeNearest2x,
                vec![x],
                Shape::new(&[dims[0], dims[1], 2 * dims[2], 2 * dims[3]], DType::F32),
            );
            g.set_outputs(vec![y]);
            let input = noise(dims.iter().product(), 3);
            outs.push(Session::new(dev).compile(g).run(&[("x", &input)]).remove(0));
        }
        let bad = outs[0]
            .iter()
            .zip(&outs[1])
            .filter(|(a, b)| (*a - *b).abs() > 1e-6)
            .count();
        println!(
            "resize2x {dims:?}: {bad} of {} values differ",
            outs[0].len()
        );
    }
}

fn main() {
    if std::env::var_os("RESIZE_SWEEP").is_some() {
        return resize_sweep();
    }
    if let Ok(p) = std::env::var("YUNET_WEIGHTS") {
        return whole(&p);
    }
    let d = [1, 8, 16, 24];
    check("conv 3x3 s2 (3->16)", &[1, 3, 32, 32], |g, x| {
        let w = g.param("w", Shape::new(&[16, 3, 3, 3], DType::F32));
        (
            g.conv2d(x, w, [3, 3], [2, 2], [1, 1], [1, 1], 1),
            vec![("w", vec![16, 3, 3, 3])],
        )
    });
    check("conv 1x1", &d, |g, x| {
        let w = g.param("w", Shape::new(&[16, 8, 1, 1], DType::F32));
        (
            g.conv2d(x, w, [1, 1], [1, 1], [0, 0], [1, 1], 1),
            vec![("w", vec![16, 8, 1, 1])],
        )
    });
    check("conv 3x3 depthwise", &d, |g, x| {
        let w = g.param("w", Shape::new(&[8, 1, 3, 3], DType::F32));
        (
            g.conv2d(x, w, [3, 3], [1, 1], [1, 1], [1, 1], 8),
            vec![("w", vec![8, 1, 3, 3])],
        )
    });
    check("conv 3x3 1->1", &[1, 1, 16, 24], |g, x| {
        let w = g.param("w", Shape::new(&[1, 1, 3, 3], DType::F32));
        (
            g.conv2d(x, w, [3, 3], [1, 1], [1, 1], [1, 1], 1),
            vec![("w", vec![1, 1, 3, 3])],
        )
    });
    check("bias add [1,C,1,1]", &d, |g, x| {
        let b = g.param("b", Shape::new(&[8], DType::F32));
        let b = g.reshape_(b, vec![1, 8, 1, 1]);
        (g.add(x, b), vec![("b", vec![8])])
    });
    check("relu", &d, |g, x| (g.relu(x), vec![]));
    check("maxpool 2x2", &d, |g, x| {
        (
            g.pool2d(x, rlx_ir::op::ReduceOp::Max, [2, 2], [2, 2], [0, 0]),
            vec![],
        )
    });
    check("resize nearest 2x", &d, |g, x| {
        (
            g.add_node(
                rlx_ir::op::Op::ResizeNearest2x,
                vec![x],
                Shape::new(&[1, 8, 32, 48], DType::F32),
            ),
            vec![],
        )
    });
    check("transpose+reshape", &[1, 4, 16, 24], |g, x| {
        let y = g.transpose_(x, vec![0, 2, 3, 1]);
        (g.reshape_(y, vec![1, -1, 4]), vec![])
    });
    check("sigmoid", &d, |g, x| (g.sigmoid(x), vec![]));
    check("conv dw + bias + relu", &d, |g, x| {
        let w = g.param("w", Shape::new(&[8, 1, 3, 3], DType::F32));
        let b = g.param("b", Shape::new(&[8], DType::F32));
        let y = g.conv2d(x, w, [3, 3], [1, 1], [1, 1], [1, 1], 8);
        let b = g.reshape_(b, vec![1, 8, 1, 1]);
        let y = g.add(y, b);
        (g.relu(y), vec![("w", vec![8, 1, 3, 3]), ("b", vec![8])])
    });
    // The neck: a coarse map upsampled and added to a finer one (both computed in the graph).
    check("add(skip, resize2x(pool(x)))", &d, |g, x| {
        let small = g.pool2d(x, rlx_ir::op::ReduceOp::Max, [2, 2], [2, 2], [0, 0]);
        let up = g.add_node(
            rlx_ir::op::Op::ResizeNearest2x,
            vec![small],
            Shape::new(&[1, 8, 16, 24], DType::F32),
        );
        let skip = g.relu(x);
        (g.add(skip, up), vec![])
    });
    check("add(resize2x(pool(x)), skip)", &d, |g, x| {
        let small = g.pool2d(x, rlx_ir::op::ReduceOp::Max, [2, 2], [2, 2], [0, 0]);
        let up = g.add_node(
            rlx_ir::op::Op::ResizeNearest2x,
            vec![small],
            Shape::new(&[1, 8, 16, 24], DType::F32),
        );
        let skip = g.relu(x);
        (g.add(up, skip), vec![])
    });
    check("resize2x(relu(pool)) + skip -> conv1x1", &d, |g, x| {
        let small = g.pool2d(x, rlx_ir::op::ReduceOp::Max, [2, 2], [2, 2], [0, 0]);
        let small = g.relu(small);
        let up = g.add_node(
            rlx_ir::op::Op::ResizeNearest2x,
            vec![small],
            Shape::new(&[1, 8, 16, 24], DType::F32),
        );
        let y = g.add(x, up);
        let w = g.param("w", Shape::new(&[8, 8, 1, 1], DType::F32));
        (
            g.conv2d(y, w, [1, 1], [1, 1], [0, 0], [1, 1], 1),
            vec![("w", vec![8, 8, 1, 1])],
        )
    });
    check("conv+bias+relu -> resize2x", &[1, 8, 6, 10], |g, x| {
        let w = g.param("w", Shape::new(&[8, 1, 3, 3], DType::F32));
        let b = g.param("b", Shape::new(&[8], DType::F32));
        let y = g.conv2d(x, w, [3, 3], [1, 1], [1, 1], [1, 1], 8);
        let b = g.reshape_(b, vec![1, 8, 1, 1]);
        let y = g.add(y, b);
        let y = g.relu(y);
        (
            g.add_node(
                rlx_ir::op::Op::ResizeNearest2x,
                vec![y],
                Shape::new(&[1, 8, 12, 20], DType::F32),
            ),
            vec![("w", vec![8, 1, 3, 3]), ("b", vec![8])],
        )
    });
    check("x+bias -> resize2x", &[1, 8, 6, 10], |g, x| {
        let b = g.param("b", Shape::new(&[8], DType::F32));
        let b = g.reshape_(b, vec![1, 8, 1, 1]);
        let y = g.add(x, b);
        (
            g.add_node(
                rlx_ir::op::Op::ResizeNearest2x,
                vec![y],
                Shape::new(&[1, 8, 12, 20], DType::F32),
            ),
            vec![("b", vec![8])],
        )
    });
    check("relu(x) -> resize2x", &[1, 8, 6, 10], |g, x| {
        let y = g.relu(x);
        (
            g.add_node(
                rlx_ir::op::Op::ResizeNearest2x,
                vec![y],
                Shape::new(&[1, 8, 12, 20], DType::F32),
            ),
            vec![],
        )
    });
    check("relu(x+bias) -> resize2x", &[1, 8, 6, 10], |g, x| {
        let b = g.param("b", Shape::new(&[8], DType::F32));
        let b = g.reshape_(b, vec![1, 8, 1, 1]);
        let y = g.add(x, b);
        let y = g.relu(y);
        (
            g.add_node(
                rlx_ir::op::Op::ResizeNearest2x,
                vec![y],
                Shape::new(&[1, 8, 12, 20], DType::F32),
            ),
            vec![("b", vec![8])],
        )
    });
    check("conv 1x1 -> transpose", &d, |g, x| {
        let w = g.param("w", Shape::new(&[4, 8, 1, 1], DType::F32));
        let y = g.conv2d(x, w, [1, 1], [1, 1], [0, 0], [1, 1], 1);
        let y = g.transpose_(y, vec![0, 2, 3, 1]);
        (g.reshape_(y, vec![1, -1, 4]), vec![("w", vec![4, 8, 1, 1])])
    });
}

/// The whole network (weights from `YUNET_WEIGHTS`), output by output.
fn whole(path: &str) {
    let w = rlx_yunet::Weights::load(std::path::Path::new(path)).expect("weights");
    let (h, wd) = (192, 320);
    let input: Vec<f32> = noise(3 * h * wd, 5)
        .iter()
        .map(|v| (v + 0.5) * 255.0)
        .collect();
    let mut outs = Vec::new();
    for dev in [Device::Cpu, Device::Metal] {
        let mut net = rlx_yunet::YuNet::new(w.clone(), dev);
        outs.push(net.run(&input, h, wd).expect("run"));
    }
    // Layer by layer.
    let mut taps = Vec::new();
    for dev in [Device::Cpu, Device::Metal] {
        let mut c = Session::new(dev).compile(rlx_yunet::model::build_graph_with_taps(h, wd));
        for (k, v) in &w.tensors {
            c.set_param(k, v);
        }
        taps.push(c.run(&[("input", &input)]));
    }
    for (i, name) in rlx_yunet::model::TAPS.iter().enumerate() {
        let j = rlx_yunet::OUTPUTS.len() + i;
        let worst = taps[0][j]
            .iter()
            .zip(&taps[1][j])
            .map(|(a, b)| (a - b).abs() / (1.0 + a.abs()))
            .fold(0.0f32, f32::max);
        println!(
            "tap   {name:8} {:>6} values  max rel |cpu - metal| = {worst:.2e}{}",
            taps[0][j].len(),
            if worst > 1e-3 { "   <-- differs" } else { "" }
        );
        if worst > 1e-3 {
            let bad: Vec<usize> = (0..taps[0][j].len())
                .filter(|&k| {
                    (taps[0][j][k] - taps[1][j][k]).abs() / (1.0 + taps[0][j][k].abs()) > 1e-3
                })
                .collect();
            println!(
                "        {} wrong, first at {}, last at {}; wrong per quarter: {:?}",
                bad.len(),
                bad[0],
                bad[bad.len() - 1],
                (0..4)
                    .map(|q| bad
                        .iter()
                        .filter(|&&k| k * 4 / taps[0][j].len() == q)
                        .count())
                    .collect::<Vec<_>>()
            );
        }
    }
    for (i, name) in rlx_yunet::OUTPUTS.iter().enumerate() {
        let worst = outs[0][i]
            .iter()
            .zip(&outs[1][i])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "yunet {name:8} {:>6} values  max |cpu - metal| = {worst:.2e}{}",
            outs[0][i].len(),
            if worst > 1e-3 { "   <-- differs" } else { "" }
        );
    }
}
