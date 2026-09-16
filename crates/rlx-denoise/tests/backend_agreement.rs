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

//! A network is only portable if every backend computes the same thing.
//!
//! Compiling for a backend proves nothing about its arithmetic. These run the
//! same graph on the CPU and on the accelerator and require the results to
//! match, one op at a time, so a disagreement names the op responsible instead
//! of showing up as a network that trains on one machine and not another.

#![cfg(feature = "metal")]

use rlx_ir::{DType, Graph, GraphExt, PadMode, Shape};
use rlx_runtime::{Device, Session};

/// Largest absolute difference between running `build` on the CPU and on
/// `device`.
fn divergence(
    name: &str,
    device: Device,
    dims: &[usize],
    build: impl Fn(&mut Graph, rlx_ir::NodeId) -> rlx_ir::NodeId,
) -> f32 {
    let n: usize = dims.iter().product();
    let input: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
        .collect();

    let run = |device: Device| -> Vec<f32> {
        let mut graph = Graph::new(name);
        let x = graph.input("x", Shape::new(dims, DType::F32));
        let y = build(&mut graph, x);
        graph.set_outputs(vec![y]);
        let mut compiled = Session::new(device).compile(graph);
        compiled
            .run(&[("x", &input)])
            .into_iter()
            .next()
            .unwrap_or_default()
    };

    let cpu = run(Device::Cpu);
    let other = run(device);
    assert_eq!(
        cpu.len(),
        other.len(),
        "{name}: cpu returned {} values, {device:?} returned {}",
        cpu.len(),
        other.len()
    );
    cpu.iter()
        .zip(&other)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Is Metal's softmax simply ignoring the axis and always using the last one?
///
/// If so, `metal(axis=k)` equals `cpu(axis=rank-1)` for every k, which says the
/// axis argument never reaches the kernel — a different fault from an axis that
/// arrives but is mishandled.
#[test]
fn metal_softmax_axis_is_reported() {
    let dims = [2usize, 8, 6, 6];
    let n: usize = dims.iter().product();
    let input: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
        .collect();

    let run = |device: Device, axis: i32| -> Vec<f32> {
        let mut graph = Graph::new("sm");
        let x = graph.input("x", Shape::new(&dims, DType::F32));
        let y = graph.sm(x, axis);
        graph.set_outputs(vec![y]);
        Session::new(device)
            .compile(graph)
            .run(&[("x", &input)])
            .into_iter()
            .next()
            .unwrap_or_default()
    };

    let cpu_last = run(Device::Cpu, 3);
    for axis in 0..4 {
        let metal = run(Device::Metal, axis);
        let vs_last = cpu_last
            .iter()
            .zip(&metal)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let cpu_here = run(Device::Cpu, axis);
        let vs_here = cpu_here
            .iter()
            .zip(&metal)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mut shifted = f32::NAN;
        if axis + 1 < 4 {
            let cpu_next = run(Device::Cpu, axis + 1);
            shifted = cpu_next
                .iter()
                .zip(&metal)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
        }
        println!(
            "axis {axis}: vs cpu({axis}) = {vs_here:e}   vs cpu({}) = {shifted:e}   vs cpu(last) = {vs_last:e}",
            axis + 1
        );
    }
}

/// Every op the kernel-predicting head introduces, one at a time.
///
/// The direct head uses only convolution, ReLU and concat, and agrees across
/// backends. The kernel head adds a softmax, a replicate pad, spatial narrows
/// and an elementwise multiply — so when the two heads disagree about a
/// backend, the fault is in one of these.
#[test]
fn metal_matches_cpu_on_every_kernel_head_op() {
    const TOLERANCE: f32 = 1e-5;
    let dims = [2usize, 8, 6, 6];
    let mut failures = Vec::new();

    let checks: Vec<(
        &str,
        Box<dyn Fn(&mut Graph, rlx_ir::NodeId) -> rlx_ir::NodeId>,
    )> = vec![
        ("softmax(axis=0)", Box::new(|g: &mut Graph, x| g.sm(x, 0))),
        ("softmax(axis=1)", Box::new(|g: &mut Graph, x| g.sm(x, 1))),
        ("softmax(axis=2)", Box::new(|g: &mut Graph, x| g.sm(x, 2))),
        ("softmax(axis=3)", Box::new(|g: &mut Graph, x| g.sm(x, 3))),
        ("softmax(axis=-1)", Box::new(|g: &mut Graph, x| g.sm(x, -1))),
        (
            "pad(replicate)",
            Box::new(|g: &mut Graph, x| {
                g.pad_(x, vec![[0, 0], [0, 0], [2, 2], [2, 2]], PadMode::Replicate)
            }),
        ),
        (
            "narrow(spatial)",
            Box::new(|g: &mut Graph, x| {
                let rows = g.narrow_(x, 2, 1, 4);
                g.narrow_(rows, 3, 2, 3)
            }),
        ),
        (
            "concat(repeated)",
            Box::new(|g: &mut Graph, x| {
                let one = g.narrow_(x, 1, 3, 1);
                g.concat_(vec![one, one, one], 1)
            }),
        ),
        (
            "mul(elementwise)",
            Box::new(|g: &mut Graph, x| {
                let a = g.narrow_(x, 1, 0, 3);
                let b = g.narrow_(x, 1, 3, 3);
                g.mul(a, b)
            }),
        ),
        (
            "pad+narrow+mul",
            Box::new(|g: &mut Graph, x| {
                let colour = g.narrow_(x, 1, 0, 3);
                let padded = g.pad_(
                    colour,
                    vec![[0, 0], [0, 0], [1, 1], [1, 1]],
                    PadMode::Replicate,
                );
                let rows = g.narrow_(padded, 2, 0, 6);
                let shifted = g.narrow_(rows, 3, 2, 6);
                let tap = g.narrow_(x, 1, 5, 1);
                let tap = g.concat_(vec![tap, tap, tap], 1);
                g.mul(shifted, tap)
            }),
        ),
    ];

    for (name, build) in checks {
        let delta = divergence(name, Device::Metal, &dims, |g, x| build(g, x));
        println!("{name:<20} max |cpu - metal| = {delta:e}");
        // NaN must count as a failure, so compare the other way round rather
        // than negating `<=`.
        if delta.partial_cmp(&TOLERANCE).is_none_or(|o| o.is_gt()) {
            failures.push(format!("{name}: {delta:e}"));
        }
    }

    assert!(
        failures.is_empty(),
        "metal disagrees with the cpu on: {}",
        failures.join(", ")
    );
}
