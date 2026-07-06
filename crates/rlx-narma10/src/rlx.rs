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

//! NARMA-10 series generation compiled to RLX backends (CPU / Metal / MLX / CUDA / …).

use crate::host::{Coefficients, ORDER, Series, generate_with_coeff};
use anyhow::{Context, Result, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{CompiledGraph, Device, Session, is_available};
use std::collections::HashMap;

const F32: DType = DType::F32;

/// RLX device slots exercised by [`generate_on_device`] parity tests.
pub const BACKEND_DEVICES: [Device; 7] = [
    Device::Cpu,
    Device::Metal,
    Device::Mlx,
    Device::Cuda,
    Device::Rocm,
    Device::Gpu,
    Device::Vulkan,
];

fn coeff_params(coeff: Coefficients) -> HashMap<String, Vec<f32>> {
    HashMap::from([
        ("alpha".into(), vec![coeff.alpha as f32]),
        ("beta".into(), vec![coeff.beta as f32]),
        ("gamma".into(), vec![coeff.gamma as f32]),
        ("delta".into(), vec![coeff.delta as f32]),
        ("zero".into(), vec![0.0f32]),
    ])
}

/// Build an unrolled NARMA-10 graph: input `u` `[total]`, output `y` `[total]`.
pub fn build_series_graph(total: usize, coeff: Coefficients) -> Graph {
    assert!(total > ORDER, "total must exceed ORDER");
    let mut g = Graph::new("narma10_series");
    let u = g.input("u", Shape::new(&[total], F32));

    let alpha = g.param("alpha", Shape::new(&[1], F32));
    let beta = g.param("beta", Shape::new(&[1], F32));
    let gamma = g.param("gamma", Shape::new(&[1], F32));
    let delta = g.param("delta", Shape::new(&[1], F32));
    let zero = g.param("zero", Shape::new(&[1], F32));

    let mut y_nodes: Vec<NodeId> = Vec::with_capacity(total);
    y_nodes.push(zero);

    for t in 0..total - 1 {
        let yt = y_nodes[t];
        let mut sum = yt;
        for i in 1..ORDER {
            if t >= i {
                sum = g.add(sum, y_nodes[t - i]);
            }
        }
        let u_t = g.narrow_(u, 0, t, 1);
        let u_lag = if t >= ORDER - 1 {
            g.narrow_(u, 0, t - (ORDER - 1), 1)
        } else {
            zero
        };
        let u_prod = g.mul(u_t, u_lag);
        let ar = g.mul(yt, alpha);
        let yt_sum = g.mul(yt, sum);
        let nl = g.mul(yt_sum, beta);
        let inp = g.mul(u_prod, gamma);
        let ar_nl = g.add(ar, nl);
        let inp_d = g.add(inp, delta);
        let y_next = g.add(ar_nl, inp_d);
        y_nodes.push(y_next);
    }

    let y_out = g.concat_(y_nodes, 0);
    g.set_outputs(vec![y_out]);
    let _ = coeff;
    g
}

/// Compiled NARMA-10 forward for a fixed series length.
pub struct SeriesRunner {
    total: usize,
    compiled: CompiledGraph,
}

impl SeriesRunner {
    /// Compile the unrolled recurrence on `device`.
    pub fn new(device: Device, n_timesteps: usize, coeff: Coefficients) -> Result<Self> {
        ensure!(n_timesteps > 0, "n_timesteps must be positive");
        if device != Device::Cpu {
            ensure!(
                is_available(device),
                "RLX backend {device:?} is not available in this build"
            );
        }
        let total = n_timesteps + ORDER;
        let graph = build_series_graph(total, coeff);
        let params = coeff_params(coeff);
        let session = Session::new(device);
        let mut compiled = session.compile(graph);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        compiled.finalize_params();
        Ok(Self { total, compiled })
    }

    /// Run with flat `u` (`total` samples); returns full `y` (`total` samples).
    pub fn run(&mut self, inputs: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            inputs.len() == self.total,
            "expected {} inputs, got {}",
            self.total,
            inputs.len()
        );
        let outs = self.compiled.run(&[("u", inputs)]);
        outs.into_iter()
            .next()
            .context("NARMA-10 graph produced no output")
    }
}

/// Generate NARMA-10 on an RLX backend (host RNG for `u`, device recurrence).
pub fn generate_on_device(device: Device, n_timesteps: usize, seed: u64) -> Result<Series> {
    generate_on_device_with_coeff(device, n_timesteps, seed, Coefficients::default())
}

/// Like [`generate_on_device`] with custom coefficients.
pub fn generate_on_device_with_coeff(
    device: Device,
    n_timesteps: usize,
    seed: u64,
    coeff: Coefficients,
) -> Result<Series> {
    let host = generate_with_coeff(n_timesteps, seed, coeff);
    let inputs_f32: Vec<f32> = host.inputs.iter().map(|&v| v as f32).collect();
    let mut runner = SeriesRunner::new(device, n_timesteps, coeff)?;
    let y = runner.run(&inputs_f32)?;
    let y_f64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
    Ok(Series {
        inputs: host.inputs,
        outputs: y_f64.clone(),
        targets: y_f64[ORDER..].to_vec(),
    })
}

/// Max absolute difference between two equal-length slices.
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_rlx_matches_host_reference() {
        let host = crate::host::generate(128, 7);
        let device = generate_on_device(Device::Cpu, 128, 7).unwrap();
        let err = max_abs_diff(&host.targets, &device.targets);
        assert!(
            err < 1e-5,
            "host vs RLX CPU NARMA-10 mismatch: max |Δ| = {err:e}"
        );
    }
}
