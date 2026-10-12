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

//! The RKHS gradient evaluation compiled to RLX backends.
//!
//! Section 4.1's gradient, `g(x) = (1/n) Σ c_i K(X_i, x)`, has to be evaluated
//! at every leaf centre on every attempt, and refinement doubles the leaf count
//! each time — so this one kernel dominates the run. It is a dense
//! `leaves × samples` Gaussian kernel followed by a matrix-vector product,
//! which is exactly the shape a backend wants.
//!
//! The squared distances are formed as `‖c‖² − 2c·x + ‖x‖²` so the whole thing
//! reduces to one `mm`. The training features are graph parameters (fixed for
//! the run) while the residuals and leaf centres are inputs (they change every
//! step and every refinement); graphs are cached per leaf count, since that is
//! the only shape that varies.
//!
//! Compute is `f32` here against the host path's `f64`, so
//! `tests/backend_parity.rs` compares at single-precision tolerances.

use crate::algorithm::{Functional, GradientApprox};
use crate::rep::RegressionTree;
use crate::tasks::rkhs::{Dataset, Residuals, RkhsRegression};
use anyhow::{Context, Result, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompiledGraph, Device, Session, is_available};
use std::cell::RefCell;
use std::collections::HashMap;

const F32: DType = DType::F32;

/// RLX device slots exercised by the parity tests.
pub const BACKEND_DEVICES: [Device; 7] = [
    Device::Cpu,
    Device::Metal,
    Device::Mlx,
    Device::Cuda,
    Device::Rocm,
    Device::Gpu,
    Device::Vulkan,
];

/// Build the kernel-gradient graph for `leaves` evaluation points.
///
/// Inputs `centers` `[leaves, dim]` and `coefficients` `[samples, 1]`, parameter
/// `data` `[samples, dim]`, output `[leaves, 1]`.
pub fn build_gradient_graph(leaves: usize, samples: usize, dim: usize, gamma: f64) -> Graph {
    let mut g = Graph::new("rkhs_gradient");
    let centers = g.input("centers", Shape::new(&[leaves, dim], F32));
    let coefficients = g.input("coefficients", Shape::new(&[samples, 1], F32));
    let data = g.param("data", Shape::new(&[samples, dim], F32));

    // ‖c‖² per centre → [leaves, 1]; ‖x‖² per sample → [1, samples].
    let center_sq = g.mul(centers, centers);
    let center_sq = g.sum(center_sq, vec![1], true);
    let data_sq = g.mul(data, data);
    let data_sq = g.sum(data_sq, vec![1], true);
    let data_sq = g.transpose_(data_sq, vec![1, 0]);

    // cross = centers · dataᵀ → [leaves, samples].
    let data_t = g.transpose_(data, vec![1, 0]);
    let cross = g.mm(centers, data_t);

    // squared distance = ‖c‖² + ‖x‖² − 2·cross.
    let two = g.constant(2.0, F32);
    let cross2 = g.mul(cross, two);
    let sums = g.add(center_sq, data_sq);
    let distance = g.sub(sums, cross2);

    // K = exp(−γ·d²); then g = K · c / n.
    let minus_gamma = g.constant(-gamma, F32);
    let scaled = g.mul(distance, minus_gamma);
    let kernel = g.exp(scaled);
    let weighted = g.mm(kernel, coefficients);
    let inv_n = g.constant(1.0 / samples as f64, F32);
    let out = g.mul(weighted, inv_n);

    g.set_outputs(vec![out]);
    g
}

/// A compiled evaluator for Section 4.1's gradient on one device.
pub struct KernelGradient {
    device: Device,
    gamma: f64,
    dim: usize,
    samples: usize,
    data: Vec<f32>,
    cache: HashMap<usize, CompiledGraph>,
}

impl KernelGradient {
    /// Compile against `train`; graphs are built lazily per leaf count.
    pub fn new(device: Device, train: &Dataset, gamma: f64) -> Result<Self> {
        if device != Device::Cpu {
            ensure!(
                is_available(device),
                "RLX backend {device:?} is not available in this build"
            );
        }
        ensure!(!train.is_empty(), "training set must be non-empty");
        let data = (0..train.len())
            .flat_map(|i| train.row(i).iter().map(|&v| v as f32).collect::<Vec<_>>())
            .collect();
        Ok(Self {
            device,
            gamma,
            dim: train.dim(),
            samples: train.len(),
            data,
            cache: HashMap::new(),
        })
    }

    /// The device this evaluator runs on.
    pub fn device(&self) -> Device {
        self.device
    }

    fn compiled_for(&mut self, leaves: usize) -> &mut CompiledGraph {
        let (device, samples, dim, gamma) = (self.device, self.samples, self.dim, self.gamma);
        let data = &self.data;
        self.cache.entry(leaves).or_insert_with(|| {
            let graph = build_gradient_graph(leaves, samples, dim, gamma);
            let session = Session::new(device);
            let mut compiled = session.compile(graph);
            compiled.set_param("data", data);
            compiled.finalize_params();
            compiled
        })
    }

    /// Evaluate `(1/n) Σ c_i K(X_i, ·)` at every row of `centers`.
    pub fn evaluate(&mut self, centers: &[f32], coefficients: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            coefficients.len() == self.samples,
            "expected {} coefficients, got {}",
            self.samples,
            coefficients.len()
        );
        ensure!(
            centers.len().is_multiple_of(self.dim),
            "centers must be a multiple of the feature dimension"
        );
        let leaves = centers.len() / self.dim;
        let compiled = self.compiled_for(leaves);
        let outputs = compiled.run(&[("centers", centers), ("coefficients", coefficients)]);
        outputs
            .into_iter()
            .next()
            .context("RKHS gradient graph produced no output")
    }
}

/// Algorithm 1's `g_t ← approximate ∇L(f_t)` step, evaluated on a backend.
///
/// The error bound `U_t` stays on the host: it is three scalars
/// (`mean|c| · Lip(K) · max leaf radius`) and moving it would cost more than it
/// saves.
pub fn approximate_gradient_on(
    runner: &mut KernelGradient,
    problem: &RkhsRegression,
    f: &RegressionTree,
    state: &Residuals,
) -> Result<GradientApprox<RegressionTree>> {
    let centers: Vec<f32> = (0..f.leaves())
        .flat_map(|k| {
            f.leaf_center(k)
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<_>>()
        })
        .collect();
    let coefficients: Vec<f32> = state.coefficients.iter().map(|&c| c as f32).collect();
    let values = runner.evaluate(&centers, &coefficients)?;

    let mut gradient = f.clone();
    ensure!(
        values.len() == gradient.leaves(),
        "backend returned {} values for {} leaves",
        values.len(),
        gradient.leaves()
    );
    for (slot, value) in gradient.values_mut().iter_mut().zip(&values) {
        *slot = *value as f64;
    }

    let error_bound = state.mean_absolute * problem.kernel.lipschitz() * gradient.max_leaf_radius();
    Ok(GradientApprox {
        norm: gradient.sup_norm(),
        gradient,
        error_bound,
    })
}

/// Largest absolute difference between two equal-length slices.
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// [`RkhsRegression`] with its gradient evaluated on an RLX backend.
///
/// Wraps rather than replaces the host problem: the residuals, the error bound
/// and the loss stay where they are, and only the kernel evaluation moves. That
/// keeps `tests/backend_parity.rs` comparing one thing at a time, and means a
/// backend that is unavailable costs nothing but a different constructor.
pub struct DeviceRkhsRegression {
    inner: RkhsRegression,
    // `Functional::approximate_gradient` takes `&self`, but the graph cache
    // grows as the tree refines; the runner is single-threaded either way.
    runner: RefCell<KernelGradient>,
}

impl DeviceRkhsRegression {
    /// Compile `problem`'s gradient kernel for `device`.
    pub fn new(problem: RkhsRegression, device: Device) -> Result<Self> {
        let runner = KernelGradient::new(device, &problem.train, problem.kernel.gamma)?;
        Ok(Self {
            inner: problem,
            runner: RefCell::new(runner),
        })
    }

    /// The underlying host problem.
    pub fn problem(&self) -> &RkhsRegression {
        &self.inner
    }

    /// The device the gradient runs on.
    pub fn device(&self) -> Device {
        self.runner.borrow().device()
    }
}

impl Functional for DeviceRkhsRegression {
    type Rep = RegressionTree;
    type StepState = Residuals;

    fn begin_step(&self, f: &Self::Rep) -> Self::StepState {
        self.inner.begin_step(f)
    }

    fn approximate_gradient(
        &self,
        f: &Self::Rep,
        state: &Self::StepState,
        _seed: u64,
    ) -> GradientApprox<Self::Rep> {
        approximate_gradient_on(&mut self.runner.borrow_mut(), &self.inner, f, state)
            .expect("RKHS gradient kernel failed on the device")
    }

    fn loss(&self, f: &Self::Rep) -> Option<f64> {
        self.inner.loss(f)
    }

    fn constants(&self) -> Option<crate::theory::Constants> {
        self.inner.constants()
    }
}
