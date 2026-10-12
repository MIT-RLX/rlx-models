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

//! Section 4.1's gradient kernel on RLX backends, against the host reference.

use rlx_fgd::algorithm::{Functional, Representation};
use rlx_fgd::rep::RegressionTree;
use rlx_fgd::rlx::{BACKEND_DEVICES, KernelGradient, approximate_gradient_on, max_abs_diff};
use rlx_fgd::tasks::rkhs::{Dataset, PointwiseLoss, RbfKernel, RkhsRegression};
use rlx_runtime::{Device, is_available};

/// `f32` compute against an `f64` host reference over a `240`-term kernel sum.
const TOLERANCE: f64 = 2e-6;

fn problem() -> RkhsRegression {
    let data = Dataset::synthetic(240, 9);
    let (lower, upper) = data.bounds();
    RkhsRegression::new(
        data.normalized(&lower, &upper),
        RbfKernel::new(12.0),
        PointwiseLoss::SquaredError,
    )
}

fn perturbed_tree(problem: &RkhsRegression, depth: u32) -> RegressionTree {
    let mut tree = problem.initial_tree(depth);
    let mut rng = rlx_fgd::algorithm::Rng::new(5);
    for value in tree.values_mut() {
        *value = 0.5 * (rng.uniform01() - 0.5);
    }
    tree
}

fn check(device: Device) {
    if device != Device::Cpu && !is_available(device) {
        eprintln!("skipping {device:?}: not available in this build");
        return;
    }
    let problem = problem();
    let mut runner = KernelGradient::new(device, &problem.train, problem.kernel.gamma)
        .unwrap_or_else(|e| panic!("{device:?}: {e}"));

    // Several depths, so the per-leaf-count graph cache is exercised as well.
    let mut f = perturbed_tree(&problem, 2);
    for _ in 0..4 {
        let state = problem.begin_step(&f);
        let host = problem.approximate_gradient(&f, &state, 0);
        let backend = approximate_gradient_on(&mut runner, &problem, &f, &state)
            .unwrap_or_else(|e| panic!("{device:?}: {e}"));

        let error = max_abs_diff(host.gradient.values(), backend.gradient.values());
        assert!(
            error < TOLERANCE,
            "{device:?} at {} leaves: max |Δ| = {error:e}",
            f.leaves()
        );
        assert!(
            (host.norm - backend.norm).abs() < TOLERANCE,
            "{device:?}: ‖g‖ {} vs {}",
            host.norm,
            backend.norm
        );
        // The error bound is a host-side scalar computation either way, so it
        // must agree exactly.
        assert_eq!(host.error_bound, backend.error_bound);
        f = f.refine();
    }
}

#[test]
fn cpu_backend_matches_the_host_reference() {
    check(Device::Cpu);
}

#[test]
fn every_available_backend_matches_the_host_reference() {
    for device in BACKEND_DEVICES {
        check(device);
    }
}

/// Algorithm 1's accept/refine decisions are made from `‖g‖` and `U`; if a
/// backend shifts either, the *schedule* diverges and not just the numbers.
/// This checks the decisions agree, not merely the values they came from.
#[test]
fn backends_make_the_same_refinement_decisions() {
    let problem = problem();
    let tolerance = 0.5;
    for device in BACKEND_DEVICES {
        if device != Device::Cpu && !is_available(device) {
            continue;
        }
        let mut runner = KernelGradient::new(device, &problem.train, problem.kernel.gamma)
            .unwrap_or_else(|e| panic!("{device:?}: {e}"));
        let mut f = perturbed_tree(&problem, 2);
        for _ in 0..6 {
            let state = problem.begin_step(&f);
            let host = problem.approximate_gradient(&f, &state, 0);
            let backend = approximate_gradient_on(&mut runner, &problem, &f, &state)
                .unwrap_or_else(|e| panic!("{device:?}: {e}"));
            assert_eq!(
                host.accepts(tolerance),
                backend.accepts(tolerance),
                "{device:?} at {} leaves disagreed on whether to refine",
                f.leaves()
            );
            f = f.refine();
        }
    }
}
