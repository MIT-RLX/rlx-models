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

//! **Functional gradient descent with adaptive representations** — Csillag,
//! Schuller, Dall'Antonia, Guibas, Velho and Novello, NeurIPS 2026
//! ([arXiv:2606.16926](https://arxiv.org/abs/2606.16926)).
//!
//! Gradient descent run directly in function space, rather than on the
//! parameters of a fixed representation. The obstacle is that a functional
//! gradient is an infinite-dimensional object: it cannot be stored, so it must
//! be approximated, and a *fixed* approximation leaves a floor of error that
//! stops the iteration short of the minimiser no matter how long it runs. The
//! paper's answer is to refine the representation during optimisation, using a
//! test built only from quantities you can actually compute, and to fold that
//! approximation into the convergence analysis so the guarantees survive it.
//!
//! # Layout
//!
//! - [`algorithm`] — Algorithm 1 and its traits ([`Representation`], [`Functional`])
//! - [`theory`] — Section 3.1: the descent lemma, the two convergence
//!   propositions, and the admissible `(ε, η)` region
//! - [`rep`] — the refinable representations: dyadic grids, decision trees,
//!   frequency grids, voxel grids with spherical-harmonic colour
//! - [`tasks`] — the functionals: `L²` fitting, RKHS regression, the wave
//!   equation, radiance fields
//! - [`render`], [`scene`] — the volumetric renderer, cameras, and
//!   NeRF-Synthetic loading that Section 4.3 needs
//! - [`baseline`] — the neural-network comparison of Figure 1
//! - [`linalg`] — the small symmetric solvers the tasks need
//! - [`rlx`] — the RKHS gradient kernel compiled to RLX backends
//!
//! # The shape of a run
//!
//! ```no_run
//! use rlx_fgd::algorithm::{FgdConfig, Refinement, run};
//! use rlx_fgd::rep::DyadicGrid;
//! use rlx_fgd::tasks::fit::L2Fit;
//!
//! let problem = L2Fit::default();
//! let report = run(
//!     &problem,
//!     DyadicGrid::<2>::unit(3),
//!     FgdConfig {
//!         learning_rate: 0.9,
//!         tolerance: 0.5,
//!         steps: 100,
//!         refinement: Refinement::Adaptive { max_level: 10 },
//!         seed: 0,
//!     },
//! );
//!
//! // Theorem 3.10 applies exactly when the tolerance test held at every step.
//! assert!(report.guarantees_hold());
//! ```
//!
//! Swapping [`Refinement::Fixed`] in gives the paper's “Approx. FGD” baseline
//! on the same code path, which is how the adaptive-versus-fixed comparison is
//! made without two implementations that might differ for uninteresting reasons.
//!
//! # Two tolerances
//!
//! Algorithm 1's `ε` and the `RelErr` bound in Propositions 3.6 and 3.8 are
//! different numbers: running with tolerance `ε` guarantees
//! `RelErr ≤ ε/(1+ε)` (Theorem 3.10(i)). Convert with
//! [`theory::guaranteed_relative_error`] rather than by hand.
//!
//! # Scope
//!
//! All four experiments are implemented and tested: Figure 1's `L²` fitting,
//! Section 4.1's RKHS regression, Section 4.2's wave equation, and Section
//! 4.3's radiance fields. The crate README lists where this port departs from
//! the authors' reference code, and why.

pub mod algorithm;
pub mod baseline;
pub mod linalg;
pub mod render;
pub mod rep;
pub mod rlx;
pub mod scene;
pub mod tasks;
pub mod theory;

pub use algorithm::{
    FgdConfig, Functional, GradientApprox, Refinement, Report, Representation, Rng, StepRecord, run,
};
pub use render::{Ray, march, render_ray};
pub use rep::{Complex, DyadicGrid, FourierGrid, RegressionTree, VoxelGrid};
pub use scene::{Camera, Image, Scene, View};
pub use tasks::fit::{L2Fit, Spiral};
pub use tasks::radiance::{Occupancy, RadianceField};
pub use tasks::rkhs::{Dataset, PointwiseLoss, RbfKernel, RkhsRegression};
pub use tasks::wave::{SincCubeBumps, WaveEquation};
pub use theory::{Constants, guaranteed_relative_error, relative_error, tolerance_in_scope};
