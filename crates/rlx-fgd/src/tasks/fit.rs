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

//! `L²` function fitting — Figure 1 and Appendix A.1.1.
//!
//! `L_FIT(f) = ½‖f − f*‖²_{L²([0,1]²)}`, whose functional gradient is just the
//! residual `f − f*` (Proposition A.1). This is the cleanest instantiation:
//! `B = H = L²`, so Assumptions 3.3 and 3.4 hold with `α = β = 1`, the loss is
//! `1`-smooth and `1`-Polyak-Łojasiewicz, and Proposition 3.8's linear rate
//! applies with every constant known.
//!
//! It is also where the error bound is at its sharpest. On a cell, the best
//! constant is the mean of `f − f*` and the resulting squared `L²` error is
//! exactly the cell's variance times its volume — so `U_t` is not a bound that
//! happens to hold, it is the approximation error, estimated by Monte Carlo.

use crate::algorithm::{Functional, GradientApprox, Representation, Rng};
use crate::rep::{CellFit, DyadicGrid};
use crate::theory::Constants;

/// The spiral target of Figure 1: `cos(ωr − mθ)·exp(−r²/σ²)` on `[0,1]²`,
/// in coordinates recentred to `[−0.75, 0.75]²`.
///
/// Its detail is concentrated near the centre and it oscillates faster than any
/// coarse grid can hold, which is what makes a *fixed* representation plateau.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spiral {
    /// Radial frequency.
    pub omega: f64,
    /// Angular winding number.
    pub arms: f64,
    /// Gaussian envelope width.
    pub sigma: f64,
}

impl Default for Spiral {
    fn default() -> Self {
        Self {
            omega: 20.0,
            arms: 3.0,
            sigma: 0.5,
        }
    }
}

impl Spiral {
    /// Evaluate the target at `(x, y) ∈ [0,1]²`.
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        let xx = 1.5 * x - 0.75;
        let yy = 1.5 * y - 0.75;
        let r = (xx * xx + yy * yy).sqrt();
        let theta = yy.atan2(xx);
        let phase = self.omega * r - self.arms * theta;
        let envelope = (-(r * r) / (self.sigma * self.sigma)).exp();
        phase.cos() * envelope
    }
}

/// Default resolution of the quadrature grid on which losses are reported.
///
/// Reporting `L(f)` by quadrature on `f`'s *own* cells would make the number
/// mean something different at every refinement level, and so make the adaptive
/// and fixed runs incomparable — the exact comparison Figure 1 is about. The
/// quadrature is pinned instead, at this resolution by default. `f` itself is
/// evaluated exactly at each point (it is piecewise constant), so the only
/// quadrature error comes from `f*`'s variation across one sample cell, and
/// that error is identical for every method at a given resolution.
pub const LOSS_QUADRATURE: usize = 256;

/// `L_FIT` over piecewise-constant grids on `[0,1]²`.
#[derive(Debug, Clone)]
pub struct L2Fit {
    /// The target `f*`.
    pub target: Spiral,
    /// Monte-Carlo samples per cell used to fit it and to estimate `U_t`.
    pub samples_per_cell: usize,
    /// Resolution of the quadrature that reports `L(f)`.
    ///
    /// Fixed for the whole run, and deliberately not tied to the grid: a
    /// quadrature that followed the refinement level would change scale exactly
    /// when the representation did, turning each refinement into a step change
    /// in the reported loss that has nothing to do with the iterate. Should be
    /// at least the finest grid the run can reach, so no cell goes unsampled —
    /// see [`L2Fit::quadrature_resolves`].
    pub loss_quadrature: usize,
}

impl Default for L2Fit {
    fn default() -> Self {
        Self {
            target: Spiral::default(),
            samples_per_cell: 8,
            loss_quadrature: LOSS_QUADRATURE,
        }
    }
}

impl L2Fit {
    /// A fit problem for `target` with the reference implementation's eight
    /// samples per cell.
    pub fn new(target: Spiral) -> Self {
        Self {
            target,
            ..Self::default()
        }
    }

    /// A fit problem whose loss quadrature resolves grids up to `max_level`.
    pub fn with_max_level(target: Spiral, max_level: u32) -> Self {
        Self {
            target,
            loss_quadrature: LOSS_QUADRATURE.max(1usize << max_level),
            ..Self::default()
        }
    }

    /// Whether the loss quadrature is fine enough to place a point in every
    /// cell of `f`. When false, the reported loss under-resolves the residual.
    pub fn quadrature_resolves(&self, f: &DyadicGrid<2>) -> bool {
        self.loss_quadrature >= f.cells_per_axis()
    }

    /// `μ = 1`: Proposition A.1 shows `L_FIT(f) − L* = ½‖∇L_FIT(f)‖²_{L²}`.
    pub const POLYAK_LOJASIEWICZ: f64 = 1.0;

    /// `L* = 0`, attained at `f = f*`.
    pub const GLOBAL_MINIMUM: f64 = 0.0;

    /// `½‖f − f*‖²_{L²}` by midpoint quadrature at [`L2Fit::loss_quadrature`].
    pub fn exact_loss(&self, f: &DyadicGrid<2>) -> f64 {
        let n = self.loss_quadrature;
        let cell = 1.0 / n as f64;
        let mut total = 0.0;
        for i in 0..n {
            let x = (i as f64 + 0.5) * cell;
            for j in 0..n {
                let y = (j as f64 + 0.5) * cell;
                let residual = f.eval([x, y]) - self.target.eval(x, y);
                total += residual * residual;
            }
        }
        0.5 * total * cell * cell
    }

    /// `L_FIT(f_t)` recovered from the two quantities Algorithm 1 already has.
    ///
    /// `∇L_FIT(f) = f − f*`, so `L_FIT(f) = ½‖∇L_FIT(f)‖²_{L²}`. The
    /// approximation `g` is the cell-wise mean of `∇L_FIT(f)` and the error
    /// `∇L_FIT(f) − g` has zero mean on every cell, so the two are `L²`
    /// orthogonal and
    ///
    /// ```text
    /// L_FIT(f) = ½(‖g‖² + ‖g − ∇L_FIT(f)‖²) = ½(‖g‖² + U²).
    /// ```
    ///
    /// The loss therefore costs nothing beyond the gradient step — and, read
    /// the other way, this identity is a check on `U`: agreement with
    /// [`L2Fit::exact_loss`] says the error bound really is measuring the
    /// approximation error and not something adjacent to it.
    ///
    /// Exact only for the true cell means. Both terms are estimated from the
    /// same `N` samples, which leaves an `O(1/N)` bias — the sample mean
    /// carries `σ²/N` of spurious energy into `‖g‖²`, and
    /// [`crate::rep::CellFit::from_samples`] corrects `U²` upward by a matching
    /// `σ²/N` rather than letting the two cancel.
    pub fn loss_from_gradient(&self, norm: f64, error_bound: f64) -> f64 {
        0.5 * (norm * norm + error_bound * error_bound)
    }

    /// The offsets in `[0,1)²` used inside every cell of one attempt.
    ///
    /// One draw is shared by all cells, and by every refinement attempt within a
    /// step: refinement then competes against a fixed sample draw instead of
    /// against resampling noise, so a step never refines merely because the
    /// dice moved.
    fn offsets(&self, seed: u64) -> Vec<[f64; 2]> {
        let mut rng = Rng::new(seed);
        (0..self.samples_per_cell)
            .map(|_| [rng.uniform01(), rng.uniform01()])
            .collect()
    }
}

impl Functional for L2Fit {
    type Rep = DyadicGrid<2>;
    /// `∇L_FIT(f) = f − f*` is pointwise in `f`; nothing to precompute.
    type StepState = ();

    fn begin_step(&self, _f: &Self::Rep) -> Self::StepState {}

    fn approximate_gradient(
        &self,
        f: &Self::Rep,
        _state: &Self::StepState,
        seed: u64,
    ) -> GradientApprox<Self::Rep> {
        let offsets = self.offsets(seed);
        let widths = f.cell_widths();
        let mut gradient = f.clone();

        let mut samples = Vec::with_capacity(offsets.len());
        let error_squared = gradient.fill_from(|flat| {
            let origin = f.cell_origin(flat);
            // `∇L_FIT(f) = f − f*`, and `f` is constant across this cell — the
            // gradient is approximated at the iterate's own resolution — so the
            // cell's value is read once rather than looked up per sample. That
            // also sidesteps a rounding edge: a sample sitting exactly on the
            // cell's lower face can index the neighbour instead.
            let here = f.values()[flat];
            samples.clear();
            for offset in &offsets {
                let x = origin[0] + offset[0] * widths[0];
                let y = origin[1] + offset[1] * widths[1];
                samples.push(here - self.target.eval(x, y));
            }
            CellFit::from_samples(&samples)
        });

        GradientApprox {
            norm: gradient.l2_norm(),
            gradient,
            error_bound: error_squared.sqrt(),
        }
    }

    fn loss(&self, f: &Self::Rep) -> Option<f64> {
        Some(self.exact_loss(f))
    }

    fn constants(&self) -> Option<Constants> {
        // B = H = L², so α = β = 1; L_FIT is 1-smooth since ∇L_FIT = id − f*.
        Some(Constants::hilbert(1.0))
    }
}

/// A grid holding the target itself, for rendering the reference panel.
pub fn target_grid(target: Spiral, level: u32) -> DyadicGrid<2> {
    let mut grid = DyadicGrid::<2>::unit(level);
    let widths = grid.cell_widths();
    let centers: Vec<f64> = (0..grid.dofs())
        .map(|flat| {
            let origin = grid.cell_origin(flat);
            target.eval(origin[0] + 0.5 * widths[0], origin[1] + 0.5 * widths[1])
        })
        .collect();
    grid.values_mut().copy_from_slice(&centers);
    grid
}
