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

//! Piecewise-constant dyadic grids — the `L²` representation of Figure 1.
//!
//! A [`DyadicGrid`] holds one value per cell of a uniform `2^level` grid over an
//! axis-aligned box. Refinement splits every cell in two along every axis and
//! copies the parent value into each child, which leaves the encoded function
//! untouched — the property [`crate::algorithm::Representation`] requires.
//!
//! The reason this representation is a good fit for `L²` is that the best
//! piecewise-constant approximation of a function on a cell *is* its mean over
//! that cell, and the resulting squared error is exactly the cell's variance
//! times its volume. So the error bound of Equation 5 is not a loose surrogate
//! here: [`CellFit::error_density`] estimates the approximation error itself.

/// A uniform `2^level`-per-axis piecewise-constant grid over a `D`-dimensional box.
#[derive(Debug, Clone, PartialEq)]
pub struct DyadicGrid<const D: usize> {
    lower: [f64; D],
    upper: [f64; D],
    level: u32,
    values: Vec<f64>,
}

/// The per-cell outcome of fitting a function into one grid cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellFit {
    /// The constant stored in the cell — the sample mean, which is the
    /// `L²`-optimal choice.
    pub mean: f64,
    /// `E_x[(mean − target(x))²]` over the cell: the squared `L²` error of that
    /// constant, per unit volume.
    pub error_density: f64,
}

/// Unbiased estimate of the mean squared error of a least-squares fit.
///
/// `rss` is the residual sum of squares of fitting `parameters` coefficients to
/// `observations` samples. The fit is itself estimated, so it is off by
/// `O(σ√(p/m))`, and the error density it leaves behind has expectation
/// `σ²(1 + p/m)` for the true noise level `σ²`; the unbiased estimate of `σ²`
/// is `rss/(m − p)`. Hence `(rss/(m − p))·(1 + p/m)`.
///
/// At `p = 1` this is Bessel's correction times `(m+1)/m`, which is the
/// per-cell estimator of [`CellFit::from_samples`]. With no spare degrees of
/// freedom the result is `+∞`: an unmeasured error is not an absent one, and
/// reporting zero would let Algorithm 1 accept any representation at all.
pub fn unbiased_error_density(rss: f64, observations: usize, parameters: usize) -> f64 {
    if observations <= parameters {
        return f64::INFINITY;
    }
    let m = observations as f64;
    let p = parameters as f64;
    (rss / (m - p)) * (1.0 + p / m)
}

impl CellFit {
    /// Fit a cell from samples of the target drawn uniformly inside it.
    ///
    /// The stored constant is the sample mean `x̄`. Its squared error density is
    /// `(x̄ − μ)² + σ²` for the cell's true mean and variance, whose expectation
    /// is `σ²(1 + 1/N)`; with `s²` the Bessel-corrected sample variance, the
    /// unbiased estimate is therefore `s²(1 + 1/N)`.
    ///
    /// The `1 + 1/N` matters because `x̄` is not the true mean — the fitted
    /// constant is itself off by `O(σ/√N)`, and an error bound that ignored
    /// that would under-report. The authors' code instead divides the squared
    /// deviations by `N`, which understates `‖g − ∇L(f)‖²` by a factor
    /// `(N−1)/(N+1)` — 29% at their `N = 8`. (It has a compensating virtue: the
    /// same `1/N` bias appears with the opposite sign in `‖g‖²`, so that choice
    /// makes [`crate::tasks::fit::L2Fit::loss_from_gradient`] exact in
    /// expectation. Algorithm 1's test needs `‖g − ∇L(f)‖` on its own, so this
    /// estimates that instead.)
    ///
    /// Fewer than two samples gives no information about the spread, and the
    /// error density is reported as `+∞` rather than as zero — an unmeasured
    /// error is not an absent one, and reporting zero would let Algorithm 1
    /// accept any representation at all.
    pub fn from_samples(samples: &[f64]) -> Self {
        let count = samples.len();
        if count == 0 {
            return Self {
                mean: 0.0,
                error_density: f64::INFINITY,
            };
        }
        let mean = samples.iter().sum::<f64>() / count as f64;
        if count == 1 {
            return Self {
                mean,
                error_density: f64::INFINITY,
            };
        }
        let deviations: f64 = samples.iter().map(|s| (s - mean) * (s - mean)).sum();
        Self {
            mean,
            error_density: unbiased_error_density(deviations, count, 1),
        }
    }

    /// The same estimator from running sums, for callers that cannot keep the
    /// samples — the radiance task accumulates over views inside a hot loop.
    pub fn from_moments(sum: f64, sum_of_squares: f64, count: usize) -> Self {
        if count == 0 {
            return Self {
                mean: 0.0,
                error_density: f64::INFINITY,
            };
        }
        let mean = sum / count as f64;
        let deviations = (sum_of_squares - sum * mean).max(0.0);
        Self {
            mean,
            error_density: unbiased_error_density(deviations, count, 1),
        }
    }
}

impl<const D: usize> DyadicGrid<D> {
    /// A zero grid with `2^level` cells per axis over `[lower, upper]`.
    pub fn zeros(lower: [f64; D], upper: [f64; D], level: u32) -> Self {
        let cells = 1usize << level;
        Self {
            lower,
            upper,
            level,
            values: vec![0.0; cells.pow(D as u32)],
        }
    }

    /// A zero grid over the unit box `[0, 1]^D`.
    pub fn unit(level: u32) -> Self {
        Self::zeros([0.0; D], [1.0; D], level)
    }

    /// Cells per axis, `2^level`.
    pub fn cells_per_axis(&self) -> usize {
        1usize << self.level
    }

    /// Refinement level.
    pub fn level(&self) -> u32 {
        self.level
    }

    /// The box's lower corner.
    pub fn lower(&self) -> [f64; D] {
        self.lower
    }

    /// The box's upper corner.
    pub fn upper(&self) -> [f64; D] {
        self.upper
    }

    /// Cell values, row-major with the last axis fastest.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Mutable cell values.
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// The width of one cell along each axis.
    pub fn cell_widths(&self) -> [f64; D] {
        let cells = self.cells_per_axis() as f64;
        let mut widths = [0.0; D];
        for axis in 0..D {
            widths[axis] = (self.upper[axis] - self.lower[axis]) / cells;
        }
        widths
    }

    /// The volume of one cell.
    pub fn cell_volume(&self) -> f64 {
        self.cell_widths().iter().product()
    }

    /// Half the length of a cell's diagonal — the radius used by `L^∞`
    /// Lipschitz error bounds.
    pub fn cell_radius(&self) -> f64 {
        0.5 * self.cell_widths().iter().map(|w| w * w).sum::<f64>().sqrt()
    }

    /// The multi-index of cell `flat`.
    pub fn cell_index(&self, flat: usize) -> [usize; D] {
        let cells = self.cells_per_axis();
        let mut index = [0usize; D];
        let mut rest = flat;
        for axis in (0..D).rev() {
            index[axis] = rest % cells;
            rest /= cells;
        }
        index
    }

    /// The lower corner of cell `flat`.
    pub fn cell_origin(&self, flat: usize) -> [f64; D] {
        let index = self.cell_index(flat);
        let widths = self.cell_widths();
        let mut origin = [0.0; D];
        for axis in 0..D {
            origin[axis] = self.lower[axis] + index[axis] as f64 * widths[axis];
        }
        origin
    }

    /// The centre of cell `flat`.
    pub fn cell_center(&self, flat: usize) -> [f64; D] {
        let origin = self.cell_origin(flat);
        let widths = self.cell_widths();
        let mut center = [0.0; D];
        for axis in 0..D {
            center[axis] = origin[axis] + 0.5 * widths[axis];
        }
        center
    }

    /// The value of the encoded function at `x`, clamped to the box.
    pub fn eval(&self, x: [f64; D]) -> f64 {
        self.values[self.flat_index(x)]
    }

    /// The flat index of the cell containing `x`, clamped to the box.
    pub fn flat_index(&self, x: [f64; D]) -> usize {
        let cells = self.cells_per_axis();
        let mut flat = 0usize;
        for axis in 0..D {
            let span = self.upper[axis] - self.lower[axis];
            let t = (x[axis] - self.lower[axis]) / span;
            let i = (t * cells as f64).floor();
            let i = if i.is_nan() {
                0
            } else {
                i.clamp(0.0, (cells - 1) as f64) as usize
            };
            flat = flat * cells + i;
        }
        flat
    }

    /// `‖f‖²_{L²}` — the sum of squared cell values times the cell volume.
    pub fn l2_norm_squared(&self) -> f64 {
        self.cell_volume() * self.values.iter().map(|v| v * v).sum::<f64>()
    }

    /// `‖f‖_{L²}`.
    pub fn l2_norm(&self) -> f64 {
        self.l2_norm_squared().sqrt()
    }

    /// `‖f‖_{L^∞}` — the largest absolute cell value.
    pub fn sup_norm(&self) -> f64 {
        self.values.iter().fold(0.0f64, |a, v| a.max(v.abs()))
    }

    /// Fill every cell from `fit`, returning the total squared `L²` error,
    /// `Σ_cells vol · error_density`.
    ///
    /// This is the per-step work of Algorithm 1's `g_t ← approximate ∇L(f_t)`
    /// together with its `U_t`, for an `L²` problem.
    pub fn fill_from<G: FnMut(usize) -> CellFit>(&mut self, mut fit: G) -> f64 {
        let volume = self.cell_volume();
        let mut error_squared = 0.0;
        for flat in 0..self.values.len() {
            let cell = fit(flat);
            self.values[flat] = cell.mean;
            error_squared += volume * cell.error_density;
        }
        error_squared
    }

    /// Double the resolution along every axis, copying each parent value into
    /// its `2^D` children. The encoded function is unchanged.
    pub fn refined(&self) -> Self {
        let coarse = self.cells_per_axis();
        let fine = coarse * 2;
        let mut values = vec![0.0; fine.pow(D as u32)];
        for (flat, value) in values.iter_mut().enumerate() {
            // Map the fine multi-index back to its parent by halving each axis.
            let mut rest = flat;
            let mut parent = 0usize;
            let mut axis_index = [0usize; D];
            for axis in (0..D).rev() {
                axis_index[axis] = rest % fine;
                rest /= fine;
            }
            for axis in 0..D {
                parent = parent * coarse + axis_index[axis] / 2;
            }
            *value = self.values[parent];
        }
        Self {
            lower: self.lower,
            upper: self.upper,
            level: self.level + 1,
            values,
        }
    }

    /// `self ← self − η·gradient`.
    pub fn descend_by(&mut self, learning_rate: f64, gradient: &Self) {
        assert_eq!(
            self.values.len(),
            gradient.values.len(),
            "gradient must be at the iterate's refinement level"
        );
        for (value, g) in self.values.iter_mut().zip(&gradient.values) {
            *value -= learning_rate * g;
        }
    }

    /// Sample the encoded function on a uniform `resolution^D` grid of cell
    /// centres, row-major. Used for rendering and for error metrics against a
    /// reference solution.
    pub fn sample_uniform(&self, resolution: usize) -> Vec<f64> {
        let total = resolution.pow(D as u32);
        let mut out = Vec::with_capacity(total);
        for flat in 0..total {
            let mut rest = flat;
            let mut index = [0usize; D];
            for axis in (0..D).rev() {
                index[axis] = rest % resolution;
                rest /= resolution;
            }
            let mut x = [0.0; D];
            for axis in 0..D {
                let span = self.upper[axis] - self.lower[axis];
                x[axis] = self.lower[axis] + span * (index[axis] as f64 + 0.5) / resolution as f64;
            }
            out.push(self.eval(x));
        }
        out
    }
}

impl<const D: usize> crate::algorithm::Representation for DyadicGrid<D> {
    fn dofs(&self) -> usize {
        self.values.len()
    }

    fn level(&self) -> u32 {
        self.level
    }

    fn refine(&self) -> Self {
        self.refined()
    }

    fn descend(&mut self, learning_rate: f64, gradient: &Self) {
        self.descend_by(learning_rate, gradient);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::Representation;

    /// Every cell's centre has to land back in that cell. `cell_origin`,
    /// `cell_index` and `flat_index` each walk the row-major layout
    /// independently, so an axis-order slip between them would misplace values
    /// without changing any shape.
    #[test]
    fn cell_indexing_round_trips_in_two_and_three_dimensions() {
        let two = DyadicGrid::<2>::zeros([-1.0, 2.0], [3.0, 5.0], 3);
        for flat in 0..two.dofs() {
            assert_eq!(two.flat_index(two.cell_center(flat)), flat);
        }
        let three = DyadicGrid::<3>::zeros([0.0; 3], [1.0, 2.0, 4.0], 2);
        for flat in 0..three.dofs() {
            assert_eq!(three.flat_index(three.cell_center(flat)), flat);
        }
    }

    #[test]
    fn refinement_preserves_values_in_three_dimensions() {
        let mut grid = DyadicGrid::<3>::zeros([0.0; 3], [1.0, 2.0, 4.0], 2);
        for (i, value) in grid.values_mut().iter_mut().enumerate() {
            *value = i as f64;
        }
        let fine = grid.refined();
        assert_eq!(fine.dofs(), 8 * grid.dofs());
        for flat in 0..fine.dofs() {
            let centre = fine.cell_center(flat);
            assert_eq!(fine.eval(centre), grid.eval(centre));
        }
    }

    #[test]
    fn evaluation_outside_the_box_clamps_to_the_nearest_cell() {
        let mut grid = DyadicGrid::<2>::unit(1);
        grid.values_mut().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(grid.eval([-5.0, -5.0]), 1.0);
        assert_eq!(grid.eval([5.0, 5.0]), 4.0);
        assert_eq!(grid.eval([f64::NAN, 0.9]), 2.0);
    }

    #[test]
    fn norms_account_for_cell_volume() {
        let mut grid = DyadicGrid::<2>::zeros([0.0, 0.0], [2.0, 2.0], 1);
        grid.values_mut().copy_from_slice(&[1.0, -3.0, 2.0, 0.0]);
        // Four cells of area 1 each.
        assert!((grid.cell_volume() - 1.0).abs() < 1e-15);
        assert!((grid.l2_norm_squared() - 14.0).abs() < 1e-12);
        assert!((grid.sup_norm() - 3.0).abs() < 1e-15);
        // Half the diagonal of a 1×1 cell.
        assert!((grid.cell_radius() - 0.5 * 2f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn cell_fit_reports_an_unmeasured_spread_as_infinite() {
        assert_eq!(CellFit::from_samples(&[]).error_density, f64::INFINITY);
        assert_eq!(CellFit::from_samples(&[4.0]).error_density, f64::INFINITY);
        assert_eq!(CellFit::from_samples(&[4.0]).mean, 4.0);
        // Constant samples say the cell is exactly representable.
        let flat = CellFit::from_samples(&[2.0; 8]);
        assert_eq!(flat.mean, 2.0);
        assert!(flat.error_density.abs() < 1e-15);
    }

    /// `s²(1 + 1/N)` has to be unbiased for the true squared error density
    /// `(x̄ − μ)² + σ²`, or Algorithm 1's test is calibrated against the wrong
    /// number.
    #[test]
    fn cell_fit_error_density_is_unbiased() {
        let mut rng = crate::algorithm::Rng::new(99);
        let (mu, sigma) = (0.3, 0.7);
        let n = 8;
        let trials = 400_000;
        let mut estimated = 0.0;
        let mut actual = 0.0;
        for _ in 0..trials {
            // Uniform samples, whose variance is known exactly.
            let samples: Vec<f64> = (0..n)
                .map(|_| mu + sigma * 12f64.sqrt() * (rng.uniform01() - 0.5))
                .collect();
            let fit = CellFit::from_samples(&samples);
            estimated += fit.error_density;
            actual += (fit.mean - mu) * (fit.mean - mu) + sigma * sigma;
        }
        let estimated = estimated / trials as f64;
        let actual = actual / trials as f64;
        assert!(
            (estimated / actual - 1.0).abs() < 0.01,
            "estimator averaged {estimated:.6} against a true {actual:.6}"
        );
        // The authors' ÷N form would sit at (N−1)/(N+1) of this.
        assert!(estimated > sigma * sigma);
    }

    #[test]
    fn sample_uniform_walks_the_same_layout_as_the_cells() {
        let mut grid = DyadicGrid::<2>::unit(2);
        for (i, value) in grid.values_mut().iter_mut().enumerate() {
            *value = i as f64;
        }
        let sampled = grid.sample_uniform(4);
        assert_eq!(sampled, grid.values());
    }
}
