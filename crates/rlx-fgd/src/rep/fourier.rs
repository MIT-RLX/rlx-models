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

//! Frequency-space grids — the `H²(R³)` representation of Section 4.2.
//!
//! Functional gradients in a Sobolev space are awkward to compute directly, but
//! Proposition 4.2 gives the gradient's *Fourier transform* in closed form. So
//! the representation stores `f̂` rather than `f`: a piecewise-constant complex
//! grid over a frequency box `[a, b]³` in `(τ, ξ, ζ)`. The function itself is
//! recovered by [`FourierGrid::evaluate_spatial_grid`], which integrates each
//! frequency bin exactly — a box in frequency transforms to a `sinc` envelope
//! times a carrier, so no FFT and no periodic wrap-around are involved.
//!
//! Proposition 4.2's gradient needs two marginals of `f̂` along `τ`,
//! `∫f̂(τ',ξ,ζ)dτ'` and `∫τ'f̂(τ',ξ,ζ)dτ'`, at every evaluation. Recomputing
//! them per point would make each gradient evaluation `O(n_t)`; they are cached
//! alongside the grid instead. Both are linear in the grid, so they survive the
//! descent step `f ← f − ηg` and the repeat-into-children refinement without
//! needing to be rebuilt — an invariant [`FourierGrid::marginals_are_consistent`]
//! checks.
//!
//! The `H²` norm is diagonal in frequency, weighted by the Sobolev symbol
//! [`sobolev_symbol`]; that makes both `‖g‖_B` and the error bound `U` of
//! Equation 5 straightforward sums over cells.

use crate::rep::complex::{Complex, sinc};

/// The `H²(R³)` Fourier symbol `1 + |ω|² + |ω|⁴`, with `|ω|² = τ² + ξ² + ζ²`.
///
/// Proposition 4.2 divides by exactly this to turn an `L²` residual into an
/// `H²` gradient, and the same weight makes `‖·‖_{H²}` a plain weighted sum over
/// frequency cells. (The authors' reference code writes the *gradient's*
/// denominator as `(1 + |ω|²)²` instead, which differs in the cross term; the
/// paper's form is used here because it is the one that matches the `H²` inner
/// product the norm below also uses.)
pub fn sobolev_symbol(omega_squared: f64) -> f64 {
    1.0 + omega_squared + omega_squared * omega_squared
}

/// A piecewise-constant complex grid over a frequency box `[lower, upper]³`.
#[derive(Debug, Clone, PartialEq)]
pub struct FourierGrid {
    lower: f64,
    upper: f64,
    level: u32,
    /// `f̂` on the `(τ, ξ, ζ)` grid, row-major with `ζ` fastest.
    grid: Vec<Complex>,
    /// `∫ f̂(τ', ξ, ζ) dτ'`, one entry per `(ξ, ζ)`.
    integral_tau: Vec<Complex>,
    /// `∫ τ' f̂(τ', ξ, ζ) dτ'`, one entry per `(ξ, ζ)`.
    integral_tau_weighted: Vec<Complex>,
}

impl FourierGrid {
    /// A zero spectrum with `2^level` cells per axis over `[lower, upper]³`.
    pub fn zeros(lower: f64, upper: f64, level: u32) -> Self {
        assert!(upper > lower, "frequency box must be non-degenerate");
        let n = 1usize << level;
        Self {
            lower,
            upper,
            level,
            grid: vec![Complex::ZERO; n * n * n],
            integral_tau: vec![Complex::ZERO; n * n],
            integral_tau_weighted: vec![Complex::ZERO; n * n],
        }
    }

    /// Cells per axis, `2^level`.
    pub fn cells_per_axis(&self) -> usize {
        1usize << self.level
    }

    /// Refinement level.
    pub fn level(&self) -> u32 {
        self.level
    }

    /// The frequency box bounds, the same on all three axes.
    pub fn bounds(&self) -> (f64, f64) {
        (self.lower, self.upper)
    }

    /// The width of one frequency cell along each axis.
    pub fn cell_width(&self) -> f64 {
        (self.upper - self.lower) / self.cells_per_axis() as f64
    }

    /// The volume of one frequency cell.
    pub fn cell_volume(&self) -> f64 {
        self.cell_width().powi(3)
    }

    /// The spectrum values.
    pub fn grid(&self) -> &[Complex] {
        &self.grid
    }

    /// The `(τ, ξ, ζ)` index of flat index `flat`.
    pub fn cell_index(&self, flat: usize) -> [usize; 3] {
        let n = self.cells_per_axis();
        [flat / (n * n), (flat / n) % n, flat % n]
    }

    /// The lower corner of cell `flat` in frequency space.
    pub fn cell_origin(&self, flat: usize) -> [f64; 3] {
        let index = self.cell_index(flat);
        let width = self.cell_width();
        [
            self.lower + index[0] as f64 * width,
            self.lower + index[1] as f64 * width,
            self.lower + index[2] as f64 * width,
        ]
    }

    /// The centre of cell `flat` in frequency space.
    pub fn cell_center(&self, flat: usize) -> [f64; 3] {
        let origin = self.cell_origin(flat);
        let half = 0.5 * self.cell_width();
        [origin[0] + half, origin[1] + half, origin[2] + half]
    }

    /// `f̂(ω)`, and **zero outside the frequency box** — the spectrum is only
    /// modelled on `[lower, upper]³`, so extrapolating a cell value past the
    /// boundary would invent energy the representation never resolved.
    pub fn eval(&self, omega: [f64; 3]) -> Complex {
        match self.try_flat_index(omega) {
            Some(flat) => self.grid[flat],
            None => Complex::ZERO,
        }
    }

    /// `∫ f̂(τ', ξ, ζ) dτ'` at `(ξ, ζ)`, clamped to the box.
    pub fn integral_tau(&self, xi: f64, zeta: f64) -> Complex {
        self.integral_tau[self.clamped_plane_index(xi, zeta)]
    }

    /// `∫ τ' f̂(τ', ξ, ζ) dτ'` at `(ξ, ζ)`, clamped to the box.
    pub fn integral_tau_weighted(&self, xi: f64, zeta: f64) -> Complex {
        self.integral_tau_weighted[self.clamped_plane_index(xi, zeta)]
    }

    fn try_flat_index(&self, omega: [f64; 3]) -> Option<usize> {
        let n = self.cells_per_axis();
        let span = self.upper - self.lower;
        let mut flat = 0usize;
        for value in omega {
            let t = (value - self.lower) / span * n as f64;
            if t.is_nan() || t < 0.0 || t >= n as f64 {
                return None;
            }
            flat = flat * n + t.floor() as usize;
        }
        Some(flat)
    }

    fn clamped_plane_index(&self, xi: f64, zeta: f64) -> usize {
        let n = self.cells_per_axis();
        let span = self.upper - self.lower;
        let axis = |value: f64| {
            let t = (value - self.lower) / span * n as f64;
            if t.is_nan() {
                0
            } else {
                t.floor().clamp(0.0, (n - 1) as f64) as usize
            }
        };
        axis(xi) * n + axis(zeta)
    }

    /// Overwrite the spectrum from `value(flat)` and rebuild both `τ` marginals.
    ///
    /// The marginals use the midpoint rule over `τ` cells: `∫f̂ dτ' ≈ Σ f̂ · Δτ`,
    /// and `∫τ'f̂ dτ' ≈ Σ f̂ · (τ²_{i+1} − τ²_i)/2`, which integrates the weight
    /// `τ'` exactly across each cell rather than sampling it at the centre.
    pub fn fill_from<G: FnMut(usize) -> Complex>(&mut self, mut value: G) {
        for flat in 0..self.grid.len() {
            self.grid[flat] = value(flat);
        }
        self.rebuild_marginals();
    }

    /// Recompute both `τ` marginals from the stored spectrum.
    pub fn rebuild_marginals(&mut self) {
        let n = self.cells_per_axis();
        let width = self.cell_width();
        self.integral_tau
            .iter_mut()
            .for_each(|v| *v = Complex::ZERO);
        self.integral_tau_weighted
            .iter_mut()
            .for_each(|v| *v = Complex::ZERO);
        for i in 0..n {
            let tau_lo = self.lower + i as f64 * width;
            let tau_hi = tau_lo + width;
            let weight = 0.5 * (tau_hi * tau_hi - tau_lo * tau_lo);
            for plane in 0..n * n {
                let cell = self.grid[i * n * n + plane];
                self.integral_tau[plane] += cell.scale(width);
                self.integral_tau_weighted[plane] += cell.scale(weight);
            }
        }
    }

    /// Whether the cached marginals still match the stored spectrum.
    ///
    /// Both are linear in the grid, so refinement and the descent step preserve
    /// them; this is the check that says so.
    pub fn marginals_are_consistent(&self, tolerance: f64) -> bool {
        let mut rebuilt = self.clone();
        rebuilt.rebuild_marginals();
        self.integral_tau
            .iter()
            .zip(&rebuilt.integral_tau)
            .chain(
                self.integral_tau_weighted
                    .iter()
                    .zip(&rebuilt.integral_tau_weighted),
            )
            .all(|(a, b)| (*a - *b).abs() <= tolerance)
    }

    /// `‖f‖²_{H²}` — `Σ_cells vol · symbol(|ω|²) · |f̂|²` at cell centres.
    pub fn sobolev_norm_squared(&self) -> f64 {
        let volume = self.cell_volume();
        (0..self.grid.len())
            .map(|flat| {
                let c = self.cell_center(flat);
                let omega2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
                volume * sobolev_symbol(omega2) * self.grid[flat].norm_squared()
            })
            .sum()
    }

    /// `‖f‖_{H²}`.
    pub fn sobolev_norm(&self) -> f64 {
        self.sobolev_norm_squared().sqrt()
    }

    /// Double the resolution on every axis, copying each parent value into its
    /// children. The encoded spectrum, and both marginals, are unchanged.
    pub fn refined(&self) -> Self {
        let coarse = self.cells_per_axis();
        let fine = coarse * 2;
        let mut grid = vec![Complex::ZERO; fine * fine * fine];
        for (flat, cell) in grid.iter_mut().enumerate() {
            let i = flat / (fine * fine);
            let j = (flat / fine) % fine;
            let k = flat % fine;
            *cell = self.grid[(i / 2) * coarse * coarse + (j / 2) * coarse + (k / 2)];
        }
        let plane = |source: &[Complex]| {
            let mut out = vec![Complex::ZERO; fine * fine];
            for (flat, cell) in out.iter_mut().enumerate() {
                let j = flat / fine;
                let k = flat % fine;
                *cell = source[(j / 2) * coarse + (k / 2)];
            }
            out
        };
        Self {
            lower: self.lower,
            upper: self.upper,
            level: self.level + 1,
            integral_tau: plane(&self.integral_tau),
            integral_tau_weighted: plane(&self.integral_tau_weighted),
            grid,
        }
    }

    /// `self ← self − η·gradient`, on the spectrum and both marginals.
    pub fn descend_by(&mut self, learning_rate: f64, gradient: &Self) {
        assert_eq!(
            self.grid.len(),
            gradient.grid.len(),
            "gradient must be at the iterate's refinement level"
        );
        for (value, g) in self.grid.iter_mut().zip(&gradient.grid) {
            *value -= g.scale(learning_rate);
        }
        for (value, g) in self.integral_tau.iter_mut().zip(&gradient.integral_tau) {
            *value -= g.scale(learning_rate);
        }
        for (value, g) in self
            .integral_tau_weighted
            .iter_mut()
            .zip(&gradient.integral_tau_weighted)
        {
            *value -= g.scale(learning_rate);
        }
    }

    /// Evaluate `f` on the separable spatial grid `ts × xs × ys`, row-major.
    ///
    /// Each frequency cell contributes `f̂(centre)·Δ·sinc(Δ·t)·e^{2πi·centre·t}`
    /// per axis: the exact integral of a constant spectrum over a box, not a
    /// point sample of it. The three axes factor, so this contracts one axis at
    /// a time rather than forming the full six-index product.
    pub fn evaluate_spatial_grid(&self, ts: &[f64], xs: &[f64], ys: &[f64]) -> Vec<f64> {
        let n = self.cells_per_axis();
        let width = self.cell_width();
        let tau = |i: usize| self.lower + (i as f64 + 0.5) * width;
        let two_pi = std::f64::consts::TAU;

        // Contract τ: [i,j,k] × [i,m] → [m,j,k].
        let mut after_t = vec![Complex::ZERO; ts.len() * n * n];
        for i in 0..n {
            let center = tau(i);
            for (m, &t) in ts.iter().enumerate() {
                let phase = Complex::from_angle(two_pi * center * t);
                let row = &self.grid[i * n * n..(i + 1) * n * n];
                let out = &mut after_t[m * n * n..(m + 1) * n * n];
                for (o, g) in out.iter_mut().zip(row) {
                    *o += *g * phase;
                }
            }
        }

        // Contract ξ: [m,j,k] × [j,n] → [m,n,k].
        let mut after_x = vec![Complex::ZERO; ts.len() * xs.len() * n];
        for m in 0..ts.len() {
            for j in 0..n {
                let center = tau(j);
                for (nn, &x) in xs.iter().enumerate() {
                    let phase = Complex::from_angle(two_pi * center * x);
                    for k in 0..n {
                        after_x[(m * xs.len() + nn) * n + k] +=
                            after_t[(m * n + j) * n + k] * phase;
                    }
                }
            }
        }

        // Contract ζ: [m,n,k] × [k,p] → [m,n,p], then apply the envelopes.
        let mut out = vec![0.0; ts.len() * xs.len() * ys.len()];
        for m in 0..ts.len() {
            let env_t = width * sinc(width * ts[m]);
            for nn in 0..xs.len() {
                let env_x = width * sinc(width * xs[nn]);
                for (p, &y) in ys.iter().enumerate() {
                    let env_y = width * sinc(width * y);
                    let mut acc = Complex::ZERO;
                    for k in 0..n {
                        let center = tau(k);
                        acc += after_x[(m * xs.len() + nn) * n + k]
                            * Complex::from_angle(two_pi * center * y);
                    }
                    out[(m * xs.len() + nn) * ys.len() + p] = acc.re * env_t * env_x * env_y;
                }
            }
        }
        out
    }
}

impl crate::algorithm::Representation for FourierGrid {
    fn dofs(&self) -> usize {
        self.grid.len()
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

    fn constant(level: u32, value: f64) -> FourierGrid {
        let mut grid = FourierGrid::zeros(-5.0, 5.0, level);
        grid.fill_from(|_| Complex::real(value));
        grid
    }

    #[test]
    fn cell_indexing_round_trips() {
        let grid = FourierGrid::zeros(-3.0, 4.0, 3);
        for flat in 0..grid.dofs() {
            let centre = grid.cell_center(flat);
            let origin = grid.cell_origin(flat);
            for axis in 0..3 {
                assert!(centre[axis] > origin[axis]);
                assert!(centre[axis] < origin[axis] + grid.cell_width());
            }
        }
    }

    #[test]
    fn a_constant_spectrum_integrates_to_the_box_width() {
        let grid = constant(3, 1.0);
        for (xi, zeta) in [(-4.0, 0.0), (0.5, 3.9), (4.9, -4.9)] {
            // ∫ 1 dτ over [−5, 5].
            assert!((grid.integral_tau(xi, zeta).re - 10.0).abs() < 1e-12);
            assert!(grid.integral_tau(xi, zeta).im.abs() < 1e-15);
            // ∫ τ dτ over a symmetric interval vanishes.
            assert!(grid.integral_tau_weighted(xi, zeta).abs() < 1e-12);
        }
    }

    /// The `τ` weight is integrated across each cell rather than sampled at its
    /// centre, which is only visible on a spectrum that is not symmetric in `τ`.
    #[test]
    fn the_weighted_marginal_integrates_tau_exactly() {
        let n = 8usize;
        let mut grid = FourierGrid::zeros(-5.0, 5.0, 3);
        // One on the positive half of the τ axis, zero on the negative half.
        grid.fill_from(|flat| {
            if flat / (n * n) >= n / 2 {
                Complex::real(1.0)
            } else {
                Complex::ZERO
            }
        });
        // ∫₀⁵ τ dτ = 12.5, and ∫₀⁵ dτ = 5.
        assert!((grid.integral_tau_weighted(0.0, 0.0).re - 12.5).abs() < 1e-12);
        assert!((grid.integral_tau(0.0, 0.0).re - 5.0).abs() < 1e-12);
    }

    #[test]
    fn the_spectrum_is_zero_outside_the_box_and_the_marginals_clamp() {
        let grid = constant(2, 2.0);
        assert_eq!(grid.eval([0.0, 0.0, 0.0]), Complex::real(2.0));
        assert_eq!(grid.eval([6.0, 0.0, 0.0]), Complex::ZERO);
        assert_eq!(grid.eval([0.0, -5.001, 0.0]), Complex::ZERO);
        // Marginals clamp instead, since they stand for a plane the grid covers.
        assert!((grid.integral_tau(99.0, -99.0).re - 20.0).abs() < 1e-12);
    }

    /// At the origin the inverse transform is just the integral of the
    /// spectrum, which pins down the `Δ·sinc(Δ·x)` envelope's scale.
    #[test]
    fn the_inverse_transform_at_the_origin_is_the_spectrums_integral() {
        for level in 1..5 {
            let grid = constant(level, 1.0);
            let value = grid.evaluate_spatial_grid(&[0.0], &[0.0], &[0.0])[0];
            assert!(
                (value - 1000.0).abs() < 1e-9,
                "level {level}: got {value}, expected the box volume 10³"
            );
        }
    }

    #[test]
    fn the_sobolev_symbol_is_the_h2_weight() {
        assert_eq!(sobolev_symbol(0.0), 1.0);
        assert_eq!(sobolev_symbol(2.0), 1.0 + 2.0 + 4.0);
        // Strictly increasing, so higher frequencies are damped harder.
        assert!(sobolev_symbol(1.0) < sobolev_symbol(1.5));
    }

    #[test]
    fn the_sobolev_norm_scales_quadratically_with_the_spectrum() {
        let one = constant(3, 1.0);
        let three = constant(3, 3.0);
        assert!((three.sobolev_norm_squared() / one.sobolev_norm_squared() - 9.0).abs() < 1e-9);
        assert!(one.sobolev_norm() > 0.0);
    }
}
