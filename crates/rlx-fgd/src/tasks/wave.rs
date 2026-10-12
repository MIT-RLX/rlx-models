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

//! Solving the wave equation — Section 4.2 and Appendix A.1.3.
//!
//! The PDE `∂²f/∂t² = c²Δf` with `f(0,·) = h`, `∂_t f(0,·) = 0` is recast as
//! minimising
//!
//! ```text
//! L_PDE(f) = ½∫|∂²_t f − c²Δf|² + ½∫|f(0,·) − h|² + ½∫|∂_t f(0,·)|²
//! ```
//!
//! over `H = H²(R³)`, which vanishes exactly on solutions (Equation 9). Nothing
//! about that is a neural network: there is no parametrisation to fit, only a
//! function to descend on.
//!
//! Sobolev gradients are hard to write down, but Proposition 4.2 gives the
//! gradient's *Fourier transform* in closed form, so the whole task runs in
//! frequency space on a [`crate::rep::FourierGrid`]. Each of the three loss
//! terms transforms into something local: the PDE residual into a pointwise
//! multiplier `(τ² − c²(ξ²+ζ²))²`, and the two initial conditions into the `τ`
//! marginals the grid already caches.
//!
//! Following the authors' code, frequencies use the `e^{2πiω·x}` convention, in
//! which the `2π` factors of Proposition 4.2's statement are absorbed into the
//! transform and do not appear here.
//!
//! One consequence worth knowing when reading a run's output: starting from
//! `f̂ = 0`, the velocity term stays at zero to machine precision for the rest
//! of the optimisation. That is not a term that failed to engage. The Sobolev
//! symbol and the PDE multiplier are both even in `τ`, and the displacement
//! term is `τ`-independent, so every iterate stays even in `τ` — which is
//! exactly `f(t) = f(−t)`, the symmetry `∂_t f(0,·) = 0` demands. The condition
//! holds because the geometry of the problem enforces it, not because it is
//! being descended on.

use crate::algorithm::{Functional, GradientApprox, Rng};
use crate::rep::complex::{Complex, sinc};
use crate::rep::{FourierGrid, sobolev_symbol};

/// An initial displacement `h`, supplied through its Fourier transform.
pub trait InitialCondition {
    /// `ĥ(ξ, ζ)`.
    fn spectrum(&self, xi: f64, zeta: f64) -> Complex;

    /// `h(x, y)`, for plotting and for round-trip checks of the transform.
    fn spatial(&self, x: f64, y: f64) -> f64;
}

/// The quadratic B-spline `B₂`, supported on `[−3/2, 3/2]` — the Fourier
/// transform of `sinc³`.
pub fn bspline2(t: f64) -> f64 {
    if t <= -1.5 {
        0.0
    } else if t <= -0.5 {
        0.5 * (t + 1.5) * (t + 1.5)
    } else if t <= 0.5 {
        0.75 - t * t
    } else if t <= 1.5 {
        0.5 * (t - 1.5) * (t - 1.5)
    } else {
        0.0
    }
}

/// Section 4.2's initial displacement: a sum of shifted `sinc³` bumps,
/// `h(x,y) = Σ_p sinc³(s·x + p₀)·sinc³(s·y + p₁)`.
///
/// Chosen because its transform is exactly a shifted B-spline — compactly
/// supported in frequency — so the whole initial condition is resolved by a
/// finite frequency box and the only approximation left is the one Algorithm 1
/// is measuring.
#[derive(Debug, Clone, PartialEq)]
pub struct SincCubeBumps {
    /// Bump offsets `p`.
    pub positions: Vec<[f64; 2]>,
    /// Spatial dilation `s`.
    pub scale: f64,
}

impl SincCubeBumps {
    /// `count` bumps at reproducible pseudo-random offsets in `[−3, 3]²`,
    /// dilated by `scale` — the authors' configuration at `count = 3`,
    /// `scale = 2`.
    pub fn sampled(count: usize, scale: f64, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let positions = (0..count)
            .map(|_| [-3.0 + 6.0 * rng.uniform01(), -3.0 + 6.0 * rng.uniform01()])
            .collect();
        Self { positions, scale }
    }
}

impl InitialCondition for SincCubeBumps {
    fn spectrum(&self, xi: f64, zeta: f64) -> Complex {
        // h(x) = g(s·x) ⇒ ĥ(ξ) = ĝ(ξ/s)/s, per axis; a spatial shift by p
        // becomes the modulation e^{2πi p ξ}.
        let u = xi / self.scale;
        let v = zeta / self.scale;
        let envelope = bspline2(u) * bspline2(v) / (self.scale * self.scale);
        let two_pi = std::f64::consts::TAU;
        self.positions
            .iter()
            .fold(Complex::ZERO, |acc, p| {
                acc + Complex::from_angle(two_pi * (p[0] * u + p[1] * v))
            })
            .scale(envelope)
    }

    fn spatial(&self, x: f64, y: f64) -> f64 {
        self.positions
            .iter()
            .map(|p| sinc(self.scale * x + p[0]).powi(3) * sinc(self.scale * y + p[1]).powi(3))
            .sum()
    }
}

/// Default cells per axis of the quadrature that reports `L_PDE`.
pub const LOSS_QUADRATURE: usize = 32;

/// `L_PDE` over frequency-space grids.
#[derive(Debug, Clone)]
pub struct WaveEquation<H> {
    /// Wave speed `c`.
    pub speed: f64,
    /// The initial displacement `h`.
    pub initial: H,
    /// Monte-Carlo samples per frequency cell used to estimate `U_t`.
    pub samples_per_cell: usize,
    /// Cells per axis of the quadrature that reports `L_PDE`.
    ///
    /// Pinned for the whole run rather than taken from the iterate's own grid.
    /// Two of the three terms of Equation 9 involve `ĥ`, which the quadrature
    /// samples; a quadrature that followed the refinement level would resolve
    /// `ĥ` better after each refinement and so report a *larger* loss for an
    /// unchanged iterate. That is a change of measuring stick, not of solution
    /// quality, and it makes every loss comparison across a refinement
    /// meaningless. Should be at least as fine as the finest grid a run
    /// reaches.
    pub loss_quadrature: usize,
}

/// The three terms of Equation 9, reported separately.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossTerms {
    /// `½∫|∂²_t f − c²Δf|²` — how badly the PDE itself is violated.
    pub residual: f64,
    /// `½∫|f(0,·) − h|²` — the displacement initial condition.
    pub displacement: f64,
    /// `½∫|∂_t f(0,·)|²` — the zero-velocity initial condition.
    pub velocity: f64,
}

impl LossTerms {
    /// `L_PDE(f)`.
    pub fn total(&self) -> f64 {
        self.residual + self.displacement + self.velocity
    }
}

impl<H: InitialCondition> WaveEquation<H> {
    /// A wave problem with speed `c` and initial displacement `h`.
    pub fn new(speed: f64, initial: H) -> Self {
        Self {
            speed,
            initial,
            samples_per_cell: 8,
            loss_quadrature: LOSS_QUADRATURE,
        }
    }

    /// A wave problem whose loss quadrature resolves grids up to `max_level`.
    pub fn with_max_level(speed: f64, initial: H, max_level: u32) -> Self {
        Self {
            loss_quadrature: LOSS_QUADRATURE.max(1usize << max_level),
            ..Self::new(speed, initial)
        }
    }

    /// Whether the loss quadrature is at least as fine as `f`'s grid.
    pub fn quadrature_resolves(&self, f: &FourierGrid) -> bool {
        self.loss_quadrature >= f.cells_per_axis()
    }

    /// `[∇L_PDE(f)]^(ω)` at one frequency (Proposition 4.2).
    pub fn gradient_at(&self, f: &FourierGrid, omega: [f64; 3]) -> Complex {
        let [tau, xi, zeta] = omega;
        let dispersion = tau * tau - self.speed * self.speed * (xi * xi + zeta * zeta);
        let pde = f.eval(omega).scale(dispersion * dispersion);
        let displacement = f.integral_tau(xi, zeta) - self.initial.spectrum(xi, zeta);
        let velocity = f.integral_tau_weighted(xi, zeta).scale(tau);
        let omega_squared = tau * tau + xi * xi + zeta * zeta;
        // Dividing by the H² symbol is the Riesz map: it turns the L² residual
        // above into the gradient with respect to the H² inner product.
        (pde + displacement + velocity).scale(1.0 / sobolev_symbol(omega_squared))
    }

    /// `L_PDE(f)` term by term, by midpoint quadrature over the frequency box.
    ///
    /// Plancherel makes each term a weighted sum over the box: the PDE residual
    /// over the volume, and the two initial conditions over the `(ξ, ζ)` plane
    /// through the `τ` marginals. Both `f̂` and its marginals are piecewise
    /// constant, so they are read by lookup and the quadrature resolution is
    /// free to be [`WaveEquation::loss_quadrature`] rather than the iterate's.
    pub fn loss_terms(&self, f: &FourierGrid) -> LossTerms {
        let n = self.loss_quadrature;
        let (lower, upper) = f.bounds();
        let width = (upper - lower) / n as f64;
        let volume = width * width * width;
        let area = width * width;
        let center = |i: usize| lower + (i as f64 + 0.5) * width;

        let mut residual = 0.0;
        let mut displacement = 0.0;
        let mut velocity = 0.0;
        for j in 0..n {
            let xi = center(j);
            for k in 0..n {
                let zeta = center(k);
                for i in 0..n {
                    let tau = center(i);
                    let dispersion = tau * tau - self.speed * self.speed * (xi * xi + zeta * zeta);
                    residual +=
                        volume * dispersion * dispersion * f.eval([tau, xi, zeta]).norm_squared();
                }
                let gap = f.integral_tau(xi, zeta) - self.initial.spectrum(xi, zeta);
                displacement += area * gap.norm_squared();
                velocity += area * f.integral_tau_weighted(xi, zeta).norm_squared();
            }
        }

        LossTerms {
            residual: 0.5 * residual,
            displacement: 0.5 * displacement,
            velocity: 0.5 * velocity,
        }
    }

    /// The offsets in `[0,1)³` shared by every cell of one attempt.
    fn offsets(&self, seed: u64) -> Vec<[f64; 3]> {
        let mut rng = Rng::new(seed);
        (0..self.samples_per_cell)
            .map(|_| [rng.uniform01(), rng.uniform01(), rng.uniform01()])
            .collect()
    }
}

impl<H: InitialCondition> Functional for WaveEquation<H> {
    type Rep = FourierGrid;
    /// Proposition 4.2 reads `f̂` and its marginals directly; both are already
    /// stored, so there is nothing to precompute per step.
    type StepState = ();

    fn begin_step(&self, _f: &Self::Rep) -> Self::StepState {}

    fn approximate_gradient(
        &self,
        f: &Self::Rep,
        _state: &Self::StepState,
        seed: u64,
    ) -> GradientApprox<Self::Rep> {
        let mut gradient = f.clone();
        gradient.fill_from(|flat| self.gradient_at(f, f.cell_center(flat)));

        // Both the norm and the error are measured in H², on the same sample
        // points, so their ratio is a like-for-like relative error even though
        // each is only a quadrature estimate.
        let offsets = self.offsets(seed);
        let width = f.cell_width();
        let volume = f.cell_volume();
        let count = offsets.len() as f64;
        let mut norm_squared = 0.0;
        let mut error_squared = 0.0;

        for flat in 0..f.grid().len() {
            let origin = f.cell_origin(flat);
            // The approximation is constant across this cell by construction.
            let fit = gradient.grid()[flat];
            let mut cell_norm = 0.0;
            let mut cell_error = 0.0;
            for offset in &offsets {
                let point = [
                    origin[0] + offset[0] * width,
                    origin[1] + offset[1] * width,
                    origin[2] + offset[2] * width,
                ];
                let symbol =
                    sobolev_symbol(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
                let exact = self.gradient_at(f, point);
                cell_norm += symbol * fit.norm_squared();
                cell_error += symbol * (fit - exact).norm_squared();
            }
            norm_squared += volume * cell_norm / count;
            error_squared += volume * cell_error / count;
        }

        GradientApprox {
            gradient,
            norm: norm_squared.sqrt(),
            error_bound: error_squared.sqrt(),
        }
    }

    fn loss(&self, f: &Self::Rep) -> Option<f64> {
        Some(self.loss_terms(f).total())
    }
}
