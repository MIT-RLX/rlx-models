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

//! Convergence theory of approximate FGD, Section 3.1 of the paper.
//!
//! Everything here is a closed-form consequence of the paper's four assumptions;
//! none of it is needed to *run* [`crate::algorithm`], but it is what makes the
//! run meaningful, so the driver reports against it and the tests check the
//! guarantees empirically.
//!
//! | Quantity | Paper |
//! |----------|-------|
//! | [`relative_error`] | Equation 4 |
//! | [`Constants::descent_coefficient`] | Lemma 3.5 bracket, Proposition 3.6 `r` |
//! | [`Constants::descent_bound`] | Lemma 3.5 |
//! | [`Constants::max_relative_error`] | Proposition 3.6, `ε < min{1/2, α/(α+β)}` |
//! | [`Constants::max_learning_rate`] | Proposition 3.6, `η ≤ 2(α−(α+β)ε)/(K(2ε+1))` |
//! | [`Constants::stationary_bound`] | Proposition 3.6 |
//! | [`Constants::polyak_lojasiewicz_bound`] | Proposition 3.8 |
//! | [`guaranteed_relative_error`] | Theorem 3.10(i), `ε/(1+ε)` |

/// `RelErr(g, ∇L(f)) = ‖g − ∇L(f)‖_B / ‖g‖_B` (Equation 4).
///
/// The denominator is the norm of *our own* approximation, so it is computable
/// exactly; only the numerator needs the upper bound `U` of Equation 5. Follows
/// the paper's footnote convention that `0/0 = 0` and `a/0 = +∞` for `a > 0`.
pub fn relative_error(error: f64, gradient_norm: f64) -> f64 {
    if gradient_norm > 0.0 {
        error / gradient_norm
    } else if error > 0.0 {
        f64::INFINITY
    } else {
        0.0
    }
}

/// Theorem 3.10(i): running Algorithm 1 with tolerance `ε` guarantees
/// `RelErr(g_t, ∇L(f_t)) ≤ ε/(1+ε)` at every step.
///
/// The tolerance the *algorithm* takes and the relative error the *convergence
/// propositions* take are therefore different numbers; mixing them up silently
/// weakens every rate below. Use this to go from one to the other.
///
/// Theorem 3.10 is stated for `ε ∈ (0,1)`, which is what makes the resulting
/// bound smaller than the `1/2` Lemma 3.5 needs — see [`tolerance_in_scope`].
pub fn guaranteed_relative_error(tolerance: f64) -> f64 {
    tolerance / (1.0 + tolerance)
}

/// Whether `ε` is inside Theorem 3.10's stated range `(0,1)`.
///
/// Worth checking, because a loose tolerance is tempting: it refines rarely and
/// so runs fast. But at `ε ≥ 1` the guaranteed relative error reaches `1/2`,
/// which is the exact point where Lemma 3.5 stops applying, and past it
/// Algorithm 1 is still a perfectly reasonable heuristic with no theorem behind
/// it. (The authors' own `ε = 3` teaser and `ε = 50` wave-equation runs are in
/// that regime.)
pub fn tolerance_in_scope(tolerance: f64) -> bool {
    tolerance > 0.0 && tolerance < 1.0
}

/// Inverse of [`guaranteed_relative_error`]: the Algorithm 1 tolerance needed to
/// hold `RelErr` at `target`. Infinite for `target ≥ 1`.
pub fn tolerance_for_relative_error(target: f64) -> f64 {
    if target >= 1.0 {
        f64::INFINITY
    } else {
        target / (1.0 - target)
    }
}

/// The problem constants of Assumptions 3.2–3.4.
///
/// - `alpha` — Assumption 3.3, `H` descends in `B`: `DL(f; ∇L(f)) ≥ α‖∇L(f)‖²_B`.
/// - `beta` — Assumption 3.4, gradient compatibility: `‖DL(f;·)‖_{B*} ≤ β‖∇L(f)‖_B`.
/// - `smoothness` — `K` of Assumption 3.2.
///
/// When `B = H` the first two hold with `α = β = 1` ([`Constants::hilbert`]);
/// [`crate::tasks::rkhs`] is the interesting case where they do not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constants {
    /// `α > 0` of Assumption 3.3.
    pub alpha: f64,
    /// `β > 0` of Assumption 3.4.
    pub beta: f64,
    /// `K > 0` of Assumption 3.2.
    pub smoothness: f64,
}

impl Constants {
    /// The `B = H` case: `α = β = 1`, as noted under Assumption 3.4.
    pub fn hilbert(smoothness: f64) -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            smoothness,
        }
    }

    /// Proposition A.2 for `L_REG` over `B = L^∞`: `α = λ_min(K)/(nκ²)`,
    /// `β = n/λ_min(K)`, with `κ = sup|K(x,x')|` and `n` the sample count.
    pub fn rkhs_regression(
        samples: usize,
        min_kernel_eigenvalue: f64,
        kernel_bound: f64,
        smoothness: f64,
    ) -> Self {
        let n = samples as f64;
        Self {
            alpha: min_kernel_eigenvalue / (n * kernel_bound * kernel_bound),
            beta: n / min_kernel_eigenvalue,
            smoothness,
        }
    }

    /// The bracket of Lemma 3.5, `α − Kη/2 − (β + 3Kη/2)·e/(1−e)` with
    /// `e = RelErr`. This is Proposition 3.6's `r` when `e` is the uniform bound
    /// `ε`. Descent is guaranteed exactly while this is strictly positive.
    pub fn descent_coefficient(&self, learning_rate: f64, relative_error: f64) -> f64 {
        if relative_error >= 1.0 {
            return f64::NEG_INFINITY;
        }
        let ratio = relative_error / (1.0 - relative_error);
        self.alpha
            - self.smoothness * learning_rate / 2.0
            - (self.beta + 1.5 * self.smoothness * learning_rate) * ratio
    }

    /// Lemma 3.5 in full: the guaranteed value of `L(f_{t+1})` given `L(f_t)` and
    /// `‖∇L(f_t)‖_B`. Only valid for `RelErr ≤ 1/2`.
    pub fn descent_bound(
        &self,
        loss: f64,
        learning_rate: f64,
        relative_error: f64,
        gradient_norm: f64,
    ) -> f64 {
        loss - learning_rate
            * self.descent_coefficient(learning_rate, relative_error)
            * gradient_norm
            * gradient_norm
    }

    /// `ε < min{1/2, α/(α+β)}` — above this no learning rate gives descent,
    /// because `descent_coefficient` is negative even in the limit `η → 0`.
    pub fn max_relative_error(&self) -> f64 {
        0.5f64.min(self.alpha / (self.alpha + self.beta))
    }

    /// Proposition 3.6: `η ≤ 2(α − (α+β)ε) / (K(2ε+1))`, the largest learning
    /// rate that keeps `r > 0` at relative error `ε`. Non-positive when `ε`
    /// exceeds [`Constants::max_relative_error`].
    pub fn max_learning_rate(&self, relative_error: f64) -> f64 {
        2.0 * (self.alpha - (self.alpha + self.beta) * relative_error)
            / (self.smoothness * (2.0 * relative_error + 1.0))
    }

    /// Proposition 3.6: the bound on `min_t ‖∇L(f_t)‖²_B` after `steps`
    /// iterations, `(L(f_0) − L*) / (T·η·r)`.
    pub fn stationary_bound(
        &self,
        initial_loss: f64,
        optimal_loss: f64,
        steps: usize,
        learning_rate: f64,
        relative_error: f64,
    ) -> f64 {
        let r = self.descent_coefficient(learning_rate, relative_error);
        if steps == 0 || r <= 0.0 {
            return f64::INFINITY;
        }
        (initial_loss - optimal_loss) / (steps as f64 * learning_rate * r)
    }

    /// Proposition 3.8: `L(f_T) − L* ≤ [1 − 2ηβ⁻²μr]^T (L(f_0) − L*)` for a
    /// `μ`-Polyak-Łojasiewicz loss (Assumption 3.7).
    pub fn polyak_lojasiewicz_bound(
        &self,
        mu: f64,
        initial_gap: f64,
        steps: usize,
        learning_rate: f64,
        relative_error: f64,
    ) -> f64 {
        let r = self.descent_coefficient(learning_rate, relative_error);
        if r <= 0.0 {
            return f64::INFINITY;
        }
        let contraction = 1.0 - 2.0 * learning_rate * mu * r / (self.beta * self.beta);
        initial_gap * contraction.max(0.0).powi(steps as i32)
    }
}
