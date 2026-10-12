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

//! Regression in an RKHS — Section 4.1 and Appendix A.1.2.
//!
//! `L_REG(f) = (1/n) Σ ℓ(f(X_i), Y_i)` over an RKHS `H_K`, with gradient
//! `∇L_REG(f)(x) = (1/n) Σ ℓ'(f(X_i), Y_i) K(X_i, x)` (Proposition 4.1).
//!
//! This is the case that justifies the paper's whole Hilbert-inside-Banach
//! setup. The gradient lives in `H_K`, but `H_K` cannot hold a piecewise
//! constant, so there is no way to approximate it *inside* `H_K` with a tree.
//! Approximating in `B = L^∞` instead is legitimate because Proposition A.2
//! supplies the bridging constants — [`RkhsRegression::constants`] — at the
//! price of `α` and `β` that depend on `λ_min(K)` rather than both being `1`.
//!
//! The error bound `U_t` is a Lipschitz estimate: a tree is exact at each leaf
//! centre, so over a leaf of radius `r` it is off by at most `Lip(∇L) · r`, and
//! `Lip(∇L) ≤ mean|ℓ'| · Lip(K)`. Refinement halves leaf diameters, so `U → 0`
//! and Lemma 3.9 applies.

use crate::algorithm::{Functional, GradientApprox};
use crate::linalg::smallest_eigenvalue;
use crate::rep::RegressionTree;
use crate::theory::Constants;

/// The pointwise loss `ℓ(ŷ, y)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointwiseLoss {
    /// `½(ŷ − y)²`. Strongly convex with `μ = 1` and `1`-smooth, so
    /// Proposition 4.1 gives both the Polyak-Łojasiewicz and smoothness halves.
    SquaredError,
    /// Binary cross-entropy on *logits*: `−y·logσ(ŷ) − (1−y)·logσ(−ŷ)`.
    ///
    /// The paper writes cross-entropy in probability form; the authors' code
    /// takes `f` to be a logit, which is both what the `1/4`-smoothness below
    /// refers to and the only form that stays finite for an unconstrained `f`.
    BinaryCrossEntropy,
}

impl PointwiseLoss {
    /// `ℓ(ŷ, y)`.
    pub fn value(&self, prediction: f64, target: f64) -> f64 {
        match self {
            Self::SquaredError => {
                let d = prediction - target;
                0.5 * d * d
            }
            Self::BinaryCrossEntropy => {
                -(target * log_sigmoid(prediction) + (1.0 - target) * log_sigmoid(-prediction))
            }
        }
    }

    /// `ℓ'(ŷ, y)`, the derivative in the first argument.
    pub fn derivative(&self, prediction: f64, target: f64) -> f64 {
        match self {
            Self::SquaredError => prediction - target,
            Self::BinaryCrossEntropy => sigmoid(prediction) - target,
        }
    }

    /// The smoothness constant `K` of `ℓ` in its first argument.
    pub fn smoothness(&self) -> f64 {
        match self {
            Self::SquaredError => 1.0,
            Self::BinaryCrossEntropy => 0.25,
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

/// The Gaussian kernel `K(a, b) = exp(−γ‖a − b‖²)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RbfKernel {
    /// `γ > 0`.
    pub gamma: f64,
}

impl RbfKernel {
    /// A kernel with bandwidth `γ`.
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }

    /// `K(a, b)`.
    pub fn eval(&self, a: &[f64], b: &[f64]) -> f64 {
        let d2: f64 = a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum();
        (-self.gamma * d2).exp()
    }

    /// `sup_x ‖∇_x K(·, x)‖ = √(2γ/e)`.
    ///
    /// `‖∇_x K‖ = 2γ·r·e^{−γr²}` at distance `r`, maximised at `r = 1/√(2γ)`.
    /// This is what converts a leaf radius into an `L^∞` error bound.
    pub fn lipschitz(&self) -> f64 {
        (2.0 * self.gamma / std::f64::consts::E).sqrt()
    }

    /// `κ = sup|K(x, x')| = 1` for a Gaussian kernel — the `κ` of
    /// Proposition A.2.
    pub fn bound(&self) -> f64 {
        1.0
    }
}

/// A design matrix with scalar targets.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    features: Vec<f64>,
    targets: Vec<f64>,
    dim: usize,
}

impl Dataset {
    /// Build from a row-major feature block and matching targets.
    pub fn new(features: Vec<f64>, targets: Vec<f64>, dim: usize) -> Self {
        assert!(dim > 0, "dimension must be positive");
        assert_eq!(
            features.len(),
            targets.len() * dim,
            "features and targets must agree in row count"
        );
        Self {
            features,
            targets,
            dim,
        }
    }

    /// Row count.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Whether there are no rows.
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Feature dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Row `i`.
    pub fn row(&self, i: usize) -> &[f64] {
        &self.features[i * self.dim..(i + 1) * self.dim]
    }

    /// Targets.
    pub fn targets(&self) -> &[f64] {
        &self.targets
    }

    /// Keep only the first `dim` features of each row, as Section 4.1 does.
    pub fn truncate_features(&self, dim: usize) -> Self {
        assert!(dim <= self.dim, "cannot widen a dataset");
        let features = (0..self.len())
            .flat_map(|i| self.row(i)[..dim].to_vec())
            .collect();
        Self::new(features, self.targets.clone(), dim)
    }

    /// Per-axis min and max over the rows.
    pub fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let mut lower = vec![f64::INFINITY; self.dim];
        let mut upper = vec![f64::NEG_INFINITY; self.dim];
        for i in 0..self.len() {
            for (axis, value) in self.row(i).iter().enumerate() {
                lower[axis] = lower[axis].min(*value);
                upper[axis] = upper[axis].max(*value);
            }
        }
        (lower, upper)
    }

    /// Rescale features into `[0, 1]^d` using the supplied bounds.
    ///
    /// The representation is a tree over the unit box, so the data has to be
    /// put there. Train and test must share one set of bounds — computed on
    /// train — or the test points land in different leaves than they should.
    pub fn normalized(&self, lower: &[f64], upper: &[f64]) -> Self {
        let features = (0..self.len())
            .flat_map(|i| {
                self.row(i)
                    .iter()
                    .enumerate()
                    .map(|(axis, value)| {
                        let span = upper[axis] - lower[axis];
                        if span > 0.0 {
                            (value - lower[axis]) / span
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        Self::new(features, self.targets.clone(), self.dim)
    }

    /// A deterministic shuffle split into `(train, test)`.
    pub fn split(&self, test_fraction: f64, seed: u64) -> (Self, Self) {
        let n = self.len();
        let mut order: Vec<usize> = (0..n).collect();
        let mut rng = crate::algorithm::Rng::new(seed);
        for i in (1..n).rev() {
            let j = (rng.uniform01() * (i + 1) as f64) as usize;
            order.swap(i, j.min(i));
        }
        let test_len = ((n as f64) * test_fraction).round() as usize;
        let take = |ids: &[usize]| {
            let features = ids.iter().flat_map(|&i| self.row(i).to_vec()).collect();
            let targets = ids.iter().map(|&i| self.targets[i]).collect();
            Self::new(features, targets, self.dim)
        };
        (take(&order[test_len..]), take(&order[..test_len]))
    }

    /// The synthetic set of the authors' `get_synthetic_data`: `n` uniform
    /// points in `[0,1]²` with `y = x₀·sin(20·x₀)`.
    pub fn synthetic(n: usize, seed: u64) -> Self {
        let mut rng = crate::algorithm::Rng::new(seed);
        let mut features = Vec::with_capacity(2 * n);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let x0 = rng.uniform01();
            let x1 = rng.uniform01();
            features.push(x0);
            features.push(x1);
            targets.push(x0 * (20.0 * x0).sin());
        }
        Self::new(features, targets, 2)
    }

    /// Parse LIBSVM/SVMlight text, the format Section 4.1's cod-rna set ships in.
    ///
    /// `label_map` rewrites the raw label; cod-rna labels are `±1` and the
    /// authors map them to `{0, 1}` with `(y + 1)/2`.
    pub fn from_libsvm(
        text: &str,
        dim: usize,
        label_map: impl Fn(f64) -> f64,
    ) -> anyhow::Result<Self> {
        let mut features = Vec::new();
        let mut targets = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let label: f64 = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("line {}: missing label", lineno + 1))?
                .parse()?;
            let mut row = vec![0.0; dim];
            for entry in parts {
                let (index, value) = entry.split_once(':').ok_or_else(|| {
                    anyhow::anyhow!("line {}: malformed entry {entry:?}", lineno + 1)
                })?;
                let index: usize = index.parse()?;
                if index >= 1 && index <= dim {
                    row[index - 1] = value.parse()?;
                }
            }
            features.extend_from_slice(&row);
            targets.push(label_map(label));
        }
        Ok(Self::new(features, targets, dim))
    }
}

/// `L_REG` over decision trees in `L^∞`.
#[derive(Debug, Clone)]
pub struct RkhsRegression {
    /// Training data, already normalised into the tree's box.
    pub train: Dataset,
    /// The kernel defining `H_K`.
    pub kernel: RbfKernel,
    /// The pointwise loss `ℓ`.
    pub loss: PointwiseLoss,
}

/// Per-step quantities: the residuals `c_i = ℓ'(f(X_i), Y_i)`.
///
/// These depend on `f` as a *function*, not on how finely it is represented, so
/// refining mid-step cannot change them — which is exactly why Algorithm 1's
/// inner loop can reuse them across attempts.
#[derive(Debug, Clone)]
pub struct Residuals {
    /// `c_i` for each training point.
    pub coefficients: Vec<f64>,
    /// `(1/n) Σ |c_i|`, the factor in the Lipschitz error bound.
    pub mean_absolute: f64,
}

impl RkhsRegression {
    /// A regression problem on `train`.
    pub fn new(train: Dataset, kernel: RbfKernel, loss: PointwiseLoss) -> Self {
        Self {
            train,
            kernel,
            loss,
        }
    }

    /// A root tree over the unit box of the right dimension.
    pub fn initial_tree(&self, depth: u32) -> RegressionTree {
        RegressionTree::unit(self.train.dim(), depth)
    }

    /// `(1/n) Σ ℓ(f(X_i), Y_i)` on an arbitrary dataset.
    pub fn dataset_loss(&self, f: &RegressionTree, data: &Dataset) -> f64 {
        let n = data.len();
        if n == 0 {
            return 0.0;
        }
        (0..n)
            .map(|i| self.loss.value(f.eval(data.row(i)), data.targets()[i]))
            .sum::<f64>()
            / n as f64
    }

    /// The empirical kernel matrix over the training points, row-major.
    ///
    /// `O(n²)` entries; intended for the small-`n` checks of Proposition A.2,
    /// not for the training loop.
    pub fn kernel_matrix(&self) -> Vec<f64> {
        let n = self.train.len();
        let mut matrix = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let k = self.kernel.eval(self.train.row(i), self.train.row(j));
                matrix[i * n + j] = k;
                matrix[j * n + i] = k;
            }
        }
        matrix
    }

    /// `λ_min(K)` of the empirical kernel matrix, by cyclic Jacobi rotations.
    ///
    /// Proposition A.2's `α` and `β` both hinge on this, and the proposition
    /// assumes it is strictly positive. `O(n³)` — a diagnostic, not a step.
    pub fn min_kernel_eigenvalue(&self) -> f64 {
        smallest_eigenvalue(&self.kernel_matrix(), self.train.len())
    }
}

impl Functional for RkhsRegression {
    type Rep = RegressionTree;
    type StepState = Residuals;

    fn begin_step(&self, f: &Self::Rep) -> Self::StepState {
        let n = self.train.len();
        let coefficients: Vec<f64> = (0..n)
            .map(|i| {
                self.loss
                    .derivative(f.eval(self.train.row(i)), self.train.targets()[i])
            })
            .collect();
        let mean_absolute = if n == 0 {
            0.0
        } else {
            coefficients.iter().map(|c| c.abs()).sum::<f64>() / n as f64
        };
        Residuals {
            coefficients,
            mean_absolute,
        }
    }

    fn approximate_gradient(
        &self,
        f: &Self::Rep,
        state: &Self::StepState,
        _seed: u64,
    ) -> GradientApprox<Self::Rep> {
        let n = self.train.len().max(1) as f64;
        let mut gradient = f.clone();

        // A tree is constant per leaf, so the L^∞-best representable value is
        // the gradient at the leaf's own centre — no sampling needed, which is
        // why this instantiation is deterministic and ignores `seed`.
        for k in 0..gradient.leaves() {
            let center = gradient.leaf_center(k);
            let value = state
                .coefficients
                .iter()
                .enumerate()
                .map(|(i, c)| c * self.kernel.eval(self.train.row(i), &center))
                .sum::<f64>()
                / n;
            gradient.values_mut()[k] = value;
        }

        // |g(x) − ∇L(f)(x)| ≤ Lip(∇L)·‖x − centre‖ ≤ mean|c|·Lip(K)·radius.
        let error_bound =
            state.mean_absolute * self.kernel.lipschitz() * gradient.max_leaf_radius();

        GradientApprox {
            norm: gradient.sup_norm(),
            gradient,
            error_bound,
        }
    }

    fn loss(&self, f: &Self::Rep) -> Option<f64> {
        Some(self.dataset_loss(f, &self.train))
    }

    fn constants(&self) -> Option<Constants> {
        let lambda = self.min_kernel_eigenvalue();
        if lambda.is_nan() || lambda <= 0.0 {
            // Proposition A.2 needs a strictly positive definite kernel matrix;
            // without it neither α nor β is defined.
            return None;
        }
        Some(Constants::rkhs_regression(
            self.train.len(),
            lambda,
            self.kernel.bound(),
            self.loss.smoothness(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `λ_min(K)` decides Proposition A.2's `α` and `β`, so a wrong eigenvalue
    /// would silently rescale both constants rather than fail anywhere visible.
    #[test]
    fn jacobi_finds_the_smallest_eigenvalue() {
        // Diagonal: the spectrum is the diagonal.
        let diagonal = [3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        assert!((smallest_eigenvalue(&diagonal, 3) - 1.0).abs() < 1e-12);

        // [[2,1],[1,2]] has eigenvalues 1 and 3.
        let two = [2.0, 1.0, 1.0, 2.0];
        assert!((smallest_eigenvalue(&two, 2) - 1.0).abs() < 1e-12);

        // A 4×4 with spectrum {1, 2, 3, 4}, rotated out of diagonal form by a
        // Householder reflection so the solver has real work to do.
        let v = [0.5, 0.5, 0.5, 0.5];
        let eigenvalues = [1.0, 2.0, 3.0, 4.0];
        let mut a = [0.0; 16];
        for i in 0..4 {
            for j in 0..4 {
                // (I − 2vvᵀ) Λ (I − 2vvᵀ), with ‖v‖ = 1.
                let mut sum = 0.0;
                for k in 0..4 {
                    let qik = if i == k { 1.0 } else { 0.0 } - 2.0 * v[i] * v[k];
                    let qjk = if j == k { 1.0 } else { 0.0 } - 2.0 * v[j] * v[k];
                    sum += qik * eigenvalues[k] * qjk;
                }
                a[i * 4 + j] = sum;
            }
        }
        assert!((smallest_eigenvalue(&a, 4) - 1.0).abs() < 1e-10);
        assert_eq!(smallest_eigenvalue(&[], 0), 0.0);
    }

    /// `Lip(K) = √(2γ/e)` is what turns a leaf radius into the `L^∞` error
    /// bound; if it were too small, `U` would stop being an upper bound.
    #[test]
    fn rbf_lipschitz_constant_matches_the_steepest_slope() {
        for gamma in [0.5, 3.0, 20.0, 100.0] {
            let kernel = RbfKernel::new(gamma);
            let origin = [0.0, 0.0];
            let mut steepest = 0.0f64;
            let steps = 200_000;
            for i in 0..steps {
                // ‖∇ₓK‖ = 2γ·r·e^{−γr²}; sweep r past its maximum at 1/√(2γ).
                let r = 4.0 * (i as f64 + 0.5) / (steps as f64 * gamma.sqrt());
                let h = 1e-6;
                let slope = (kernel.eval(&[r + h, 0.0], &origin)
                    - kernel.eval(&[r - h, 0.0], &origin))
                    / (2.0 * h);
                steepest = steepest.max(slope.abs());
            }
            let claimed = kernel.lipschitz();
            assert!(
                steepest <= claimed * (1.0 + 1e-6),
                "γ = {gamma}: measured slope {steepest} exceeded the claimed {claimed}"
            );
            assert!(
                steepest > claimed * 0.999,
                "γ = {gamma}: the bound {claimed} is loose against {steepest}"
            );
        }
    }

    #[test]
    fn pointwise_loss_derivatives_match_finite_differences() {
        let h = 1e-6;
        for loss in [
            PointwiseLoss::SquaredError,
            PointwiseLoss::BinaryCrossEntropy,
        ] {
            for &prediction in &[-3.0, -0.4, 0.0, 0.7, 2.5] {
                for &target in &[0.0, 1.0] {
                    let numeric = (loss.value(prediction + h, target)
                        - loss.value(prediction - h, target))
                        / (2.0 * h);
                    let exact = loss.derivative(prediction, target);
                    assert!(
                        (numeric - exact).abs() < 1e-6,
                        "{loss:?} at ({prediction}, {target}): {exact} vs {numeric}"
                    );
                }
            }
        }
    }

    #[test]
    fn log_sigmoid_stays_finite_at_the_tails() {
        for x in [-800.0, -50.0, 0.0, 50.0, 800.0] {
            assert!(log_sigmoid(x).is_finite(), "log σ({x}) was not finite");
            assert!((0.0..=1.0).contains(&sigmoid(x)));
        }
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn split_is_a_permutation_and_normalization_lands_in_the_unit_box() {
        let data = Dataset::synthetic(101, 7);
        let (train, test) = data.split(0.2, 3);
        assert_eq!(train.len() + test.len(), data.len());

        let (lower, upper) = data.bounds();
        let normalized = data.normalized(&lower, &upper);
        for i in 0..normalized.len() {
            for &value in normalized.row(i) {
                assert!((0.0..=1.0).contains(&value), "{value} left the unit box");
            }
        }
        assert_eq!(normalized.targets(), data.targets());
    }

    #[test]
    fn libsvm_parsing_fills_missing_indices_with_zero() {
        let text = "-1 1:0.5 3:2.0\n+1 2:1.0\n\n# a comment\n";
        let data = Dataset::from_libsvm(text, 3, |y| (y + 1.0) / 2.0).expect("parses");
        assert_eq!(data.len(), 2);
        assert_eq!(data.row(0), &[0.5, 0.0, 2.0]);
        assert_eq!(data.row(1), &[0.0, 1.0, 0.0]);
        assert_eq!(data.targets(), &[0.0, 1.0]);
    }
}
