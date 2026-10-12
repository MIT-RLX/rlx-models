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

//! The paper's empirical claims, and the invariants the representations owe
//! Algorithm 1.

use rlx_fgd::algorithm::{FgdConfig, Functional, Refinement, Representation, Rng, run};
use rlx_fgd::rep::complex::Complex;
use rlx_fgd::rep::{DyadicGrid, FourierGrid, RegressionTree};
use rlx_fgd::tasks::fit::L2Fit;
use rlx_fgd::tasks::rkhs::{Dataset, PointwiseLoss, RbfKernel, RkhsRegression};
use rlx_fgd::tasks::wave::{InitialCondition, SincCubeBumps, WaveEquation};
use rlx_fgd::theory::guaranteed_relative_error;

// ── Representation invariants ───────────────────────────────────────────────
//
// Algorithm 1 refines *between* fixing `f_t` and stepping from it, so a
// refinement that changed the encoded function would move the iterate the
// convergence proofs track. Each representation owes that guarantee.

#[test]
fn grid_refinement_preserves_the_encoded_function() {
    let mut grid = DyadicGrid::<2>::unit(3);
    let mut rng = Rng::new(1);
    for value in grid.values_mut() {
        *value = rng.uniform01() - 0.5;
    }
    let fine = grid.refine();
    assert_eq!(fine.cells_per_axis(), 2 * grid.cells_per_axis());
    assert_eq!(fine.dofs(), 4 * grid.dofs());
    for _ in 0..2000 {
        let x = [rng.uniform01(), rng.uniform01()];
        assert_eq!(grid.eval(x), fine.eval(x), "refinement moved f at {x:?}");
    }
}

#[test]
fn tree_refinement_preserves_the_encoded_function() {
    let mut tree = RegressionTree::unit(3, 4);
    let mut rng = Rng::new(2);
    for value in tree.values_mut() {
        *value = rng.uniform01() - 0.5;
    }
    let fine = tree.refine();
    assert_eq!(fine.leaves(), 2 * tree.leaves());
    assert!(fine.max_leaf_radius() < tree.max_leaf_radius());
    for _ in 0..2000 {
        let x: Vec<f64> = (0..3).map(|_| rng.uniform01()).collect();
        assert_eq!(tree.eval(&x), fine.eval(&x), "refinement moved f at {x:?}");
    }
}

#[test]
fn tree_leaves_partition_the_box() {
    let tree = RegressionTree::unit(3, 5);
    let mut volume = 0.0;
    for k in 0..tree.leaves() {
        let (lo, hi) = tree.leaf_bounds(k);
        volume += lo.iter().zip(hi).map(|(a, b)| b - a).product::<f64>();
        // Every leaf must contain its own centre, or the error bound's radius
        // is measured against a box the value does not come from.
        assert_eq!(tree.leaf_of(&tree.leaf_center(k)), k);
    }
    assert!(
        (volume - 1.0).abs() < 1e-12,
        "leaf volumes summed to {volume}"
    );
}

#[test]
fn fourier_refinement_and_descent_keep_the_marginals_consistent() {
    let mut grid = FourierGrid::zeros(-4.0, 4.0, 2);
    let mut rng = Rng::new(3);
    grid.fill_from(|_| Complex::new(rng.uniform01() - 0.5, rng.uniform01() - 0.5));
    assert!(grid.marginals_are_consistent(1e-12));

    let fine = grid.refine();
    assert!(
        fine.marginals_are_consistent(1e-12),
        "refinement broke the cached τ marginals"
    );
    // Refinement preserves the spectrum, so both marginals are unchanged too.
    for (xi, zeta) in [(-3.0, 1.0), (0.5, -2.0), (2.5, 2.5)] {
        assert!((grid.integral_tau(xi, zeta) - fine.integral_tau(xi, zeta)).abs() < 1e-12);
    }

    let mut iterate = fine.clone();
    let mut gradient = fine.clone();
    gradient.fill_from(|_| Complex::new(rng.uniform01(), rng.uniform01()));
    iterate.descend(0.3, &gradient);
    assert!(
        iterate.marginals_are_consistent(1e-12),
        "the descent step broke the cached τ marginals"
    );
}

// ── `L²` fitting: Figure 1 ──────────────────────────────────────────────────

/// `U` is supposed to be the approximation error, not a number that merely
/// behaves like one. For `L_FIT` there is an independent handle on it: the
/// gradient splits orthogonally into its cell means and the rest, so
/// `½(‖g‖² + U²)` must equal the loss computed by quadrature. The gap is the
/// `O(1/N)` sampling bias, and has to shrink like `1/N`.
#[test]
fn error_bound_reconstructs_the_loss_through_the_orthogonal_split() {
    let f = DyadicGrid::<2>::unit(6);
    let mut previous = f64::INFINITY;
    for samples in [8usize, 64, 512] {
        let problem = L2Fit {
            samples_per_cell: samples,
            ..L2Fit::with_max_level(Default::default(), 6)
        };
        let approx = problem.approximate_gradient(&f, &(), 11);
        let reconstructed = problem.loss_from_gradient(approx.norm, approx.error_bound);
        let exact = problem.exact_loss(&f);
        let gap = ((reconstructed - exact) / exact).abs();
        assert!(
            gap < 20.0 / samples as f64,
            "N = {samples}: ½(‖g‖²+U²) was off by {gap:e}"
        );
        assert!(gap < previous, "the gap must shrink with more samples");
        previous = gap;
    }
}

/// The paper's central claim, in the form that isolates the mechanism: run the
/// same optimiser, from the same starting representation, with the same step
/// size, and vary only whether refinement is allowed.
#[test]
fn fixed_representation_plateaus_where_adaptive_keeps_descending() {
    let max_level = 9;
    let problem = L2Fit::with_max_level(Default::default(), max_level);
    let config = |refinement| FgdConfig {
        learning_rate: 0.05,
        tolerance: 0.5,
        steps: 80,
        refinement,
        seed: 0,
    };

    let fixed = run(
        &problem,
        DyadicGrid::<2>::unit(4),
        config(Refinement::Fixed),
    );
    let adaptive = run(
        &problem,
        DyadicGrid::<2>::unit(4),
        config(Refinement::Adaptive { max_level }),
    );

    let fixed_loss = fixed.final_loss().expect("losses are reported");
    let adaptive_loss = adaptive.final_loss().expect("losses are reported");
    assert!(
        adaptive_loss < fixed_loss / 100.0,
        "adaptive {adaptive_loss:e} should be far below fixed {fixed_loss:e}"
    );

    // The fixed run stalls: its last quarter buys almost nothing, because the
    // representation's own error is all that is left.
    let fixed_curve: Vec<f64> = fixed.history.iter().filter_map(|s| s.loss).collect();
    assert!(
        fixed_curve[fixed_curve.len() - 1] > 0.99 * fixed_curve[3 * fixed_curve.len() / 4],
        "the fixed run was still making progress, so it had not plateaued yet"
    );

    // The adaptive run is still making progress at the same point.
    let adaptive_curve: Vec<f64> = adaptive.history.iter().filter_map(|s| s.loss).collect();
    assert!(
        adaptive_curve[adaptive_curve.len() - 1]
            < 0.9 * adaptive_curve[3 * adaptive_curve.len() / 4],
        "the adaptive run plateaued too"
    );

    // It started small and grew into the resolution it needed.
    assert!(adaptive.history[0].dofs < adaptive.history[adaptive.history.len() - 1].dofs);
    assert!(adaptive.final_level() > 4);
}

/// “Strictly outperforms FGD with any fixed representation.” Read as a claim
/// about the accuracy-per-work frontier, since a fixed grid fine enough to
/// match the adaptive run's *final* resolution can of course reach a similar
/// loss — by paying for that resolution on every step from the first.
#[test]
fn adaptive_sits_ahead_of_the_fixed_representation_frontier() {
    let max_level = 9;
    let problem = L2Fit::with_max_level(Default::default(), max_level);
    let config = |refinement| FgdConfig {
        learning_rate: 0.05,
        tolerance: 0.5,
        steps: 80,
        refinement,
        seed: 0,
    };

    let adaptive = run(
        &problem,
        DyadicGrid::<2>::unit(4),
        config(Refinement::Adaptive { max_level }),
    );
    let adaptive_loss = adaptive.final_loss().expect("losses are reported");
    let fixed: Vec<_> = (4..=max_level)
        .map(|level| {
            run(
                &problem,
                DyadicGrid::<2>::unit(level),
                config(Refinement::Fixed),
            )
        })
        .collect();

    // Against every fixed run that costs no more than the adaptive one, the
    // adaptive run is not merely ahead but ahead by a wide margin.
    for report in &fixed {
        if report.total_work() <= adaptive.total_work() {
            let loss = report.final_loss().expect("losses are reported");
            assert!(
                adaptive_loss < loss / 4.0,
                "a fixed run at {} dofs matched the adaptive one: {loss:e} vs {adaptive_loss:e}",
                report.history[0].dofs
            );
        }
    }

    // The only fixed run that beats it is the one held at the adaptive run's
    // own final resolution throughout — and that costs substantially more for
    // a loss in the same place.
    let finest = fixed.last().expect("at least one fixed run");
    assert_eq!(finest.final_level(), adaptive.final_level());
    assert!(
        adaptive.total_work() < finest.total_work() * 3 / 4,
        "adaptive work {} was not below the fixed run's {}",
        adaptive.total_work(),
        finest.total_work()
    );
    assert!(
        adaptive_loss < 2.0 * finest.final_loss().expect("losses are reported"),
        "adaptive {adaptive_loss:e} fell well short of the finest fixed run"
    );
}

/// Theorem 3.10(i): every step of Algorithm 1 satisfies `RelErr ≤ ε/(1+ε)`.
#[test]
fn algorithm_1_holds_the_relative_error_it_promises() {
    for tolerance in [0.1, 0.3, 0.9] {
        let problem = L2Fit::with_max_level(Default::default(), 11);
        let report = run(
            &problem,
            DyadicGrid::<2>::unit(3),
            FgdConfig {
                learning_rate: 0.05,
                tolerance,
                steps: 30,
                refinement: Refinement::Adaptive { max_level: 11 },
                seed: 5,
            },
        );
        assert!(
            report.guarantees_hold(),
            "ε = {tolerance}: the run exhausted its refinement budget"
        );
        let bound = guaranteed_relative_error(tolerance);
        assert!(
            report.worst_relative_error() <= bound,
            "ε = {tolerance}: worst RelErr {} exceeded {bound}",
            report.worst_relative_error()
        );
    }
}

#[test]
fn a_loose_tolerance_never_refines_and_a_tight_one_always_does() {
    let problem = L2Fit::with_max_level(Default::default(), 10);
    let base = FgdConfig {
        learning_rate: 0.05,
        steps: 10,
        seed: 2,
        tolerance: 0.0,
        refinement: Refinement::Adaptive { max_level: 10 },
    };

    // A tolerance this loose accepts any approximation whose error is under the
    // gradient's own size, which a 32×32 grid already manages here.
    let lazy = run(
        &problem,
        DyadicGrid::<2>::unit(5),
        FgdConfig {
            tolerance: 1e6,
            ..base
        },
    );
    assert_eq!(
        lazy.final_level(),
        5,
        "nothing should have forced a refinement"
    );
    assert!(lazy.history.iter().all(|s| s.refinements == 0));

    let eager = run(
        &problem,
        DyadicGrid::<2>::unit(5),
        FgdConfig {
            tolerance: 0.05,
            ..base
        },
    );
    assert!(eager.final_level() > 5, "a tight tolerance must refine");
    assert!(eager.history[0].refinements > 0);
}

// ── RKHS regression: Section 4.1 ────────────────────────────────────────────

fn regression_problem(loss: PointwiseLoss) -> RkhsRegression {
    let data = Dataset::synthetic(240, 9);
    let (lower, upper) = data.bounds();
    RkhsRegression::new(data.normalized(&lower, &upper), RbfKernel::new(20.0), loss)
}

/// The Lipschitz bound of Section 4.1 claims to be a genuine `L^∞` upper bound,
/// not an estimate. Check it the only way that means anything: against the
/// actual gradient, at points that are not leaf centres.
#[test]
fn rkhs_error_bound_really_bounds_the_sup_norm_error() {
    let problem = regression_problem(PointwiseLoss::SquaredError);
    let mut f = problem.initial_tree(3);
    let mut rng = Rng::new(17);
    for value in f.values_mut() {
        *value = 0.4 * (rng.uniform01() - 0.5);
    }

    for depth in 0..4 {
        let state = problem.begin_step(&f);
        let approx = problem.approximate_gradient(&f, &state, 0);

        let n = problem.train.len() as f64;
        let mut worst = 0.0f64;
        for _ in 0..4000 {
            let x: Vec<f64> = (0..problem.train.dim()).map(|_| rng.uniform01()).collect();
            let exact = (0..problem.train.len())
                .map(|i| state.coefficients[i] * problem.kernel.eval(problem.train.row(i), &x))
                .sum::<f64>()
                / n;
            worst = worst.max((approx.gradient.eval(&x) - exact).abs());
        }
        assert!(
            worst <= approx.error_bound,
            "depth {}: sup error {worst:e} exceeded U = {:e}",
            f.depth(),
            approx.error_bound
        );
        // Lemma 3.9 needs U → 0 under refinement; here it halves per level,
        // because splitting the widest side halves the leaf diameter.
        if depth > 0 {
            assert!(approx.error_bound > 0.0);
        }
        f = f.refine();
    }
}

#[test]
fn rkhs_error_bound_decays_under_refinement() {
    let problem = regression_problem(PointwiseLoss::SquaredError);
    let mut f = problem.initial_tree(2);
    let state = problem.begin_step(&f);
    let mut previous = f64::INFINITY;
    for _ in 0..8 {
        let approx = problem.approximate_gradient(&f, &state, 0);
        assert!(
            approx.error_bound < previous,
            "U did not decrease: {previous:e} → {:e}",
            approx.error_bound
        );
        previous = approx.error_bound;
        f = f.refine();
    }
    // Eight refinements in two dimensions shrink the diameter by 2^4.
    let first = problem.approximate_gradient(&problem.initial_tree(2), &state, 0);
    assert!(previous < first.error_bound / 10.0);
}

/// Proposition A.2, numerically. `α` and `β` are what let the method
/// approximate an RKHS gradient inside `L^∞` at all, so they are worth
/// checking against the definitions in Assumptions 3.3 and 3.4 rather than
/// taking on trust.
#[test]
fn proposition_a2_constants_bridge_the_rkhs_and_sup_norm_gradients() {
    // Small enough for an O(n³) eigenvalue solve and a dense sup-norm search.
    let data = Dataset::synthetic(60, 4);
    let (lower, upper) = data.bounds();
    let problem = RkhsRegression::new(
        data.normalized(&lower, &upper),
        RbfKernel::new(3.0),
        PointwiseLoss::SquaredError,
    );
    let constants = problem
        .constants()
        .expect("a strictly positive definite kernel matrix");
    assert!(constants.alpha > 0.0 && constants.beta > 0.0);

    let f = problem.initial_tree(2);
    let state = problem.begin_step(&f);
    let n = problem.train.len();
    let coefficients = &state.coefficients;
    let gradient = |x: &[f64]| {
        (0..n)
            .map(|i| coefficients[i] * problem.kernel.eval(problem.train.row(i), x))
            .sum::<f64>()
            / n as f64
    };

    // ‖∇L(f)‖_{L^∞} over the training points plus a dense sweep of the box.
    let mut sup = 0.0f64;
    for i in 0..n {
        sup = sup.max(gradient(problem.train.row(i)).abs());
    }
    let grid = 200;
    for i in 0..grid {
        for j in 0..grid {
            let x = [i as f64 / (grid - 1) as f64, j as f64 / (grid - 1) as f64];
            sup = sup.max(gradient(&x).abs());
        }
    }

    // DL(f; v) = (1/n) Σ cᵢ v(Xᵢ); its (L^∞)* dual norm is (1/n)‖c‖₁.
    let directional: f64 = (0..n)
        .map(|i| coefficients[i] * gradient(problem.train.row(i)))
        .sum::<f64>()
        / n as f64;
    let dual_norm = coefficients.iter().map(|c| c.abs()).sum::<f64>() / n as f64;

    // Assumption 3.4. A sampled supremum can only *under*-estimate ‖∇L(f)‖,
    // which makes this inequality harder to satisfy, not easier — so passing it
    // is real evidence.
    assert!(
        dual_norm <= constants.beta * sup,
        "gradient compatibility failed: {dual_norm:e} > {} · {sup:e}",
        constants.beta
    );
    // Assumption 3.3.
    assert!(
        directional >= constants.alpha * sup * sup,
        "H did not descend in B: {directional:e} < {} · {sup:e}²",
        constants.alpha
    );
}

#[test]
fn rkhs_regression_descends_under_both_losses() {
    for loss in [
        PointwiseLoss::SquaredError,
        PointwiseLoss::BinaryCrossEntropy,
    ] {
        let data = Dataset::synthetic(400, 21);
        let targets: Vec<f64> = match loss {
            // A binary problem for cross-entropy; the synthetic regression
            // target thresholded at its own midpoint.
            PointwiseLoss::BinaryCrossEntropy => data
                .targets()
                .iter()
                .map(|&y| if y > 0.0 { 1.0 } else { 0.0 })
                .collect(),
            PointwiseLoss::SquaredError => data.targets().to_vec(),
        };
        let rebuilt = Dataset::new(
            (0..data.len()).flat_map(|i| data.row(i).to_vec()).collect(),
            targets,
            data.dim(),
        );
        let (lower, upper) = rebuilt.bounds();
        let normalized = rebuilt.normalized(&lower, &upper);
        let (train, test) = normalized.split(0.25, 13);

        let problem = RkhsRegression::new(train, RbfKernel::new(20.0), loss);
        let before_train = problem.dataset_loss(&problem.initial_tree(2), &problem.train);
        let before_test = problem.dataset_loss(&problem.initial_tree(2), &test);

        let report = run(
            &problem,
            problem.initial_tree(2),
            FgdConfig {
                // Cross-entropy is 1/4-smooth against squared error's 1, so a
                // step scaled by 1/K puts both losses on the same footing —
                // the reason the authors' own runs use 10 for one and 50 for
                // the other.
                learning_rate: 10.0 / loss.smoothness(),
                tolerance: 0.5,
                steps: 60,
                refinement: Refinement::Adaptive { max_level: 14 },
                seed: 0,
            },
        );

        let after_train = report.final_loss().expect("losses are reported");
        let after_test = problem.dataset_loss(&report.solution, &test);
        assert!(
            after_train < before_train * 0.6,
            "{loss:?}: train loss {before_train:e} → {after_train:e}"
        );
        assert!(
            after_test < before_test,
            "{loss:?}: test loss {before_test:e} → {after_test:e}"
        );
    }
}

// ── Wave equation: Section 4.2 ──────────────────────────────────────────────

/// `h` and `ĥ` have to be an actual Fourier pair: the PDE's whole initial
/// condition enters the gradient through `ĥ` alone, so a scale or sign slip
/// there would be invisible until the solution came out wrong.
#[test]
fn initial_condition_spectrum_inverts_to_its_spatial_form() {
    let initial = SincCubeBumps::sampled(3, 2.0, 0);
    // ĥ is supported on |ξ| ≤ 1.5·scale, so the inverse transform is a proper
    // integral over a bounded box rather than a truncation.
    let limit = 1.5 * initial.scale;
    let steps = 400;
    let width = 2.0 * limit / steps as f64;
    let two_pi = std::f64::consts::TAU;

    for &(x, y) in &[(0.0, 0.0), (0.35, -0.2), (-0.6, 0.45), (1.1, 0.9)] {
        let mut acc = Complex::ZERO;
        for i in 0..steps {
            let xi = -limit + (i as f64 + 0.5) * width;
            for j in 0..steps {
                let zeta = -limit + (j as f64 + 0.5) * width;
                acc +=
                    initial.spectrum(xi, zeta) * Complex::from_angle(two_pi * (xi * x + zeta * y));
            }
        }
        let recovered = acc.re * width * width;
        let expected = initial.spatial(x, y);
        assert!(
            (recovered - expected).abs() < 2e-3 * expected.abs().max(1.0),
            "at ({x}, {y}): inverse transform gave {recovered:.6} for h = {expected:.6}"
        );
        assert!(
            acc.im.abs() * width * width < 1e-6,
            "a real-valued h must have a conjugate-symmetric spectrum"
        );
    }
}

/// The closed-form `sinc` envelope in [`FourierGrid::evaluate_spatial_grid`]
/// against brute-force quadrature of the same piecewise-constant spectrum.
///
/// The closed form integrates each frequency cell exactly, so the comparison is
/// really a convergence claim: refining the quadrature has to walk towards it,
/// at the midpoint rule's second-order rate.
#[test]
fn spatial_evaluation_is_the_limit_of_direct_quadrature() {
    let mut rng = Rng::new(31);
    let mut grid = FourierGrid::zeros(-2.0, 2.0, 3);
    grid.fill_from(|_| Complex::new(rng.uniform01() - 0.5, rng.uniform01() - 0.5));

    let (t, x, y) = (0.31, -0.4, 0.17);
    let closed_form = grid.evaluate_spatial_grid(&[t], &[x], &[y])[0];
    let two_pi = std::f64::consts::TAU;
    let width = grid.cell_width();

    let quadrature = |sub: usize| {
        let step = width / sub as f64;
        let mut acc = Complex::ZERO;
        for flat in 0..grid.grid().len() {
            let origin = grid.cell_origin(flat);
            let value = grid.grid()[flat];
            for a in 0..sub {
                let tau = origin[0] + (a as f64 + 0.5) * step;
                for b in 0..sub {
                    let xi = origin[1] + (b as f64 + 0.5) * step;
                    for c in 0..sub {
                        let zeta = origin[2] + (c as f64 + 0.5) * step;
                        acc += value * Complex::from_angle(two_pi * (tau * t + xi * x + zeta * y));
                    }
                }
            }
        }
        acc.re * step * step * step
    };

    let coarse = (quadrature(4) - closed_form).abs();
    let fine = (quadrature(8) - closed_form).abs();
    let finer = (quadrature(16) - closed_form).abs();
    assert!(
        fine < coarse / 3.0 && finer < fine / 3.0,
        "quadrature did not converge on the closed form: {coarse:e} → {fine:e} → {finer:e}"
    );
    assert!(
        finer < 1e-2 * closed_form.abs().max(1e-4),
        "at the finest quadrature the gap was still {finer:e}"
    );
}

#[test]
fn wave_equation_loss_descends() {
    let max_level = 5;
    let problem = WaveEquation {
        samples_per_cell: 4,
        ..WaveEquation::with_max_level(2.0, SincCubeBumps::sampled(3, 2.0, 0), max_level)
    };
    let initial = FourierGrid::zeros(-5.0, 5.0, 3);
    let before = problem.loss_terms(&initial);
    // The zero spectrum solves the PDE and the zero-velocity condition exactly;
    // all of the initial loss is the displacement it fails to match.
    assert_eq!(before.residual, 0.0);
    assert_eq!(before.velocity, 0.0);
    assert!(before.displacement > 0.0);

    let report = run(
        &problem,
        initial,
        FgdConfig {
            learning_rate: 0.1,
            tolerance: 0.5,
            steps: 60,
            refinement: Refinement::Adaptive { max_level },
            seed: 0,
        },
    );

    let mut previous = before.total();
    for record in &report.history {
        let loss = record.loss.expect("the wave loss is reported");
        assert!(
            loss <= previous + 1e-12,
            "step {}: L_PDE rose, {previous:e} → {loss:e}",
            record.step
        );
        previous = loss;
    }
    // The velocity term stays at zero throughout: every operator involved is
    // even in τ, so the iterates are too — which is exactly the symmetry
    // ∂_t f(0,·) = 0 asks for. It holds by geometry, not by being descended on.
    let after = problem.loss_terms(&report.solution);
    assert!(
        after.velocity < 1e-20 * before.displacement,
        "the velocity term left zero: {:e}",
        after.velocity
    );
    assert!(
        after.total() < 0.5 * before.total(),
        "L_PDE {:e} → {:e}",
        before.total(),
        after.total()
    );
    assert!(
        after.displacement < before.displacement,
        "the initial condition should be better matched, not worse"
    );
}
