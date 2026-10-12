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

//! Section 3.1 checked both ways: the closed forms against the algebra of the
//! proofs, and the guarantees against an actual run.

use rlx_fgd::algorithm::{FgdConfig, Functional, Refinement, run};
use rlx_fgd::rep::DyadicGrid;
use rlx_fgd::tasks::fit::L2Fit;
use rlx_fgd::theory::{
    Constants, guaranteed_relative_error, relative_error, tolerance_for_relative_error,
    tolerance_in_scope,
};

#[test]
fn relative_error_follows_the_papers_division_convention() {
    assert_eq!(relative_error(0.0, 0.0), 0.0, "0/0 = 0");
    assert_eq!(relative_error(1.0, 0.0), f64::INFINITY, "a/0 = +∞");
    assert!((relative_error(0.25, 0.5) - 0.5).abs() < 1e-15);
}

#[test]
fn algorithm_tolerance_and_relative_error_bound_are_inverse() {
    for tolerance in [0.01, 0.1, 0.5, 1.0, 3.0, 50.0] {
        let bound = guaranteed_relative_error(tolerance);
        let recovered = tolerance_for_relative_error(bound);
        assert!(
            (recovered - tolerance).abs() < 1e-9 * tolerance.max(1.0),
            "round trip failed for ε = {tolerance}"
        );
    }
}

/// Theorem 3.10 is stated for `ε ∈ (0,1)`, and that restriction is not
/// cosmetic: it is exactly what keeps the guaranteed relative error under the
/// `1/2` Lemma 3.5 requires. A looser tolerance refines less often and runs
/// faster, but leaves the theorem behind — as the authors' own `ε = 3` and
/// `ε = 50` experiment settings do.
#[test]
fn tolerances_at_or_above_one_leave_the_theorem_behind() {
    for tolerance in [0.01, 0.1, 0.5, 0.99] {
        assert!(tolerance_in_scope(tolerance));
        assert!(
            guaranteed_relative_error(tolerance) < 0.5,
            "ε = {tolerance} should stay under Lemma 3.5's ceiling"
        );
    }
    assert!(!tolerance_in_scope(1.0));
    assert_eq!(guaranteed_relative_error(1.0), 0.5);
    assert!(!tolerance_in_scope(3.0));
    assert!(guaranteed_relative_error(3.0) > 0.5);
}

#[test]
fn max_learning_rate_is_exactly_where_the_descent_coefficient_vanishes() {
    // Proposition 3.6 states η ≤ 2(α − (α+β)ε)/(K(2ε+1)); the proof derives the
    // condition as r > 0. The two agree only if that η is r's root.
    for &(alpha, beta, smoothness) in &[
        (1.0, 1.0, 1.0),
        (0.5, 2.0, 3.0),
        (2.0, 0.25, 0.75),
        (0.02, 13.0, 4.5),
    ] {
        let constants = Constants {
            alpha,
            beta,
            smoothness,
        };
        for relative_error in [0.0, 0.01, 0.05, 0.1] {
            if relative_error >= constants.max_relative_error() {
                continue;
            }
            let eta = constants.max_learning_rate(relative_error);
            assert!(eta > 0.0, "admissible rate must be positive");
            let r = constants.descent_coefficient(eta, relative_error);
            assert!(
                r.abs() < 1e-12 * (alpha + beta + smoothness),
                "r({eta}, {relative_error}) = {r}, expected 0"
            );
            // Just inside the bound descent is guaranteed; just outside it is not.
            assert!(constants.descent_coefficient(eta * 0.99, relative_error) > 0.0);
            assert!(constants.descent_coefficient(eta * 1.01, relative_error) < 0.0);
        }
    }
}

#[test]
fn no_learning_rate_works_past_the_relative_error_ceiling() {
    let constants = Constants {
        alpha: 1.0,
        beta: 3.0,
        smoothness: 1.0,
    };
    // ε < min{1/2, α/(α+β)} = 1/4 here.
    assert!((constants.max_relative_error() - 0.25).abs() < 1e-15);
    assert!(constants.max_learning_rate(0.25).abs() < 1e-15);
    assert!(constants.max_learning_rate(0.30) < 0.0);
    // Even a vanishing step cannot rescue it: r stays negative as η → 0.
    assert!(constants.descent_coefficient(1e-12, 0.30) < 0.0);
}

#[test]
fn hilbert_case_recovers_the_classical_descent_lemma() {
    // The paper's tightness remark: with B = H, g = ∇L(f) and η = 1/K, the
    // bound must collapse to L(f − ηg) ≤ L(f) − (η/2)‖∇L(f)‖².
    let constants = Constants::hilbert(4.0);
    let eta = 1.0 / constants.smoothness;
    let bound = constants.descent_bound(10.0, eta, 0.0, 2.0);
    let classical = 10.0 - 0.5 * eta * 2.0 * 2.0;
    assert!((bound - classical).abs() < 1e-12, "{bound} vs {classical}");
}

#[test]
fn polyak_lojasiewicz_contraction_is_a_proper_rate() {
    let constants = Constants::hilbert(1.0);
    let eta = constants.max_learning_rate(0.1) * 0.5;
    let after_one = constants.polyak_lojasiewicz_bound(1.0, 1.0, 1, eta, 0.1);
    let after_ten = constants.polyak_lojasiewicz_bound(1.0, 1.0, 10, eta, 0.1);
    assert!(after_one < 1.0, "one step must contract");
    assert!(after_ten < after_one, "the rate must compound");
    assert!(after_ten > 0.0);
    // An inadmissible relative error gives no bound at all rather than a bogus one.
    assert_eq!(
        constants.polyak_lojasiewicz_bound(1.0, 1.0, 10, eta, 0.9),
        f64::INFINITY
    );
}

#[test]
fn stationary_bound_decays_like_one_over_t() {
    let constants = Constants::hilbert(1.0);
    let eta = 0.5;
    let at_100 = constants.stationary_bound(1.0, 0.0, 100, eta, 0.05);
    let at_1000 = constants.stationary_bound(1.0, 0.0, 1000, eta, 0.05);
    assert!((at_100 / at_1000 - 10.0).abs() < 1e-9);
    assert_eq!(
        constants.stationary_bound(1.0, 0.0, 0, eta, 0.05),
        f64::INFINITY
    );
}

/// The descent lemma, checked against a run rather than re-derived.
///
/// Lemma 3.5 bounds `L(f_{t+1})` using `‖∇L(f_t)‖_B`, which we never compute.
/// The reverse triangle inequality gives `‖∇L(f_t)‖ ≥ ‖g_t‖ − U_t`, and the
/// bound is decreasing in that norm while `r > 0`, so substituting the lower
/// bound yields a valid — merely weaker — prediction to test against.
#[test]
fn descent_lemma_holds_at_every_step_of_a_real_run() {
    let problem = L2Fit::with_max_level(Default::default(), 10);
    let constants = problem.constants().expect("L_FIT has known constants");
    let tolerance = 0.5;
    let learning_rate = 0.05;
    assert!(
        learning_rate <= constants.max_learning_rate(guaranteed_relative_error(tolerance)),
        "the test's η must itself be admissible"
    );

    let initial = DyadicGrid::<2>::unit(4);
    let mut previous = problem.exact_loss(&initial);
    let report = run(
        &problem,
        initial,
        FgdConfig {
            learning_rate,
            tolerance,
            steps: 40,
            refinement: Refinement::Adaptive { max_level: 10 },
            seed: 7,
        },
    );

    let mut checked = 0;
    for record in &report.history {
        let loss = record.loss.expect("L_FIT reports its loss");
        // Lemma 3.5 is conditioned on the relative error being controlled; a
        // step that exhausted the refinement budget instead is outside it.
        if record.tolerance_met {
            let r = constants.descent_coefficient(learning_rate, record.relative_error);
            assert!(r > 0.0, "an accepted step must have a positive r, got {r}");
            let gradient_floor = (record.gradient_norm - record.error_bound).max(0.0);
            let predicted = previous - learning_rate * r * gradient_floor * gradient_floor;
            assert!(
                loss <= predicted + 1e-12,
                "step {}: L = {loss:e} exceeded the Lemma 3.5 bound {predicted:e}",
                record.step
            );
            assert!(
                loss <= previous + 1e-12,
                "step {}: the loss rose, {previous:e} → {loss:e}",
                record.step
            );
            checked += 1;
        }
        previous = loss;
    }
    assert!(
        checked == report.history.len(),
        "{checked} of {} steps stayed inside the tolerance",
        report.history.len()
    );
}

/// Proposition 3.8's linear rate, checked on the one task where every constant
/// is known: `L_FIT` is `1`-smooth and `1`-Polyak-Łojasiewicz (Proposition A.1)
/// with `α = β = 1`, and `L* = 0`.
#[test]
fn polyak_lojasiewicz_rate_bounds_a_real_run() {
    let problem = L2Fit::with_max_level(Default::default(), 10);
    let constants = problem.constants().expect("L_FIT has known constants");
    let tolerance = 0.5;
    let relative_error = guaranteed_relative_error(tolerance);
    // Admissibility is not the binding constraint here. A large step drives
    // ‖g‖ towards the representation's error floor within a handful of
    // iterations, so the refinement budget has to double every few steps to
    // keep the tolerance test satisfied; a small step keeps `U/‖g‖` stable for
    // long stretches. This is why the authors run the teaser at η = 0.05.
    let learning_rate = 0.05;
    assert!(learning_rate <= constants.max_learning_rate(relative_error));

    let initial = DyadicGrid::<2>::unit(4);
    let initial_gap = problem.exact_loss(&initial) - L2Fit::GLOBAL_MINIMUM;
    let steps = 40;
    let report = run(
        &problem,
        initial,
        FgdConfig {
            learning_rate,
            tolerance,
            steps,
            refinement: Refinement::Adaptive { max_level: 10 },
            seed: 3,
        },
    );
    assert!(
        report.guarantees_hold(),
        "the run left Theorem 3.10's scope, so Proposition 3.8 does not apply"
    );

    let predicted = constants.polyak_lojasiewicz_bound(
        L2Fit::POLYAK_LOJASIEWICZ,
        initial_gap,
        steps,
        learning_rate,
        relative_error,
    );
    let achieved = report.final_loss().expect("losses are reported") - L2Fit::GLOBAL_MINIMUM;
    assert!(
        achieved <= predicted,
        "achieved gap {achieved:e} exceeded Proposition 3.8's bound {predicted:e}"
    );
}
