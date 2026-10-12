// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adam's bias correction is indexed by a step counter. `Optimizer::step` is
//! called once per *parameter*, so a counter incremented there advances by the
//! number of trainables on every optimizer step rather than by one — and the
//! correction it computes is then wrong for every parameter but the first.
//!
//! PyTorch keeps `state['step']` per parameter for exactly this reason.

use rlx_tune::{Adam, Optimizer};

/// Two parameters, identical initial value, identical gradient, same optimizer,
/// same step. Adam is elementwise and stateless across parameters, so they must
/// move by exactly the same amount. If the step counter is shared and advanced
/// per parameter, the second one is corrected as though it were further along
/// in training and moves less — a real asymmetry decided by iteration order.
#[test]
fn two_parameters_with_the_same_gradient_take_the_same_step() {
    let mut opt = Adam::new(1.0);
    let mut a = [0.0f32];
    let mut b = [0.0f32];
    opt.step("a", &mut a, &[1.0]);
    opt.step("b", &mut b, &[1.0]);
    assert!(
        (a[0] - b[0]).abs() < 1e-6,
        "identical gradients produced different updates: a {} vs b {} — the \
         bias-correction step counter is shared across parameters",
        a[0],
        b[0]
    );
}

/// Adam's first update has magnitude ~`lr` regardless of gradient scale: the
/// bias correction is exactly what makes `m_hat / sqrt(v_hat)` equal 1 at
/// `t = 1`. This pins the property that the correction is actually applied,
/// which the test above cannot distinguish from "correction disabled for
/// everyone" (two uncorrected parameters also agree with each other).
#[test]
fn the_first_adam_update_has_magnitude_lr() {
    for &g in &[1.0f32, 37.0, 0.002] {
        let mut opt = Adam::new(0.1);
        let mut p = [0.0f32];
        opt.step("p", &mut p, &[g]);
        assert!(
            (p[0].abs() - 0.1).abs() < 1e-4,
            "first update for grad {g} was {}, expected magnitude lr=0.1 — \
             bias correction is not being applied",
            p[0].abs()
        );
    }
}

/// The counter must advance per optimizer step, so a parameter stepped `n`
/// times is corrected at `t = n`. With a shared counter, stepping ONE parameter
/// repeatedly is indistinguishable from the correct behaviour — the bug only
/// shows up with several parameters, which is why it survives a single-tensor
/// smoke test. Drive two and check the second still warms up like Adam.
#[test]
fn bias_correction_warms_up_per_parameter_not_per_call() {
    let mut one = Adam::new(0.1);
    let mut solo = [0.0f32];
    for _ in 0..5 {
        one.step("solo", &mut solo, &[1.0]);
    }

    let mut many = Adam::new(0.1);
    let (mut x, mut y) = ([0.0f32], [0.0f32]);
    for _ in 0..5 {
        many.step("x", &mut x, &[1.0]);
        many.step("y", &mut y, &[1.0]);
    }

    assert!(
        (solo[0] - y[0]).abs() < 1e-5,
        "a parameter stepped 5 times moved {} on its own but {} alongside \
         another parameter — the step counter is being shared",
        solo[0],
        y[0]
    );
}
