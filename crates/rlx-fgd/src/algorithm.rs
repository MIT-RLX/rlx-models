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

//! Algorithm 1 — functional gradient descent with adaptive representations.
//!
//! ```text
//! Require: loss L : H → R, learning rate η, initial condition f₀, tolerance ε
//!   Choose an initial representation
//!   for t = 0, …, T−1 do
//!       while true do
//!           g_t ← approximate ∇L(f_t) using the current representation
//!           U_t ← tight upper bound on ‖g_t − ∇L(f_t)‖_B
//!           if (1 + ε)·U_t < ε·‖g_t‖_B then break
//!           Refine the current representation
//!       f_{t+1} ← f_t − η·g_t
//!   return f_T
//! ```
//!
//! The one thing that makes this implementable — and the paper's contribution —
//! is that the `while` test uses only computable quantities: `‖g_t‖_B` is the
//! norm of our own approximation, and `U_t` is a bound the representation can
//! produce without ever materialising `∇L(f_t)`. Lemma 3.9 guarantees the loop
//! terminates whenever refinement drives `U → 0`.
//!
//! Setting [`Refinement::Fixed`] turns the same driver into the paper's
//! fixed-representation baseline (“Approx. FGD”), which is how
//! `tests/adaptive_vs_fixed.rs` reproduces the resolution-dependent error floor.

use crate::theory::{self, Constants};

/// A finite-dimensional representation of a function, refinable without bound.
///
/// Refinement must be *function-preserving*: `refine()` encodes the same
/// function in more degrees of freedom. Algorithm 1 refines the representation
/// mid-step, after `f_t` is fixed but before the descent step, so a refinement
/// that perturbed `f_t` would change the iterate the convergence proofs track.
pub trait Representation: Clone {
    /// Degrees of freedom currently stored.
    fn dofs(&self) -> usize;

    /// Refinement level. `refine` increments it; the initial level is whatever
    /// the constructor chose.
    fn level(&self) -> u32;

    /// A strictly finer representation encoding the same function.
    fn refine(&self) -> Self;

    /// The descent step `f ← f − η·g` of Equation 3.
    ///
    /// `gradient` is always at this representation's own level, because
    /// Algorithm 1 approximates the gradient *in the current representation*.
    fn descend(&mut self, learning_rate: f64, gradient: &Self);
}

/// The output of “approximate `∇L(f_t)` using the current representation”,
/// together with the two computable quantities Algorithm 1's test needs.
#[derive(Debug, Clone)]
pub struct GradientApprox<R> {
    /// `g_t` — the approximation itself.
    pub gradient: R,
    /// `‖g_t‖_B`, computed exactly from the representation.
    pub norm: f64,
    /// `U_t`, an upper bound on `‖g_t − ∇L(f_t)‖_B` (Equation 5).
    pub error_bound: f64,
}

impl<R> GradientApprox<R> {
    /// `RelErr(g_t, ∇L(f_t))` as bounded through `U_t` (Equation 4).
    pub fn relative_error(&self) -> f64 {
        theory::relative_error(self.error_bound, self.norm)
    }

    /// Algorithm 1's acceptance test, `(1 + ε)·U_t < ε·‖g_t‖_B`.
    ///
    /// Written in the paper's multiplied-out form rather than as
    /// `RelErr ≤ ε/(1+ε)` so that a zero gradient norm is handled by the
    /// comparison itself instead of by a division convention.
    pub fn accepts(&self, tolerance: f64) -> bool {
        (1.0 + tolerance) * self.error_bound < tolerance * self.norm
    }
}

/// How the driver treats the `while` loop of Algorithm 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refinement {
    /// Algorithm 1 proper: refine until the tolerance test passes, or until
    /// `max_level` is reached.
    ///
    /// The cap exists only so a misconfigured run cannot exhaust memory; a step
    /// that hits it is recorded in [`StepRecord::tolerance_met`] as a failure,
    /// and [`Report::guarantees_hold`] then reports that the run is outside the
    /// scope of Theorem 3.10.
    Adaptive { max_level: u32 },
    /// The paper's baseline: never refine. The representation stays at whatever
    /// level `f₀` was built with, and `U_t` is whatever it is.
    Fixed,
}

/// Configuration for [`run`].
#[derive(Debug, Clone, Copy)]
pub struct FgdConfig {
    /// `η`.
    pub learning_rate: f64,
    /// `ε` — Algorithm 1's tolerance, *not* the `RelErr` bound of
    /// Propositions 3.6/3.8. See [`theory::guaranteed_relative_error`].
    pub tolerance: f64,
    /// `T`.
    pub steps: usize,
    /// Adaptive (the method) or fixed (the baseline).
    pub refinement: Refinement,
    /// Seed for the per-step sampling seeds handed to the functional.
    pub seed: u64,
}

impl Default for FgdConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            tolerance: 0.5,
            steps: 100,
            refinement: Refinement::Adaptive { max_level: 12 },
            seed: 0,
        }
    }
}

/// A functional `L : H → R` whose gradients are approximated inside a
/// representation living in the larger Banach space `B`.
pub trait Functional {
    /// The representation used for both the iterate and the gradient.
    type Rep: Representation;

    /// Anything that depends on `f_t` but not on the refinement level.
    ///
    /// Algorithm 1's inner loop refines the representation without changing the
    /// function it encodes, so per-iterate work — the residuals
    /// `ℓ'(f(X_i), Y_i)` of Proposition 4.1, say — is valid for every attempt
    /// within a step and is computed once here.
    type StepState;

    /// Compute the per-step state, once, before the refinement loop.
    fn begin_step(&self, f: &Self::Rep) -> Self::StepState;

    /// Approximate `∇L(f)` at `f`'s current level and bound the error.
    ///
    /// `seed` is fixed across all attempts within one step, so refinement is
    /// compared against a fixed sample draw rather than chasing sampling noise.
    fn approximate_gradient(
        &self,
        f: &Self::Rep,
        state: &Self::StepState,
        seed: u64,
    ) -> GradientApprox<Self::Rep>;

    /// Restore any constraint of the problem that refinement can leave stated
    /// only approximately. Called once after each refinement, before the
    /// gradient is re-approximated.
    ///
    /// Refinement is function-preserving by construction, which is all the
    /// convergence proofs need. But a *constraint* can be stated more sharply
    /// on a finer grid than a coarse one could state it, and then the
    /// inherited values violate it. Section 4.3 is the case: a coarse voxel
    /// straddling the object's silhouette counts as occupied, and after
    /// refinement its children outside the visual hull inherit a density that
    /// the gradient is identically zero on — so nothing will ever remove it
    /// again, and it sits in the reconstruction as a permanent blocky halo.
    fn after_refine(&self, _f: &mut Self::Rep) {}

    /// `L(f)`, when it is cheap enough to report every step.
    fn loss(&self, _f: &Self::Rep) -> Option<f64> {
        None
    }

    /// The problem's `(α, β, K)`, when they are known in closed form.
    fn constants(&self) -> Option<Constants> {
        None
    }
}

/// One iteration of the outer loop.
#[derive(Debug, Clone, Copy)]
pub struct StepRecord {
    /// `t`.
    pub step: usize,
    /// How many times the representation was refined during this step.
    pub refinements: u32,
    /// The level the descent step was taken at.
    pub level: u32,
    /// Degrees of freedom at that level.
    pub dofs: usize,
    /// Degrees of freedom fitted during this step, counting every refinement
    /// attempt — including the ones that were rejected and refined away.
    ///
    /// A machine-independent cost proxy. It is what the adaptive schedule is
    /// spending: a step that refines three times pays for four fits, and the
    /// case for starting coarse is that those four together still cost less
    /// than one fit at the final resolution would have, for every step before
    /// it was needed.
    pub work: usize,
    /// `‖g_t‖_B`.
    pub gradient_norm: f64,
    /// `U_t`.
    pub error_bound: f64,
    /// `RelErr(g_t, ∇L(f_t))` via `U_t`.
    pub relative_error: f64,
    /// Whether `(1 + ε)·U_t < ε·‖g_t‖_B` held when the step was taken. False
    /// only when the level cap stopped refinement early.
    pub tolerance_met: bool,
    /// `L(f_{t+1})`, if the functional reports it.
    pub loss: Option<f64>,
}

/// The result of a run.
#[derive(Debug, Clone)]
pub struct Report<R> {
    /// `f_T`.
    pub solution: R,
    /// One entry per outer iteration.
    pub history: Vec<StepRecord>,
    /// The configuration that produced it.
    pub config: FgdConfig,
}

impl<R> Report<R> {
    /// Whether Algorithm 1's test passed at *every* step — the precondition of
    /// Theorem 3.10. A `false` here means the run exceeded its level cap and
    /// the convergence guarantees do not apply to it.
    pub fn guarantees_hold(&self) -> bool {
        self.history.iter().all(|s| s.tolerance_met)
    }

    /// The worst `RelErr` over the run. Theorem 3.10(i) caps this at
    /// `ε/(1+ε)` whenever [`Report::guarantees_hold`].
    pub fn worst_relative_error(&self) -> f64 {
        self.history
            .iter()
            .map(|s| s.relative_error)
            .fold(0.0f64, f64::max)
    }

    /// `Σ_t work(t)` — total degrees of freedom fitted across the run.
    pub fn total_work(&self) -> usize {
        self.history.iter().map(|s| s.work).sum()
    }

    /// `L(f_T)`, if the functional reports losses.
    pub fn final_loss(&self) -> Option<f64> {
        self.history.last().and_then(|s| s.loss)
    }

    /// The final refinement level.
    pub fn final_level(&self) -> u32 {
        self.history.last().map_or(0, |s| s.level)
    }
}

/// Run Algorithm 1.
pub fn run<F: Functional>(problem: &F, initial: F::Rep, config: FgdConfig) -> Report<F::Rep> {
    let mut f = initial;
    let mut history = Vec::with_capacity(config.steps);
    let mut seeds = SeedStream::new(config.seed);

    for step in 0..config.steps {
        let state = problem.begin_step(&f);
        // One draw per outer step, reused by every refinement attempt.
        let seed = seeds.next_seed();

        let mut refinements = 0;
        let mut work = f.dofs();
        let mut approx = problem.approximate_gradient(&f, &state, seed);
        let mut tolerance_met = approx.accepts(config.tolerance);

        if let Refinement::Adaptive { max_level } = config.refinement {
            while !tolerance_met && f.level() < max_level {
                f = f.refine();
                problem.after_refine(&mut f);
                refinements += 1;
                work += f.dofs();
                approx = problem.approximate_gradient(&f, &state, seed);
                tolerance_met = approx.accepts(config.tolerance);
            }
        }

        let record = StepRecord {
            step,
            refinements,
            level: f.level(),
            dofs: f.dofs(),
            work,
            gradient_norm: approx.norm,
            error_bound: approx.error_bound,
            relative_error: approx.relative_error(),
            tolerance_met,
            loss: None,
        };

        f.descend(config.learning_rate, &approx.gradient);

        history.push(StepRecord {
            loss: problem.loss(&f),
            ..record
        });
    }

    Report {
        solution: f,
        history,
        config,
    }
}

/// SplitMix64, used only to derive one reproducible sampling seed per step.
#[derive(Debug, Clone)]
pub struct SeedStream(u64);

impl SeedStream {
    /// Start the stream. Any seed, including `0`, is valid.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next seed.
    pub fn next_seed(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// SplitMix64 as a uniform `[0, 1)` source, for the Monte-Carlo cell sampling.
#[derive(Debug, Clone)]
pub struct Rng(SeedStream);

impl Rng {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        Self(SeedStream::new(seed))
    }

    /// Next value in `[0, 1)`.
    pub fn uniform01(&mut self) -> f64 {
        (self.0.next_seed() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}
