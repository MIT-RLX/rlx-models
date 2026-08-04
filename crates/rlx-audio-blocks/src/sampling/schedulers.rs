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

//! Noise schedules and denoise steppers shared by the diffusion / flow-matching
//! audio generators (Stable-Audio, Seed-VC, ACE-Step, the VibeVoice diffusion
//! head, …). These are the model-agnostic *math*: the network that predicts the
//! velocity / noise is the model's; the schedule and the update rule live here.

use core::f32::consts::PI;

/// Variance-preserving `beta` schedules (DDPM family).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BetaSchedule {
    /// Linear in `beta` from `beta_start` to `beta_end` (Ho et al. 2020).
    Linear,
    /// Linear in `sqrt(beta)` (the Stable-Diffusion "scaled_linear").
    ScaledLinear,
    /// Cosine `alpha_cumprod` schedule (Nichol & Dhariwal 2021).
    Cosine,
}

/// Per-timestep `beta_t` for `num_train_timesteps` steps.
pub fn betas(
    schedule: BetaSchedule,
    num_train_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
) -> Vec<f32> {
    let n = num_train_timesteps.max(1);
    match schedule {
        BetaSchedule::Linear => (0..n)
            .map(|i| beta_start + (beta_end - beta_start) * i as f32 / (n - 1).max(1) as f32)
            .collect(),
        BetaSchedule::ScaledLinear => {
            let a = beta_start.sqrt();
            let b = beta_end.sqrt();
            (0..n)
                .map(|i| {
                    let s = a + (b - a) * i as f32 / (n - 1).max(1) as f32;
                    s * s
                })
                .collect()
        }
        BetaSchedule::Cosine => {
            // Derive betas from the cosine alpha_cumprod, clamped as in the paper.
            let acp = cosine_alphas_cumprod(n, 0.008);
            let mut out = Vec::with_capacity(n);
            for t in 0..n {
                let prev = if t == 0 { 1.0 } else { acp[t - 1] };
                let beta = 1.0 - acp[t] / prev;
                out.push(beta.clamp(0.0, 0.999));
            }
            out
        }
    }
}

/// Cumulative product of `alpha_t = 1 - beta_t`.
pub fn alphas_cumprod(betas: &[f32]) -> Vec<f32> {
    let mut acc = 1.0f32;
    betas
        .iter()
        .map(|&b| {
            acc *= 1.0 - b;
            acc
        })
        .collect()
}

/// `alpha_cumprod` for the cosine schedule (`s` is the small offset, 0.008).
fn cosine_alphas_cumprod(n: usize, s: f32) -> Vec<f32> {
    let f = |t: f32| {
        let x = (t + s) / (1.0 + s) * PI / 2.0;
        let c = x.cos();
        c * c
    };
    let f0 = f(0.0);
    (0..n)
        .map(|i| {
            let t = (i + 1) as f32 / n as f32;
            (f(t) / f0).clamp(1.0e-8, 1.0)
        })
        .collect()
}

/// A flow-matching Euler sampler over a descending `sigma` schedule.
///
/// The velocity field `v = model(x, sigma)` is integrated with the explicit
/// Euler rule `x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v`. `sigmas` has
/// `num_steps + 1` entries, descending from `sigma_max` to `0`.
#[derive(Debug, Clone)]
pub struct FlowMatchEuler {
    pub sigmas: Vec<f32>,
}

impl FlowMatchEuler {
    /// Uniform sigma schedule from `1.0` down to `0.0` in `num_steps` steps.
    pub fn uniform(num_steps: usize) -> Self {
        let steps = num_steps.max(1);
        let sigmas = (0..=steps).map(|i| 1.0 - i as f32 / steps as f32).collect();
        Self { sigmas }
    }

    /// Ascending sigma schedule from `0.0` up to `1.0` in `num_steps` steps — the
    /// conditional-flow-matching convention that integrates noise → data (used by
    /// VoxCPM / Seed-VC style acoustic heads), the reverse of [`uniform`].
    ///
    /// [`uniform`]: FlowMatchEuler::uniform
    pub fn ascending(num_steps: usize) -> Self {
        let steps = num_steps.max(1);
        let sigmas = (0..=steps).map(|i| i as f32 / steps as f32).collect();
        Self { sigmas }
    }

    /// Construct from an explicit sigma schedule (length ≥ 2).
    pub fn from_sigmas(sigmas: Vec<f32>) -> Self {
        Self { sigmas }
    }

    /// Number of denoise steps (`sigmas.len() - 1`).
    pub fn num_steps(&self) -> usize {
        self.sigmas.len().saturating_sub(1)
    }

    /// One Euler update at step `i`: `x + (sigma[i+1] - sigma[i]) * v`.
    /// `x` and `v` must have equal length; returns the updated latent.
    pub fn step(&self, i: usize, x: &[f32], v: &[f32]) -> Vec<f32> {
        assert!(
            i + 1 < self.sigmas.len(),
            "flow-match step {i} out of range"
        );
        assert_eq!(x.len(), v.len(), "x/v length mismatch");
        let dt = self.sigmas[i + 1] - self.sigmas[i];
        x.iter().zip(v).map(|(&xi, &vi)| xi + dt * vi).collect()
    }
}

/// The SD3 / Flux / ACE-Step discrete-flow **timestep shift**:
/// `sigma' = shift · sigma / (1 + (shift − 1) · sigma)`. `shift = 1` is the
/// identity; `shift > 1` allocates more steps to high-noise sigmas. Endpoints are
/// fixed (`0 → 0`, `1 → 1`).
pub fn sd3_time_shift(sigma: f32, shift: f32) -> f32 {
    (shift * sigma) / (1.0 + (shift - 1.0) * sigma)
}

/// A descending `sigma` schedule (`steps + 1` points, `1 → 0`) built from a linear
/// schedule remapped by [`sd3_time_shift`]. Feed to [`FlowMatchEuler::from_sigmas`].
pub fn sd3_shifted_sigmas(steps: usize, shift: f32) -> Vec<f32> {
    let points = steps.max(1) + 1;
    let mut sigmas: Vec<f32> = (0..points)
        .map(|i| {
            let linear = 1.0 - i as f32 / (points - 1) as f32;
            sd3_time_shift(linear, shift)
        })
        .collect();
    let n = sigmas.len();
    sigmas[0] = 1.0;
    sigmas[n - 1] = 0.0;
    sigmas
}

/// One DDPM ancestral reverse step at timestep index `t`.
///
/// Given the model's predicted noise `eps` for `x_t`, returns the mean of
/// `x_{t-1}`; the caller adds `sigma_t * z` noise (with `z` standard normal)
/// for a stochastic sampler, or uses the mean directly for the deterministic
/// posterior mean. `acp` is `alphas_cumprod`, `betas[t]` the step's beta.
pub fn ddpm_posterior_mean(
    x_t: &[f32],
    eps: &[f32],
    t: usize,
    betas: &[f32],
    acp: &[f32],
) -> Vec<f32> {
    assert_eq!(x_t.len(), eps.len(), "x_t/eps length mismatch");
    let beta_t = betas[t];
    let alpha_t = 1.0 - beta_t;
    let acp_t = acp[t];
    // coefficient on x_t and on eps for the posterior mean:
    // 1/sqrt(alpha_t) * ( x_t - beta_t / sqrt(1 - acp_t) * eps )
    let inv_sqrt_alpha = 1.0 / alpha_t.sqrt();
    let eps_coef = beta_t / (1.0 - acp_t).max(1.0e-8).sqrt();
    x_t.iter()
        .zip(eps)
        .map(|(&x, &e)| inv_sqrt_alpha * (x - eps_coef * e))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_betas_are_monotonic_and_bounded() {
        let b = betas(BetaSchedule::Linear, 1000, 1e-4, 0.02);
        assert_eq!(b.len(), 1000);
        assert!((b[0] - 1e-4).abs() < 1e-6);
        assert!((b[999] - 0.02).abs() < 1e-6);
        assert!(b.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn alphas_cumprod_decreasing_in_unit_interval() {
        let b = betas(BetaSchedule::Linear, 1000, 1e-4, 0.02);
        let acp = alphas_cumprod(&b);
        assert!(acp[0] <= 1.0 && acp[0] > 0.999);
        assert!(*acp.last().unwrap() > 0.0);
        assert!(acp.windows(2).all(|w| w[1] < w[0]));
    }

    #[test]
    fn cosine_alphas_cumprod_starts_near_one_and_decreases() {
        let b = betas(BetaSchedule::Cosine, 1000, 0.0, 0.0);
        let acp = alphas_cumprod(&b);
        assert!(acp[0] > 0.99, "acp0={}", acp[0]);
        assert!(acp.windows(2).all(|w| w[1] <= w[0] + 1e-6));
        assert!(*acp.last().unwrap() >= 0.0);
    }

    #[test]
    fn flow_match_constant_velocity_integrates_linearly() {
        // With sigma descending 1 -> 0 and a constant velocity v, the total
        // displacement is (sigma_end - sigma_start) * v = -1 * v.
        let sched = FlowMatchEuler::uniform(20);
        let v = vec![2.0f32, -3.0];
        let mut x = vec![10.0f32, 10.0];
        for i in 0..sched.num_steps() {
            x = sched.step(i, &x, &v);
        }
        assert!((x[0] - (10.0 - 2.0)).abs() < 1e-4, "x0={}", x[0]);
        assert!((x[1] - (10.0 + 3.0)).abs() < 1e-4, "x1={}", x[1]);
        // Schedule endpoints are exactly 1 and 0.
        assert!((sched.sigmas[0] - 1.0).abs() < 1e-6);
        assert!(sched.sigmas.last().unwrap().abs() < 1e-6);
    }

    #[test]
    fn ascending_schedule_goes_zero_to_one() {
        let s = FlowMatchEuler::ascending(8);
        assert_eq!(s.num_steps(), 8);
        assert_eq!(s.sigmas[0], 0.0);
        assert_eq!(*s.sigmas.last().unwrap(), 1.0);
        assert!(s.sigmas.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn sd3_shift_fixes_endpoints_and_biases_high_noise() {
        assert_eq!(sd3_time_shift(0.0, 3.0), 0.0);
        assert!((sd3_time_shift(1.0, 3.0) - 1.0).abs() < 1e-6);
        // shift = 1 is the identity.
        assert!((sd3_time_shift(0.4, 1.0) - 0.4).abs() < 1e-6);
        // shift > 1 pushes an interior sigma upward (toward more noise).
        assert!(sd3_time_shift(0.5, 3.0) > 0.5);
    }

    #[test]
    fn sd3_shifted_sigmas_descend_with_pinned_endpoints() {
        let s = sd3_shifted_sigmas(20, 3.0);
        assert_eq!(s.len(), 21);
        assert_eq!(s[0], 1.0);
        assert_eq!(*s.last().unwrap(), 0.0);
        assert!(s.windows(2).all(|w| w[1] <= w[0] + 1e-6));
    }

    #[test]
    fn ddpm_posterior_mean_is_finite_and_shaped() {
        let b = betas(BetaSchedule::Linear, 100, 1e-4, 0.02);
        let acp = alphas_cumprod(&b);
        let x = vec![0.5f32; 8];
        let eps = vec![0.1f32; 8];
        let mean = ddpm_posterior_mean(&x, &eps, 50, &b, &acp);
        assert_eq!(mean.len(), 8);
        assert!(mean.iter().all(|v| v.is_finite()));
    }
}
