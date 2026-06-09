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

//! Weighted Diamond Maps renoising (Section 5).

use crate::flux_time;

/// t′ from SNR factor (reference `compute_t_prime`).
pub fn t_prime_from_snr(t_flux: f32, snr_factor: f32) -> f32 {
    let t_n = t_flux.clamp(1e-6, 1.0 - 1e-6);
    let sqrt_lambda = snr_factor.sqrt();
    let t_prime = (sqrt_lambda * t_n) / (sqrt_lambda * t_n + 1.0 - t_n);
    t_prime.clamp(0.0, 0.9999)
}

/// Renoising parameters for x_{t′} | x_t (FLUX linear schedule).
pub fn renoise_params(t_flux: f32, t_prime_flux: f32) -> (f32, f32) {
    let t = t_flux.clamp(1e-6, 1.0);
    let t_prime = t_prime_flux.clamp(0.0, t_flux - 1e-6);
    let alpha_t = 1.0 - t;
    let var_t = t * t;
    let alpha_prev = 1.0 - t_prime;
    let var_prev = t_prime * t_prime;
    let scale_factor = alpha_prev / (alpha_t + 1e-8);
    let var_q = (var_prev - scale_factor.powi(2) * var_t).max(1e-8);
    (scale_factor, var_q.sqrt())
}

/// Apply renoising: μ_q + std_q * ε.
pub fn renoise(x_t: &[f32], scale_factor: f32, std_q: f32, eps: &[f32]) -> Vec<f32> {
    assert_eq!(x_t.len(), eps.len());
    x_t.iter()
        .zip(eps.iter())
        .map(|(&x, &e)| scale_factor * x + std_q * e)
        .collect()
}

/// Score for Gaussian path: -(x - α x0) / var.
pub fn score(x: f32, alpha: f32, x0: f32, var: f32, min_var: f32) -> f32 {
    let v = var.max(min_var);
    (-(x - alpha * x0) / v).clamp(-1000.0, 1000.0)
}

/// Particle logit for weighted aggregation (reward-only branch).
pub fn particle_logit_reward_only(reward: f32, reward_scale: f32) -> f32 {
    reward * reward_scale
}

/// Full weighted logit (Proposition 5.1 style).
pub fn particle_logit_full(
    reward: f32,
    reward_scale: f32,
    log_p: f32,
    gamma_k: f32,
    eps_norm: f32,
    temperature: f32,
) -> f32 {
    (reward * reward_scale + log_p + gamma_k + eps_norm) / temperature.max(1e-8)
}

/// FLUX guidance coefficient with clip.
pub fn guidance_b(sigma: f32, max_abs: f32) -> f32 {
    flux_time::flux_guidance_coefficient(sigma, max_abs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renoise_variance_positive() {
        let (scale, std) = renoise_params(0.6, 0.3);
        assert!(scale.is_finite());
        assert!(std > 0.0);
    }

    #[test]
    fn t_prime_before_t() {
        let t = 0.5f32;
        let tp = t_prime_from_snr(t, 0.25);
        assert!(tp < t);
    }
}
