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

//! GLASS inner-flow helpers (Holderrieth et al. 2025; Diamond Maps §4.1).

use crate::interpolant::{self, clip_zero};

const STABLE: f32 = 1e-4;

/// DDPM re-scaling ρ(t, t′).
#[inline]
pub fn rho(t: f32, t_prime: f32) -> f32 {
    interpolant::alpha(t) * interpolant::sigma(t_prime)
        / clip_zero(interpolant::alpha(t_prime) * interpolant::sigma(t))
}

#[inline]
pub fn gamma(t: f32, t_prime: f32) -> f32 {
    rho(t, t_prime) * interpolant::sigma(t_prime) / clip_zero(interpolant::sigma(t))
}

#[inline]
pub fn inner_sigma(t: f32, t_prime: f32, s: f32, sigma_0: f32) -> f32 {
    let term = interpolant::sigma(t_prime).powi(2) * (1.0 - rho(t, t_prime).powi(2));
    (1.0 - s) * sigma_0.sqrt() + s * term.max(0.0).sqrt()
}

#[inline]
pub fn inner_alpha(t: f32, t_prime: f32, s: f32) -> f32 {
    s * (interpolant::alpha(t_prime) - gamma(t, t_prime) * interpolant::alpha(t))
}

#[inline]
pub fn inner_sigma_dot(_t: f32, t_prime: f32, _s: f32, sigma_0: f32) -> f32 {
    -sigma_0.sqrt()
        + interpolant::sigma(t_prime).powi(2) * (1.0 - rho(_t, t_prime).powi(2)).max(0.0).sqrt()
}

#[inline]
pub fn inner_alpha_dot(t: f32, t_prime: f32, _s: f32) -> f32 {
    interpolant::alpha(t_prime) - gamma(t, t_prime) * interpolant::alpha(t)
}

/// Inner early-stop time s* for Diamond DDPM (reference `calc_s`).
pub fn calc_s(t: f32, t_prime: f32) -> f32 {
    if (t_prime - 1.0).abs() < 1e-6 {
        return 1.0;
    }
    let diff = interpolant::g(t) - interpolant::g(t_prime);
    let fraction = interpolant::g(t_prime) * interpolant::g(t) / clip_zero(diff);
    interpolant::g_inv(fraction)
}

/// 2×2 covariance for (x_t, x_s) and its inverse (reference `_mu_cov` / `_stable_inv`).
fn mu_cov(t: f32, t_prime: f32, s: f32) -> ([f32; 2], [[f32; 2]; 2]) {
    let g = gamma(t, t_prime);
    let mu0 = interpolant::alpha(t);
    let mu1 = inner_alpha(t, t_prime, s) + g * interpolant::alpha(t);
    let cross = interpolant::sigma(t).powi(2) * g;
    let cov00 = interpolant::sigma(t).powi(2);
    let cov11 = inner_sigma(t, t_prime, s, 1.0).powi(2) + g.powi(2) * interpolant::sigma(t).powi(2);
    let mu = [mu0, mu1];
    let cov = [[cov00, cross], [cross, cov11]];
    (mu, cov)
}

fn inv2x2(m: [[f32; 2]; 2]) -> [[f32; 2]; 2] {
    let a = m[0][0] + STABLE;
    let b = m[0][1];
    let c = m[1][0];
    let d = m[1][1] + STABLE;
    let det = clip_zero(a * d - b * c);
    [[d / det, -b / det], [-c / det, a / det]]
}

/// Sufficient statistic S_{s,t}(x̄_s, x_t) and denominator (Eq. 18–19).
pub fn sufficient_stat(t: f32, t_prime: f32, s: f32, x_t: f32, x_s: f32) -> (f32, f32) {
    let (mu, cov) = mu_cov(t, t_prime, s);
    let cov_inv = inv2x2(cov);
    let denom = clip_zero(
        mu[0] * (cov_inv[0][0] * mu[0] + cov_inv[0][1] * mu[1])
            + mu[1] * (cov_inv[1][0] * mu[0] + cov_inv[1][1] * mu[1]),
    );
    let num = mu[0] * (cov_inv[0][0] * x_t + cov_inv[0][1] * x_s)
        + mu[1] * (cov_inv[1][0] * x_t + cov_inv[1][1] * x_s);
    (num / denom, denom)
}

/// Reparameterized time t* from GLASS (reference `_glass_denoiser`).
pub fn reparam_time(t: f32, t_prime: f32, s: f32, x_t: f32, x_s: f32) -> f32 {
    let (_suff, denom) = sufficient_stat(t, t_prime, s, x_t, x_s);
    interpolant::g_inv(1.0 / clip_zero(denom))
}

/// Reparameterized input α(t*) S_{s,t}(·).
pub fn reparam_input(t: f32, t_prime: f32, s: f32, x_t: f32, x_s: f32) -> (f32, f32) {
    let (suff, denom) = sufficient_stat(t, t_prime, s, x_t, x_s);
    let t_star = interpolant::g_inv(1.0 / clip_zero(denom));
    (t_star, interpolant::alpha(t_star) * suff)
}

/// Diamond early-stop: x_{t′} from inner state (Proposition 4.3).
pub fn early_stop_ddpm(t: f32, t_prime: f32, s: f32, x_t: f32, x_s: f32) -> f32 {
    let (suff, _) = sufficient_stat(t, t_prime, s, x_t, x_s);
    interpolant::alpha(t_prime) * suff
}

/// GLASS inner velocity weights applied to (x_s, denoised, x_t).
pub fn glass_velocity(t: f32, t_prime: f32, s: f32, x_t: f32, x_s: f32, denoised: f32) -> f32 {
    let sig = inner_sigma(t, t_prime, s, 1.0);
    let sig_dot = inner_sigma_dot(t, t_prime, s, 1.0);
    let alp = inner_alpha(t, t_prime, s);
    let alp_dot = inner_alpha_dot(t, t_prime, s);
    let w1 = sig_dot / clip_zero(sig);
    let w2 = alp_dot - alp * w1;
    let w3 = -gamma(t, t_prime) * w1;
    w1 * x_s + w2 * denoised + w3 * x_t
}

/// Sample inner state x̄_s (reference `calc_xbar_s`).
pub fn sample_inner_state(t: f32, t_prime: f32, s: f32, x_t: f32, eps: f32, rescale: f32) -> f32 {
    gamma(t, t_prime) * x_t + inner_sigma(t, t_prime, s, 1.0) * rescale * eps
}

/// Element-wise GLASS helpers over flat tensors.
pub fn sufficient_stat_vec(
    t: f32,
    t_prime: f32,
    s: f32,
    x_t: &[f32],
    x_s: &[f32],
) -> (Vec<f32>, f32) {
    assert_eq!(x_t.len(), x_s.len());
    let mut out = vec![0.0f32; x_t.len()];
    let mut denom_acc = 0.0f32;
    for i in 0..x_t.len() {
        let (suff, d) = sufficient_stat(t, t_prime, s, x_t[i], x_s[i]);
        out[i] = suff;
        denom_acc += d;
    }
    let denom = denom_acc / x_t.len() as f32;
    (out, denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sufficient_stat_at_s_one() {
        let t = 0.3;
        let t_prime = 1.0;
        let s = 1.0;
        let x_t = 0.5;
        let x_s = 0.9;
        let (suff, _) = sufficient_stat(t, t_prime, s, x_t, x_s);
        assert!(suff.is_finite());
    }

    #[test]
    fn early_stop_finite() {
        let out = early_stop_ddpm(0.2, 0.25, calc_s(0.2, 0.25), 0.1, 0.3);
        assert!(out.is_finite());
    }
}
