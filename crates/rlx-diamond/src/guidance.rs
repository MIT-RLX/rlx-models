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

//! Reward-guided flow step (Algorithm 1).

use crate::interpolant;

/// Apply guided velocity: u^r = u + b_t ∇V.
pub fn guided_velocity(base_velocity: f32, t_paper: f32, grad_v: f32) -> f32 {
    base_velocity + interpolant::guidance_coefficient(t_paper) * grad_v
}

/// Euler update x_{t+h} = x_t + h u^r (paper outer time).
pub fn euler_step(x: f32, dt: f32, velocity: f32) -> f32 {
    x + dt * velocity
}

/// Element-wise guided Euler on flat state.
pub fn euler_step_vec(x: &mut [f32], dt: f32, velocity: &[f32]) {
    assert_eq!(x.len(), velocity.len());
    for (xi, &v) in x.iter_mut().zip(velocity.iter()) {
        *xi += dt * v;
    }
}

/// FLUX rectified-flow guided update: latents += (σ_next - σ) * v_guided.
pub fn flux_guided_euler_step(
    latents: &mut [f32],
    base_velocity: &[f32],
    grad_v: &[f32],
    sigma: f32,
    sigma_next: f32,
    max_b: f32,
) {
    assert_eq!(latents.len(), base_velocity.len());
    assert_eq!(latents.len(), grad_v.len());
    let t_paper = crate::flux_time::flux_sigma_to_paper_t(sigma);
    let b = crate::flux_time::flux_guidance_coefficient(sigma, max_b);
    let dt = sigma_next - sigma;
    let _ = t_paper;
    for ((l, &u), &g) in latents
        .iter_mut()
        .zip(base_velocity.iter())
        .zip(grad_v.iter())
    {
        *l += dt * (u + b * g);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_increments() {
        assert!((euler_step(1.0, 0.1, 2.0) - 1.2).abs() < 1e-6);
    }
}
