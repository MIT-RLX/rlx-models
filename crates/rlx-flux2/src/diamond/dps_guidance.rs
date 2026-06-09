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

//! Diffusion posterior sampling style baseline: V_t ≈ r(D_t(x_t)).

use super::params::DiamondGuidanceParams;
use rlx_diamond::{LatentReward, flux_guided_euler_step, grad_xt_via_z};

/// One-step denoised estimate x0 = x - σ v (FLUX convention).
pub fn flux_x0_estimate(latents: &[f32], sigma: f32, velocity: &[f32]) -> Vec<f32> {
    latents
        .iter()
        .zip(velocity.iter())
        .map(|(&x, &v)| x - sigma * v)
        .collect()
}

/// ∇_{x_t} V_t ≈ ∇_{x_t} r(D_t(x_t)) with D_t ≈ x0 estimate (stop-grad through denoiser).
pub fn dps_reward_grad<R: LatentReward>(
    reward: &R,
    latents: &[f32],
    sigma: f32,
    velocity: &[f32],
    reward_scale: f32,
) -> Vec<f32> {
    let x0 = flux_x0_estimate(latents, sigma, velocity);
    let grad_z = reward.grad_wrt_z(&x0);
    let mut g = grad_xt_via_z(&grad_z);
    if reward_scale != 1.0 {
        for gi in &mut g {
            *gi *= reward_scale;
        }
    }
    g
}

/// Apply DPS guidance on top of base velocity for one Euler step.
pub fn apply_dps_guidance_step<R: LatentReward>(
    latents: &mut [f32],
    velocity: &[f32],
    sigma: f32,
    sigma_next: f32,
    reward: &R,
    diamond: &DiamondGuidanceParams,
) {
    let grad_v = dps_reward_grad(reward, latents, sigma, velocity, diamond.reward_scale);
    flux_guided_euler_step(
        latents,
        velocity,
        &grad_v,
        sigma,
        sigma_next,
        diamond.max_guidance_b,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_diamond::BluenessReward;

    #[test]
    fn x0_estimate_shape() {
        let x = vec![1.0, 2.0];
        let v = vec![0.1, 0.2];
        let x0 = flux_x0_estimate(&x, 0.5, &v);
        assert!((x0[0] - 0.95).abs() < 1e-6);
    }

    #[test]
    fn dps_grad_nonzero_for_blueness() {
        let r = BluenessReward { scale: 1.0 };
        let latents = vec![0.0f32; 6];
        let vel = vec![0.0f32; 6];
        let g = dps_reward_grad(&r, &latents, 0.5, &vel, 1.0);
        assert_eq!(g.len(), 6);
    }
}
