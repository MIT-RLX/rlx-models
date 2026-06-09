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

//! Rectified-flow / Flow-Match Euler scheduler (diffusers-compatible sigmas).

/// `num_inference_steps + 1` sigmas from 1.0 → 0.0 (linear).
pub fn flow_match_sigmas(num_inference_steps: usize) -> Vec<f32> {
    let n = num_inference_steps.max(1);
    (0..=n).map(|i| 1.0 - i as f32 / n as f32).collect()
}

/// One Euler step: `latents += (sigma_next - sigma) * noise_pred`.
pub fn flow_match_euler_step(latents: &mut [f32], noise_pred: &[f32], sigma: f32, sigma_next: f32) {
    let dt = sigma_next - sigma;
    for (l, n) in latents.iter_mut().zip(noise_pred.iter()) {
        *l += dt * n;
    }
}

/// img2img start step: `max(1, floor(steps * strength))` (mflux `Config.init_time_step`).
pub fn flow_match_init_timestep(image_strength: f32, num_inference_steps: usize) -> usize {
    let strength = image_strength.clamp(0.0, 1.0);
    if strength <= 0.0 {
        return 0;
    }
    (num_inference_steps as f32 * strength).floor().max(1.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmas_end_at_zero() {
        let s = flow_match_sigmas(4);
        assert_eq!(s.len(), 5);
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!((s[4]).abs() < 1e-6);
    }

    #[test]
    fn init_timestep_matches_mflux() {
        assert_eq!(flow_match_init_timestep(0.0, 20), 0);
        assert_eq!(flow_match_init_timestep(0.5, 20), 10);
        assert_eq!(flow_match_init_timestep(1.0, 4), 4);
    }

    #[test]
    fn euler_step_updates() {
        let mut lat = vec![1.0f32, 2.0];
        let pred = vec![0.5f32, 1.0];
        flow_match_euler_step(&mut lat, &pred, 1.0, 0.5);
        assert!((lat[0] - 0.75).abs() < 1e-5);
    }
}
