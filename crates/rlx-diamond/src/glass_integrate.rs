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

//! GLASS inner ODE integration over flat tensors.

use crate::glass;

/// denoiser reference: given reparameterized time and spatial state, write clean estimate.
pub trait DenoiserReference: Send + Sync {
    fn denoise(&self, t_star: f32, x_star: &[f32], out: &mut [f32]);
}

/// Integrate GLASS inner flow from s=0..1, returning posterior sample z at t′=1.
pub fn sample_posterior<G: DenoiserReference>(
    denoiser_ref: &G,
    t: f32,
    t_prime: f32,
    x_t: &[f32],
    inner_steps: usize,
    noise: &[f32],
    out_z: &mut [f32],
) {
    assert_eq!(x_t.len(), out_z.len());
    assert_eq!(x_t.len(), noise.len());
    assert!(inner_steps >= 1);
    let n = x_t.len();
    let mut x_s = vec![0.0f32; n];
    for i in 0..n {
        x_s[i] = glass::sample_inner_state(t, t_prime, 0.0, x_t[i], noise[i], 1.0);
    }
    let ds = 1.0 / inner_steps as f32;
    let mut x_in = vec![0.0f32; n];
    let mut denoised = vec![0.0f32; n];
    for step in 0..inner_steps {
        let s = step as f32 * ds;
        let s_next = ((step + 1) as f32 * ds).min(1.0);
        let mut t_star_acc = 0.0f32;
        for i in 0..n {
            let (t_star, xi) = glass::reparam_input(t, t_prime, s, x_t[i], x_s[i]);
            t_star_acc += t_star;
            x_in[i] = xi;
        }
        let t_star_mean = t_star_acc / n as f32;
        denoiser_ref.denoise(t_star_mean, &x_in, &mut denoised);
        for i in 0..n {
            let vel = glass::glass_velocity(t, t_prime, s, x_t[i], x_s[i], denoised[i]);
            x_s[i] += (s_next - s) * vel;
        }
    }
    out_z.copy_from_slice(&x_s);
}
