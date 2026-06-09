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

//! Host-side Diamond Maps math (no model weights).

use rlx_diamond::{
    BluenessReward, DenoiserReference, LatentReward, glass, glass_integrate::sample_posterior,
    guidance_coefficient, log_mean_exp, softmax_grad_aggregate,
};

struct LinearDenoiser;

impl DenoiserReference for LinearDenoiser {
    fn denoise(&self, _t_star: f32, x_star: &[f32], out: &mut [f32]) {
        for (o, &x) in out.iter_mut().zip(x_star.iter()) {
            *o = x * 0.9;
        }
    }
}

#[test]
fn glass_posterior_runs() {
    let x_t = vec![0.5f32; 8];
    let noise = vec![0.1f32; 8];
    let mut z = vec![0.0f32; 8];
    sample_posterior(&LinearDenoiser, 0.3, 1.0, &x_t, 5, &noise, &mut z);
    assert!(z.iter().all(|v| v.is_finite()));
}

#[test]
fn blueness_reward_increases_with_blue_channel() {
    let r = BluenessReward { scale: 1.0 };
    let low = vec![0.0f32, 0.0, 0.1, 0.0, 0.0, 0.1];
    let high = vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0];
    assert!(r.reward(&high) > r.reward(&low));
}

#[test]
fn value_softmax_grad() {
    let rewards = [0.0f32, 1.0, 0.5];
    let grads = [vec![1.0f32, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
    let g = softmax_grad_aggregate(&rewards, &grads);
    assert_eq!(g.len(), 2);
    let v = log_mean_exp(&rewards);
    assert!(v > 0.5);
}

#[test]
fn guidance_coeff_positive_midtime() {
    let b = guidance_coefficient(0.4);
    assert!(b > 0.0);
}

#[test]
fn early_stop_ddpm_finite() {
    let y = glass::early_stop_ddpm(0.2, 0.3, glass::calc_s(0.2, 0.3), 0.1, 0.2);
    assert!(y.is_finite());
}
