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

//! Weighted Diamond Maps via renoising + x0 lookahead (single-timestep denoiser).
//!
//! Full dual-timestep flow-map LoRA is not required; this uses renoising at t′ with
//! velocity-based x0 estimates (paper §5, single-forward variant).

use super::dps_guidance::flux_x0_estimate;
use super::flow_map::flow_map_predict;
use super::params::DiamondGuidanceParams;
use crate::runner::Flux2Runner;
use anyhow::Result;
use rlx_diamond::{
    LatentReward, flux_guided_euler_step, grad_xt_via_z, particle_logit_full,
    particle_logit_reward_only, renoise, renoise_params, score, softmax_weights, t_prime_from_snr,
};

fn fill_gaussian(out: &mut [f32], seed: u64) {
    let mut state = seed.wrapping_add(1);
    for v in out {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = (state as f32) / (u32::MAX as f32);
        let r = (-2.0 * u.max(1e-7).ln()).sqrt();
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u2 = (state as f32) / (u32::MAX as f32);
        let theta = 2.0 * std::f32::consts::PI * u2;
        *v = r * theta.cos();
    }
}

fn normalize_grad_by_norm(grad: &[f32], batch: usize) -> Vec<f32> {
    let dim = grad.len() / batch.max(1);
    let mut unit = vec![0.0f32; grad.len()];
    for b in 0..batch {
        let start = b * dim;
        let end = start + dim;
        let n: f32 = grad[start..end]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-8);
        for i in start..end {
            unit[i] = grad[i] / n;
        }
    }
    unit
}

/// Compute weighted Diamond gradient w.r.t. `latents` at noise level `sigma`.
pub fn weighted_diamond_grad<R: LatentReward>(
    runner: &Flux2Runner,
    latents: &[f32],
    sigma: f32,
    encoder: &[f32],
    guidance: Option<&[f32]>,
    img_ids: &[f32],
    txt_ids: &[f32],
    reward: &R,
    diamond: &DiamondGuidanceParams,
    main_velocity: &[f32],
    sigma_next: f32,
    particle_seed: u64,
) -> Result<Vec<f32>> {
    let batch = runner.batch();
    let t_prime = t_prime_from_snr(sigma, diamond.snr_factor);
    let (scale_factor, std_q) = renoise_params(sigma, t_prime);
    let alpha_t = 1.0 - sigma;
    let var_t = sigma * sigma;
    let alpha_prev = 1.0 - t_prime;
    let var_prev = t_prime * t_prime;

    let x0_main = flux_x0_estimate(latents, sigma, main_velocity);
    let mut score_t = vec![0.0f32; latents.len()];
    if diamond.include_score || diamond.include_weights {
        for i in 0..latents.len() {
            score_t[i] = score(latents[i], alpha_t, x0_main[i], var_t, 1e-4);
        }
    }

    let mut reward_grads = Vec::with_capacity(diamond.mc_samples);
    let mut likelihood_grads = Vec::with_capacity(diamond.mc_samples);
    let mut score_grads = Vec::with_capacity(diamond.mc_samples);
    let mut logits = Vec::with_capacity(diamond.mc_samples);

    let mut eps_buf = vec![0.0f32; latents.len()];

    for k in 0..diamond.mc_samples {
        fill_gaussian(&mut eps_buf, particle_seed.wrapping_add(k as u64));
        let renoised = renoise(latents, scale_factor, std_q, &eps_buf);

        let (x0_hat, x0_inst) = if diamond.use_flow_map {
            let pred = flow_map_predict(
                runner, &renoised, t_prime, sigma_next, encoder, guidance, img_ids, txt_ids,
            )?;
            (
                pred.x0_hat,
                flux_x0_estimate(&renoised, t_prime, &pred.noise_pred),
            )
        } else {
            let timestep = vec![t_prime; batch];
            let v = runner
                .forward(&renoised, encoder, &timestep, guidance, img_ids, txt_ids)?
                .noise_pred;
            let inst = flux_x0_estimate(&renoised, t_prime, &v);
            (inst.clone(), inst)
        };

        let r_val = reward.reward(&x0_hat) * diamond.reward_scale;
        let grad_z = reward.grad_wrt_z(&x0_hat);
        let mut grad_xt = grad_xt_via_z(&grad_z);
        for g in &mut grad_xt {
            *g *= scale_factor;
        }

        let mut likelihood_g = vec![0.0f32; latents.len()];
        if diamond.include_likelihood {
            for i in 0..latents.len() {
                likelihood_g[i] = -(latents[i] - alpha_t * x0_inst[i]) / var_t.max(1e-4);
            }
        }

        let mut score_g = vec![0.0f32; latents.len()];
        let mut score_tp = vec![0.0f32; latents.len()];
        if diamond.include_score || diamond.include_weights {
            for i in 0..latents.len() {
                score_tp[i] = score(renoised[i], alpha_prev, x0_inst[i], var_prev, 1e-4);
            }
        }
        if diamond.include_score {
            for i in 0..latents.len() {
                score_g[i] = (score_tp[i] - score_t[i]) * scale_factor;
            }
        }

        let log_p: f32 = latents
            .iter()
            .zip(x0_inst.iter())
            .map(|(&x, &x0)| {
                let res = x - alpha_t * x0;
                -0.5 * res * res / var_t.max(1e-4)
            })
            .sum();

        let eps_norm: f32 = eps_buf.iter().map(|e| 0.5 * e * e).sum();

        let gamma_k: f32 = if diamond.include_weights {
            latents
                .iter()
                .zip(renoised.iter())
                .zip(score_t.iter())
                .zip(score_tp.iter())
                .map(|(((&xt, &rn), &st), &stp)| 0.5 * (st + stp) * (rn - xt))
                .sum()
        } else {
            0.0
        };

        let logit = if diamond.include_weights {
            particle_logit_full(
                r_val,
                1.0,
                log_p,
                gamma_k,
                eps_norm,
                diamond.weight_temperature,
            )
        } else {
            particle_logit_reward_only(r_val, 1.0)
        };

        reward_grads.push(grad_xt);
        likelihood_grads.push(likelihood_g);
        score_grads.push(score_g);
        logits.push(logit);
    }

    let weights = softmax_weights(&logits);
    let dim = latents.len();
    let mut w_reward = vec![0.0f32; dim];
    let mut w_like = vec![0.0f32; dim];
    let mut w_score = vec![0.0f32; dim];
    for (w, (gr, (gl, gs))) in weights.iter().zip(
        reward_grads
            .iter()
            .zip(likelihood_grads.iter().zip(score_grads.iter())),
    ) {
        for i in 0..dim {
            w_reward[i] += w * gr[i];
            w_like[i] += w * gl[i];
            w_score[i] += w * gs[i];
        }
    }

    let ur = normalize_grad_by_norm(&w_reward, batch);
    let ul = if diamond.include_likelihood {
        normalize_grad_by_norm(&w_like, batch)
    } else {
        vec![0.0f32; dim]
    };
    let us = if diamond.include_score {
        normalize_grad_by_norm(&w_score, batch)
    } else {
        vec![0.0f32; dim]
    };

    let mut combined = vec![0.0f32; dim];
    for i in 0..dim {
        combined[i] = (ur[i] + ul[i] + us[i]) * diamond.gradient_norm_scale;
    }
    Ok(combined)
}

/// One guided Euler step with weighted Diamond gradient.
pub fn apply_weighted_guidance_step<R: LatentReward>(
    runner: &Flux2Runner,
    latents: &mut [f32],
    velocity: &[f32],
    sigma: f32,
    sigma_next: f32,
    encoder: &[f32],
    guidance: Option<&[f32]>,
    img_ids: &[f32],
    txt_ids: &[f32],
    reward: &R,
    diamond: &DiamondGuidanceParams,
    step_index: usize,
) -> Result<()> {
    let grad = weighted_diamond_grad(
        runner,
        latents,
        sigma,
        encoder,
        guidance,
        img_ids,
        txt_ids,
        reward,
        diamond,
        velocity,
        sigma_next,
        diamond.seed.wrapping_add(step_index as u64),
    )?;
    flux_guided_euler_step(
        latents,
        velocity,
        &grad,
        sigma,
        sigma_next,
        diamond.max_guidance_b,
    );
    Ok(())
}
