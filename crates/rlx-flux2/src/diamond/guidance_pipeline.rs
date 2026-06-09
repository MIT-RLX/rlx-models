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

//! Rectified-flow sampling with Diamond Maps reward guidance.

use super::decode_reward::hybrid_reward;
use super::dps_guidance::apply_dps_guidance_step;
use super::glass_sampler::glass_posterior_sample;
use super::params::{DiamondGuidanceParams, DiamondMethod};
use super::weighted_guidance::apply_weighted_guidance_step;
use crate::latent_ops::{
    concat_latent_ids, concat_packed_latents, prepare_latent_ids, slice_gen_noise,
};
use crate::pipeline::{Flux2SampleOutput, Flux2SampleParams, init_latent_noise};
use crate::runner::Flux2Runner;
use crate::scheduler::{flow_match_euler_step, flow_match_sigmas};
use anyhow::{Result, ensure};
use rlx_diamond::{
    LatentReward, flux_guided_euler_step, grad_xt_via_z, log_mean_exp, softmax_grad_aggregate,
};

fn apply_glass_guidance<R: LatentReward>(
    runner: &Flux2Runner,
    latents: &mut [f32],
    noise: &[f32],
    sigma: f32,
    sigma_next: f32,
    encoder: &[f32],
    guidance: Option<&[f32]>,
    img_ids: &[f32],
    txt_ids: &[f32],
    gen_seq: usize,
    step_index: usize,
    diamond: &DiamondGuidanceParams,
    reward: &R,
) -> Result<()> {
    let mut rewards = Vec::with_capacity(diamond.mc_samples);
    let mut grads = Vec::with_capacity(diamond.mc_samples);
    let dim = latents.len();
    let mut z_buf = vec![0.0f32; dim];

    for k in 0..diamond.mc_samples {
        let pseed = diamond
            .seed
            .wrapping_add((step_index as u64) << 16)
            .wrapping_add(k as u64);
        glass_posterior_sample(
            runner, diamond, sigma, latents, encoder, guidance, img_ids, txt_ids, gen_seq, pseed,
            &mut z_buf,
        )?;
        let r_val = reward.reward(&z_buf) * diamond.reward_scale;
        let grad_z = reward.grad_wrt_z(&z_buf);
        rewards.push(r_val);
        grads.push(grad_xt_via_z(&grad_z));
    }

    let grad_v = softmax_grad_aggregate(&rewards, &grads);
    let _v = log_mean_exp(&rewards);
    flux_guided_euler_step(
        latents,
        noise,
        &grad_v,
        sigma,
        sigma_next,
        diamond.max_guidance_b,
    );
    Ok(())
}

/// Sample with Diamond Maps reward guidance on the last `guidance_steps` steps.
pub fn sample_rectified_flow_diamond<R: LatentReward>(
    runner: &Flux2Runner,
    params: &Flux2SampleParams<'_>,
    diamond: &DiamondGuidanceParams,
    reward: &R,
) -> Result<Flux2SampleOutput> {
    let cfg = runner.config();
    let batch = runner.batch();
    let txt_seq = runner.txt_seq();
    let gen_seq = params.latent_h * params.latent_w;
    ensure!(params.encoder_hidden_states.len() == batch * txt_seq * cfg.joint_attention_dim);

    let gen_ids = prepare_latent_ids(batch, params.latent_h, params.latent_w);
    let (img_ids, _total_seq) = if let Some(r) = params.reference {
        (
            concat_latent_ids(&gen_ids, &r.img_ids, batch),
            gen_seq + r.seq,
        )
    } else {
        (gen_ids.clone(), gen_seq)
    };

    runner.warmup_denoiser(&img_ids, params.txt_ids)?;

    let mut latents = if let Some(init) = params.initial_latents {
        init.to_vec()
    } else {
        init_latent_noise(batch, gen_seq, cfg.in_channels, params.seed)
    };
    ensure!(latents.len() == batch * gen_seq * cfg.in_channels);

    let sigmas = flow_match_sigmas(params.num_inference_steps);
    let default_guidance = vec![3.5f32; batch];
    let guidance = params.guidance.unwrap_or(&default_guidance);
    let init_step = params.init_timestep.min(params.num_inference_steps);
    let guidance_start = params
        .num_inference_steps
        .saturating_sub(diamond.guidance_steps)
        .max(init_step);

    for i in init_step..params.num_inference_steps {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];
        let timestep = vec![sigma; batch];

        let hidden = if let Some(r) = params.reference {
            concat_packed_latents(&latents, &r.packed, batch, cfg.in_channels)
        } else {
            latents.clone()
        };

        let noise = if params.cfg_scale > 1.0 {
            if let (Some(neg_e), Some(neg_ids)) = (params.encoder_negative, params.neg_txt_ids) {
                runner
                    .forward_cfg(
                        &hidden,
                        params.encoder_hidden_states,
                        neg_e,
                        &timestep,
                        Some(guidance),
                        &img_ids,
                        params.txt_ids,
                        neg_ids,
                        params.cfg_scale,
                    )?
                    .noise_pred
            } else {
                runner
                    .forward(
                        &hidden,
                        params.encoder_hidden_states,
                        &timestep,
                        Some(guidance),
                        &img_ids,
                        params.txt_ids,
                    )?
                    .noise_pred
            }
        } else {
            runner
                .forward(
                    &hidden,
                    params.encoder_hidden_states,
                    &timestep,
                    Some(guidance),
                    &img_ids,
                    params.txt_ids,
                )?
                .noise_pred
        };

        let noise = if params.reference.is_some() {
            slice_gen_noise(&noise, batch, cfg.in_channels, gen_seq)
        } else {
            noise
        };
        ensure!(noise.len() == latents.len());

        let do_guidance = i >= guidance_start && diamond.mc_samples > 0;

        if do_guidance {
            match diamond.method {
                DiamondMethod::Glass => {
                    apply_glass_guidance(
                        runner,
                        &mut latents,
                        &noise,
                        sigma,
                        sigma_next,
                        params.encoder_hidden_states,
                        Some(guidance),
                        &img_ids,
                        params.txt_ids,
                        gen_seq,
                        i,
                        diamond,
                        reward,
                    )?;
                }
                DiamondMethod::Weighted => {
                    let hybrid = hybrid_reward(
                        reward,
                        runner,
                        &gen_ids,
                        params.latent_h,
                        params.latent_w,
                        diamond.decode_reward,
                    );
                    apply_weighted_guidance_step(
                        runner,
                        &mut latents,
                        &noise,
                        sigma,
                        sigma_next,
                        params.encoder_hidden_states,
                        Some(guidance),
                        &img_ids,
                        params.txt_ids,
                        &hybrid,
                        diamond,
                        i,
                    )?;
                }
                DiamondMethod::Dps => {
                    apply_dps_guidance_step(
                        &mut latents,
                        &noise,
                        sigma,
                        sigma_next,
                        reward,
                        diamond,
                    );
                }
            }
        } else {
            flow_match_euler_step(&mut latents, &noise, sigma, sigma_next);
        }
    }

    Ok(Flux2SampleOutput {
        latents,
        img_ids: gen_ids,
        img_seq: gen_seq,
    })
}
