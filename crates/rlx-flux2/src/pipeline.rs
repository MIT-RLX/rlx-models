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

//! Multi-step rectified-flow sampling and end-to-end generate helpers.

use super::conditioning::Flux2ReferenceConditioning;
use super::latent_ops::{
    concat_latent_ids, concat_packed_latents, prepare_latent_ids, slice_gen_noise,
};
use super::scheduler::{flow_match_euler_step, flow_match_sigmas};
use crate::runner::Flux2Runner;
use anyhow::{Result, ensure};

/// Sampling / generation options.
#[derive(Debug, Clone)]
pub struct Flux2SampleParams<'a> {
    pub encoder_hidden_states: &'a [f32],
    pub encoder_negative: Option<&'a [f32]>,
    pub txt_ids: &'a [f32],
    pub neg_txt_ids: Option<&'a [f32]>,
    pub num_inference_steps: usize,
    pub cfg_scale: f32,
    pub guidance: Option<&'a [f32]>,
    pub latent_h: usize,
    pub latent_w: usize,
    pub seed: u64,
    /// img2img: starting step index (from `flow_match_init_timestep`).
    pub init_timestep: usize,
    /// img2img: pre-blended latents (skips fresh noise init when set).
    pub initial_latents: Option<&'a [f32]>,
    /// Edit mode: fixed reference tokens concatenated before each forward.
    pub reference: Option<&'a Flux2ReferenceConditioning>,
}

#[derive(Debug)]
pub struct Flux2SampleOutput {
    pub latents: Vec<f32>,
    pub img_ids: Vec<f32>,
    pub img_seq: usize,
}

/// Initialize Gaussian latents `[batch, img_seq, in_channels]`.
pub fn init_latent_noise(batch: usize, img_seq: usize, channels: usize, seed: u64) -> Vec<f32> {
    let n = batch * img_seq * channels;
    let mut out = vec![0.0f32; n];
    let mut state = seed.wrapping_add(1);
    for v in &mut out {
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
        *v = r * theta.cos() * 0.5;
    }
    out
}

fn img_seq_from_ids(img_ids: &[f32], batch: usize) -> usize {
    img_ids.len() / (batch * 4)
}

/// Run Flow-Match Euler steps on the denoiser (compiled or native per [`Flux2Runner`]).
pub fn sample_rectified_flow(
    runner: &Flux2Runner,
    params: &Flux2SampleParams<'_>,
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
        flow_match_euler_step(&mut latents, &noise, sigma, sigma_next);
    }

    Ok(Flux2SampleOutput {
        latents,
        img_ids: gen_ids,
        img_seq: gen_seq,
    })
}

/// Sample then VAE-decode to RGB u8 when VAE weights are loaded.
pub fn generate_to_rgb(
    runner: &Flux2Runner,
    params: &Flux2SampleParams<'_>,
) -> Result<(Vec<u8>, u32, u32)> {
    let sample = sample_rectified_flow(runner, params)?;
    let (rgb, h, w) = runner.decode_to_rgb(
        &sample.latents,
        &sample.img_ids,
        params.latent_h,
        params.latent_w,
    )?;
    Ok((rgb, h, w))
}

/// Write planar RGB u8 HWC to a simple PPM (for `rlx-flux2 --output`).
pub fn write_ppm(path: &std::path::Path, rgb: &[u8], width: u32, height: u32) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "P6")?;
    writeln!(f, "{width} {height}")?;
    writeln!(f, "255")?;
    let w = width as usize;
    let h = height as usize;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            f.write_all(&rgb[i..i + 3])?;
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn _img_seq_helper(ids: &[f32], batch: usize) -> usize {
    img_seq_from_ids(ids, batch)
}
