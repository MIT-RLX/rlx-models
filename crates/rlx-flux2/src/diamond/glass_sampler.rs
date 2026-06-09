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

//! GLASS posterior sampling using [`Flux2Runner`] as denoiser reference.

use super::params::DiamondGuidanceParams;
use crate::runner::Flux2Runner;
use anyhow::{Result, ensure};
use rlx_diamond::{
    DenoiserReference, flux_sigma_to_paper_t, paper_t_to_flux_sigma, sample_posterior,
};

/// RNG for particle / inner noise (xorshift).
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

/// Reference wrapper for one FLUX denoiser forward at a given paper time.
pub struct FluxGlassReference<'a> {
    pub runner: &'a Flux2Runner,
    pub encoder: &'a [f32],
    pub guidance: Option<&'a [f32]>,
    pub img_ids: &'a [f32],
    pub txt_ids: &'a [f32],
    pub batch: usize,
}

impl DenoiserReference for FluxGlassReference<'_> {
    fn denoise(&self, t_star: f32, x_star: &[f32], out: &mut [f32]) {
        let sigma = paper_t_to_flux_sigma(t_star);
        let timestep = vec![sigma; self.batch];
        let noise_pred = self
            .runner
            .forward(
                x_star,
                self.encoder,
                &timestep,
                self.guidance,
                self.img_ids,
                self.txt_ids,
            )
            .map(|o| o.noise_pred)
            .unwrap_or_else(|_| vec![0.0; x_star.len()]);
        let n = out.len().min(x_star.len()).min(noise_pred.len());
        for i in 0..n {
            out[i] = x_star[i] - sigma * noise_pred[i];
        }
    }
}

/// Sample z ~ p_{1|t}(·|x_t) via multi-step GLASS (outer noise level `sigma_flux`).
pub fn glass_posterior_sample(
    runner: &Flux2Runner,
    params: &DiamondGuidanceParams,
    sigma_flux: f32,
    x_t: &[f32],
    encoder: &[f32],
    guidance: Option<&[f32]>,
    img_ids: &[f32],
    txt_ids: &[f32],
    gen_seq: usize,
    particle_seed: u64,
    out_z: &mut [f32],
) -> Result<()> {
    let batch = runner.batch();
    let channels = runner.config().in_channels;
    ensure!(x_t.len() == batch * gen_seq * channels);
    ensure!(out_z.len() == x_t.len());
    let _ = sigma_flux;

    let t = flux_sigma_to_paper_t(sigma_flux);
    let t_prime = 1.0f32;
    let mut noise = vec![0.0f32; x_t.len()];
    fill_gaussian(&mut noise, particle_seed);

    let denoiser_ref = FluxGlassReference {
        runner,
        encoder,
        guidance,
        img_ids,
        txt_ids,
        batch,
    };
    sample_posterior(
        &denoiser_ref,
        t,
        t_prime,
        x_t,
        params.inner_steps,
        &noise,
        out_z,
    );
    Ok(())
}
