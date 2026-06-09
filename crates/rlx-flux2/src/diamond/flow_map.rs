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

//! Flow-map style denoiser calls: dual-time embedding for (t → t′) and x0 lookahead (t → 0).

use crate::runner::Flux2Runner;
use anyhow::{Result, ensure};

/// Velocity prediction and flow-map x0 estimate at the current noise level.
#[derive(Debug, Clone)]
pub struct FlowMapPrediction {
    pub noise_pred: Vec<f32>,
    /// `x - σ v0` with v0 from dual-time forward at target σ=0.
    pub x0_hat: Vec<f32>,
}

/// One dual-timestep denoiser evaluation (native or compiled).
pub fn forward_noise_dual(
    runner: &Flux2Runner,
    hidden_states: &[f32],
    encoder_hidden_states: &[f32],
    sigma: f32,
    sigma_target: f32,
    guidance: Option<&[f32]>,
    img_ids: &[f32],
    txt_ids: &[f32],
) -> Result<Vec<f32>> {
    let batch = runner.batch();
    let timestep = vec![sigma; batch];
    let target = vec![sigma_target; batch];
    if runner.uses_compiled_denoiser() {
        runner.forward_noise_dual_compiled(
            hidden_states,
            encoder_hidden_states,
            &timestep,
            &target,
            guidance,
            img_ids,
            txt_ids,
        )
    } else {
        runner.forward_noise_dual_native(
            hidden_states,
            encoder_hidden_states,
            &timestep,
            &target,
            guidance,
            img_ids,
            txt_ids,
        )
    }
}

/// Flow-map step: predict noise for scheduler (t → t_next) and x0_hat (t → 0).
pub fn flow_map_predict(
    runner: &Flux2Runner,
    latents: &[f32],
    sigma: f32,
    sigma_next: f32,
    encoder_hidden_states: &[f32],
    guidance: Option<&[f32]>,
    img_ids: &[f32],
    txt_ids: &[f32],
) -> Result<FlowMapPrediction> {
    ensure!(latents.len() == runner.batch() * runner.img_seq() * runner.config().in_channels);
    let noise_pred = forward_noise_dual(
        runner,
        latents,
        encoder_hidden_states,
        sigma,
        sigma_next,
        guidance,
        img_ids,
        txt_ids,
    )?;
    let v0 = forward_noise_dual(
        runner,
        latents,
        encoder_hidden_states,
        sigma,
        0.0,
        guidance,
        img_ids,
        txt_ids,
    )?;
    let x0_hat = latents
        .iter()
        .zip(v0.iter())
        .map(|(&x, &v)| x - sigma * v)
        .collect();
    Ok(FlowMapPrediction { noise_pred, x0_hat })
}
