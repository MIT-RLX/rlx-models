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

//! Diamond Maps: inference-time reward alignment via stochastic flow maps.
//!
//! Implements host-side math from [arXiv:2602.05993](https://arxiv.org/abs/2602.05993)
//! (value functions, GLASS flows, weighted renoising). Model integration lives in
//! `rlx-flux2::diamond`.
//!
//! **No retraining** of the base generative model is required for GLASS-based guidance;
//! multi-step GLASS posterior sampling uses the frozen denoiser as a reference.

pub mod flux_time;
pub mod glass;
pub mod glass_integrate;
pub mod guidance;
pub mod interpolant;
pub mod reward;
pub mod value;
pub mod weighted;

pub use flux_time::{
    flux_guidance_coefficient, flux_sigma_to_paper_t, flux_x0_from_velocity,
    paper_denoiser_from_flux, paper_guidance_coefficient_from_flux_sigma, paper_t_to_flux_sigma,
};
pub use glass::{
    calc_s, early_stop_ddpm, glass_velocity, reparam_input, reparam_time, sample_inner_state,
    sufficient_stat,
};
pub use glass_integrate::{DenoiserReference, sample_posterior};
pub use guidance::{euler_step_vec, flux_guided_euler_step, guided_velocity};
pub use interpolant::{denoiser_from_velocity, guidance_coefficient, velocity_from_denoiser};
pub use reward::{BluenessReward, LatentReward, LinearMeasurementReward, grad_xt_via_z, spsa_grad};
pub use value::{LogSumExpGrad, log_mean_exp, logaddexp, softmax_grad_aggregate, softmax_weights};
pub use weighted::{
    guidance_b, particle_logit_full, particle_logit_reward_only, renoise, renoise_params, score,
    t_prime_from_snr,
};
