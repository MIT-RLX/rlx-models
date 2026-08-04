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

//! Sampling utilities for diffusion / flow-matching audio generators:
//! a seedable RNG ([`rng`]) and noise schedules + denoise steppers
//! ([`schedulers`]).

pub mod guidance;
pub mod rng;
pub mod schedulers;

pub use guidance::classifier_free_guidance;
pub use rng::Rng;
pub use schedulers::{
    BetaSchedule, FlowMatchEuler, alphas_cumprod, betas, ddpm_posterior_mean, sd3_shifted_sigmas,
    sd3_time_shift,
};
