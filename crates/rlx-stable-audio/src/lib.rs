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

//! # rlx-stable-audio
//!
//! **Stable Audio Open** on RLX — text-to-audio via a **rectified-flow DiT** over
//! an autoencoder latent, conditioned on a T5 text embedding plus timing
//! (seconds-total) features.
//!
//! Composition of existing RLX pieces:
//!
//! - **Sampler** → [`rlx_audio_blocks::sampling::FlowMatchEuler`] driven by this
//!   crate's rectified-flow [`sampler`] schedule.
//! - **Text conditioner** → a T5 encoder (native T5 lives in `rlx-parlertts`).
//! - **DiT** → reuse the flow-matching DiT patterns in `rlx-flux2` / `rlx-vlash`.
//! - **Autoencoder** → the "SAME" audio VAE (conv stack like `rlx-dac`/`rlx-encodec`).
//!
//! This crate currently provides the checkpoint-free, unit-tested core: the config
//! ([`StableAudioConfig`]) and the RF sampler schedule ([`sampler`]) with the
//! length-dependent timestep shift. Wiring the DiT + conditioner + autoencoder
//! graphs end-to-end is the next step (needs a checkpoint).

pub mod sampler;

pub use sampler::{DistributionShift, ShiftKind, effective_latent_length, make_schedule};

use rlx_audio_blocks::sampling::FlowMatchEuler;

/// Stable Audio model config (a faithful subset of the upstream config;
/// dimensional fields defaulted to `0` are read from the checkpoint).
#[derive(Debug, Clone, PartialEq)]
pub struct StableAudioConfig {
    // Audio / autoencoder.
    pub sample_rate: usize,
    pub audio_channels: usize,
    pub latent_dim: i64,
    pub downsampling_ratio: i64,
    pub io_channels: i64,
    // Conditioning.
    pub cond_dim: i64,
    pub prompt_max_length: i64,
    pub t5_hidden_size: i64,
    pub seconds_min: f32,
    pub seconds_max: f32,
    // Diffusion transformer (DiT).
    pub embed_dim: i64,
    pub depth: i64,
    pub num_heads: i64,
    pub cond_token_dim: i64,
    pub global_cond_dim: i64,
    pub diffusion_objective: String,
    pub transformer_type: String,
    pub stable_audio_open_v1: bool,
    // Sampler.
    pub distribution_shift: DistributionShift,
}

impl Default for StableAudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            audio_channels: 2,
            latent_dim: 64,
            downsampling_ratio: 2048,
            io_channels: 64,
            cond_dim: 768,
            prompt_max_length: 128,
            t5_hidden_size: 768,
            seconds_min: 0.0,
            seconds_max: 47.0,
            embed_dim: 1536,
            depth: 24,
            num_heads: 24,
            cond_token_dim: 768,
            global_cond_dim: 1536,
            diffusion_objective: "rectified_flow".to_string(),
            transformer_type: "continuous_transformer".to_string(),
            stable_audio_open_v1: true,
            distribution_shift: DistributionShift::default(),
        }
    }
}

impl StableAudioConfig {
    /// Build the rectified-flow Euler sampler for a clip of `seconds` at this
    /// model's rate, using `steps` denoise steps. Reuses the shared
    /// [`FlowMatchEuler`] over this crate's RF schedule.
    pub fn flow_match_scheduler(&self, steps: usize, seconds: f32) -> FlowMatchEuler {
        let seq_len = effective_latent_length(seconds, self.sample_rate, self.downsampling_ratio);
        FlowMatchEuler::from_sigmas(make_schedule(&self.distribution_shift, steps, seq_len, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let c = StableAudioConfig::default();
        assert_eq!(c.sample_rate, 44_100);
        assert_eq!(c.t5_hidden_size, 768);
        assert_eq!(c.diffusion_objective, "rectified_flow");
        assert_eq!(c.distribution_shift.kind, ShiftKind::LogSnr);
        assert_eq!(c.distribution_shift.base_shift, 0.5);
        assert_eq!(c.distribution_shift.max_shift, 1.15);
    }

    #[test]
    fn scheduler_reuses_flow_match_euler() {
        let c = StableAudioConfig::default();
        let steps = 16;
        let sched = c.flow_match_scheduler(steps, 10.0);
        // FlowMatchEuler holds steps+1 sigmas descending to 0.
        assert_eq!(sched.num_steps(), steps);
        assert_eq!(sched.sigmas.len(), steps + 1);
        assert_eq!(sched.sigmas[0], 1.0);
        assert_eq!(*sched.sigmas.last().unwrap(), 0.0);
        // A constant-velocity Euler integration over the schedule is finite.
        let v = vec![0.1f32; 3];
        let mut x = vec![0.0f32; 3];
        for i in 0..sched.num_steps() {
            x = sched.step(i, &x, &v);
        }
        assert!(x.iter().all(|q| q.is_finite()));
    }
}
