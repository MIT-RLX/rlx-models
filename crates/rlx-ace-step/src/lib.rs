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

//! # rlx-ace-step
//!
//! **ACE-Step** music generation on RLX — a **flow-matching DiT** over a music
//! autoencoder (DCAE) latent, conditioned on a text/tag encoder (UMT5) plus a
//! lyric encoder, sampled with an SD3-style shifted flow schedule and
//! classifier-free guidance. All native Rust, composing rlx pieces:
//!
//! - **Flow sampler** → [`rlx_audio_blocks::sampling::FlowMatchEuler`] over an
//!   SD3-shifted sigma schedule ([`sd3_shifted_sigmas`]).
//! - **Guidance** → [`classifier_free_guidance`].
//! - **Text/tag + lyric conditioner** → UMT5 (native T5 in `rlx-parlertts`).
//! - **DiT** → flow-matching transformer patterns in `rlx-flux2` / `rlx-vlash`.
//! - **Music autoencoder (DCAE)** → conv codec stack (`rlx-dac` / `rlx-encodec`).
//!
//! Checkpoint-free, unit-tested core here: the config + the shifted flow schedule +
//! the guided step. DiT/conditioner/DCAE graph wiring is the next step.

use anyhow::{Result, ensure};
use rlx_audio_blocks::sampling::{FlowMatchEuler, classifier_free_guidance, sd3_shifted_sigmas};

/// ACE-Step model config. Dimensional fields carry typical ACE-Step values; exact
/// widths come from the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct AceStepConfig {
    pub sample_rate: usize,
    /// DCAE music-latent channel count.
    pub latent_channels: usize,
    /// Autoencoder time compression (audio samples per latent frame).
    pub latent_downsample: usize,
    // Flow-matching DiT.
    pub dit_hidden: usize,
    pub dit_depth: usize,
    pub dit_heads: usize,
    /// UMT5 text/tag encoder hidden size.
    pub text_encoder_hidden: usize,
    /// Lyric-token vocabulary size.
    pub lyric_vocab_size: usize,
    /// SD3-style discrete-flow timestep shift.
    pub flow_shift: f32,
    /// Classifier-free guidance scale.
    pub guidance_scale: f32,
    /// Default number of flow-matching inference steps.
    pub default_steps: usize,
}

impl Default for AceStepConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            latent_channels: 8,
            latent_downsample: 512,
            dit_hidden: 2560,
            dit_depth: 24,
            dit_heads: 20,
            text_encoder_hidden: 768, // umt5-base
            lyric_vocab_size: 6693,
            flow_shift: 3.0,
            guidance_scale: 7.0,
            default_steps: 60,
        }
    }
}

impl AceStepConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.sample_rate > 0, "sample_rate must be > 0");
        ensure!(self.latent_channels > 0, "latent_channels must be > 0");
        ensure!(self.flow_shift > 0.0, "flow_shift must be > 0");
        ensure!(self.default_steps > 0, "default_steps must be > 0");
        Ok(())
    }

    /// Build the flow-matching Euler sampler with this model's SD3 timestep shift.
    pub fn flow_scheduler(&self, steps: usize) -> FlowMatchEuler {
        FlowMatchEuler::from_sigmas(sd3_shifted_sigmas(steps, self.flow_shift))
    }

    /// Apply this model's classifier-free guidance to a (cond, uncond) velocity.
    pub fn guided(&self, v_cond: &[f32], v_uncond: &[f32]) -> Vec<f32> {
        classifier_free_guidance(v_cond, v_uncond, self.guidance_scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_validate() {
        let c = AceStepConfig::default();
        assert_eq!(c.sample_rate, 44_100);
        assert_eq!(c.text_encoder_hidden, 768);
        assert_eq!(c.flow_shift, 3.0);
        assert_eq!(c.guidance_scale, 7.0);
        c.validate().unwrap();
    }

    #[test]
    fn flow_scheduler_uses_sd3_shift() {
        let c = AceStepConfig::default();
        let steps = 30;
        let sched = c.flow_scheduler(steps);
        assert_eq!(sched.num_steps(), steps);
        assert_eq!(sched.sigmas[0], 1.0);
        assert_eq!(*sched.sigmas.last().unwrap(), 0.0);
        // With shift 3, an interior sigma is above the linear reference.
        let linear = FlowMatchEuler::uniform(steps);
        let mid = steps / 2;
        assert!(sched.sigmas[mid] > linear.sigmas[mid]);
    }

    #[test]
    fn guided_applies_cfg_scale() {
        let c = AceStepConfig {
            guidance_scale: 2.0,
            ..Default::default()
        };
        // uncond + 2*(cond-uncond): [0]+2*(1-0)=2, [10]+2*(12-10)=14
        assert_eq!(c.guided(&[1.0, 12.0], &[0.0, 10.0]), vec![2.0, 14.0]);
    }
}
