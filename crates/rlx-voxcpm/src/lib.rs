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

//! # rlx-voxcpm
//!
//! **VoxCPM** on RLX — a **tokenizer-free** multilingual TTS with voice design.
//! Instead of predicting discrete audio codes, VoxCPM runs a **MiniCPM-style LM
//! backbone** whose hidden states condition a **local flow-matching** head that
//! generates continuous acoustic latents per frame; a vocoder renders the latents.
//!
//! Native Rust, composing rlx pieces:
//!
//! - **Backbone** → MiniCPM (Llama-shaped → `rlx-minicpm5` / `rlx-llama32`).
//! - **Local flow head** → [`rlx_audio_blocks::sampling::FlowMatchEuler::ascending`]
//!   (noise → data) + [`classifier_free_guidance`].
//! - **Vocoder** → a mel/latent vocoder (`rlx-neutts` BigVGAN / `rlx-tsac` HiFT).
//!
//! Checkpoint-free, unit-tested core here: the config, the local-flow scheduler,
//! CFG, and the LM-step → acoustic-frame expansion. Wiring the backbone + flow DiT
//! + vocoder graphs is the next step.

use anyhow::{Result, ensure};
use rlx_audio_blocks::sampling::{FlowMatchEuler, classifier_free_guidance};

/// VoxCPM model config. Backbone dims are MiniCPM-shaped; acoustic/flow fields
/// carry typical values, exact widths from the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct VoxCpmConfig {
    pub sample_rate: usize,
    pub hop_length: usize,
    // MiniCPM backbone.
    pub backbone_hidden: usize,
    pub backbone_layers: usize,
    pub backbone_heads: usize,
    pub backbone_kv_heads: usize,
    /// Continuous acoustic latent width the local flow head generates.
    pub acoustic_dim: usize,
    /// Acoustic frames emitted per backbone step (local patch size).
    pub frames_per_step: usize,
    /// Default number of local flow-matching steps.
    pub flow_steps: usize,
    /// Classifier-free guidance scale.
    pub cfg_scale: f32,
}

impl Default for VoxCpmConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            hop_length: 256,
            backbone_hidden: 1024,
            backbone_layers: 24,
            backbone_heads: 16,
            backbone_kv_heads: 16,
            acoustic_dim: 64,
            frames_per_step: 1,
            flow_steps: 10,
            cfg_scale: 2.0,
        }
    }
}

impl VoxCpmConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.acoustic_dim > 0, "acoustic_dim must be > 0");
        ensure!(self.frames_per_step > 0, "frames_per_step must be > 0");
        ensure!(self.flow_steps > 0, "flow_steps must be > 0");
        ensure!(
            self.backbone_heads.is_multiple_of(self.backbone_kv_heads),
            "backbone_heads must be divisible by backbone_kv_heads (GQA)"
        );
        Ok(())
    }

    /// Acoustic frame rate (`sample_rate / hop_length`).
    pub fn frames_per_second(&self) -> f32 {
        self.sample_rate as f32 / self.hop_length as f32
    }

    /// Number of acoustic frames produced for `lm_steps` backbone steps.
    pub fn acoustic_frames(&self, lm_steps: usize) -> usize {
        lm_steps * self.frames_per_step
    }

    /// The local flow-matching sampler (noise → data) for one acoustic patch.
    pub fn local_flow_scheduler(&self, steps: usize) -> FlowMatchEuler {
        FlowMatchEuler::ascending(steps)
    }

    /// Apply this model's classifier-free guidance to a (cond, uncond) velocity.
    pub fn guided(&self, v_cond: &[f32], v_uncond: &[f32]) -> Vec<f32> {
        classifier_free_guidance(v_cond, v_uncond, self.cfg_scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_validate() {
        let c = VoxCpmConfig::default();
        assert_eq!(c.sample_rate, 16_000);
        assert_eq!(c.acoustic_dim, 64);
        c.validate().unwrap();
    }

    #[test]
    fn gqa_divisibility_enforced() {
        let c = VoxCpmConfig {
            backbone_kv_heads: 7, // 16 % 7 != 0
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn acoustic_frame_expansion() {
        let c = VoxCpmConfig {
            frames_per_step: 4,
            ..Default::default()
        };
        assert_eq!(c.acoustic_frames(10), 40);
    }

    #[test]
    fn local_flow_is_noise_to_data() {
        let c = VoxCpmConfig::default();
        let sched = c.local_flow_scheduler(12);
        assert_eq!(sched.num_steps(), 12);
        assert_eq!(sched.sigmas[0], 0.0);
        assert_eq!(*sched.sigmas.last().unwrap(), 1.0);
    }

    #[test]
    fn guided_applies_cfg_scale() {
        let c = VoxCpmConfig {
            cfg_scale: 3.0,
            ..Default::default()
        };
        // uncond + 3*(cond-uncond): 0+3*(1-0)=3
        assert_eq!(c.guided(&[1.0], &[0.0]), vec![3.0]);
    }
}
