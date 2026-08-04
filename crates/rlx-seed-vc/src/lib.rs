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

//! # rlx-seed-vc
//!
//! **Seed-VC** zero-shot voice conversion on RLX. Seed-VC converts a source
//! utterance to a reference timbre with a **conditional flow-matching (CFM)** DiT
//! that predicts a mel/latent from three conditions — source *content* features,
//! reference *speaker* embedding, and (for the singing model) *F0* — followed by a
//! vocoder. All native Rust, composing existing rlx pieces:
//!
//! - **CFM sampler** → [`rlx_audio_blocks::sampling::FlowMatchEuler`] over this
//!   crate's `0 → 1` flow schedule, with classifier-free guidance ([`cfg_blend`]).
//! - **Speaker embedding** → CAM++ (`rlx-funasr`).
//! - **Content encoder** → a Whisper/HuBERT-style encoder (`rlx-whisper` /
//!   `rlx-neutts`'s Wav2Vec2-BERT path).
//! - **Vocoder** → BigVGAN (`rlx-neutts`).
//!
//! This crate provides the checkpoint-free, unit-tested core: the config, the CFM
//! flow schedule, and the CFG-guided Euler step. Wiring the DiT + encoders +
//! vocoder graphs end-to-end is the next step (needs a checkpoint).

use anyhow::{Result, ensure};
use rlx_audio_blocks::sampling::FlowMatchEuler;

/// Seed-VC model config. Dimensional fields carry typical Seed-VC values; exact
/// widths come from the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct SeedVcConfig {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub win_length: usize,
    /// Source content-feature width (e.g. Whisper/HuBERT hidden size).
    pub content_dim: usize,
    /// Reference speaker-embedding width (CAM++ = 192).
    pub speaker_dim: usize,
    // CFM DiT.
    pub dit_hidden: usize,
    pub dit_depth: usize,
    pub dit_heads: usize,
    /// Whether F0 conditioning is used (the singing-voice model).
    pub use_f0: bool,
    /// Default number of flow-matching inference steps.
    pub diffusion_steps: usize,
    /// Classifier-free guidance scale (`inference_cfg_rate`).
    pub cfg_rate: f32,
}

impl Default for SeedVcConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22_050,
            n_mels: 80,
            hop_length: 256,
            n_fft: 1024,
            win_length: 1024,
            content_dim: 768,
            speaker_dim: 192,
            dit_hidden: 512,
            dit_depth: 13,
            dit_heads: 8,
            use_f0: false,
            diffusion_steps: 10,
            cfg_rate: 0.7,
        }
    }
}

impl SeedVcConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.sample_rate > 0, "sample_rate must be > 0");
        ensure!(
            self.n_mels > 0 && self.hop_length > 0,
            "invalid mel geometry"
        );
        ensure!(self.speaker_dim > 0, "speaker_dim must be > 0");
        ensure!(self.diffusion_steps > 0, "diffusion_steps must be > 0");
        Ok(())
    }

    /// Mel frame rate (`sample_rate / hop_length`).
    pub fn frames_per_second(&self) -> f32 {
        self.sample_rate as f32 / self.hop_length as f32
    }

    /// The conditional-flow-matching schedule: `steps + 1` sigmas ascending from
    /// `0` (noise) to `1` (data), integrated with an explicit Euler step via the
    /// shared [`FlowMatchEuler`]. (Flow matching integrates *toward* the data, so
    /// the schedule ascends, unlike the descending diffusion denoise schedule.)
    pub fn cfm_scheduler(&self, steps: usize) -> FlowMatchEuler {
        let steps = steps.max(1);
        let sigmas = (0..=steps).map(|i| i as f32 / steps as f32).collect();
        FlowMatchEuler::from_sigmas(sigmas)
    }
}

/// Classifier-free guidance blend of a conditional and unconditional velocity:
/// `v = v_uncond + rate · (v_cond − v_uncond)`. `rate = 0` → unconditional,
/// `rate = 1` → fully conditional.
pub fn cfg_blend(v_cond: &[f32], v_uncond: &[f32], rate: f32) -> Vec<f32> {
    assert_eq!(v_cond.len(), v_uncond.len(), "cfg operand length mismatch");
    v_cond
        .iter()
        .zip(v_uncond)
        .map(|(&c, &u)| u + rate * (c - u))
        .collect()
}

/// One CFG-guided flow-matching Euler step: blend the conditional/unconditional
/// velocities, then advance the latent along the schedule.
pub fn cfm_guided_step(
    scheduler: &FlowMatchEuler,
    i: usize,
    x: &[f32],
    v_cond: &[f32],
    v_uncond: &[f32],
    rate: f32,
) -> Vec<f32> {
    let v = cfg_blend(v_cond, v_uncond, rate);
    scheduler.step(i, x, &v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_validate() {
        let c = SeedVcConfig::default();
        assert_eq!(c.sample_rate, 22_050);
        assert_eq!(c.speaker_dim, 192);
        assert_eq!(c.n_mels, 80);
        assert!((c.cfg_rate - 0.7).abs() < 1e-6);
        c.validate().unwrap();
    }

    #[test]
    fn cfm_schedule_ascends_zero_to_one() {
        let c = SeedVcConfig::default();
        let sched = c.cfm_scheduler(8);
        assert_eq!(sched.num_steps(), 8);
        assert_eq!(sched.sigmas[0], 0.0);
        assert_eq!(*sched.sigmas.last().unwrap(), 1.0);
        assert!(sched.sigmas.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn cfg_blend_endpoints_and_midpoint() {
        let cond = vec![2.0f32, 4.0];
        let uncond = vec![0.0f32, 0.0];
        assert_eq!(cfg_blend(&cond, &uncond, 0.0), uncond);
        assert_eq!(cfg_blend(&cond, &uncond, 1.0), cond);
        assert_eq!(cfg_blend(&cond, &uncond, 0.5), vec![1.0, 2.0]);
    }

    #[test]
    fn guided_step_integrates_toward_data() {
        // Constant guided velocity v=1 over the 0→1 schedule moves x by +1 total.
        let c = SeedVcConfig::default();
        let sched = c.cfm_scheduler(10);
        let v_cond = vec![1.0f32; 4];
        let v_uncond = vec![1.0f32; 4];
        let mut x = vec![0.0f32; 4];
        for i in 0..sched.num_steps() {
            x = cfm_guided_step(&sched, i, &x, &v_cond, &v_uncond, c.cfg_rate);
        }
        assert!(x.iter().all(|q| (q - 1.0).abs() < 1e-4), "x={x:?}");
    }
}
