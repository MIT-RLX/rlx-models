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

//! # rlx-index-tts
//!
//! **IndexTTS-2** on RLX — a controllable TTS with explicit **duration control**
//! and **emotion control**. A GPT-style autoregressive backbone predicts semantic
//! tokens from text (+ speaker + emotion conditioning); a **flow-matching S2A**
//! (semantic-to-acoustic) stage turns those into a mel spectrogram; a BigVGAN
//! vocoder renders the waveform.
//!
//! Native Rust, composing rlx pieces:
//!
//! - **GPT AR backbone** → Llama/GPT-style (`rlx-llama32`).
//! - **S2A flow head + guidance** → [`rlx_audio_blocks::sampling`]
//!   (`FlowMatchEuler::ascending`, `classifier_free_guidance`).
//! - **Vocoder** → BigVGAN (`rlx-neutts`).
//!
//! Checkpoint-free, unit-tested core here: the config, duration→token control, the
//! S2A flow scheduler, guidance, and emotion blending. Wiring the GPT + S2A + BigVGAN
//! graphs is the next step.

use anyhow::{Result, ensure};
use rlx_audio_blocks::sampling::{FlowMatchEuler, classifier_free_guidance};

/// IndexTTS-2 config. Dimensional fields carry typical values; exact widths come
/// from the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexTtsConfig {
    pub sample_rate: usize,
    pub hop_length: usize,
    // GPT autoregressive backbone.
    pub gpt_hidden: usize,
    pub gpt_layers: usize,
    pub gpt_heads: usize,
    /// Semantic-token vocabulary.
    pub semantic_vocab: usize,
    /// Semantic tokens emitted per second (drives duration control).
    pub semantic_token_rate: f32,
    // S2A (semantic → mel) flow-matching head.
    pub s2a_flow_steps: usize,
    pub mel_dim: usize,
    // Conditioning.
    pub speaker_dim: usize,
    pub emotion_dim: usize,
    pub cfg_scale: f32,
}

impl Default for IndexTtsConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22_050,
            hop_length: 256,
            gpt_hidden: 1024,
            gpt_layers: 24,
            gpt_heads: 16,
            semantic_vocab: 8192,
            semantic_token_rate: 25.0,
            s2a_flow_steps: 10,
            mel_dim: 100,
            speaker_dim: 512,
            emotion_dim: 512,
            cfg_scale: 2.0,
        }
    }
}

impl IndexTtsConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.semantic_vocab > 0, "semantic_vocab must be > 0");
        ensure!(
            self.semantic_token_rate > 0.0,
            "semantic_token_rate must be > 0"
        );
        ensure!(self.s2a_flow_steps > 0, "s2a_flow_steps must be > 0");
        ensure!(self.mel_dim > 0, "mel_dim must be > 0");
        Ok(())
    }

    /// Duration control: the number of semantic tokens to request for a target
    /// clip length of `seconds` (rounded to the nearest token).
    pub fn tokens_for_duration(&self, seconds: f32) -> usize {
        (seconds.max(0.0) * self.semantic_token_rate).round() as usize
    }

    /// Mel frame rate (`sample_rate / hop_length`).
    pub fn frames_per_second(&self) -> f32 {
        self.sample_rate as f32 / self.hop_length as f32
    }

    /// The S2A flow-matching sampler (semantic → mel, noise → data).
    pub fn s2a_scheduler(&self, steps: usize) -> FlowMatchEuler {
        FlowMatchEuler::ascending(steps)
    }

    /// Apply this model's classifier-free guidance to a (cond, uncond) velocity.
    pub fn guided(&self, v_cond: &[f32], v_uncond: &[f32]) -> Vec<f32> {
        classifier_free_guidance(v_cond, v_uncond, self.cfg_scale)
    }

    /// Emotion control: interpolate from a `base` emotion embedding toward a
    /// `target` by `alpha ∈ [0, 1]` (`0` → base, `1` → target). Implemented via
    /// the shared guidance blend.
    pub fn blend_emotion(&self, base: &[f32], target: &[f32], alpha: f32) -> Vec<f32> {
        classifier_free_guidance(target, base, alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_validate() {
        let c = IndexTtsConfig::default();
        assert_eq!(c.sample_rate, 22_050);
        assert_eq!(c.mel_dim, 100);
        c.validate().unwrap();
    }

    #[test]
    fn duration_control_maps_seconds_to_tokens() {
        let c = IndexTtsConfig::default(); // 25 tokens/s
        assert_eq!(c.tokens_for_duration(4.0), 100);
        assert_eq!(c.tokens_for_duration(0.0), 0);
        // rounds to nearest token
        assert_eq!(c.tokens_for_duration(1.02), 26); // 25.5 → 26 (round half up)
    }

    #[test]
    fn s2a_flow_is_noise_to_data() {
        let c = IndexTtsConfig::default();
        let s = c.s2a_scheduler(10);
        assert_eq!(s.sigmas[0], 0.0);
        assert_eq!(*s.sigmas.last().unwrap(), 1.0);
    }

    #[test]
    fn guided_and_emotion_blend() {
        let c = IndexTtsConfig {
            cfg_scale: 2.0,
            ..Default::default()
        };
        assert_eq!(c.guided(&[1.0], &[0.0]), vec![2.0]);
        // emotion blend endpoints
        let base = vec![0.0f32, 0.0];
        let target = vec![10.0f32, 20.0];
        assert_eq!(c.blend_emotion(&base, &target, 0.0), base);
        assert_eq!(c.blend_emotion(&base, &target, 1.0), target);
        assert_eq!(c.blend_emotion(&base, &target, 0.5), vec![5.0, 10.0]);
    }
}
