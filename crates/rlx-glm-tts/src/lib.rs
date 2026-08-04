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

//! # rlx-glm-tts
//!
//! **GLM-TTS** (Zhipu GLM-4-Voice family) on RLX — zero-shot TTS / voice cloning.
//! A **GLM backbone** (Llama-shaped) autoregressively emits low-rate,
//! single-codebook speech tokens, interleaved with text in a **streaming**
//! pattern (a chunk of text, then a chunk of audio); a CosyVoice-style
//! **flow-matching token→mel** decoder turns the speech tokens into a mel
//! spectrogram, and a HiFiGAN vocoder renders the waveform.
//!
//! Native Rust, composing rlx pieces:
//!
//! - **Backbone** → GLM (`rlx-glm`, Llama-shaped) / `rlx-llama32`.
//! - **token→mel flow + guidance** → [`rlx_audio_blocks::sampling`].
//! - **Vocoder** → HiFiGAN (`rlx-nanocodec`) / BigVGAN (`rlx-neutts`).
//!
//! Checkpoint-free, unit-tested core: the config, the streaming text/audio
//! interleave schedule, duration control, and the flow scheduler + CFG.

use anyhow::{Result, ensure};
use rlx_audio_blocks::sampling::{FlowMatchEuler, classifier_free_guidance};

/// GLM-TTS config. Backbone dims are GLM-shaped; exact widths from the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct GlmTtsConfig {
    pub sample_rate: usize,
    // GLM backbone (Llama-shaped).
    pub backbone_hidden: usize,
    pub backbone_layers: usize,
    pub backbone_heads: usize,
    pub backbone_kv_heads: usize,
    /// Single-codebook speech-token vocabulary.
    pub speech_vocab: usize,
    /// Speech tokens per second (low-rate supervised tokenizer, ~12.5 Hz).
    pub speech_token_rate: f32,
    /// Streaming interleave: text tokens per block.
    pub text_chunk: usize,
    /// Streaming interleave: audio tokens per block.
    pub audio_chunk: usize,
    // Flow token→mel decoder.
    pub flow_steps: usize,
    pub mel_dim: usize,
    pub cfg_scale: f32,
}

impl Default for GlmTtsConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22_050,
            backbone_hidden: 2048,
            backbone_layers: 24,
            backbone_heads: 16,
            backbone_kv_heads: 4,
            speech_vocab: 16_384,
            speech_token_rate: 12.5,
            text_chunk: 13,
            audio_chunk: 26,
            flow_steps: 10,
            mel_dim: 80,
            cfg_scale: 2.0,
        }
    }
}

/// A block of the streaming interleave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Text,
    Audio,
}

/// One `(modality, count)` block of the streaming generation schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamBlock {
    pub modality: Modality,
    pub count: usize,
}

impl GlmTtsConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.speech_vocab > 0, "speech_vocab must be > 0");
        ensure!(
            self.speech_token_rate > 0.0,
            "speech_token_rate must be > 0"
        );
        ensure!(
            self.text_chunk > 0 && self.audio_chunk > 0,
            "chunk sizes must be > 0"
        );
        ensure!(
            self.backbone_heads.is_multiple_of(self.backbone_kv_heads),
            "backbone_heads must be divisible by backbone_kv_heads (GQA)"
        );
        Ok(())
    }

    /// Number of speech tokens for a target clip length of `seconds`.
    pub fn tokens_for_duration(&self, seconds: f32) -> usize {
        (seconds.max(0.0) * self.speech_token_rate).round() as usize
    }

    /// The streaming interleave schedule for `num_text_tokens`: alternating
    /// `Text(text_chunk)` / `Audio(audio_chunk)` blocks (the last text block may be
    /// short). This is how GLM-4-Voice streams audio while still reading text.
    pub fn streaming_schedule(&self, num_text_tokens: usize) -> Vec<StreamBlock> {
        let mut blocks = Vec::new();
        let mut remaining = num_text_tokens;
        while remaining > 0 {
            let t = remaining.min(self.text_chunk);
            blocks.push(StreamBlock {
                modality: Modality::Text,
                count: t,
            });
            blocks.push(StreamBlock {
                modality: Modality::Audio,
                count: self.audio_chunk,
            });
            remaining -= t;
        }
        blocks
    }

    /// The token→mel flow-matching sampler (noise → data).
    pub fn token2mel_scheduler(&self, steps: usize) -> FlowMatchEuler {
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
        let c = GlmTtsConfig::default();
        assert_eq!(c.text_chunk, 13);
        assert_eq!(c.audio_chunk, 26);
        c.validate().unwrap();
    }

    #[test]
    fn streaming_schedule_interleaves_text_and_audio() {
        let c = GlmTtsConfig::default(); // 13 text : 26 audio
        let blocks = c.streaming_schedule(30);
        // 30 text = 13 + 13 + 4, each followed by a 26-audio block.
        let text: Vec<usize> = blocks
            .iter()
            .filter(|b| b.modality == Modality::Text)
            .map(|b| b.count)
            .collect();
        assert_eq!(text, vec![13, 13, 4]);
        let audio: Vec<usize> = blocks
            .iter()
            .filter(|b| b.modality == Modality::Audio)
            .map(|b| b.count)
            .collect();
        assert_eq!(audio, vec![26, 26, 26]);
        // total text tokens preserved
        assert_eq!(text.iter().sum::<usize>(), 30);
    }

    #[test]
    fn empty_text_yields_no_blocks() {
        let c = GlmTtsConfig::default();
        assert!(c.streaming_schedule(0).is_empty());
    }

    #[test]
    fn duration_and_flow_and_guidance() {
        let c = GlmTtsConfig::default();
        assert_eq!(c.tokens_for_duration(8.0), 100); // 12.5 * 8
        let s = c.token2mel_scheduler(10);
        assert_eq!(s.sigmas[0], 0.0);
        assert_eq!(*s.sigmas.last().unwrap(), 1.0);
        assert_eq!(c.guided(&[1.0], &[0.0]), vec![2.0]);
    }
}
