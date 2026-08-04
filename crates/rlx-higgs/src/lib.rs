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

//! # rlx-higgs
//!
//! **Higgs-Audio v2** (Boson AI) on RLX — a unified audio-language model over a
//! **Llama-3.2 backbone** with a **DualFFN** audio adapter and an **RVQ audio
//! tokenizer**. The same model does text→audio (TTS / voice cloning) and
//! audio→text (STT) by generating the other modality's tokens.
//!
//! Native Rust, composing rlx pieces:
//!
//! - **Backbone** → Llama-3.2 (`rlx-llama32`), with a per-token DualFFN branch for
//!   audio positions.
//! - **Audio tokenizer** → an RVQ codec; the `K` codebooks are generated with the
//!   shared **delay pattern** ([`rlx_audio_blocks::codec`]).
//!
//! Checkpoint-free, unit-tested core here: the config ([`HiggsConfig`]), the task
//! [`HiggsMode`], and the RVQ codebook delay helpers ([`delay_encode`] /
//! [`delay_decode`]). Wiring the backbone + DualFFN + tokenizer decode end-to-end
//! is the next step.

use anyhow::Result;
use rlx_audio_blocks::codec::{build_delay_pattern, revert_delay_pattern};

/// Which direction the unified model runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiggsMode {
    /// Text → audio tokens (TTS / voice cloning).
    TextToAudio,
    /// Audio tokens → text (STT).
    AudioToText,
}

/// Higgs-Audio v2 config. Backbone dims are the Llama-3.2-3B defaults; audio
/// tokenizer widths come from the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct HiggsConfig {
    // Llama-3.2 backbone.
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    /// Separate audio FFN branch (DualFFN).
    pub dual_ffn: bool,
    // RVQ audio tokenizer.
    pub num_codebooks: usize,
    pub codebook_size: usize,
    /// Audio token frame rate (tokens per second).
    pub frame_rate: usize,
    pub audio_sample_rate: usize,
    /// Sentinel used in delay-pattern padding (typically a reserved audio id).
    pub audio_pad_id: i32,
}

impl Default for HiggsConfig {
    fn default() -> Self {
        Self {
            hidden_size: 3072,
            num_layers: 28,
            num_heads: 24,
            num_kv_heads: 8,
            vocab_size: 128_256,
            dual_ffn: true,
            num_codebooks: 8,
            codebook_size: 1024,
            frame_rate: 25,
            audio_sample_rate: 24_000,
            audio_pad_id: -1,
        }
    }
}

impl HiggsConfig {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.num_codebooks > 0, "num_codebooks must be > 0");
        anyhow::ensure!(self.codebook_size > 0, "codebook_size must be > 0");
        anyhow::ensure!(
            self.num_heads.is_multiple_of(self.num_kv_heads),
            "num_heads must be divisible by num_kv_heads (GQA)"
        );
        Ok(())
    }

    /// Interleave RVQ `codes` (`[num_codebooks][frames]`) into the delay pattern the
    /// AR head predicts, padding with `audio_pad_id`.
    pub fn delay_encode(&self, codes: &[Vec<i32>]) -> Result<Vec<Vec<i32>>> {
        build_delay_pattern(codes, self.audio_pad_id)
    }

    /// Recover `[num_codebooks][frames]` codes from the generated delay pattern.
    pub fn delay_decode(&self, delayed: &[Vec<i32>]) -> Result<Vec<Vec<i32>>> {
        revert_delay_pattern(delayed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_validate() {
        let c = HiggsConfig::default();
        assert_eq!(c.hidden_size, 3072);
        assert_eq!(c.num_codebooks, 8);
        assert!(c.dual_ffn);
        c.validate().unwrap();
    }

    #[test]
    fn gqa_divisibility_enforced() {
        let c = HiggsConfig {
            num_kv_heads: 5, // 24 % 5 != 0
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn mode_is_explicit() {
        assert_ne!(HiggsMode::TextToAudio, HiggsMode::AudioToText);
    }

    #[test]
    fn delay_roundtrip_through_config() {
        let c = HiggsConfig::default();
        let codes = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let delayed = c.delay_encode(&codes).unwrap();
        // delayed length = frames + num_codebooks - 1
        assert_eq!(delayed[0].len(), 3 + 3 - 1);
        let back = c.delay_decode(&delayed).unwrap();
        assert_eq!(back, codes);
    }
}
