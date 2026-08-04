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

//! # rlx-confucius
//!
//! **Confucius4-TTS** — a multilingual **voice-cloning** TTS on RLX. An LM-style
//! backbone emits neural-codec tokens conditioned on target text and a reference
//! utterance (its transcript + its codec frames), then a neural codec renders the
//! waveform in the reference speaker's voice.
//!
//! Native Rust, composing rlx pieces:
//!
//! - **Backbone** → Llama-shaped LM (`rlx-llama32`).
//! - **Codec** → a neural audio codec (`rlx-dac` / `rlx-snac`).
//!
//! The checkpoint-free core here is the config plus the **clone-prompt planner**
//! ([`plan_clone`]) — the ordering of reference text / reference audio / target
//! text that primes voice cloning. LM + codec graph wiring is the next step.

use anyhow::{Result, ensure};

/// Confucius4-TTS config. Dimensional fields carry plausible values; exact widths
/// come from the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfuciusConfig {
    pub sample_rate: usize,
    pub hop_length: usize,
    // LM backbone.
    pub backbone_hidden: usize,
    pub backbone_layers: usize,
    pub backbone_heads: usize,
    // Neural codec.
    pub num_codebooks: usize,
    pub codebook_size: usize,
    /// Supported language count.
    pub num_languages: usize,
    /// Whether reference-audio voice cloning is enabled.
    pub supports_cloning: bool,
}

impl Default for ConfuciusConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24_000,
            hop_length: 320,
            backbone_hidden: 1536,
            backbone_layers: 24,
            backbone_heads: 16,
            num_codebooks: 4,
            codebook_size: 1024,
            num_languages: 12,
            supports_cloning: true,
        }
    }
}

impl ConfuciusConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.num_codebooks > 0, "num_codebooks must be > 0");
        ensure!(self.codebook_size > 0, "codebook_size must be > 0");
        Ok(())
    }

    pub fn frames_per_second(&self) -> f32 {
        self.sample_rate as f32 / self.hop_length as f32
    }
}

/// A voice-cloning request: a reference utterance (transcript + its codec frames)
/// plus the target text to speak in that voice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloneRequest<'a> {
    pub reference_text: &'a str,
    pub reference_code_frames: usize,
    pub target_text: &'a str,
}

/// A segment of the assembled clone prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    /// The reference transcript (character length).
    ReferenceText(usize),
    /// The reference audio codes (frame count).
    ReferenceAudio(usize),
    /// The target text to synthesize (character length).
    TargetText(usize),
    /// Marker for where the model begins generating target audio.
    TargetAudioStart,
}

/// Plan the clone prompt: reference transcript → reference audio → target text →
/// target-audio generation. This ordering primes the backbone with the paired
/// (text, audio) reference before it must generate audio for the target text.
pub fn plan_clone(req: &CloneRequest) -> Result<Vec<Segment>> {
    ensure!(
        !req.reference_text.trim().is_empty(),
        "clone requires a non-empty reference transcript"
    );
    ensure!(
        req.reference_code_frames > 0,
        "clone requires reference audio codes"
    );
    ensure!(
        !req.target_text.trim().is_empty(),
        "clone requires non-empty target text"
    );
    Ok(vec![
        Segment::ReferenceText(req.reference_text.chars().count()),
        Segment::ReferenceAudio(req.reference_code_frames),
        Segment::TargetText(req.target_text.chars().count()),
        Segment::TargetAudioStart,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_validate() {
        let c = ConfuciusConfig::default();
        assert!(c.supports_cloning);
        assert_eq!(c.num_codebooks, 4);
        c.validate().unwrap();
    }

    #[test]
    fn clone_plan_has_reference_before_target() {
        let req = CloneRequest {
            reference_text: "hello",
            reference_code_frames: 50,
            target_text: "world!",
        };
        let plan = plan_clone(&req).unwrap();
        assert_eq!(
            plan,
            vec![
                Segment::ReferenceText(5),
                Segment::ReferenceAudio(50),
                Segment::TargetText(6),
                Segment::TargetAudioStart,
            ]
        );
    }

    #[test]
    fn clone_plan_rejects_missing_pieces() {
        // no reference audio
        assert!(
            plan_clone(&CloneRequest {
                reference_text: "hi",
                reference_code_frames: 0,
                target_text: "yo",
            })
            .is_err()
        );
        // empty target
        assert!(
            plan_clone(&CloneRequest {
                reference_text: "hi",
                reference_code_frames: 10,
                target_text: "  ",
            })
            .is_err()
        );
    }
}
