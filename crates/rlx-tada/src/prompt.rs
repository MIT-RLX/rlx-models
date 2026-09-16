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

//! The voice prompt — TADA's `EncoderOutput`.
//!
//! A voice prompt is the reference speaker reduced to three aligned things:
//! the transcript's token ids, the 50 Hz frame each token landed on, and the
//! acoustic latent sampled at that frame. Building one costs a wav2vec2 forward
//! plus a codec encode, so it is worth caching — which is what
//! [`VoicePrompt::save`] / [`VoicePrompt::load`] are for, and what upstream's
//! `EncoderOutput.save` does with a pickle.
//!
//! Serialized as JSON metadata plus a raw little-endian f32 blob, so a cached
//! prompt stays readable without a torch install.

use anyhow::{Context, Result, bail};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"TADAPMT1";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PromptMeta {
    text: String,
    token_ids: Vec<u32>,
    token_positions: Vec<u32>,
    frames: usize,
    width: usize,
    audio_samples: usize,
    sample_rate: usize,
    /// Absent in prompts written before alignment scoring existed.
    #[serde(default = "unscored")]
    alignment_score: f32,
}

/// Sentinel for a prompt that predates alignment scoring — distinguishable from
/// a genuinely terrible score.
fn unscored() -> f32 {
    f32::NAN
}

/// A reference speaker, encoded once and reusable across utterances.
#[derive(Debug, Clone)]
pub struct VoicePrompt {
    /// Normalized transcript of the reference audio.
    pub text: String,
    /// Tokenizer ids of [`Self::text`], without special tokens.
    pub token_ids: Vec<u32>,
    /// 1-based frame index each token was aligned to. Same length as
    /// [`Self::token_ids`].
    pub token_positions: Vec<u32>,
    /// `[tokens, acoustic_dim]`, already mean/std normalized.
    pub token_values: Array2<f32>,
    /// Length of the reference audio in samples.
    pub audio_samples: usize,
    pub sample_rate: usize,
    /// Mean log-softmax probability the forced aligner's CTC head gave each
    /// token at its assigned frame — how well the transcript matches the audio.
    /// `NaN` for prompts written before this was recorded. See
    /// [`crate::align::Alignment::looks_mismatched`].
    pub alignment_score: f32,
}

impl VoicePrompt {
    pub fn num_tokens(&self) -> usize {
        self.token_ids.len()
    }

    /// Reference duration in 50 Hz frames, rounded up — upstream's
    /// `ceil(audio_len / sample_rate * 50)`.
    pub fn audio_frames(&self) -> usize {
        let frames = self.audio_samples as f64 / self.sample_rate as f64 * crate::FRAME_RATE as f64;
        frames.ceil() as usize
    }

    pub fn validate(&self) -> Result<()> {
        if self.token_ids.len() != self.token_positions.len() {
            bail!(
                "voice prompt has {} tokens but {} positions",
                self.token_ids.len(),
                self.token_positions.len()
            );
        }
        if self.token_values.nrows() != self.token_ids.len() {
            bail!(
                "voice prompt has {} tokens but {} latent rows",
                self.token_ids.len(),
                self.token_values.nrows()
            );
        }
        if self.token_ids.is_empty() {
            bail!("voice prompt is empty — the reference transcript produced no tokens");
        }
        if self.token_positions.windows(2).any(|w| w[0] >= w[1]) {
            bail!("voice prompt positions are not strictly increasing");
        }
        Ok(())
    }

    /// Per-token frame gaps, as `_generate` derives them.
    ///
    /// Returns `(before, after)`, both `num_tokens` long, where
    /// `before[i]` is the number of frames between token `i-1` and token `i`
    /// and `after[i] == before[i + 1]`. The leading value is 0 because the
    /// first token has no predecessor gap to carry, and the first *real* gap is
    /// measured from frame 1.
    pub fn frame_gaps(&self, max_classes: u32) -> (Vec<u32>, Vec<u32>) {
        let n = self.num_tokens();
        // gaps[i] = positions[i] - (i == 0 ? 1 : positions[i - 1])
        let mut gaps = Vec::with_capacity(n + 1);
        gaps.push(0);
        for i in 0..n {
            let prev = if i == 0 {
                1
            } else {
                self.token_positions[i - 1]
            };
            gaps.push(
                self.token_positions[i]
                    .saturating_sub(prev)
                    .min(max_classes - 1),
            );
        }
        let before = gaps[..n].to_vec();
        let after = gaps[1..=n].to_vec();
        (before, after)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let meta = PromptMeta {
            text: self.text.clone(),
            token_ids: self.token_ids.clone(),
            token_positions: self.token_positions.clone(),
            frames: self.token_values.nrows(),
            width: self.token_values.ncols(),
            audio_samples: self.audio_samples,
            sample_rate: self.sample_rate,
            alignment_score: self.alignment_score,
        };
        let json = serde_json::to_vec(&meta)?;
        let mut f =
            std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
        f.write_all(MAGIC)?;
        f.write_all(&(json.len() as u64).to_le_bytes())?;
        f.write_all(&json)?;
        for v in self.token_values.iter() {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut f =
            std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            bail!("{} is not a TADA voice prompt", path.display());
        }
        let mut len = [0u8; 8];
        f.read_exact(&mut len)?;
        let mut json = vec![0u8; u64::from_le_bytes(len) as usize];
        f.read_exact(&mut json)?;
        let meta: PromptMeta = serde_json::from_slice(&json)
            .with_context(|| format!("parse metadata in {}", path.display()))?;
        let mut blob = Vec::new();
        f.read_to_end(&mut blob)?;
        let want = meta.frames * meta.width * 4;
        if blob.len() != want {
            bail!(
                "{}: latent blob is {} bytes, header says {want}",
                path.display(),
                blob.len()
            );
        }
        let values: Vec<f32> = blob
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let prompt = Self {
            text: meta.text,
            token_ids: meta.token_ids,
            token_positions: meta.token_positions,
            token_values: Array2::from_shape_vec((meta.frames, meta.width), values)?,
            audio_samples: meta.audio_samples,
            sample_rate: meta.sample_rate,
            alignment_score: meta.alignment_score,
        };
        prompt.validate()?;
        Ok(prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prompt(positions: &[u32]) -> VoicePrompt {
        VoicePrompt {
            text: "Hello.".into(),
            token_ids: vec![1; positions.len()],
            token_positions: positions.to_vec(),
            token_values: Array2::zeros((positions.len(), 4)),
            alignment_score: -0.5,
            audio_samples: 24_000,
            sample_rate: 24_000,
        }
    }

    #[test]
    fn gaps_measure_the_distance_to_the_previous_token() {
        // Positions 3, 5, 10 → first gap counts from frame 1.
        let (before, after) = prompt(&[3, 5, 10]).frame_gaps(256);
        assert_eq!(before, vec![0, 2, 2]);
        assert_eq!(after, vec![2, 2, 5]);
        // after[i] is before[i + 1] — the same sequence, offset by one.
        assert_eq!(&after[..2], &before[1..]);
    }

    #[test]
    fn gaps_are_clamped_to_the_time_vocabulary() {
        let (_, after) = prompt(&[1, 9_000]).frame_gaps(256);
        assert_eq!(after[1], 255);
    }

    #[test]
    fn audio_frames_rounds_up() {
        let mut p = prompt(&[1]);
        p.audio_samples = 24_001;
        assert_eq!(p.audio_frames(), 51);
        p.audio_samples = 24_000;
        assert_eq!(p.audio_frames(), 50);
    }

    #[test]
    fn non_monotonic_positions_are_rejected() {
        assert!(prompt(&[5, 3]).validate().is_err());
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = std::env::temp_dir().join("rlx_tada_prompt_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.tadaprompt");
        let mut p = prompt(&[2, 7]);
        p.token_ids = vec![11, 22];
        p.token_values =
            Array2::from_shape_vec((2, 4), (0..8).map(|v| v as f32).collect()).unwrap();
        p.save(&path).unwrap();
        let back = VoicePrompt::load(&path).unwrap();
        assert_eq!(back.token_ids, p.token_ids);
        assert_eq!(back.token_positions, p.token_positions);
        assert_eq!(back.token_values, p.token_values);
        assert_eq!(back.audio_samples, p.audio_samples);
        std::fs::remove_file(&path).ok();
    }
}
