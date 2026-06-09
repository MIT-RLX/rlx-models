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

//! Nemotron-Omni audio adapter.
//!
//! Nemotron Omni's audio encoder is Whisper-shaped (mel front-end +
//! transformer encoder). The primary adapter,
//! [`NemotronOmniAudioEncoder`], wraps an `rlx_whisper::WhisperRunner`
//! and exposes its output via the [`rlx_vlm_base::AudioEncoder`]
//! trait. For tests + multi-encoder support, [`AudioEncoderBox`]
//! lets the multimodal runner accept any owned `AudioEncoder`.

use anyhow::Result;
use rlx_vlm_base::AudioEncoder;

/// Adapter over `rlx_whisper::WhisperRunner` (the production path).
/// Requires real whisper weights — for synthetic-weight quick-check tests
/// see [`SyntheticAudioEncoder`].
pub struct NemotronOmniAudioEncoder {
    inner: rlx_whisper::WhisperRunner,
    hidden_size: usize,
}

impl NemotronOmniAudioEncoder {
    pub fn new(whisper: rlx_whisper::WhisperRunner) -> Self {
        let hidden_size = whisper.config().d_model;
        Self {
            inner: whisper,
            hidden_size,
        }
    }

    /// Convenience: build the encoder directly from a whisper weights
    /// path, applying default config + tokenizer resolution. Returns
    /// any error from `WhisperRunner::builder().build()`.
    pub fn from_weights_path(path: impl Into<std::path::PathBuf>) -> Result<Self> {
        let runner = rlx_whisper::WhisperRunner::builder()
            .weights(path)
            .build()?;
        Ok(Self::new(runner))
    }
}

impl AudioEncoder for NemotronOmniAudioEncoder {
    fn embed_audio(&mut self, samples: &[f32], _sample_rate: u32) -> Result<Vec<f32>> {
        // Whisper internally resamples to 16 kHz mel via its mel
        // pipeline; pass the raw f32 PCM samples through.
        self.inner.encode_pcm(samples)
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

/// Owned trait-object wrapper. Lets the multimodal runner accept
/// `Box<dyn AudioEncoder>` without rolling a generic everywhere.
pub struct AudioEncoderBox(pub Box<dyn AudioEncoder>);

impl AudioEncoder for AudioEncoderBox {
    fn embed_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        self.0.embed_audio(samples, sample_rate)
    }
    fn hidden_size(&self) -> usize {
        self.0.hidden_size()
    }
}

/// Deterministic in-memory `AudioEncoder` for tests: pools the input
/// PCM into a fixed `hidden_size`-dim vector via mean + position-aware
/// hashing. **NOT a real audio encoder** — exists so the multimodal
/// runner's audio integration path can be quick-checked without
/// requiring real whisper weights. PLAN.md M7 follow-up: replace with
/// a true mel encoder once whisper-weight fixtures are wired into CI.
pub struct SyntheticAudioEncoder {
    pub hidden_size: usize,
}

impl AudioEncoder for SyntheticAudioEncoder {
    fn embed_audio(&mut self, samples: &[f32], _sample_rate: u32) -> Result<Vec<f32>> {
        if samples.is_empty() {
            return Ok(vec![0.0; self.hidden_size]);
        }
        let mut out = vec![0f32; self.hidden_size];
        for (i, &s) in samples.iter().enumerate() {
            out[i % self.hidden_size] += s;
        }
        let inv = 1.0 / samples.len() as f32;
        for v in out.iter_mut() {
            *v *= inv;
        }
        Ok(out)
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_encoder_round_trip() {
        let mut enc = SyntheticAudioEncoder { hidden_size: 8 };
        let pcm: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
        let embed = enc.embed_audio(&pcm, 16_000).unwrap();
        assert_eq!(embed.len(), enc.hidden_size());
        assert!(embed.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn box_adapter_dispatches() {
        let inner: Box<dyn AudioEncoder> = Box::new(SyntheticAudioEncoder { hidden_size: 4 });
        let mut boxed = AudioEncoderBox(inner);
        assert_eq!(boxed.hidden_size(), 4);
        let out = boxed.embed_audio(&[0.1, 0.2, 0.3, 0.4], 16_000).unwrap();
        assert_eq!(out.len(), 4);
    }
}
