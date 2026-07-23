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

//! RLX FastSpeech2 + WaveRNN text-to-speech.
//!
//! Load [`weights/tts/rlx-tts/rlx-tts.gguf`](DEFAULT_BUNDLE_DIR) (or `RLX_TTS_BUNDLE`).
//! A loose safetensors + `frontend/` directory still works; GGUF is preferred.
//! Never commit or redistribute private weights.
//!
//! WaveRNN GRU uses fused activations via `rlx_cpu::vmath` (`vvexpf`):
//! `sigmoid(x)=1/(1+exp(-x))`, `tanh(x)=2*sigmoid(2x)-1`, nested `fmaf` mix.
//! Set `RLX_WR_GRU_PLAIN=1` for plain `vvtanhf`.

pub mod fastspeech2;
pub mod frontend;
mod gguf_bundle;
mod gru_rlx;
pub use gguf_bundle::{PackReport, pack_directory, sanitize_manifest};
pub use gru_rlx::{eval as gru_rlx_eval, eval_mode as gru_rlx_eval_mode};
pub mod metrics;
pub mod native;
pub mod ops;
mod wav;
pub mod wavernn;
pub mod weights;

pub use fastspeech2::VarianceControls;
pub use metrics::{
    SpectralMetrics, apply_leading_silence_ms, apply_output_volume, apply_wavernn_mulaw_iir,
    spectral_vs_ref,
};
pub use native::{BUNDLE_EXTRACT_HINT, DEFAULT_BUNDLE_DIR, RlxTts};
pub use wav::{read_wav_f32, write_wav};
pub use wavernn::{WaveRnnOpts, WaveRnnRng};

/// Default voice id written into synthesized [`AudioOutput`] when the bundle
/// manifest does not supply one.
pub const DEFAULT_VOICE_ID: &str = "rlx-tts";

/// Captured / synthesized PCM.
#[derive(Debug, Clone)]
pub struct AudioOutput {
    /// Interleaved float32 samples in [-1, 1].
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub voice_identifier: String,
}

impl AudioOutput {
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.pcm.len() as f64 / (self.sample_rate as f64 * self.channels as f64)
    }

    pub fn peak_amplitude(&self) -> f32 {
        self.pcm
            .iter()
            .filter(|s| s.is_finite())
            .map(|s| s.abs())
            .fold(0.0f32, f32::max)
    }
}

/// Native path is always available once a bundle is present.
pub fn is_native_supported() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_voice_id_is_set() {
        assert_eq!(DEFAULT_VOICE_ID, "rlx-tts");
    }
}
