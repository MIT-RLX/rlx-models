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

//! Native log-mel frontend — Whisper-style features at 16 kHz, fully in Rust/RLX.
//!
//! Matches HF `VoxtralProcessor` (= `WhisperFeatureExtractor`): pad/trim to 30 s,
//! 128 mel bins, n_fft 400, hop 160, log10 + `max - 8` floor + `(x + 4) / 4` scale.
//! Implemented by reusing [`rlx_whisper::pcm_to_log_mel`] — no Python, no `transformers`.

use crate::config::VoxtralAudioConfig;
use anyhow::{Result, ensure};
pub use rlx_whisper::{
    N_SAMPLES, SAMPLE_RATE, SpeechSegment, load_wav_mono_f32, parse_wav_mono_f32,
    pcm_segments_by_vad,
};

/// Default mel frames for a 30 s chunk (`preprocessor_config.json`: `nb_max_frames = 3000`).
pub const N_FRAMES: usize = 3_000;

#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    pub n_mels: usize,
    pub n_frames: usize,
    pub data: Vec<f32>,
}

/// Native Whisper log-mel features for one ≤30 s utterance.
///
/// PCM is padded (or trimmed) to exactly 30 s before the STFT so the geometry is
/// always `[num_mel_bins, N_FRAMES]` — matching the HF processor's `pad_to_multiple_of`
/// behaviour. Input must be 16 kHz mono f32 (see [`load_wav_mono_f32`]).
pub fn pcm_to_mel(cfg: &VoxtralAudioConfig, pcm: &[f32]) -> Result<MelSpectrogram> {
    ensure!(cfg.num_mel_bins > 0, "num_mel_bins must be > 0");
    let padded = rlx_whisper::audio::pad_or_trim_pcm(pcm);
    let mel = rlx_whisper::pcm_to_log_mel(&padded, cfg.num_mel_bins, N_FRAMES);
    Ok(MelSpectrogram {
        n_mels: mel.n_mels,
        n_frames: mel.n_frames,
        data: mel.data,
    })
}

pub fn mel_from_flat(n_mels: usize, n_frames: usize, data: Vec<f32>) -> Result<MelSpectrogram> {
    ensure!(
        data.len() == n_mels * n_frames,
        "mel flat len {} != {n_mels}×{n_frames}",
        data.len()
    );
    Ok(MelSpectrogram {
        n_mels,
        n_frames,
        data,
    })
}
