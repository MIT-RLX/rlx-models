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

//! Audio I/O, mel spectrograms, and VAD segmentation for Whisper.

use crate::config::WhisperConfig;
use crate::mel::pcm_to_log_mel;
use crate::vad::{VadConfig, segments_by_vad};
use anyhow::{Result, anyhow, bail};
use std::fs;
use std::path::Path;

pub const SAMPLE_RATE: usize = 16_000;
pub const N_SAMPLES: usize = 30 * SAMPLE_RATE;
pub const N_FRAMES: usize = 3_000;

#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    pub n_mels: usize,
    pub n_frames: usize,
    /// Row-major `[1, n_mels, n_frames]` as f32.
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Default)]
pub struct EnergyVad {
    pub threshold: f32,
    pub min_len_samples: usize,
}

impl EnergyVad {
    pub fn to_vad_config(&self) -> VadConfig {
        VadConfig {
            kind: crate::vad::VadKind::Energy,
            threshold: self.threshold,
            min_speech_samples: self.min_len_samples.max(SAMPLE_RATE / 10),
            ..VadConfig::default()
        }
    }
}

pub fn pcm_segments_by_vad(vad: &EnergyVad, pcm: &[f32]) -> Vec<SpeechSegment> {
    segments_by_vad(&vad.to_vad_config(), pcm)
}

pub fn pcm_segments_by_vad_config(cfg: &VadConfig, pcm: &[f32]) -> Vec<SpeechSegment> {
    segments_by_vad(cfg, pcm)
}

/// Pad with zeros at the end or truncate to [`N_SAMPLES`] (OpenAI `pad_or_trim`).
pub fn pad_or_trim_pcm(pcm: &[f32]) -> Vec<f32> {
    if pcm.len() >= N_SAMPLES {
        pcm[..N_SAMPLES].to_vec()
    } else {
        let mut out = vec![0.0f32; N_SAMPLES];
        out[..pcm.len()].copy_from_slice(pcm);
        out
    }
}

pub fn pcm_to_mel(cfg: &WhisperConfig, pcm: &[f32]) -> MelSpectrogram {
    let pcm = pad_or_trim_pcm(pcm);
    pcm_to_log_mel(&pcm, cfg.num_mel_bins, N_FRAMES)
}

pub fn load_wav_mono_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).map_err(|e| anyhow!("read wav {path:?}: {e}"))?;
    parse_wav_mono_f32(&bytes)
}

pub fn parse_wav_mono_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    // Very small RIFF/WAVE PCM parser (16-bit mono).
    if bytes.len() < 44 {
        bail!("wav too small");
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file");
    }
    let mut off = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (audio_format, channels, sample_rate, bits_per_sample)
    let mut data_chunk: Option<&[u8]> = None;
    while off + 8 <= bytes.len() {
        let tag = &bytes[off..off + 4];
        let len = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        off += 8;
        if off + len > bytes.len() {
            break;
        }
        match tag {
            b"fmt " => {
                if len < 16 {
                    bail!("wav fmt chunk too small");
                }
                let audio_format = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
                let channels = u16::from_le_bytes(bytes[off + 2..off + 4].try_into().unwrap());
                let sample_rate = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
                let bits_per_sample =
                    u16::from_le_bytes(bytes[off + 14..off + 16].try_into().unwrap());
                fmt = Some((audio_format, channels, sample_rate, bits_per_sample));
            }
            b"data" => {
                data_chunk = Some(&bytes[off..off + len]);
            }
            _ => {}
        }
        off += (len + 1) & !1; // word-align
        if fmt.is_some() && data_chunk.is_some() {
            break;
        }
    }
    let (audio_format, channels, sr, bps) = fmt.ok_or_else(|| anyhow!("wav missing fmt chunk"))?;
    if audio_format != 1 {
        bail!("wav: only PCM supported (format={audio_format})");
    }
    if channels != 1 {
        bail!("wav: expected mono, got {channels} channels");
    }
    if sr as usize != SAMPLE_RATE {
        bail!("wav: expected {SAMPLE_RATE} Hz, got {sr}");
    }
    if bps != 16 {
        bail!("wav: expected 16-bit PCM, got {bps}");
    }
    let data = data_chunk.ok_or_else(|| anyhow!("wav missing data chunk"))?;
    if data.len() % 2 != 0 {
        bail!("wav data chunk not aligned");
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for i in (0..data.len()).step_by(2) {
        let s = i16::from_le_bytes([data[i], data[i + 1]]) as f32 / 32768.0;
        out.push(s);
    }
    Ok(out)
}
