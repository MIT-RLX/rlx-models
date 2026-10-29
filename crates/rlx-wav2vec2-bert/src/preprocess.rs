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

//! SeamlessM4T-compatible audio preprocessing for Wav2Vec2-BERT.

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wav2Vec2BertPreprocessConfig {
    #[serde(default = "default_sample_rate")]
    pub sampling_rate: usize,
    #[serde(default = "default_num_mels")]
    pub num_mel_bins: usize,
    /// Target frame count for fixed-shape encoders.
    #[serde(default = "default_num_frames")]
    pub num_frames: usize,
    /// Number of adjacent fbank frames concatenated for one encoder frame.
    #[serde(default = "default_stride")]
    pub stride: usize,
}

fn default_sample_rate() -> usize {
    16_000
}
fn default_num_mels() -> usize {
    80
}
fn default_num_frames() -> usize {
    3_000
}
fn default_stride() -> usize {
    2
}

impl Default for Wav2Vec2BertPreprocessConfig {
    fn default() -> Self {
        Self {
            sampling_rate: default_sample_rate(),
            num_mel_bins: default_num_mels(),
            num_frames: default_num_frames(),
            stride: default_stride(),
        }
    }
}

impl Wav2Vec2BertPreprocessConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let txt = fs::read_to_string(path).map_err(|e| anyhow!("read {path:?}: {e}"))?;
        let cfg: Self = serde_json::from_str(&txt).map_err(|e| anyhow!("parse {path:?}: {e}"))?;
        Ok(cfg)
    }

    pub fn w2v_bert_2_0() -> Self {
        Self::default()
    }

    pub fn feature_dim(&self) -> usize {
        self.num_mel_bins * self.stride
    }
}

#[derive(Debug, Clone)]
pub struct LogMelFeatures {
    pub num_mel_bins: usize,
    pub num_frames: usize,
    /// Row-major `[1, num_frames, num_mel_bins]` (batch=1).
    pub features: Vec<f32>,
    /// Row-major `[1, num_frames]` (1 = valid, 0 = padded).
    pub attention_mask: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LogMelExtractor {
    cfg: Wav2Vec2BertPreprocessConfig,
}

impl LogMelExtractor {
    pub fn new(cfg: Wav2Vec2BertPreprocessConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &Wav2Vec2BertPreprocessConfig {
        &self.cfg
    }

    /// HF `SeamlessM4TFeatureExtractor`: rlx-spectral's Kaldi fbank (computed
    /// in f64: the int16-scaled frames put quiet bins near f32's noise floor),
    /// per-bin normalization, `stride` frames stacked per row.
    pub fn extract(&self, pcm: &[f32]) -> LogMelFeatures {
        let wide: Vec<f64> = pcm.iter().map(|&v| v as f64).collect();
        let fe = rlx_spectral::KaldiFbank::<f64>::new(
            self.cfg.sampling_rate as f64,
            self.cfg.num_mel_bins,
        );
        let (x, t) = fe.input_features_stacked(&wide, self.cfg.stride);
        let stacked_dim = self.cfg.feature_dim();
        let features: Vec<f32> = x.into_iter().map(|v| v as f32).collect();
        LogMelFeatures {
            num_mel_bins: stacked_dim,
            num_frames: t,
            features,
            attention_mask: vec![1.0f32; t],
        }
    }

    pub fn pad_to_seq(&self, mut feats: LogMelFeatures, seq: usize) -> LogMelFeatures {
        if feats.num_frames == seq {
            return feats;
        }
        let m = feats.num_mel_bins;
        let mut out = vec![0.0f32; seq * m];
        let mut mask = vec![0.0f32; seq];
        let copy_t = feats.num_frames.min(seq);
        out[..copy_t * m].copy_from_slice(&feats.features[..copy_t * m]);
        for i in 0..copy_t {
            mask[i] = 1.0;
        }
        feats.num_frames = seq;
        feats.features = out;
        feats.attention_mask = mask;
        feats
    }
}

pub fn load_wav_mono_f32(path: &Path) -> Result<(Vec<f32>, usize)> {
    let bytes = fs::read(path).map_err(|e| anyhow!("read wav {path:?}: {e}"))?;
    parse_wav_mono_f32(&bytes)
}

pub fn parse_wav_mono_f32(bytes: &[u8]) -> Result<(Vec<f32>, usize)> {
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
            b"data" => data_chunk = Some(&bytes[off..off + len]),
            _ => {}
        }
        off += (len + 1) & !1;
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
    Ok((out, sr as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w2v_bert_default_stacks_to_160_features() {
        let cfg = Wav2Vec2BertPreprocessConfig::w2v_bert_2_0();
        assert_eq!(cfg.feature_dim(), 160);
        let pcm: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin())
            .collect();
        let features = LogMelExtractor::new(cfg).extract(&pcm);
        assert!(features.num_frames > 0);
        assert_eq!(features.num_mel_bins, 160);
        assert_eq!(features.features.len(), features.num_frames * 160);
        assert!(features.features.iter().all(|v| v.is_finite()));
    }
}
