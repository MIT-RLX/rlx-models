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

//! MiraTTS FastBiCodec speaker encoder (`s_encoder.onnx`) — mel → 32 global tokens.
//!
//! Matches MiraTTS / FastBiCodec `encode_audio` (no WavLM / `q_encoder`): volume
//! normalize → 6 s @ 16 kHz → mel → FSQ global tokens. Those tokens condition
//! both the LM prompt and the detokenizer.

use std::f32::consts::PI;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;

use crate::codec::{CONTEXT_LEN, SAMPLE_RATE};

/// Reference clip length FastBiCodec tiles/crops to (6 s @ 16 kHz).
pub const REF_SAMPLES: usize = 96_000;
const N_FFT: usize = 1024;
const WIN: usize = 640;
const HOP: usize = 320;
const N_MELS: usize = 128;
const F_MIN: f64 = 10.0;
const VOL_COEFF: f32 = 0.2;

/// Native `s_encoder.onnx` over RLX (mel → 32 global/speaker tokens).
pub struct MiraSpeakerEncoder {
    model: TinyModel,
    device: Device,
}

impl MiraSpeakerEncoder {
    pub fn load(decoders_dir: &Path, device: Device) -> Result<Self> {
        anyhow::ensure!(
            decoders_dir.join("s_encoder.onnx").is_file(),
            "s_encoder.onnx missing in {}",
            decoders_dir.display()
        );
        let cfg = BundleConfig {
            model: String::new(),
            sample_rate: SAMPLE_RATE,
            add_blank: false,
            language: "EN".into(),
            speakers: Default::default(),
            default_speaker: None,
            noise_scale: 0.0,
            noise_scale_w: 0.0,
            length_scale: 1.0,
            inter_channels: 0,
            gin_channels: 0,
        };
        Ok(Self {
            model: TinyModel::new(decoders_dir.to_path_buf(), cfg),
            device,
        })
    }

    /// Encode a mono 16 kHz reference clip → 32 global tokens (0..4095).
    pub fn encode_pcm(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        let mel = pcm_to_mel(pcm);
        self.encode_mel(&mel)
    }

    /// Run `s_encoder` on a precomputed mel `[T, 128]` (row-major).
    pub fn encode_mel(&self, mel_tx128: &[f32]) -> Result<Vec<u32>> {
        anyhow::ensure!(
            mel_tx128.len() % N_MELS == 0,
            "mel length {} not divisible by {N_MELS}",
            mel_tx128.len()
        );
        let t = mel_tx128.len() / N_MELS;
        anyhow::ensure!(t > 0, "empty mel");
        let mut g = self
            .model
            .compile_named(
                "s_encoder",
                self.device,
                t,
                &[("mel_time_steps", t), ("batch_size", 1)],
            )
            .map_err(|e| anyhow::anyhow!("compile s_encoder: {e:#}"))?;
        // ONNX input is `[1, T, 128]` row-major — same layout as `mel_tx128`.
        let bytes = f32_bytes(mel_tx128);
        let out = g.run_typed(&[("mel_spectrogram", bytes.as_slice(), DType::F32)]);
        let (raw, dt) = out
            .into_iter()
            .next()
            .context("s_encoder produced no output")?;
        let tokens = match dt {
            DType::I32 => raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
                .collect::<Vec<_>>(),
            DType::I64 => raw
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as u32
                })
                .collect::<Vec<_>>(),
            other => anyhow::bail!("s_encoder unexpected dtype {other:?}"),
        };
        anyhow::ensure!(!tokens.is_empty(), "s_encoder returned empty global_tokens");
        let mut out = tokens;
        out.resize(CONTEXT_LEN, 0);
        Ok(out)
    }
}

/// Volume-normalize (coeff=0.2), tile/crop to 6 s, mel `[T, 128]` row-major.
pub fn pcm_to_mel(pcm: &[f32]) -> Vec<f32> {
    let mut x = pcm.to_vec();
    if x.is_empty() {
        x.push(0.0);
    }
    let peak = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-8);
    let scale = VOL_COEFF / peak;
    for v in &mut x {
        *v *= scale;
    }
    // Tile / truncate to REF_SAMPLES.
    if x.len() < REF_SAMPLES {
        let n = x.len();
        while x.len() < REF_SAMPLES {
            let take = (REF_SAMPLES - x.len()).min(n);
            x.extend_from_slice(&x[..take].to_vec());
        }
    }
    x.truncate(REF_SAMPLES);
    mel_spectrogram(&x)
}

/// FastBiCodec mel: n_fft=1024, win=640, hop=320, n_mels=128, f_min=10, power=1.
fn mel_spectrogram(pcm: &[f32]) -> Vec<f32> {
    let n_bins = N_FFT / 2 + 1;
    let n_frames = if pcm.len() >= WIN {
        1 + (pcm.len() - WIN) / HOP
    } else {
        1
    };
    let window: Vec<f32> = (0..WIN)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / WIN as f32).cos())
        .collect();
    let fb = mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE as f64, F_MIN);
    let mut mel = vec![0f32; n_frames * N_MELS];
    let mut frame = vec![0f32; N_FFT];
    for t in 0..n_frames {
        frame.fill(0.0);
        let start = t * HOP;
        for i in 0..WIN {
            let s = start + i;
            if s < pcm.len() {
                frame[i] = pcm[s] * window[i];
            }
        }
        // Center-pad remaining n_fft-win zeros (already zero).
        let mag = rfft_mag(&frame);
        for m in 0..N_MELS {
            let mut acc = 0.0f32;
            for k in 0..n_bins {
                acc += fb[m * n_bins + k] * mag[k];
            }
            mel[t * N_MELS + m] = acc;
        }
    }
    mel
}

fn mel_filterbank(n_fft: usize, n_mels: usize, sr: f64, f_min: f64) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let f_max = sr * 0.5;
    let fftfreqs: Vec<f64> = (0..n_bins).map(|k| k as f64 * sr / n_fft as f64).collect();
    let n_pts = n_mels + 2;
    let mel_pts: Vec<f64> = (0..n_pts)
        .map(|i| {
            let mel = hz_to_mel(f_min)
                + (hz_to_mel(f_max) - hz_to_mel(f_min)) * i as f64 / (n_pts - 1) as f64;
            mel_to_hz(mel)
        })
        .collect();
    let mut fb = vec![0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let left = mel_pts[m];
        let center = mel_pts[m + 1];
        let right = mel_pts[m + 2];
        for (k, &f) in fftfreqs.iter().enumerate() {
            let v = if f < left || f > right {
                0.0
            } else if f <= center {
                ((f - left) / (center - left).max(1e-8)) as f32
            } else {
                ((right - f) / (right - center).max(1e-8)) as f32
            };
            fb[m * n_bins + k] = v;
        }
    }
    fb
}

fn hz_to_mel(hz: f64) -> f64 {
    // Slaney (HTK-ish) as used by librosa / FastBiCodec.
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / (200.0 / 3.0);
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / (200.0 / 3.0)
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / (200.0 / 3.0);
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * ((logstep * (mel - min_log_mel)).exp())
    } else {
        mel * (200.0 / 3.0)
    }
}

/// One-sided magnitude spectrum (power=1.0).
fn rfft_mag(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let n_bins = n / 2 + 1;
    let mut out = vec![0f32; n_bins];
    for k in 0..n_bins {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (n_i, &x) in frame.iter().enumerate() {
            let ang = -2.0 * PI * k as f32 * n_i as f32 / n as f32;
            re += x * ang.cos();
            im += x * ang.sin();
        }
        out[k] = (re * re + im * im).sqrt();
    }
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_frames_for_ref_length() {
        let pcm = vec![0.1f32; REF_SAMPLES];
        let mel = pcm_to_mel(&pcm);
        let t = mel.len() / N_MELS;
        // 1 + (96000-640)/320 = 299
        assert_eq!(t, 1 + (REF_SAMPLES - WIN) / HOP);
        assert_eq!(mel.len(), t * N_MELS);
    }
}
