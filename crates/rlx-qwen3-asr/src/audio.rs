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

//! Log-mel frontend (transformers `WhisperFeatureExtractor`, 128 mels, 16 kHz)
//! and the Qwen3-Omni audio-encoder downsample geometry.

use crate::config::AudioEncoderConfig;
use anyhow::{Result, ensure};
use rustfft::num_complex::Complex;
use std::sync::Arc;

pub use rlx_whisper::{SAMPLE_RATE, load_wav_mono_f32, parse_wav_mono_f32};

const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_FREQ: usize = N_FFT / 2 + 1;

/// Row-major log-mel spectrogram `[n_mels, n_frames]`.
#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    pub n_mels: usize,
    pub n_frames: usize,
    /// `data[m * n_frames + f]`.
    pub data: Vec<f32>,
}

/// Number of mel frames the feature extractor yields for `n_samples`
/// (`torch.stft(center=True)[..., :-1]` ⇒ `floor(n_samples / hop)`).
pub fn mel_frames_for_samples(n_samples: usize) -> usize {
    n_samples / HOP_LENGTH
}

/// Compute the log-mel spectrogram for one mono 16 kHz utterance.
///
/// Matches `transformers.WhisperFeatureExtractor`: reflect-pad by `n_fft/2`,
/// Hann window, power STFT, Slaney mel filterbank, `log10`, then the
/// `max(x, x.max()-8); (x+4)/4` normalization over the whole utterance.
pub fn pcm_to_log_mel(pcm: &[f32], n_mels: usize) -> Result<MelSpectrogram> {
    ensure!(
        pcm.len() > N_FFT / 2,
        "audio too short ({} samples) for the {N_FFT}-point STFT",
        pcm.len()
    );
    let n_frames = mel_frames_for_samples(pcm.len());
    ensure!(n_frames > 0, "audio yields zero mel frames");

    let filters = mel_filterbank(SAMPLE_RATE as f64, N_FFT, n_mels);
    let window = hann_window(N_FFT);
    let power = stft_power(pcm, &window, n_frames);

    let mut mel = vec![0f32; n_mels * n_frames];
    for f in 0..n_frames {
        for m in 0..n_mels {
            let mut acc = 0f32;
            for bin in 0..N_FREQ {
                acc += filters[m * N_FREQ + bin] * power[f * N_FREQ + bin];
            }
            mel[m * n_frames + f] = acc;
        }
    }

    let mut max = f32::NEG_INFINITY;
    for v in mel.iter_mut() {
        *v = v.max(1e-10).log10();
        max = max.max(*v);
    }
    let floor = max - 8.0;
    for v in mel.iter_mut() {
        *v = v.max(floor);
        *v = (*v + 4.0) / 4.0;
    }

    Ok(MelSpectrogram {
        n_mels,
        n_frames,
        data: mel,
    })
}

/// Power STFT with `center=True` reflect padding, dropping the trailing frame.
fn stft_power(pcm: &[f32], window: &[f32], n_frames: usize) -> Vec<f32> {
    let pad = N_FFT / 2;
    let mut padded = Vec::with_capacity(pcm.len() + 2 * pad);
    // reflect (mirror around the edge sample, excluding it) — np.pad(mode="reflect").
    for i in (1..=pad).rev() {
        padded.push(pcm[i.min(pcm.len() - 1)]);
    }
    padded.extend_from_slice(pcm);
    for i in 1..=pad {
        padded.push(pcm[pcm.len().saturating_sub(i + 1)]);
    }

    let fft = fft_plan();
    let mut buf: Vec<Complex<f32>> = vec![Complex { re: 0.0, im: 0.0 }; N_FFT];
    let mut data = vec![0f32; n_frames * N_FREQ];
    for fi in 0..n_frames {
        let start = fi * HOP_LENGTH;
        for (t, w) in window.iter().enumerate() {
            let s = padded.get(start + t).copied().unwrap_or(0.0);
            buf[t] = Complex { re: s * w, im: 0.0 };
        }
        fft.process(&mut buf);
        for bin in 0..N_FREQ {
            let c = buf[bin];
            data[fi * N_FREQ + bin] = c.re * c.re + c.im * c.im;
        }
    }
    data
}

fn fft_plan() -> Arc<dyn rustfft::Fft<f32>> {
    use std::sync::OnceLock;
    static PLAN: OnceLock<Arc<dyn rustfft::Fft<f32>>> = OnceLock::new();
    PLAN.get_or_init(|| rustfft::FftPlanner::new().plan_fft_forward(N_FFT))
        .clone()
}

/// Periodic Hann window (`torch.hann_window`): `sin^2(pi * n / N)`.
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / n as f32;
            x.sin().powi(2)
        })
        .collect()
}

fn hz_to_mel(hz: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - 0.0) / (200.0 / 3.0);
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / (200.0 / 3.0)
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - 0.0) / (200.0 / 3.0);
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        mel * (200.0 / 3.0)
    }
}

/// Slaney mel filterbank (`norm="slaney"`, `mel_scale="slaney"`).
fn mel_filterbank(sample_rate: f64, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let fmax = sample_rate * 0.5;
    let n_freq = n_fft / 2 + 1;
    let fftfreqs: Vec<f64> = (0..n_freq)
        .map(|k| k as f64 * sample_rate / n_fft as f64)
        .collect();
    let n_pts = n_mels + 2;
    let mel_pts: Vec<f64> = (0..n_pts)
        .map(|i| {
            let mel =
                hz_to_mel(0.0) + (hz_to_mel(fmax) - hz_to_mel(0.0)) * i as f64 / (n_pts - 1) as f64;
            mel_to_hz(mel)
        })
        .collect();
    let fdiff: Vec<f64> = (0..n_pts - 1)
        .map(|i| mel_pts[i + 1] - mel_pts[i])
        .collect();
    let mut w = vec![0f32; n_mels * n_freq];
    for m in 0..n_mels {
        for k in 0..n_freq {
            let f = fftfreqs[k];
            let lower = (f - mel_pts[m]) / fdiff[m];
            let upper = (mel_pts[m + 2] - f) / fdiff[m + 1];
            w[m * n_freq + k] = lower.min(upper).max(0.0) as f32;
        }
        let enorm = 2.0 / (mel_pts[m + 2] - mel_pts[m]) as f32;
        for k in 0..n_freq {
            w[m * n_freq + k] *= enorm;
        }
    }
    w
}

/// Output length of one stride-2, padding-1, kernel-3 convolution.
pub fn conv_len(l: usize) -> usize {
    if l == 0 { 0 } else { (l - 1) / 2 + 1 }
}

/// Post-CNN time length of a `l`-frame chunk (three stacked [`conv_len`]).
pub fn downsample_time(l: usize) -> usize {
    conv_len(conv_len(conv_len(l)))
}

/// Total post-CNN frames (== number of `<|audio_pad|>` placeholders) for a
/// `t`-frame mel — `_get_feat_extract_output_lengths` with the per-block
/// deepstack shortcut.
pub fn aftercnn_len(t: usize, chunk: usize) -> usize {
    let blocks = t / chunk;
    let rem = t % chunk;
    blocks * downsample_time(chunk) + if rem == 0 { 0 } else { downsample_time(rem) }
}

/// Resolved chunk / window geometry for a single utterance.
#[derive(Debug, Clone)]
pub struct AudioGeometry {
    /// Input mel frames (`feature_lens`).
    pub n_frames: usize,
    /// Pre-CNN chunks (`ceil(n_frames / chunk)`).
    pub num_chunks: usize,
    /// Padded width of every chunk.
    pub max_chunk_len: usize,
    /// Per-chunk post-CNN time length.
    pub t_pc: usize,
    /// Post-CNN frequency length (conv_out fan-in = `downsample_hidden_size * freq_pc`).
    pub freq_pc: usize,
    /// Per-conv `[freq, time]` output dims (for static graph shapes).
    pub conv_freq: [usize; 3],
    pub conv_time: [usize; 3],
    /// Valid (non-padding) post-CNN frames = audio token count.
    pub num_audio_tokens: usize,
    /// Per-window attention lengths (block-diagonal windowing), summing to
    /// `num_audio_tokens`.
    pub windows: Vec<usize>,
}

impl AudioGeometry {
    pub fn new(cfg: &AudioEncoderConfig, n_frames: usize) -> Result<Self> {
        ensure!(n_frames > 0, "empty mel");
        let chunk = cfg.chunk_frames();
        let num_chunks = n_frames.div_ceil(chunk);
        let max_chunk_len = if num_chunks > 1 { chunk } else { n_frames };

        let conv_freq = [
            conv_len(cfg.num_mel_bins),
            conv_len(conv_len(cfg.num_mel_bins)),
            conv_len(conv_len(conv_len(cfg.num_mel_bins))),
        ];
        let conv_time = [
            conv_len(max_chunk_len),
            conv_len(conv_len(max_chunk_len)),
            conv_len(conv_len(conv_len(max_chunk_len))),
        ];
        let t_pc = conv_time[2];
        let freq_pc = conv_freq[2];
        let num_audio_tokens = aftercnn_len(n_frames, chunk);

        // Attention windows: post-CNN frames split into `t_pc * ratio`-sized
        // blocks (matches `get_audio_cu_seqlens`).
        let ratio = cfg.n_window_infer / chunk;
        let ratio = ratio.max(1);
        let window_after = (t_pc * ratio).max(1);
        let mut windows = Vec::new();
        let mut remaining = num_audio_tokens;
        while remaining > window_after {
            windows.push(window_after);
            remaining -= window_after;
        }
        if remaining > 0 {
            windows.push(remaining);
        }

        Ok(Self {
            n_frames,
            num_chunks,
            max_chunk_len,
            t_pc,
            freq_pc,
            conv_freq,
            conv_time,
            num_audio_tokens,
            windows,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio_cfg() -> AudioEncoderConfig {
        serde_json::from_str(
            r#"{"d_model":896,"num_mel_bins":128,"num_hidden_layers":18,
                "encoder_attention_heads":14,"encoder_ffn_dim":3584,
                "downsample_hidden_size":480,"output_dim":1024,
                "max_source_positions":1500,"n_window":50,"n_window_infer":800}"#,
        )
        .unwrap()
    }

    #[test]
    fn downsample_matches_known_values() {
        assert_eq!(conv_len(128), 64);
        assert_eq!(downsample_time(100), 13);
        assert_eq!(downsample_time(50), 7);
        // freq path 128 -> 64 -> 32 -> 16
        assert_eq!(conv_len(conv_len(conv_len(128))), 16);
    }

    #[test]
    fn aftercnn_token_counts() {
        assert_eq!(aftercnn_len(100, 100), 13);
        assert_eq!(aftercnn_len(200, 100), 26);
        assert_eq!(aftercnn_len(250, 100), 33);
        assert_eq!(aftercnn_len(50, 100), 7);
    }

    #[test]
    fn geometry_windows_partition_tokens() {
        let cfg = audio_cfg();
        let g = AudioGeometry::new(&cfg, 250).unwrap();
        assert_eq!(g.num_chunks, 3);
        assert_eq!(g.max_chunk_len, 100);
        assert_eq!(g.t_pc, 13);
        assert_eq!(g.freq_pc, 16);
        assert_eq!(g.num_audio_tokens, 33);
        assert_eq!(g.windows.iter().sum::<usize>(), 33);
        // window_after = 13 * 8 = 104 ⇒ a single window of 33
        assert_eq!(g.windows, vec![33]);

        let g2 = AudioGeometry::new(&cfg, 1200).unwrap();
        // aftercnn(1200) = 12 * 13 = 156, windows of 104 -> [104, 52]
        assert_eq!(g2.num_audio_tokens, 156);
        assert_eq!(g2.windows, vec![104, 52]);
    }
}
