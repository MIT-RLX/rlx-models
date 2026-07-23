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

//! Log-mel frontend matching NeMo's `AudioToMelSpectrogramPreprocessor`
//! defaults: optional 0.97 pre-emphasis, centered reflect-padded STFT with
//! a periodic Hann window zero-padded to `n_fft`, power spectrum, a
//! Slaney mel filterbank (Slaney-normalized), natural-log with a `2^-24`
//! additive guard, then `per_feature` mean/std normalization over time.

use std::sync::Arc;

use rustfft::num_complex::Complex;

use crate::config::AsrConfig;

/// `[n_mels, n_frames]` row-major log-mel features.
#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    /// Mel bin count (rows).
    pub n_mels: usize,
    /// Time frames (columns).
    pub n_frames: usize,
    /// Row-major `[n_mels, n_frames]` values.
    pub data: Vec<f32>,
}

/// Round up mel length for compile-cache reuse (fewer encoder graphs).
///
/// Ladder covers ~2.5–80 s at 10 ms hop; beyond that, round up to 1024.
pub fn bucket_mel_frames(n_frames: usize) -> usize {
    const LADDER: &[usize] = &[
        256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
    ];
    let n = n_frames.max(1);
    for &b in LADDER {
        if n <= b {
            return b;
        }
    }
    n.div_ceil(1024).saturating_mul(1024).max(n)
}

/// Zero-pad trailing mel columns so `n_frames == target` (row-major `[n_mels, T]`).
pub fn pad_mel_to_frames(m: &MelSpectrogram, target: usize) -> MelSpectrogram {
    if m.n_frames >= target {
        return MelSpectrogram {
            n_mels: m.n_mels,
            n_frames: m.n_frames,
            data: m.data.clone(),
        };
    }
    let n_mels = m.n_mels;
    let old = m.n_frames;
    let mut data = vec![0.0f32; n_mels * target];
    for mi in 0..n_mels {
        let src = &m.data[mi * old..(mi + 1) * old];
        data[mi * target..mi * target + old].copy_from_slice(src);
    }
    MelSpectrogram {
        n_mels,
        n_frames: target,
        data,
    }
}

/// The model's own analysis frontend: its mel filterbank `[n_mels, n_freq]`
/// and STFT window, extracted from `preprocessor.featurizer.{fb,window}`.
/// Using these guarantees agreement with NeMo's preprocessor.
#[derive(Debug, Clone)]
pub struct Frontend {
    /// Mel filterbank weights, row-major `[n_mels, n_freq]`.
    pub fb: Vec<f32>,
    /// `n_fft / 2 + 1`.
    pub n_freq: usize,
    /// Analysis window of length `win_length` (zero-padded to `n_fft` at STFT).
    pub window: Vec<f32>,
}

impl Frontend {
    /// Load the stored filterbank + window from a `.nemo`.
    pub fn from_model(model: &rlx_nemo::NemoModel, cfg: &AsrConfig) -> anyhow::Result<Self> {
        let fb_t = model.tensor("preprocessor.featurizer.fb")?;
        let win_t = model.tensor("preprocessor.featurizer.window")?;
        let n_freq = cfg.n_fft / 2 + 1;
        anyhow::ensure!(
            fb_t.data.len() == cfg.n_mels * n_freq,
            "featurizer.fb {} != n_mels*n_freq {}",
            fb_t.data.len(),
            cfg.n_mels * n_freq
        );
        Ok(Self {
            fb: fb_t.data,
            n_freq,
            window: win_t.data,
        })
    }
}

/// NeMo's additive log guard, `2^-24`.
const LOG_GUARD: f32 = 5.960_464_5e-8;
/// NeMo `normalize_batch` epsilon.
const NORM_EPS: f32 = 1e-5;
/// Default pre-emphasis coefficient.
const PREEMPH: f32 = 0.97;

/// Compute log-mel features from mono 16 kHz `f32` PCM in `[-1, 1]`.
///
/// When `frontend` is `Some`, the model's own filterbank + window are used
/// (exact NeMo parity); otherwise a Slaney filterbank + periodic Hann are
/// computed from the config.
pub fn log_mel(cfg: &AsrConfig, pcm: &[f32], frontend: Option<&Frontend>) -> MelSpectrogram {
    let n_fft = cfg.n_fft;
    let hop = cfg.hop_length;
    let win = cfg.win_length;
    let n_freq = n_fft / 2 + 1;

    // Pre-emphasis: y[t] = x[t] - 0.97 * x[t-1].
    let mut sig = pcm.to_vec();
    if !sig.is_empty() {
        for t in (1..sig.len()).rev() {
            sig[t] -= PREEMPH * sig[t - 1];
        }
    }

    // center=True reflect padding by n_fft/2 on both sides.
    let padded = reflect_pad(&sig, n_fft / 2);
    let n_frames = if padded.len() >= n_fft {
        1 + (padded.len() - n_fft) / hop
    } else {
        1
    };

    let window = match frontend {
        Some(fe) => center_window(&fe.window, n_fft),
        None => hann_to_nfft(win, n_fft),
    };
    let fft = plan_fft(n_fft);
    let filters = match frontend {
        Some(fe) => fe.fb.clone(),
        None => mel_filterbank(cfg.sample_rate as f64, n_fft, cfg.n_mels),
    };

    let mut data = vec![0.0f32; cfg.n_mels * n_frames];
    let mut buf: Vec<Complex<f32>> = vec![Complex { re: 0.0, im: 0.0 }; n_fft];
    let mut power = vec![0.0f32; n_freq];

    for fi in 0..n_frames {
        let start = fi * hop;
        for j in 0..n_fft {
            let s = padded.get(start + j).copied().unwrap_or(0.0);
            buf[j] = Complex {
                re: s * window[j],
                im: 0.0,
            };
        }
        fft.process(&mut buf);
        for (bin, p) in power.iter_mut().enumerate() {
            let c = buf[bin];
            *p = c.re * c.re + c.im * c.im; // power spectrum (mag_power = 2)
        }
        for mi in 0..cfg.n_mels {
            let row = &filters[mi * n_freq..(mi + 1) * n_freq];
            let acc: f32 = row.iter().zip(&power).map(|(w, p)| w * p).sum();
            data[mi * n_frames + fi] = (acc + LOG_GUARD).ln();
        }
    }

    if cfg.normalize == "per_feature" {
        normalize_per_feature(&mut data, cfg.n_mels, n_frames);
    }

    MelSpectrogram {
        n_mels: cfg.n_mels,
        n_frames,
        data,
    }
}

/// Per-mel-bin standardization over the time axis (unbiased std + eps),
/// matching NeMo `normalize_batch(..., "per_feature")`.
fn normalize_per_feature(data: &mut [f32], n_mels: usize, n_frames: usize) {
    if n_frames == 0 {
        return;
    }
    for mi in 0..n_mels {
        let row = &mut data[mi * n_frames..(mi + 1) * n_frames];
        let mean = row.iter().sum::<f32>() / n_frames as f32;
        let var = if n_frames > 1 {
            row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (n_frames as f32 - 1.0)
        } else {
            0.0
        };
        let std = var.sqrt();
        let denom = std + NORM_EPS;
        for v in row.iter_mut() {
            *v = (*v - mean) / denom;
        }
    }
}

fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    if x.is_empty() {
        return vec![0.0; 2 * pad];
    }
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    // Left reflect: x[pad], x[pad-1], …, x[1] (np.pad 'reflect' excludes the edge).
    for i in (1..=pad).rev() {
        out.push(x[i.min(n - 1)]);
    }
    out.extend_from_slice(x);
    // Right reflect: x[n-2], x[n-3], …
    for i in 1..=pad {
        let idx = n.saturating_sub(1 + i);
        out.push(x[idx]);
    }
    out
}

/// Periodic Hann window of length `win`, centered (zero-padded) into `n_fft`.
fn hann_to_nfft(win: usize, n_fft: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n_fft];
    let left = (n_fft - win) / 2;
    for n in 0..win {
        // torch.hann_window(periodic=True): 0.5 - 0.5 cos(2πn/win).
        let v = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / win as f64).cos();
        w[left + n] = v as f32;
    }
    w
}

/// Center an existing window of length `win.len()` into an `n_fft` frame.
fn center_window(win: &[f32], n_fft: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n_fft];
    let left = n_fft.saturating_sub(win.len()) / 2;
    for (i, &v) in win.iter().enumerate() {
        if left + i < n_fft {
            w[left + i] = v;
        }
    }
    w
}

fn plan_fft(n_fft: usize) -> Arc<dyn rustfft::Fft<f32>> {
    rustfft::FftPlanner::new().plan_fft_forward(n_fft)
}

// ── Slaney mel filterbank (librosa default: htk=False, norm='slaney') ──

fn hz_to_mel(hz: f64) -> f64 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        (hz - f_min) / f_sp
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * ((mel - min_log_mel) * logstep).exp()
    } else {
        f_min + f_sp * mel
    }
}

/// `[n_mels, n_freq]` row-major triangular filterbank with Slaney norm.
fn mel_filterbank(sample_rate: f64, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_freq = n_fft / 2 + 1;
    let f_max = sample_rate / 2.0;
    let fft_freqs: Vec<f64> = (0..n_freq)
        .map(|i| i as f64 * sample_rate / n_fft as f64)
        .collect();

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let mut fb = vec![0.0f32; n_mels * n_freq];
    for m in 0..n_mels {
        let lower = hz_points[m];
        let center = hz_points[m + 1];
        let upper = hz_points[m + 2];
        // Slaney normalization: 2 / (upper - lower).
        let enorm = 2.0 / (upper - lower);
        for (k, &f) in fft_freqs.iter().enumerate() {
            let up = (f - lower) / (center - lower);
            let down = (upper - f) / (upper - center);
            let tri = up.min(down).max(0.0);
            fb[m * n_freq + k] = (tri * enorm) as f32;
        }
    }
    fb
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_nemo::NemoConfig;

    fn test_cfg() -> AsrConfig {
        let yaml = b"preprocessor:\n  features: 80\n  n_fft: 512\n  window_size: 0.025\n  window_stride: 0.01\n  normalize: per_feature\nencoder:\n  d_model: 256\n  n_layers: 2\n  n_heads: 4\n";
        AsrConfig::from_nemo(&NemoConfig::from_yaml_bytes(yaml).unwrap()).unwrap()
    }

    #[test]
    fn mel_shape_and_finite() {
        let cfg = test_cfg();
        // 1 second of a 220 Hz sine at 16 kHz.
        let pcm: Vec<f32> = (0..16_000)
            .map(|n| (2.0 * std::f32::consts::PI * 220.0 * n as f32 / 16_000.0).sin() * 0.5)
            .collect();
        let mel = log_mel(&cfg, &pcm, None);
        assert_eq!(mel.n_mels, 80);
        // center=True: n_frames == 1 + len/hop == 1 + 16000/160 == 101.
        assert_eq!(mel.n_frames, 101);
        assert_eq!(mel.data.len(), 80 * 101);
        assert!(mel.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn per_feature_normalized_has_zero_mean() {
        let cfg = test_cfg();
        let pcm: Vec<f32> = (0..8000).map(|n| (n as f32 * 0.01).sin() * 0.3).collect();
        let mel = log_mel(&cfg, &pcm, None);
        // Each mel bin should be ~zero-mean after per-feature normalization.
        for mi in 0..mel.n_mels {
            let row = &mel.data[mi * mel.n_frames..(mi + 1) * mel.n_frames];
            let mean = row.iter().sum::<f32>() / mel.n_frames as f32;
            assert!(mean.abs() < 1e-3, "bin {mi} mean {mean} not ~0");
        }
    }

    #[test]
    fn filterbank_rows_nonzero() {
        let fb = mel_filterbank(16_000.0, 512, 80);
        let n_freq = 512 / 2 + 1;
        for m in 0..80 {
            let s: f32 = fb[m * n_freq..(m + 1) * n_freq].iter().sum();
            assert!(s > 0.0, "mel filter {m} is all-zero");
        }
    }
}
