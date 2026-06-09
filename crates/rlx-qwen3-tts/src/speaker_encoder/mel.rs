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

//! Mel-spectrogram front-end matching qwen_tts `mel_spectrogram`.
//!
//! Pipeline (exactly mirrors HF):
//!   1. Reflect-pad PCM by `(n_fft - hop) // 2` on each side.
//!   2. STFT with Hann window (`n=win_size`), hop=`hop_size`, `center=False`,
//!      one-sided, complex.
//!   3. Magnitude = `sqrt(re^2 + im^2 + 1e-9)`.
//!   4. Slaney-norm librosa mel filterbank `[num_mels, n_bins]` × magnitude.
//!   5. Natural log with `clip(x, min=1e-5)`.
//!
//! Returns `[num_mels, T]` (matching HF before its `transpose(1, 2)`).

use crate::speaker_encoder::config::MelParams;
use anyhow::{Result, ensure};
use ndarray::Array2;
use rlx_fft::reference::fft_real_batch;

/// Hann window (PyTorch periodic convention): `sin(pi * i / n)^2`.
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / n as f32;
            x.sin().powi(2)
        })
        .collect()
}

/// Reflect padding ignoring the edge sample (PyTorch `reflect` semantics).
fn reflect_pad(pcm: &[f32], pad: usize) -> Vec<f32> {
    let n = pcm.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        // reflect across index 0, skipping the boundary sample itself.
        let j = pad - i;
        out.push(pcm[j]);
    }
    out.extend_from_slice(pcm);
    for i in 0..pad {
        let j = n.saturating_sub(2 + i);
        out.push(pcm[j]);
    }
    out
}

fn hz_to_mel_slaney(hz: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / (200.0 / 3.0);
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / (200.0 / 3.0)
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / (200.0 / 3.0);
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        mel * (200.0 / 3.0)
    }
}

/// Slaney-normalized mel filterbank matching `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm='slaney')`.
/// Layout: row-major `[num_mels, n_bins]`.
pub fn slaney_mel_filterbank(
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
    sample_rate: f64,
) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let fftfreqs: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * sample_rate / n_fft as f64)
        .collect();
    let mel_lo = hz_to_mel_slaney(fmin);
    let mel_hi = hz_to_mel_slaney(fmax);
    let n_pts = n_mels + 2;
    let mel_pts: Vec<f64> = (0..n_pts)
        .map(|i| {
            let mel = mel_lo + (mel_hi - mel_lo) * i as f64 / (n_pts - 1) as f64;
            mel_to_hz_slaney(mel)
        })
        .collect();
    let mut fb = vec![0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let lo = mel_pts[m];
        let ce = mel_pts[m + 1];
        let hi = mel_pts[m + 2];
        // Slaney area-normalization: enorm = 2 / (mel_pts[m+2] - mel_pts[m]).
        let enorm = 2.0 / (hi - lo).max(1e-12);
        for (k, &f) in fftfreqs.iter().enumerate() {
            let v = if f <= lo || f >= hi {
                0.0
            } else if f <= ce {
                ((f - lo) / (ce - lo).max(1e-12)) * enorm
            } else {
                ((hi - f) / (hi - ce).max(1e-12)) * enorm
            };
            fb[m * n_bins + k] = v as f32;
        }
    }
    fb
}

/// PCM (mono, sample_rate) → `[num_mels, T]` log-mel matching `qwen_tts.mel_spectrogram`.
pub fn log_mel(pcm: &[f32], params: &MelParams) -> Result<Array2<f32>> {
    let n_fft = params.n_fft;
    let win = params.win;
    let hop = params.hop;
    ensure!(win == n_fft, "win_size must equal n_fft (HF assumption)");
    let pad = (n_fft - hop) / 2;
    let padded = reflect_pad(pcm, pad);
    let n_frames = if padded.len() >= n_fft {
        1 + (padded.len() - n_fft) / hop
    } else {
        0
    };
    ensure!(n_frames > 0, "input too short for STFT");

    let window = hann_window(win);
    let n_bins = n_fft / 2 + 1;

    // Pack all frames into a `[n_frames, n_fft]` block and FFT in one shot.
    let mut block = vec![0f32; n_frames * n_fft];
    for fi in 0..n_frames {
        let start = fi * hop;
        let src = &padded[start..start + n_fft];
        let dst = &mut block[fi * n_fft..(fi + 1) * n_fft];
        for (j, (&s, &w)) in src.iter().zip(window.iter()).enumerate() {
            dst[j] = s * w;
        }
    }
    let spec = fft_real_batch(&block, n_frames, n_fft)?;

    // sqrt(re^2 + im^2 + 1e-9) per (frame, bin).
    let mut mag = vec![0f32; n_frames * n_bins];
    for fi in 0..n_frames {
        let in_base = fi * n_fft * 2;
        let out_base = fi * n_bins;
        for k in 0..n_bins {
            let re = spec[in_base + k * 2];
            let im = spec[in_base + k * 2 + 1];
            mag[out_base + k] = (re * re + im * im + 1e-9).sqrt();
        }
    }

    // mel = filterbank @ mag, then log(max(mel, 1e-5)).
    let filters = slaney_mel_filterbank(
        n_fft,
        params.num_mels,
        params.fmin,
        params.fmax,
        params.sample_rate,
    );
    let mut mel = Array2::<f32>::zeros((params.num_mels, n_frames));
    for fi in 0..n_frames {
        for m in 0..params.num_mels {
            let row = &filters[m * n_bins..(m + 1) * n_bins];
            let frame = &mag[fi * n_bins..(fi + 1) * n_bins];
            let mut acc = 0f32;
            for k in 0..n_bins {
                acc += row[k] * frame[k];
            }
            mel[[m, fi]] = acc.max(1e-5).ln();
        }
    }
    Ok(mel)
}
