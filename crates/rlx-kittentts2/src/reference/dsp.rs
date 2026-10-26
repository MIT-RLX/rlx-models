// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Front-end features of a reference clip, each matching the exact upstream
//! extractor it replaces:
//!
//! | feature | upstream | consumer |
//! |---|---|---|
//! | [`s3gen_mel`] | Matcha `mel_spectrogram` (24 kHz, n_fft 1920, hop 480, 80 bins ≤ 8 kHz) | S3Gen `prompt_feat` |
//! | [`tokenizer_log_mel`] | Whisper `log_mel_spectrogram` (16 kHz, n_fft 400, hop 160, 128 bins) | S3 tokenizer |
//! | [`kaldi_fbank`] | `torchaudio.compliance.kaldi.fbank` defaults (Povey window, 80 bins) | CAM++ |

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// `torch.hann_window(n)` (periodic).
pub fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos()) as f32)
        .collect()
}

/// Kaldi's Povey window: a symmetric Hann raised to 0.85.
fn povey(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / (n - 1) as f64).cos();
            h.powf(0.85) as f32
        })
        .collect()
}

fn hz_to_mel_slaney(hz: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / F_SP
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        mel * F_SP
    }
}

/// `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)` (Slaney scale and
/// area normalization), `[n_mels, n_fft/2 + 1]` row-major.
pub fn slaney_mel_filters(sr: f64, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let (lo, hi) = (hz_to_mel_slaney(fmin), hz_to_mel_slaney(fmax));
    let pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz_slaney(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut fb = vec![0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let (l, c, h) = (pts[m], pts[m + 1], pts[m + 2]);
        let enorm = 2.0 / (h - l);
        for k in 0..n_bins {
            let f = k as f64 * sr / n_fft as f64;
            let lower = (f - l) / (c - l);
            let upper = (h - f) / (h - c);
            fb[m * n_bins + k] = (lower.min(upper).max(0.0) * enorm) as f32;
        }
    }
    fb
}

/// `F.pad(x, (p, p), mode="reflect")` (the edge sample is not repeated).
fn reflect_pad(x: &[f32], p: usize) -> Vec<f32> {
    let n = x.len();
    let at = |i: isize| -> f32 {
        let j = if i < 0 {
            -i
        } else if i as usize >= n {
            2 * (n as isize - 1) - i
        } else {
            i
        };
        x[j.clamp(0, n as isize - 1) as usize]
    };
    (-(p as isize)..(n + p) as isize).map(at).collect()
}

/// Windowed power or magnitude spectra of `frames` hops over `x`.
struct Stft {
    n_fft: usize,
    fft: Arc<dyn Fft<f32>>,
}

impl Stft {
    fn new(n_fft: usize) -> Self {
        Self {
            n_fft,
            fft: FftPlanner::new().plan_fft_forward(n_fft),
        }
    }

    /// `|X[k]|²` for `k < n_bins` of one already-windowed, zero-padded frame.
    fn power(&self, frame: &[f32], buf: &mut Vec<Complex<f32>>, n_bins: usize, out: &mut [f64]) {
        buf.clear();
        buf.extend(
            (0..self.n_fft).map(|j| Complex::new(frame.get(j).copied().unwrap_or(0.0), 0.0)),
        );
        self.fft.process(buf);
        for (k, o) in out.iter_mut().take(n_bins).enumerate() {
            let c = buf[k];
            *o = (c.re as f64) * (c.re as f64) + (c.im as f64) * (c.im as f64);
        }
    }
}

/// S3Gen's `prompt_feat`: `[frames, 80]` log-mel of 24 kHz audio.
///
/// Reflect-pad 720 each side, 1920-point Hann frames every 480 samples (no
/// centering), `sqrt(|X|² + 1e-9)`, mel 0–8 kHz, `ln(max(·, 1e-5))`.
pub fn s3gen_mel(wav24: &[f32]) -> (usize, Vec<f32>) {
    const N_FFT: usize = 1920;
    const HOP: usize = 480;
    const N_MELS: usize = 80;
    let x = reflect_pad(wav24, (N_FFT - HOP) / 2);
    let frames = if x.len() >= N_FFT {
        1 + (x.len() - N_FFT) / HOP
    } else {
        0
    };
    let win = hann_periodic(N_FFT);
    let n_bins = N_FFT / 2 + 1;
    let fb = slaney_mel_filters(24_000.0, N_FFT, N_MELS, 0.0, 8000.0);
    let stft = Stft::new(N_FFT);
    let mut buf = Vec::with_capacity(N_FFT);
    let mut frame = vec![0f32; N_FFT];
    let mut pw = vec![0f64; n_bins];
    let mut out = vec![0f32; frames * N_MELS];
    for f in 0..frames {
        for (j, v) in frame.iter_mut().enumerate() {
            *v = x[f * HOP + j] * win[j];
        }
        stft.power(&frame, &mut buf, n_bins, &mut pw);
        let mag: Vec<f64> = pw.iter().map(|p| (p + 1e-9).sqrt()).collect();
        for m in 0..N_MELS {
            let row = &fb[m * n_bins..(m + 1) * n_bins];
            let s: f64 = row.iter().zip(&mag).map(|(&w, &v)| w as f64 * v).sum();
            out[f * N_MELS + m] = s.max(1e-5).ln() as f32;
        }
    }
    (frames, out)
}

/// The S3 tokenizer's (Whisper's) log-mel: `[128, frames]` of 16 kHz audio,
/// `frames = len / 160`.
///
/// Centered 400-point Hann STFT (reflect), the last frame dropped, power,
/// 128 Slaney mel bins, `log10(max(·, 1e-10))`, floored at `max − 8`, then
/// `(x + 4) / 4`.
pub fn tokenizer_log_mel(wav16: &[f32]) -> (usize, Vec<f32>) {
    const N_FFT: usize = 400;
    const HOP: usize = 160;
    const N_MELS: usize = 128;
    let x = reflect_pad(wav16, N_FFT / 2);
    let frames = wav16.len() / HOP;
    let win = hann_periodic(N_FFT);
    let n_bins = N_FFT / 2 + 1;
    let fb = slaney_mel_filters(16_000.0, N_FFT, N_MELS, 0.0, 8000.0);
    let stft = Stft::new(N_FFT);
    let mut buf = Vec::with_capacity(N_FFT);
    let mut frame = vec![0f32; N_FFT];
    let mut pw = vec![0f64; n_bins];
    let mut out = vec![0f32; N_MELS * frames];
    for f in 0..frames {
        for (j, v) in frame.iter_mut().enumerate() {
            *v = x[f * HOP + j] * win[j];
        }
        stft.power(&frame, &mut buf, n_bins, &mut pw);
        for m in 0..N_MELS {
            let row = &fb[m * n_bins..(m + 1) * n_bins];
            let s: f64 = row.iter().zip(&pw).map(|(&w, &v)| w as f64 * v).sum();
            out[m * frames + f] = (s.max(1e-10) as f32).log10();
        }
    }
    let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in out.iter_mut() {
        *v = (v.max(max - 8.0) + 4.0) / 4.0;
    }
    (frames, out)
}

fn kaldi_mel(f: f64) -> f64 {
    1127.0 * (1.0 + f / 700.0).ln()
}

/// Kaldi mel banks over `n_fft / 2` bins (no Nyquist), `[n_mels, n_fft/2 + 1]`
/// with the Nyquist column zero — `torchaudio`'s `get_mel_banks` + pad.
fn kaldi_mel_banks(sr: f64, n_fft: usize, n_mels: usize, low: f64, high: f64) -> Vec<f32> {
    let n_bins = n_fft / 2;
    let fft_bin_width = sr / n_fft as f64;
    let (mlo, mhi) = (kaldi_mel(low), kaldi_mel(high));
    let delta = (mhi - mlo) / (n_mels + 1) as f64;
    let mut out = vec![0f32; n_mels * (n_bins + 1)];
    for m in 0..n_mels {
        let left = mlo + m as f64 * delta;
        let center = left + delta;
        let right = center + delta;
        for k in 0..n_bins {
            let mel = kaldi_mel(fft_bin_width * k as f64);
            let up = (mel - left) / (center - left);
            let down = (right - mel) / (right - center);
            out[m * (n_bins + 1) + k] = up.min(down).max(0.0) as f32;
        }
    }
    out
}

/// `torchaudio.compliance.kaldi.fbank(x, num_mel_bins=80)` at 16 kHz:
/// `[frames, 80]`, `frames = 1 + (len − 400) / 160` (snip edges).
///
/// Per frame: remove DC, pre-emphasis 0.97 (first sample against itself),
/// Povey window, zero-pad to 512, power spectrum, Kaldi mel 20 Hz–8 kHz,
/// `ln(max(·, f32::EPSILON))`. No dither.
pub fn kaldi_fbank(wav16: &[f32]) -> (usize, Vec<f32>) {
    const LEN: usize = 400;
    const SHIFT: usize = 160;
    const N_FFT: usize = 512;
    const N_MELS: usize = 80;
    let frames = if wav16.len() >= LEN {
        1 + (wav16.len() - LEN) / SHIFT
    } else {
        0
    };
    let win = povey(LEN);
    let banks = kaldi_mel_banks(16_000.0, N_FFT, N_MELS, 20.0, 8000.0);
    let n_bins = N_FFT / 2 + 1;
    let stft = Stft::new(N_FFT);
    let mut buf = Vec::with_capacity(N_FFT);
    let mut frame = vec![0f32; LEN];
    let mut pw = vec![0f64; n_bins];
    let mut out = vec![0f32; frames * N_MELS];
    for f in 0..frames {
        frame.copy_from_slice(&wav16[f * SHIFT..f * SHIFT + LEN]);
        let mean = frame.iter().sum::<f32>() / LEN as f32;
        frame.iter_mut().for_each(|v| *v -= mean);
        for j in (1..LEN).rev() {
            frame[j] -= 0.97 * frame[j - 1];
        }
        frame[0] -= 0.97 * frame[0];
        frame.iter_mut().zip(&win).for_each(|(v, w)| *v *= w);
        stft.power(&frame, &mut buf, n_bins, &mut pw);
        for m in 0..N_MELS {
            let row = &banks[m * n_bins..(m + 1) * n_bins];
            let s: f64 = row.iter().zip(&pw).map(|(&w, &v)| w as f64 * v).sum();
            out[f * N_MELS + m] = (s as f32).max(f32::EPSILON).ln();
        }
    }
    (frames, out)
}

/// Subtract each bin's mean over time (CAM++'s input normalization).
pub fn mean_normalize(feats: &mut [f32], frames: usize, dim: usize) {
    for c in 0..dim {
        let m = (0..frames).map(|t| feats[t * dim + c]).sum::<f32>() / frames as f32;
        (0..frames).for_each(|t| feats[t * dim + c] -= m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_pad_skips_the_edge_sample() {
        assert_eq!(
            reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2),
            [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
    }

    #[test]
    fn frame_counts_follow_the_upstream_extractors() {
        let x = vec![0.01f32; 16_000];
        assert_eq!(tokenizer_log_mel(&x).0, 100);
        assert_eq!(kaldi_fbank(&x).0, 98);
        let y = vec![0.01f32; 24_000];
        assert_eq!(s3gen_mel(&y).0, 50);
    }
}
