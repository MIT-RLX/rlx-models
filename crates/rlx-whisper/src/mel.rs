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

//! CPU mel spectrogram (OpenAI Whisper frontend).

use crate::audio::{MelSpectrogram, SAMPLE_RATE};
use rustfft::num_complex::Complex;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_FREQ: usize = N_FFT / 2 + 1;

/// Mel frames needed for `n_samples` of 16 kHz PCM (OpenAI hop 160, fft 400).
pub fn mel_frames_from_samples(n_samples: usize) -> usize {
    use crate::audio::{N_FRAMES, N_SAMPLES};
    let n = n_samples.min(N_SAMPLES);
    if n < N_FFT {
        return 1;
    }
    let n_stft = (n - N_FFT) / HOP_LENGTH + 1;
    n_stft.clamp(1, N_FRAMES)
}

/// Bucketed mel frame count for a PCM utterance (compile-cache friendly).
pub fn mel_geometry_frames_for_pcm(pcm: &[f32]) -> usize {
    use crate::audio::pad_or_trim_pcm;
    let pcm = pad_or_trim_pcm(pcm);
    bucket_mel_frames(mel_frames_from_samples(pcm.len()))
}

/// Inverse of [`crate::config::WhisperConfig::encoder_seq_len`] (conv stride-2).
///
/// Used to size align-hidden compile graphs from a sliced encoder sequence length.
pub fn mel_frames_for_enc_seq(enc_seq: usize) -> usize {
    enc_seq.saturating_mul(2).saturating_sub(1).max(1)
}

/// Round up mel length for compile-cache reuse (fewer encoder graphs).
pub fn bucket_mel_frames(n_frames: usize) -> usize {
    use crate::audio::N_FRAMES;
    let mut b = 8usize;
    while b < n_frames {
        b = b.saturating_mul(2);
    }
    b.clamp(8, N_FRAMES)
}

/// PCM sample count that yields at least `mel_frames` STFT columns (OpenAI hop 160, fft 400).
pub fn samples_for_mel_frames(mel_frames: usize) -> usize {
    if mel_frames <= 1 {
        return N_FFT;
    }
    (mel_frames - 1) * HOP_LENGTH + N_FFT
}

/// Pad mel spectrogram to `target` frames (zero-fill trailing columns).
pub fn pad_mel_to_frames(m: &MelSpectrogram, target: usize) -> MelSpectrogram {
    if m.n_frames >= target {
        return m.clone();
    }
    let n_mels = m.n_mels;
    let old_frames = m.n_frames;
    let mut data = vec![0f32; n_mels * target];
    for mi in 0..n_mels {
        for fi in 0..old_frames {
            data[mi * target + fi] = m.data[mi * old_frames + fi];
        }
    }
    MelSpectrogram {
        n_mels,
        n_frames: target,
        data,
    }
}

/// Build log-mel for one utterance: row-major `[1, n_mels, n_frames]` (padded/truncated to `n_frames`).
pub fn pcm_to_log_mel(pcm: &[f32], n_mels: usize, n_frames: usize) -> MelSpectrogram {
    let filters = cached_mel_filterbank(n_mels);
    let window = cached_hann_window(N_FFT);
    let stft = stft_mag(pcm, window.as_slice());
    let n_stft_frames = stft.n_frames;
    let mut mel = vec![0f32; n_mels * n_frames];
    for fi in 0..n_stft_frames.min(n_frames) {
        for mi in 0..n_mels {
            let mut acc = 0f32;
            for bin in 0..N_FREQ {
                acc += filters[mi * N_FREQ + bin] * stft.data[fi * N_FREQ + bin];
            }
            mel[mi * n_frames + fi] = acc;
        }
    }
    let mut max = f32::NEG_INFINITY;
    for v in mel.iter_mut() {
        *v = (*v).max(1e-10).log10();
        max = max.max(*v);
    }
    let floor = max - 8.0;
    for v in mel.iter_mut() {
        *v = (*v).max(floor);
        *v = (*v + 4.0) / 4.0;
    }
    MelSpectrogram {
        n_mels,
        n_frames,
        data: mel,
    }
}

/// Stack `N` mels into `[N, n_mels, n_frames]` row-major.
pub fn stack_mels(mels: &[MelSpectrogram]) -> Vec<f32> {
    if mels.is_empty() {
        return Vec::new();
    }
    let n_mels = mels[0].n_mels;
    let n_frames = mels[0].n_frames;
    let plane = n_mels * n_frames;
    let mut out = vec![0f32; mels.len() * plane];
    for (i, m) in mels.iter().enumerate() {
        debug_assert_eq!(m.n_mels, n_mels);
        debug_assert_eq!(m.n_frames, n_frames);
        out[i * plane..(i + 1) * plane].copy_from_slice(&m.data);
    }
    out
}

struct StftMag {
    n_frames: usize,
    data: Vec<f32>,
}

fn stft_mag(pcm: &[f32], window: &[f32]) -> StftMag {
    let pad = N_FFT / 2;
    let mut padded = Vec::with_capacity(pcm.len() + 2 * pad);
    for i in (1..=pad).rev() {
        let j = pcm.len().saturating_sub(i);
        padded.push(pcm[j]);
    }
    padded.extend_from_slice(pcm);
    for i in 1..=pad {
        let j = pcm.len().saturating_sub(i);
        padded.push(pcm[j]);
    }
    let n_frames = if padded.len() >= N_FFT {
        1 + (padded.len() - N_FFT) / HOP_LENGTH
    } else {
        0
    };
    let mut data = vec![0f32; n_frames * N_FREQ];
    // Cached forward FFT plan (length N_FFT). rustfft's forward transform computes
    // `Σ s·w·e^{-iωt}`, matching the previous naive DFT (`re += s·w·cos`,
    // `im -= s·w·sin`) within float rounding; only bins `0..N_FREQ` are kept.
    let fft = fft_plan();
    let mut buf: Vec<Complex<f32>> = vec![Complex { re: 0.0, im: 0.0 }; N_FFT];
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
    StftMag { n_frames, data }
}

/// Shared forward FFT plan of length [`N_FFT`] (rustfft plans are `Send + Sync`).
fn fft_plan() -> Arc<dyn rustfft::Fft<f32>> {
    static PLAN: OnceLock<Arc<dyn rustfft::Fft<f32>>> = OnceLock::new();
    PLAN.get_or_init(|| rustfft::FftPlanner::new().plan_fft_forward(N_FFT))
        .clone()
}

/// Hann window, memoized by length (deterministic; recomputed otherwise per call).
fn cached_hann_window(n: usize) -> Arc<Vec<f32>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<Vec<f32>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    guard
        .entry(n)
        .or_insert_with(|| Arc::new(hann_window(n)))
        .clone()
}

/// Mel filterbank, memoized by `n_mels` (constant `SAMPLE_RATE`/`N_FFT`).
fn cached_mel_filterbank(n_mels: usize) -> Arc<Vec<f32>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<Vec<f32>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    guard
        .entry(n_mels)
        .or_insert_with(|| Arc::new(mel_filterbank(SAMPLE_RATE as f64, N_FFT, n_mels)))
        .clone()
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / n as f32;
            (x.sin()).powi(2)
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
        min_log_hz * ((logstep * (mel - min_log_mel)).exp())
    } else {
        mel * (200.0 / 3.0)
    }
}

fn mel_filterbank(sample_rate: f64, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let fmax = sample_rate * 0.5;
    let n_freq = n_fft / 2 + 1;
    let fftfreqs: Vec<f64> = (0..n_freq)
        .map(|k| k as f64 * sample_rate / n_fft as f64)
        .collect();
    let n_mel_points = n_mels + 2;
    let mel_pts: Vec<f64> = (0..n_mel_points)
        .map(|i| {
            let mel = hz_to_mel(0.0)
                + (hz_to_mel(fmax) - hz_to_mel(0.0)) * i as f64 / (n_mel_points - 1) as f64;
            mel_to_hz(mel)
        })
        .collect();
    let mut fdiff = vec![0f64; n_mel_points - 1];
    for i in 0..fdiff.len() {
        fdiff[i] = mel_pts[i + 1] - mel_pts[i];
    }
    let mut weights = vec![0f32; n_mels * n_freq];
    for m in 0..n_mels {
        for k in 0..n_freq {
            let f = fftfreqs[k];
            let lower = (f - mel_pts[m]) / fdiff[m];
            let upper = (mel_pts[m + 2] - f) / fdiff[m + 1];
            let v = lower.min(upper).max(0.0) as f32;
            weights[m * n_freq + k] = v;
        }
        let enorm = 2.0 / (mel_pts[m + 2] - mel_pts[m]) as f32;
        for k in 0..n_freq {
            weights[m * n_freq + k] *= enorm;
        }
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::N_FRAMES;

    #[test]
    fn mel_nonempty_for_tone() {
        let sr = SAMPLE_RATE;
        let pcm: Vec<f32> = (0..sr)
            .map(|i| (440.0 * 2.0 * std::f32::consts::PI * i as f32 / sr as f32).sin() * 0.2)
            .collect();
        let mel = pcm_to_log_mel(&pcm, 80, N_FRAMES);
        assert_eq!(mel.n_mels, 80);
        assert_eq!(mel.n_frames, N_FRAMES);
        let energy: f32 = mel.data.iter().map(|x| x.abs()).sum();
        assert!(energy > 0.0);
    }

    /// Reference naive O(N^2) DFT (the pre-FFT implementation) for parity locking.
    fn stft_mag_naive(pcm: &[f32], window: &[f32]) -> (usize, Vec<f32>) {
        let pad = N_FFT / 2;
        let mut padded = Vec::with_capacity(pcm.len() + 2 * pad);
        for i in (1..=pad).rev() {
            padded.push(pcm[pcm.len().saturating_sub(i)]);
        }
        padded.extend_from_slice(pcm);
        for i in 1..=pad {
            padded.push(pcm[pcm.len().saturating_sub(i)]);
        }
        let n_frames = if padded.len() >= N_FFT {
            1 + (padded.len() - N_FFT) / HOP_LENGTH
        } else {
            0
        };
        let mut data = vec![0f32; n_frames * N_FREQ];
        for fi in 0..n_frames {
            let start = fi * HOP_LENGTH;
            for bin in 0..N_FREQ {
                let mut re = 0f32;
                let mut im = 0f32;
                let omega = 2.0 * std::f32::consts::PI * bin as f32 / N_FFT as f32;
                for (t, w) in window.iter().enumerate() {
                    let s = padded.get(start + t).copied().unwrap_or(0.0);
                    let ang = omega * t as f32;
                    re += s * w * ang.cos();
                    im -= s * w * ang.sin();
                }
                data[fi * N_FREQ + bin] = re * re + im * im;
            }
        }
        (n_frames, data)
    }

    #[test]
    fn fft_stft_matches_naive_dft() {
        let sr = SAMPLE_RATE;
        let pcm: Vec<f32> = (0..sr)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.3 * (440.0 * 2.0 * std::f32::consts::PI * t).sin()
                    + 0.1 * (1300.0 * 2.0 * std::f32::consts::PI * t).sin()
            })
            .collect();
        let window = hann_window(N_FFT);
        let fast = stft_mag(&pcm, &window);
        let (n_naive, naive) = stft_mag_naive(&pcm, &window);
        assert_eq!(fast.n_frames, n_naive);
        let peak = naive.iter().cloned().fold(0f32, f32::max).max(1e-12);
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for (a, b) in fast.data.iter().zip(naive.iter()) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            // Relative error is only meaningful on non-null bins; near-null bins are
            // dominated by the naive DFT's own f32 cancellation noise (the FFT is more
            // accurate there) and are floored out by the downstream log-mel clamp.
            if *b > 1e-4 * peak {
                max_rel = max_rel.max(d / b.abs());
            }
        }
        assert!(
            max_abs < 1e-3 * peak,
            "stft max_abs diff too high: {max_abs}"
        );
        assert!(max_rel < 1e-3, "stft max_rel diff too high: {max_rel}");
    }
}
