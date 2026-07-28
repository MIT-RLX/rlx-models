// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! 80-bin log-mel frontend.
//!
//! Matches `tools/audio_io.py` defaults where practical:
//! 25 ms / 10 ms frames @ 16 kHz, dither + pre-emphasis, povey-ish window.
//! Pitch / audio-analytics fusion is still stubbed; Python path also applies
//! a silence-fbank calibration affine before e5 distill.

use crate::spec::MEL_BINS;
use anyhow::{Result, bail};

/// Frame shift / length at 16 kHz (10 ms / 25 ms) — Kaldi-style fbank defaults from mini.json.
pub const FRAME_SHIFT: usize = 160;
pub const FRAME_LENGTH: usize = 400;
pub const SAMPLE_RATE: u32 = 16_000;
/// Kaldi dither on int16-scale samples (see `fbank-with-audio-analytics.dither`).
pub const DITHER: f32 = 1.0;
pub const PREEMPH: f32 = 0.97;

/// Compute `[n_frames, MEL_BINS]` log-mel filterbank from mono PCM f32 in [-1, 1].
pub fn log_mel_fbank(pcm: &[f32], sample_rate: u32) -> Result<Vec<Vec<f32>>> {
    if sample_rate != 8_000 && sample_rate != 16_000 {
        bail!("unsupported sample rate {sample_rate}");
    }
    let mut pcm = if sample_rate == 8_000 {
        upsample_2x(pcm)
    } else {
        pcm.to_vec()
    };
    if pcm.len() < FRAME_LENGTH {
        return Ok(Vec::new());
    }
    // int16-domain dither + DC remove + pre-emphasis (tools/audio_io.py)
    for x in pcm.iter_mut() {
        *x *= 32768.0;
    }
    // deterministic light dither; production can inject RNG
    for (i, x) in pcm.iter_mut().enumerate() {
        let u =
            ((i as u32).wrapping_mul(1664525).wrapping_add(1013904223) >> 8) as f32 / 16777216.0;
        *x += DITHER * (u - 0.5) * 2.0;
    }
    let mean = pcm.iter().sum::<f32>() / pcm.len() as f32;
    for x in pcm.iter_mut() {
        *x -= mean;
    }
    for i in (1..pcm.len()).rev() {
        pcm[i] -= PREEMPH * pcm[i - 1];
    }
    let n_fft = 512;
    let filters = mel_filterbank(SAMPLE_RATE, n_fft, MEL_BINS);
    let window: Vec<f32> = (0..FRAME_LENGTH)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / (FRAME_LENGTH as f32 - 1.0);
            let hann = 0.5 - 0.5 * x.cos();
            hann.powf(0.85) // Kaldi "povey"
        })
        .collect();
    let n_frames = 1 + (pcm.len() - FRAME_LENGTH) / FRAME_SHIFT;
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let start = f * FRAME_SHIFT;
        let mut frame = vec![0f32; n_fft];
        for i in 0..FRAME_LENGTH {
            frame[i] = pcm[start + i] * window[i];
        }
        let power = rfft_power(&frame);
        let mut mel = vec![0f32; MEL_BINS];
        for (b, filt) in filters.iter().enumerate() {
            let mut e = 0f32;
            for (k, &w) in filt.iter().enumerate() {
                e += w * power[k];
            }
            mel[b] = (e.max(1e-10)).ln();
        }
        out.push(mel);
    }
    Ok(out)
}

fn upsample_2x(pcm: &[f32]) -> Vec<f32> {
    let mut o = Vec::with_capacity(pcm.len() * 2);
    for i in 0..pcm.len() {
        o.push(pcm[i]);
        let n = if i + 1 < pcm.len() {
            pcm[i + 1]
        } else {
            pcm[i]
        };
        o.push(0.5 * (pcm[i] + n));
    }
    o
}

fn mel_filterbank(sr: u32, n_fft: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let fmin = 20.0;
    let fmax = (sr as f32) / 2.0;
    let n_freqs = n_fft / 2 + 1;
    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |m: f32| 700.0 * (10f32.powf(m / 2595.0) - 1.0);
    let mmin = hz_to_mel(fmin);
    let mmax = hz_to_mel(fmax);
    let mut mels = Vec::with_capacity(n_mels + 2);
    for i in 0..n_mels + 2 {
        mels.push(mmin + (mmax - mmin) * i as f32 / (n_mels as f32 + 1.0));
    }
    let hz: Vec<f32> = mels.iter().map(|&m| mel_to_hz(m)).collect();
    let bins: Vec<usize> = hz
        .iter()
        .map(|&f| ((n_fft as f32 + 1.0) * f / sr as f32).floor() as usize)
        .collect();
    let mut filters = vec![vec![0f32; n_freqs]; n_mels];
    for m in 1..=n_mels {
        let left = bins[m - 1];
        let center = bins[m];
        let right = bins[m + 1];
        for k in left..center {
            if center != left && k < n_freqs {
                filters[m - 1][k] = (k - left) as f32 / (center - left) as f32;
            }
        }
        for k in center..right {
            if right != center && k < n_freqs {
                filters[m - 1][k] = (right - k) as f32 / (right - center) as f32;
            }
        }
    }
    filters
}

fn rfft_power(frame: &[f32]) -> Vec<f32> {
    // Naive DFT magnitude-squared (fine for frontend scaffolding; swap to rlx-fft later).
    let n = frame.len();
    let n_freqs = n / 2 + 1;
    let mut out = vec![0f32; n_freqs];
    for k in 0..n_freqs {
        let mut re = 0f32;
        let mut im = 0f32;
        for (t, &x) in frame.iter().enumerate() {
            let ang = -2.0 * std::f32::consts::PI * k as f32 * t as f32 / n as f32;
            re += x * ang.cos();
            im += x * ang.sin();
        }
        out[k] = re * re + im * im;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fbank_shape() {
        let pcm = vec![0.1f32; 16_000];
        let m = log_mel_fbank(&pcm, 16_000).unwrap();
        assert!(!m.is_empty());
        assert_eq!(m[0].len(), MEL_BINS);
    }
}
