// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Streaming mel frontend (16 kHz, 10 ms hop).

use alloc::vec;
use alloc::vec::Vec;
use core::f32::consts::{PI, TAU};

pub const SAMPLE_RATE_16K: usize = 16_000;

#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub f_min: f32,
    pub f_max: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: SAMPLE_RATE_16K,
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            n_mels: 32,
            f_min: 0.0,
            f_max: 8000.0,
        }
    }
}

/// Streaming mel extractor with `(x / 10) + 2` normalization.
pub struct MelFrontend {
    cfg: MelConfig,
    window: Vec<f32>,
    mel_fb: Vec<f32>,
    leftover: Vec<f32>,
}

impl MelFrontend {
    pub fn new(cfg: MelConfig) -> Self {
        let window = hann(cfg.win_length);
        let n_freqs = cfg.n_fft / 2 + 1;
        let mel_fb = mel_filterbank(
            cfg.n_mels,
            n_freqs,
            cfg.sample_rate as f32,
            cfg.f_min,
            cfg.f_max,
            cfg.n_fft,
        );
        Self {
            cfg,
            window,
            mel_fb,
            leftover: Vec::new(),
        }
    }

    pub fn config(&self) -> &MelConfig {
        &self.cfg
    }

    pub fn reset(&mut self) {
        self.leftover.clear();
    }

    pub fn n_mels(&self) -> usize {
        self.cfg.n_mels
    }

    /// Push PCM; returns mel frames as `[n_frames * n_mels]` (frame-major).
    pub fn push(&mut self, pcm: &[f32]) -> Vec<f32> {
        self.leftover.extend_from_slice(pcm);
        let mut frames = Vec::new();
        let win = self.cfg.win_length;
        let hop = self.cfg.hop_length;
        let n_fft = self.cfg.n_fft;
        let n_mels = self.cfg.n_mels;
        let n_freqs = n_fft / 2 + 1;

        while self.leftover.len() >= win {
            let frame = &self.leftover[..win];
            let mut windowed = vec![0.0f32; n_fft];
            for i in 0..win {
                windowed[i] = frame[i] * self.window[i];
            }
            let power = rfft_power(&windowed);
            let mut mel = vec![0.0f32; n_mels];
            for m in 0..n_mels {
                let mut s = 0.0f32;
                let row = m * n_freqs;
                for f in 0..n_freqs {
                    s += self.mel_fb[row + f] * power[f];
                }
                let logv = (s.max(1e-10)).ln();
                mel[m] = logv / 10.0 + 2.0;
            }
            frames.extend_from_slice(&mel);
            self.leftover.drain(..hop);
        }
        frames
    }
}

fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = PI * 2.0 * i as f32 / n as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn mel_filterbank(
    n_mels: usize,
    n_freqs: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
    n_fft: usize,
) -> Vec<f32> {
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..n_mels + 2 {
        let m = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
        mel_points.push(mel_to_hz(m));
    }
    let mut bins = Vec::with_capacity(n_mels + 2);
    for hz in &mel_points {
        bins.push(((n_fft as f32 + 1.0) * hz / sample_rate).floor() as usize);
    }
    let mut fb = vec![0.0f32; n_mels * n_freqs];
    for m in 1..=n_mels {
        let left = bins[m - 1];
        let center = bins[m];
        let right = bins[m + 1];
        for f in left..center {
            if f < n_freqs && center > left {
                fb[(m - 1) * n_freqs + f] = (f - left) as f32 / (center - left) as f32;
            }
        }
        for f in center..right {
            if f < n_freqs && right > center {
                fb[(m - 1) * n_freqs + f] = (right - f) as f32 / (right - center) as f32;
            }
        }
    }
    fb
}

fn rfft_power(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let n_freqs = n / 2 + 1;
    let mut out = vec![0.0f32; n_freqs];
    for k in 0..n_freqs {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        let ang0 = -TAU * k as f32 / n as f32;
        for (t, &xv) in x.iter().enumerate() {
            let ang = ang0 * t as f32;
            re += xv * ang.cos();
            im += xv * ang.sin();
        }
        out[k] = re * re + im * im;
    }
    out
}
