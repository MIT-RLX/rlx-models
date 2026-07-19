//! Spectral comparison vs a CPU (or clean-clone) reference.

use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

use crate::wav::{cosine, resample_linear};

const COMPARE_SR: u32 = 16_000;
const N_FFT: usize = 512;
const HOP: usize = 160;
const N_MELS: usize = 40;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralMetrics {
    pub stft_cosine: f64,
    pub logmel_cosine: f64,
    pub band_low_ratio: f64,
    pub band_mid_ratio: f64,
    pub band_high_ratio: f64,
}

pub fn spectral_vs_ref(pcm: &[f32], sr: u32, ref_pcm: &[f32], ref_sr: u32) -> SpectralMetrics {
    let a = resample_linear(pcm, sr, COMPARE_SR);
    let b = resample_linear(ref_pcm, ref_sr, COMPARE_SR);
    let n = a.len().min(b.len());
    if n < N_FFT {
        return SpectralMetrics {
            stft_cosine: 0.0,
            logmel_cosine: 0.0,
            band_low_ratio: 0.0,
            band_mid_ratio: 0.0,
            band_high_ratio: 0.0,
        };
    }
    let a = &a[..n];
    let b = &b[..n];
    let ma = stft_mag(a);
    let mb = stft_mag(b);
    let stft_cosine = cosine(&ma, &mb);
    let la = log_mel(&ma);
    let lb = log_mel(&mb);
    let logmel_cosine = cosine(&la, &lb);
    let (al, am, ah) = band_energy(&ma);
    let (bl, bm, bh) = band_energy(&mb);
    SpectralMetrics {
        stft_cosine,
        logmel_cosine,
        band_low_ratio: ratio(al, bl),
        band_mid_ratio: ratio(am, bm),
        band_high_ratio: ratio(ah, bh),
    }
}

fn ratio(a: f64, b: f64) -> f64 {
    if b.abs() < 1e-12 {
        return 0.0;
    }
    a / b
}

fn stft_mag(pcm: &[f32]) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut out = Vec::new();
    let mut i = 0;
    while i + N_FFT <= pcm.len() {
        let mut buf: Vec<Complex<f32>> = (0..N_FFT)
            .map(|k| {
                let w = 0.5
                    - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / (N_FFT as f32 - 1.0)).cos();
                Complex {
                    re: pcm[i + k] * w,
                    im: 0.0,
                }
            })
            .collect();
        fft.process(&mut buf);
        for c in buf.iter().take(N_FFT / 2 + 1) {
            out.push(c.norm());
        }
        i += HOP;
    }
    out
}

fn log_mel(stft: &[f32]) -> Vec<f32> {
    let bins = N_FFT / 2 + 1;
    if stft.is_empty() || bins == 0 {
        return Vec::new();
    }
    let frames = stft.len() / bins;
    let mut out = Vec::with_capacity(frames * N_MELS);
    for f in 0..frames {
        let frame = &stft[f * bins..(f + 1) * bins];
        for m in 0..N_MELS {
            let start = m * bins / N_MELS;
            let end = ((m + 1) * bins / N_MELS).max(start + 1);
            let e: f32 =
                frame[start..end.min(frame.len())].iter().sum::<f32>() / (end - start) as f32;
            out.push((e + 1e-10).ln());
        }
    }
    out
}

fn band_energy(stft: &[f32]) -> (f64, f64, f64) {
    let bins = N_FFT / 2 + 1;
    if stft.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let frames = stft.len() / bins;
    let mut low = 0.0f64;
    let mut mid = 0.0f64;
    let mut high = 0.0f64;
    for f in 0..frames {
        let frame = &stft[f * bins..(f + 1) * bins];
        for (i, &v) in frame.iter().enumerate() {
            let e = (v as f64).powi(2);
            let third = bins / 3;
            if i < third {
                low += e;
            } else if i < 2 * third {
                mid += e;
            } else {
                high += e;
            }
        }
    }
    (low, mid, high)
}
