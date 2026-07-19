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

//! DSP for OpenVoice: the magnitude spectrogram (`spectrogram_torch`) + resample.

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

pub const N_FFT: usize = 1024;
pub const HOP: usize = 256;
pub const N_FREQ: usize = N_FFT / 2 + 1; // 513
pub const SR: u32 = 22050;

/// Periodic Hann window (`torch.hann_window(n)`, periodic=True).
fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / n as f32;
            x.sin().powi(2)
        })
        .collect()
}

/// Reflect-pad `x` by `pad` samples on each side (`mode="reflect"`).
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        out.push(x[pad - i]);
    }
    out.extend_from_slice(x);
    for i in 0..pad {
        out.push(x[n - 2 - i]);
    }
    out
}

pub struct Spectrogram {
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl Spectrogram {
    pub fn new() -> Self {
        Self {
            window: hann_periodic(N_FFT),
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        }
    }

    /// `spectrogram_torch(y, 1024, _, 256, 1024, center=False)` → magnitude
    /// `[N_FREQ, T]` (row-major, freq-major). Pads by `(n_fft-hop)/2 = 384`.
    pub fn magnitude(&self, wav: &[f32]) -> (Vec<f32>, usize) {
        let pad = (N_FFT - HOP) / 2; // 384
        let padded = reflect_pad(wav, pad);
        let t = if padded.len() >= N_FFT {
            (padded.len() - N_FFT) / HOP + 1
        } else {
            0
        };
        let mut spec = vec![0.0f32; N_FREQ * t];
        let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
        for frame in 0..t {
            let start = frame * HOP;
            for i in 0..N_FFT {
                buf[i] = Complex32::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for f in 0..N_FREQ {
                let re = buf[f].re;
                let im = buf[f].im;
                spec[f * t + frame] = (re * re + im * im + 1e-6).sqrt();
            }
        }
        (spec, t)
    }
}

/// Linear resample `x` from `from` Hz to `to` Hz.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as u64 * to as u64 / from as u64).max(1) as usize;
    (0..n)
        .map(|i| {
            let s = i as f64 * from as f64 / to as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = x[idx.min(x.len() - 1)];
            let b = x[(idx + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}
