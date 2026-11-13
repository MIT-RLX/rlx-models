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

//! The encoder's STFT features (`LiquidCodecSTFT`), on the host.
//!
//! Reflect-pad by `(win - hop) / 2` on both sides, frame at the hop with a
//! periodic Hann window (`center=False`), and for each one-sided bin emit
//! `log1p(|S|)`, `Re S / max(|S|, 1e-5)` and `Im S / max(|S|, 1e-5)`: the
//! magnitude and the unit phase vector. Rows are frames, `[T, 3 * bins]`
//! with the three groups in that order.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::config::LiquidCodecConfig;

/// The window and FFT plan, reused across frames.
pub struct Stft {
    n_fft: usize,
    hop: usize,
    bins: usize,
    window: Vec<f64>,
    fft: Arc<dyn Fft<f64>>,
    buf: Vec<Complex<f64>>,
}

impl Stft {
    pub fn new(cfg: &LiquidCodecConfig) -> Self {
        let n_fft = cfg.stft_n_fft;
        Self {
            n_fft,
            hop: cfg.stft_hop_length,
            bins: cfg.bins(),
            window: (0..n_fft)
                .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos())
                .collect(),
            fft: FftPlanner::<f64>::new().plan_fft_forward(n_fft),
            buf: vec![Complex::new(0.0, 0.0); n_fft],
        }
    }

    /// Reflect padding on each side, `(n_fft - hop) / 2`.
    pub fn pad(&self) -> usize {
        (self.n_fft - self.hop) / 2
    }

    /// One past the last (unpadded) sample frame `t` reads.
    pub fn end_of(&self, t: usize) -> usize {
        t * self.hop + self.n_fft - self.pad()
    }

    /// Features of frame `t` into `row` (`3 * bins`); `sample(i)` is the
    /// padded signal at unpadded index `i` (negative = left padding).
    pub fn frame(&mut self, t: usize, sample: impl Fn(isize) -> f32, row: &mut [f32]) {
        let bins = self.bins;
        let start = (t * self.hop) as isize - self.pad() as isize;
        for (i, b) in self.buf.iter_mut().enumerate() {
            *b = Complex::new(sample(start + i as isize) as f64 * self.window[i], 0.0);
        }
        self.fft.process(&mut self.buf);
        for k in 0..bins {
            let z = self.buf[k];
            let mag = z.norm();
            let den = mag.max(1e-5);
            row[k] = mag.ln_1p() as f32;
            row[bins + k] = (z.re / den) as f32;
            row[2 * bins + k] = (z.im / den) as f32;
        }
    }
}

/// `audio[i]` with torch's reflect padding at both ends.
pub fn reflect(audio: &[f32], i: isize) -> f32 {
    let n = audio.len() as isize;
    let i = if i < 0 {
        -i
    } else if i >= n {
        2 * (n - 1) - i
    } else {
        i
    };
    audio[i as usize]
}

/// Features of `audio` (whole frames: `len % frame_size == 0`): `audio.len()
/// / hop` rows of `3 * bins`.
pub fn features(audio: &[f32], cfg: &LiquidCodecConfig) -> Vec<f32> {
    let mut stft = Stft::new(cfg);
    let pad = stft.pad();
    let n = audio.len();
    assert!(n > pad, "{n} samples: too short to reflect-pad by {pad}");
    let steps = (n + 2 * pad - cfg.stft_n_fft) / cfg.stft_hop_length + 1;
    let row = 3 * cfg.bins();
    let mut out = vec![0f32; steps * row];
    for (t, r) in out.chunks_exact_mut(row).enumerate() {
        stft.frame(t, |i| reflect(audio, i), r);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_row_per_hop() {
        let cfg = LiquidCodecConfig::default();
        let audio: Vec<f32> = (0..2 * cfg.frame_size())
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let f = features(&audio, &cfg);
        assert_eq!(f.len(), audio.len() / cfg.stft_hop_length * 3 * cfg.bins());
        // The phase part is a unit vector wherever the magnitude is not tiny.
        let b = cfg.bins();
        let (re, im) = (f[b + 3], f[2 * b + 3]);
        assert!(((re * re + im * im) - 1.0).abs() < 1e-5);
    }
}
