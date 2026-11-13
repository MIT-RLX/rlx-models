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

//! The codec's inverse STFT, whole-clip ([`istft`]) or streaming ([`Istft`]).
//!
//! Each step of the codec output is `[2 * bins]`: log-magnitudes, then
//! phases. A step becomes `irfft` × periodic Hann, overlap-added at the hop
//! and divided by the summed squared window; `(win - hop) / 2` samples are
//! trimmed from both ends of the stream. Pushing the steps in pieces gives
//! the same samples as one call (`LiquidCodecModel.decode` with a
//! `LiquidCodecStreamingCache`).

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

use crate::config::CODEC_HOP;

/// Streaming inverse STFT, window length = `n_fft`.
pub struct Istft {
    bins: usize,
    hop: usize,
    window: Vec<f64>,
    ifft: Arc<dyn Fft<f64>>,
    /// Overlap-add sums of the last `win - hop` samples, still open.
    audio: Vec<f64>,
    env: Vec<f64>,
    /// Leading samples still to drop (the stream's first `trim`).
    skip: usize,
    started: bool,
}

impl Istft {
    /// An empty stream over steps of `n_fft / 2 + 1` frequency bins, `hop`
    /// samples apart.
    pub fn new(n_fft: usize, hop: usize) -> Self {
        assert!(hop > 0 && hop <= n_fft, "hop {hop} for n_fft {n_fft}");
        let window = (0..n_fft)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos())
            .collect();
        let tail = n_fft - hop;
        Self {
            bins: n_fft / 2 + 1,
            hop,
            window,
            ifft: FftPlanner::<f64>::new().plan_fft_inverse(n_fft),
            audio: vec![0.0; tail],
            env: vec![0.0; tail],
            skip: tail / 2,
            started: false,
        }
    }

    /// Add `spec` (`[steps, 2 * bins]`) and return the samples no later step
    /// changes: `hop` per step, less the stream's leading trim.
    pub fn push(&mut self, spec: &[f32]) -> Vec<f32> {
        let (bins, hop, win) = (self.bins, self.hop, self.window.len());
        let n_fft = win;
        let steps = spec.len() / (2 * bins);
        if steps == 0 {
            return Vec::new();
        }
        let tail = win - hop;
        let size = steps * hop + tail;
        self.audio.resize(size, 0.0);
        self.env.resize(size, 0.0);
        let mut buf = vec![Complex::new(0.0, 0.0); n_fft];
        for t in 0..steps {
            let row = &spec[t * 2 * bins..(t + 1) * 2 * bins];
            for k in 0..bins {
                let mag = (row[k] as f64).exp();
                let ph = row[bins + k] as f64;
                let mut z = Complex::from_polar(mag, ph);
                if k == 0 || k == bins - 1 {
                    // `irfft` reads only the real part of DC and Nyquist.
                    z.im = 0.0;
                }
                buf[k] = z;
                if k > 0 && k < bins - 1 {
                    buf[n_fft - k] = z.conj();
                }
            }
            self.ifft.process(&mut buf);
            for i in 0..win {
                let w = self.window[i];
                self.audio[t * hop + i] += buf[i].re / n_fft as f64 * w;
                self.env[t * hop + i] += w * w;
            }
        }
        let done = steps * hop;
        let skip = self.skip.min(done);
        self.skip -= skip;
        self.started = true;
        let out = (skip..done)
            .map(|i| (self.audio[i] / self.env[i]) as f32)
            .collect();
        self.audio.drain(..done);
        self.env.drain(..done);
        out
    }

    /// End the stream: the open samples, less the trailing trim. Empty when
    /// nothing was pushed.
    pub fn finish(self) -> Vec<f32> {
        if !self.started {
            return Vec::new();
        }
        let trim = (self.window.len() - self.hop) / 2;
        let keep = self.audio.len() - trim;
        (self.skip.min(keep)..keep)
            .map(|i| (self.audio[i] / self.env[i]) as f32)
            .collect()
    }
}

/// Inverse STFT of a whole clip's `[steps, 2 * bins]` codec output (hop
/// [`CODEC_HOP`]).
pub fn istft(spec: &[f32], bins: usize) -> Vec<f32> {
    let mut s = Istft::new((bins - 1) * 2, CODEC_HOP);
    let mut out = s.push(spec);
    out.extend(s.finish());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_match_one_call() {
        let bins = 641;
        let steps = 23;
        let spec: Vec<f32> = (0..steps * 2 * bins)
            .map(|i| ((i as f32) * 0.37).sin() * if i % (2 * bins) < bins { 0.5 } else { 3.0 })
            .collect();
        let whole = istft(&spec, bins);
        assert_eq!(whole.len(), steps * CODEC_HOP);
        let mut s = Istft::new(1280, CODEC_HOP);
        let mut out = Vec::new();
        let mut at = 0;
        for n in [1, 5, 2, 9, 6] {
            out.extend(s.push(&spec[at * 2 * bins..(at + n) * 2 * bins]));
            at += n;
        }
        out.extend(s.finish());
        assert_eq!(out.len(), whole.len());
        let err = out
            .iter()
            .zip(&whole)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-6, "max |Δ| {err}");
    }
}
