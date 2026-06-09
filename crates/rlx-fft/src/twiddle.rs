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

//! Twiddle-factor initialization for butterfly stages.

use crate::config::{FftLearnConfig, TransformDir};
use std::f32::consts::TAU;

/// Which learned butterfly a twiddle belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwiddleSet {
    /// Shared forward twiddles (`twiddle.s*.b*.*`).
    Shared,
    /// Encoder / forward FFT twiddles (`encoder.twiddle.s*.b*.*`).
    Encoder,
    /// Decoder / inverse FFT twiddles (`decoder.twiddle.s*.b*.*`).
    Decoder,
}

/// Flat twiddle buffer: for each stage `s` and butterfly `b`, store `(re, im)`.
/// Length = `num_stages * (n_fft/2) * 2`.
pub fn exact_twiddles(cfg: &FftLearnConfig) -> Vec<f32> {
    let half = cfg.n_fft / 2;
    let stages = cfg.num_stages();
    let mut out = vec![0f32; stages * half * 2];
    for s in 0..stages {
        let stride = 1usize << s;
        for b in 0..half {
            let k = b % stride;
            let m = (2 * stride) as f32;
            let exp = -TAU * k as f32 / m;
            let base = (s * half + b) * 2;
            out[base] = exp.cos();
            out[base + 1] = exp.sin();
        }
    }
    out
}

pub fn exact_twiddles_dir(cfg: &FftLearnConfig, dir: TransformDir) -> Vec<f32> {
    let _ = dir;
    exact_twiddles(cfg)
}

pub fn twiddle_name(stage: usize, butterfly: usize, part: &str) -> String {
    twiddle_name_set(TwiddleSet::Shared, stage, butterfly, part)
}

pub fn twiddle_name_set(set: TwiddleSet, stage: usize, butterfly: usize, part: &str) -> String {
    let prefix = match set {
        TwiddleSet::Shared => "twiddle",
        TwiddleSet::Encoder => "encoder.twiddle",
        TwiddleSet::Decoder => "decoder.twiddle",
    };
    format!("{prefix}.s{stage}.b{butterfly}.{part}")
}

pub fn twiddle_name_dir(dir: TransformDir, stage: usize, butterfly: usize, part: &str) -> String {
    let _ = dir;
    twiddle_name(stage, butterfly, part)
}

pub fn twiddle_index(stage: usize, butterfly: usize, half: usize, part: usize) -> usize {
    (stage * half + butterfly) * 2 + part
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twiddle_magnitude_is_one() {
        let cfg = FftLearnConfig::new(64, 1).unwrap();
        let tw = exact_twiddles(&cfg);
        for chunk in tw.chunks(2) {
            let mag = (chunk[0] * chunk[0] + chunk[1] * chunk[1]).sqrt();
            assert!((mag - 1.0).abs() < 1e-6, "mag={mag}");
        }
    }
}
