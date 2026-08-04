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

//! # rlx-roformer-sep
//!
//! **BS-RoFormer** and **Mel-Band-RoFormer** music source separation on RLX. Both
//! STFT the mixture, **split the frequency bins into bands**, run a RoFormer
//! (RoPE transformer) alternating attention across time and across bands, estimate
//! a **complex mask** per band, apply it, and ISTFT back to per-stem waveforms.
//! The two variants differ only in the band-split scheme (fixed widths vs
//! mel-spaced).
//!
//! Native Rust. The STFT/ISTFT reuses [`rlx-fft`](https://docs.rs/rlx-fft); this
//! crate contributes the checkpoint-free, unit-tested DSP glue: the config, the two
//! band-split schemes ([`fixed_band_ranges`], [`mel_band_ranges`]), and complex
//! mask application ([`apply_complex_mask`]). The RoFormer graph + mask estimation
//! is the next step.

use anyhow::{Result, ensure};

/// The band-split scheme.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BandSplit {
    /// BS-RoFormer: explicit per-band bin widths (low bands narrow, high bands wide).
    Fixed(Vec<usize>),
    /// Mel-Band-RoFormer: `n` mel-spaced bands over the frequency axis.
    Mel(usize),
}

/// Separation model config.
#[derive(Debug, Clone, PartialEq)]
pub struct RoformerSepConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub band_split: BandSplit,
    // RoFormer.
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    /// Number of output stems (e.g. vocals/drums/bass/other = 4).
    pub num_stems: usize,
}

impl Default for RoformerSepConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            n_fft: 2048,
            hop_length: 512,
            win_length: 2048,
            band_split: BandSplit::Mel(62),
            dim: 384,
            depth: 12,
            heads: 8,
            num_stems: 4,
        }
    }
}

impl RoformerSepConfig {
    /// Number of one-sided STFT frequency bins (`n_fft/2 + 1`).
    pub fn num_freqs(&self) -> usize {
        self.n_fft / 2 + 1
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.n_fft > 0, "n_fft must be > 0");
        ensure!(self.hop_length > 0, "hop_length must be > 0");
        ensure!(self.num_stems > 0, "num_stems must be > 0");
        Ok(())
    }

    /// The `(start, end)` bin ranges for this config's band-split scheme, covering
    /// all of `[0, num_freqs)`.
    pub fn band_ranges(&self) -> Result<Vec<(usize, usize)>> {
        match &self.band_split {
            BandSplit::Fixed(widths) => fixed_band_ranges(self.num_freqs(), widths),
            BandSplit::Mel(n) => {
                mel_band_ranges(self.num_freqs(), *n, self.sample_rate, self.n_fft)
            }
        }
    }
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Partition `num_freqs` bins into contiguous bands of the given `widths`. The
/// final band absorbs any remaining bins so the ranges always cover `[0, num_freqs)`.
pub fn fixed_band_ranges(num_freqs: usize, widths: &[usize]) -> Result<Vec<(usize, usize)>> {
    ensure!(num_freqs > 0, "num_freqs must be > 0");
    ensure!(!widths.is_empty(), "need at least one band width");
    let sum: usize = widths.iter().sum();
    ensure!(
        sum <= num_freqs,
        "band widths sum {sum} exceed num_freqs {num_freqs}"
    );
    let mut ranges = Vec::with_capacity(widths.len());
    let mut cursor = 0;
    for (i, &w) in widths.iter().enumerate() {
        let end = if i + 1 == widths.len() {
            num_freqs // last band absorbs the remainder
        } else {
            cursor + w
        };
        ranges.push((cursor, end));
        cursor = end;
    }
    Ok(ranges)
}

/// Partition `num_freqs` bins into `n_bands` mel-spaced contiguous bands covering
/// `[0, num_freqs)`.
pub fn mel_band_ranges(
    num_freqs: usize,
    n_bands: usize,
    sample_rate: usize,
    n_fft: usize,
) -> Result<Vec<(usize, usize)>> {
    ensure!(num_freqs > 0, "num_freqs must be > 0");
    ensure!(n_bands > 0, "n_bands must be > 0");
    ensure!(
        n_bands <= num_freqs,
        "n_bands {n_bands} exceed num_freqs {num_freqs}"
    );

    let nyquist = sample_rate as f32 / 2.0;
    let mel_max = hz_to_mel(nyquist);
    // n_bands + 1 mel-spaced edges → hz → bin index.
    let mut edges: Vec<usize> = (0..=n_bands)
        .map(|i| {
            let mel = mel_max * i as f32 / n_bands as f32;
            let hz = mel_to_hz(mel);
            let bin = (hz * n_fft as f32 / sample_rate as f32).round() as i64;
            bin.clamp(0, num_freqs as i64) as usize
        })
        .collect();
    // Force endpoints and enforce a strictly non-decreasing, coverage-complete set.
    edges[0] = 0;
    edges[n_bands] = num_freqs;
    for i in 1..=n_bands {
        if edges[i] < edges[i - 1] {
            edges[i] = edges[i - 1];
        }
    }
    Ok((0..n_bands).map(|i| (edges[i], edges[i + 1])).collect())
}

/// Apply a complex mask to a complex spectrogram, element-wise:
/// `out = spec * mask` where each is `(re, im)`. All four slices share a length.
pub fn apply_complex_mask(
    spec_re: &[f32],
    spec_im: &[f32],
    mask_re: &[f32],
    mask_im: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let n = spec_re.len();
    ensure!(
        spec_im.len() == n && mask_re.len() == n && mask_im.len() == n,
        "complex mask operands must share a length"
    );
    let mut out_re = vec![0.0f32; n];
    let mut out_im = vec![0.0f32; n];
    for i in 0..n {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        out_re[i] = spec_re[i] * mask_re[i] - spec_im[i] * mask_im[i];
        out_im[i] = spec_re[i] * mask_im[i] + spec_im[i] * mask_re[i];
    }
    Ok((out_re, out_im))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_and_num_freqs() {
        let c = RoformerSepConfig::default();
        assert_eq!(c.num_freqs(), 2048 / 2 + 1);
        c.validate().unwrap();
    }

    #[test]
    fn fixed_bands_cover_all_bins_contiguously() {
        let r = fixed_band_ranges(10, &[2, 3, 5]).unwrap();
        assert_eq!(r, vec![(0, 2), (2, 5), (5, 10)]);
        // last band absorbs the remainder
        let r2 = fixed_band_ranges(10, &[2, 3]).unwrap();
        assert_eq!(r2, vec![(0, 2), (2, 10)]);
        // over-wide is rejected
        assert!(fixed_band_ranges(4, &[3, 3]).is_err());
    }

    #[test]
    fn mel_bands_are_contiguous_and_cover_range() {
        let n_freqs = 513; // n_fft = 1024
        let ranges = mel_band_ranges(n_freqs, 8, 44_100, 1024).unwrap();
        assert_eq!(ranges.len(), 8);
        assert_eq!(ranges[0].0, 0);
        assert_eq!(ranges[7].1, n_freqs);
        // contiguous + non-decreasing
        for w in ranges.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
        for (s, e) in &ranges {
            assert!(s <= e);
        }
    }

    #[test]
    fn mel_scale_is_monotonic() {
        assert!(hz_to_mel(1000.0) > hz_to_mel(100.0));
        // round-trip
        let hz = 2000.0;
        assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 1.0);
    }

    #[test]
    fn complex_mask_multiplies() {
        // (1+2i) * (3+4i) = (3-8) + (4+6)i = -5 + 10i
        let (re, im) = apply_complex_mask(&[1.0], &[2.0], &[3.0], &[4.0]).unwrap();
        assert_eq!(re, vec![-5.0]);
        assert_eq!(im, vec![10.0]);
        // identity mask (1+0i) is a no-op
        let (re2, im2) =
            apply_complex_mask(&[7.0, 8.0], &[9.0, 1.0], &[1.0, 1.0], &[0.0, 0.0]).unwrap();
        assert_eq!(re2, vec![7.0, 8.0]);
        assert_eq!(im2, vec![9.0, 1.0]);
    }
}
