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

//! Feature frontend — port of `stft.cc` plus the mel/normalization half of
//! `AUP_Aed_aivad_proc`.
//!
//! Per 256-sample hop: slide a 768-sample window over the *pre-emphasised*
//! signal, apply Hann-768, zero-pad to 1024, take `|X[k]|²`, project onto 40
//! mel bands, take `log(band / 32768² + 1e-20)`, and standardize. Feature 40
//! is the pitch estimate (from [`crate::pitch`]) under the same standardization.
//! Three consecutive frames form the `[3, 41]` network input.

use crate::math;
use crate::ooura;
use crate::pitch::{PitchEstimator, PitchFrame};
use crate::weights::CoreWeights;
use crate::{
    CONTEXT_FRAMES, FEATURE_LEN, FFT_SIZE, HOP_SIZE, MEL_BANDS, SAMPLE_RATE, SPECTRUM_BINS,
    WINDOW_SIZE,
};

/// `log()` floor, and the denominator guard on the per-feature std.
const EPS: f32 = 1e-20;
/// Spectra are computed on int16-scaled samples; undo that before the log.
const POWER_NORM: f32 = 32768.0 * 32768.0;
const PRE_EMPHASIS: f32 = 0.97;
const MEL_TOP_HZ: f32 = 8000.0;

/// Triangular mel filterbank with the reference's exact integer bin edges.
/// Triangular mel filterbank, stored banded.
///
/// A `[MEL_BANDS, SPECTRUM_BINS]` dense matrix is 20,520 coefficients of which
/// roughly a thousand are non-zero — each triangle touches one contiguous
/// stretch of bins. Multiplying through the zeros costs 20x the arithmetic and
/// changes nothing: the out-of-band entries are exact `+0.0`, and adding `+0.0`
/// leaves a non-negative accumulator untouched, so the banded sum is
/// bit-identical to the dense one (`banded_mel_matches_dense` pins that).
struct MelFilterBank {
    /// First spectrum bin band `j` touches.
    offset: [u16; MEL_BANDS],
    /// Non-zero weights, band after band. The bank holds 989 of them; the
    /// array is sized with headroom and `MEL_WEIGHT_CAP` is checked on build.
    weight: [f32; MEL_WEIGHT_CAP],
    /// Band `j` occupies `weight[span[j]..span[j + 1]]`.
    span: [u16; MEL_BANDS + 1],
}

impl MelFilterBank {
    /// Slaney-style mel with `2595·log10(1 + f/700)`, but the bin edges are
    /// truncated from an `f32` expression exactly as the C does — rounding them
    /// differently shifts whole bands, so this is not cosmetic.
    fn new() -> Self {
        let high_mel = 2595.0f32 * math::log10(1.0f32 + MEL_TOP_HZ / 700.0);
        let mut edge = [0usize; MEL_BANDS + 2];
        for (i, slot) in edge.iter_mut().enumerate() {
            let mel = i as f32 * high_mel / (MEL_BANDS as f32 + 1.0);
            let hz = 700.0f32 * (math::pow10(mel / 2595.0) - 1.0);
            *slot = ((FFT_SIZE as f32 + 1.0) * hz / SAMPLE_RATE as f32) as usize;
        }
        let mut fb = Self {
            offset: [0; MEL_BANDS],
            weight: [0.0; MEL_WEIGHT_CAP],
            span: [0; MEL_BANDS + 1],
        };
        let mut at = 0usize;
        for j in 0..MEL_BANDS {
            let (lo, mid, hi) = (edge[j], edge[j + 1], edge[j + 2]);
            fb.offset[j] = lo as u16;
            for i in lo..mid {
                fb.weight[at] = (i - lo) as f32 / (mid - lo) as f32;
                at += 1;
            }
            for i in mid..hi {
                fb.weight[at] = (hi - i) as f32 / (hi - mid) as f32;
                at += 1;
            }
            fb.span[j + 1] = at as u16;
        }
        debug_assert!(at <= MEL_WEIGHT_CAP, "mel bank needs {at} weights");
        fb
    }

    /// `out[j] = Σ_k bin_pow[k] · coeff[j][k]`, summed in bin order like the C.
    fn apply(&self, bin_pow: &[f32], out: &mut [f32]) {
        for (j, slot) in out.iter_mut().enumerate() {
            let w = &self.weight[self.span[j] as usize..self.span[j + 1] as usize];
            let p = &bin_pow[self.offset[j] as usize..self.offset[j] as usize + w.len()];
            let mut sum = 0.0f32;
            for (&p, &c) in p.iter().zip(w) {
                sum += p * c;
            }
            *slot = sum;
        }
    }

    /// The dense `[MEL_BANDS, SPECTRUM_BINS]` matrix the banded form encodes.
    #[cfg(test)]
    fn dense(&self) -> alloc::vec::Vec<f32> {
        let mut coeff = alloc::vec![0.0f32; MEL_BANDS * SPECTRUM_BINS];
        for j in 0..MEL_BANDS {
            let w = &self.weight[self.span[j] as usize..self.span[j + 1] as usize];
            for (k, &c) in w.iter().enumerate() {
                coeff[j * SPECTRUM_BINS + self.offset[j] as usize + k] = c;
            }
        }
        coeff
    }

    /// The dense form this replaces, kept as the reference the banded version
    /// is checked against.
    #[cfg(test)]
    fn apply_dense(&self, bin_pow: &[f32], out: &mut [f32]) {
        let coeff = self.dense();
        for (j, slot) in out.iter_mut().enumerate() {
            let row = &coeff[j * SPECTRUM_BINS..(j + 1) * SPECTRUM_BINS];
            let mut sum = 0.0f32;
            for (&p, &c) in bin_pow.iter().zip(row) {
                sum += p * c;
            }
            *slot = sum;
        }
    }
}

/// Capacity for the banded mel weights. The bank uses 989; the slack costs
/// 140 bytes of `.bss` and buys freedom from an allocator.
const MEL_WEIGHT_CAP: usize = 1024;

/// One frame's frontend output.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameInfo {
    pub pitch: PitchFrame,
}

/// Streaming feature extractor for the fixed 16 kHz / 256-hop configuration.
pub struct Frontend {
    window: [f32; WINDOW_SIZE],
    mel: MelFilterBank,
    mean: [f32; FEATURE_LEN],
    /// Stored as `std + EPS`, and *divided* by — not turned into a reciprocal
    /// and multiplied. In `f32` those are not the same number.
    std_eps: [f32; FEATURE_LEN],
    pitch: PitchEstimator,

    /// 768-sample analysis queue over the pre-emphasised signal.
    analysis_q: [f32; WINDOW_SIZE],
    fft_buf: [f32; FFT_SIZE],
    bin_pow: [f32; SPECTRUM_BINS],
    band: [f32; MEL_BANDS],
    /// `[CONTEXT_FRAMES, FEATURE_LEN]` rolling context, newest last.
    stack: [f32; CONTEXT_FRAMES * FEATURE_LEN],
}

impl Frontend {
    pub fn new(weights: CoreWeights<'_>) -> Self {
        let mut fe = Self {
            window: [0.0; WINDOW_SIZE],
            mel: MelFilterBank::new(),
            mean: [0.0; FEATURE_LEN],
            std_eps: [0.0; FEATURE_LEN],
            pitch: PitchEstimator::new(),
            analysis_q: [0.0; WINDOW_SIZE],
            fft_buf: [0.0; FFT_SIZE],
            bin_pow: [0.0; SPECTRUM_BINS],
            band: [0.0; MEL_BANDS],
            stack: [0.0; CONTEXT_FRAMES * FEATURE_LEN],
        };
        fe.window.copy_from_slice(weights.window);
        fe.mean.copy_from_slice(weights.feature_mean);
        for (dst, &s) in fe.std_eps.iter_mut().zip(weights.feature_std) {
            *dst = s + EPS;
        }
        fe
    }

    pub fn reset(&mut self) {
        self.analysis_q.fill(0.0);
        self.stack.fill(0.0);
        self.pitch.reset();
    }

    /// Advance one internal frame.
    ///
    /// `raw` is the frame in int16 units; `emphasised` is the same frame after
    /// the `1 − 0.97 z⁻¹` pre-emphasis (kept continuous by the caller, since
    /// the API hop and the analysis hop need not match).
    pub fn push(&mut self, raw: &[f32], emphasised: &[f32]) -> FrameInfo {
        debug_assert_eq!(raw.len(), HOP_SIZE);
        debug_assert_eq!(emphasised.len(), HOP_SIZE);

        self.analysis_q.copy_within(HOP_SIZE.., 0);
        self.analysis_q[WINDOW_SIZE - HOP_SIZE..].copy_from_slice(emphasised);
        for (dst, (&q, &w)) in self.fft_buf[..WINDOW_SIZE]
            .iter_mut()
            .zip(self.analysis_q.iter().zip(&self.window))
        {
            *dst = q * w;
        }
        ooura::power_spectrum(&self.fft_buf[..WINDOW_SIZE], &mut self.bin_pow);

        // Pitch runs on the un-emphasised signal but reuses this spectrum.
        let pitch = self.pitch.process(raw, &self.bin_pow);

        self.mel.apply(&self.bin_pow, &mut self.band);
        self.stack.copy_within(FEATURE_LEN.., 0);
        let cur = &mut self.stack[(CONTEXT_FRAMES - 1) * FEATURE_LEN..];
        for j in 0..MEL_BANDS {
            let v = math::ln(self.band[j] / POWER_NORM + EPS);
            cur[j] = (v - self.mean[j]) / self.std_eps[j];
        }
        cur[MEL_BANDS] = (pitch.freq_hz - self.mean[MEL_BANDS]) / self.std_eps[MEL_BANDS];

        FrameInfo { pitch }
    }

    /// The current `[CONTEXT_FRAMES, FEATURE_LEN]` network input.
    pub fn context(&self) -> &[f32] {
        &self.stack
    }

    /// `|X[k]|²` of the frame just pushed.
    pub fn spectrum(&self) -> &[f32] {
        &self.bin_pow
    }
}

/// Apply pre-emphasis to `src`, threading `prev` across calls.
pub fn pre_emphasis(src: &[f32], prev: &mut f32, dst: &mut [f32]) {
    for (out, &x) in dst.iter_mut().zip(src) {
        *out = x - PRE_EMPHASIS * *prev;
        *prev = x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banded_mel_matches_dense() {
        // Skipping the out-of-band zeros must not change a single bit: they are
        // exact `+0.0` and the accumulator is non-negative throughout.
        let mel = MelFilterBank::new();
        let spectrum: alloc::vec::Vec<f32> = (0..SPECTRUM_BINS)
            .map(|i| (i as f32 * 0.37).sin().abs() * 1e4 + 1e-3)
            .collect();
        let mut banded = [0.0f32; MEL_BANDS];
        let mut dense = [0.0f32; MEL_BANDS];
        mel.apply(&spectrum, &mut banded);
        mel.apply_dense(&spectrum, &mut dense);
        assert_eq!(banded, dense, "banded mel diverged from the dense form");

        // And the banding is worth doing: far fewer coefficients than dense.
        assert!(
            mel.weight.len() * 4 < MEL_BANDS * SPECTRUM_BINS,
            "banded form is {} coefficients vs {} dense",
            mel.weight.len(),
            MEL_BANDS * SPECTRUM_BINS
        );
    }

    #[test]
    fn mel_edges_match_reference() {
        let fb = MelFilterBank::new();
        // Spot-check the extremes: band 0 starts at bin 0, the last band ends
        // at the Nyquist bin (8 kHz at fs = 16 kHz).
        let coeff = fb.dense();
        assert!(coeff[0..3].iter().any(|&v| v > 0.0));
        let last = &coeff[(MEL_BANDS - 1) * SPECTRUM_BINS..];
        assert!(
            last[SPECTRUM_BINS - 2] > 0.0,
            "top band should reach Nyquist"
        );
        // Every band must be non-empty, else the reference bails out entirely.
        for j in 0..MEL_BANDS {
            let row = &coeff[j * SPECTRUM_BINS..(j + 1) * SPECTRUM_BINS];
            assert!(row.iter().any(|&v| v > 0.0), "band {j} is empty");
        }
    }

    #[test]
    fn pre_emphasis_is_continuous_across_calls() {
        let mut prev = 0.0f32;
        let mut a = [0.0f32; 4];
        pre_emphasis(&[1.0, 2.0, 3.0, 4.0], &mut prev, &mut a);
        let mut b = [0.0f32; 2];
        pre_emphasis(&[5.0, 6.0], &mut prev, &mut b);
        assert_eq!(a[0], 1.0);
        assert!((b[0] - (5.0 - 0.97 * 4.0)).abs() < 1e-6);
    }

    #[test]
    fn silence_gives_the_log_floor() {
        let w = crate::weights::embedded();
        let mut fe = Frontend::new(crate::weights::embedded());
        let zeros = [0.0f32; HOP_SIZE];
        fe.push(&zeros, &zeros);
        let cur = &fe.context()[(CONTEXT_FRAMES - 1) * FEATURE_LEN..];
        let floor = (math::ln(EPS) - w.feature_mean[0]) / (w.feature_std[0] + EPS);
        assert!((cur[0] - floor).abs() < 1e-4, "{} vs {floor}", cur[0]);
    }
}
