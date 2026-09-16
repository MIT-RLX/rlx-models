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

//! Pitch estimator — port of TEN-VAD's `pitch_est.cc`.
//!
//! The estimate is feature 40 of the 41 the network consumes, so it has to
//! track the reference closely. Pipeline per 256-sample frame:
//!
//! 1. 18-band energies off the power spectrum → log → DCT → cepstrum.
//! 2. Cepstrum → band gains → autocorrelation → order-16 LPC (Levinson).
//! 3. Inverse-filter the (un-pre-emphasised) signal, lowpass, decimate to 4 kHz.
//! 4. Two half-frame normalized cross-correlations over periods 0..64.
//! 5. Viterbi max-path across the last 3 frames (6 half-frames), then a
//!    weighted linear regression of the backtracked contour.
//!
//! Steps 1–2 and 4–5 are derived from LPCNet's `compute_frame_features()` and
//! `process_superframe()` (Mozilla, BSD-2-Clause / BSD-3-Clause) by way of the
//! TEN-VAD C source; the `SIDXT` transition window and the sharpening pass
//! below follow TEN-VAD rather than upstream LPCNet.

use crate::biquad::{Biquad, LOWPASS_4KHZ};
use crate::math;
#[cfg_attr(feature = "fast-pitch", allow(unused_imports))]
use crate::ooura;

/// Internal analysis hop; `ten_vad.cc` pins this regardless of the API hop.
pub const HOP: usize = 256;
/// Analysis window length (used only as the LPC noise-floor bias here).
pub const WINDOW: usize = 768;
/// Spectrum bins the estimator is fed (`fft/2 + 1`).
pub const NBINS: usize = 513;
const FFT_SIZE: usize = 1024;

const NB_BANDS: usize = 18;
const LPC_ORDER: usize = 16;
// The circular LPC history masks instead of dividing.
const _: () = assert!(LPC_ORDER.is_power_of_two());
const PROC_FS: f32 = 4000.0;
/// 16 kHz → 4 kHz.
const RESAMPLE: usize = 4;
const MIN_PERIOD: usize = 32 / RESAMPLE;
const MAX_PERIOD: usize = 256 / RESAMPLE;
const DIF_PERIOD: usize = MAX_PERIOD - MIN_PERIOD;
/// Correlation window: half a decimated hop.
const CORR_HALF: usize = HOP / (RESAMPLE * 2);
const EXC_LEN: usize = MAX_PERIOD + HOP / RESAMPLE + 1;
const XCORR_TRAINING_OFFSET: usize = 80;
const INQ_LEN: usize = if XCORR_TRAINING_OFFSET > HOP {
    XCORR_TRAINING_OFFSET + HOP
} else {
    HOP + HOP
};
/// 40 ms of context → 3 frames → 6 half-frames.
const N_FEAT: usize = 3;
const N_SUB: usize = N_FEAT * 2;
/// Per-step transition penalty of the max-path search.
const MAXPATH_W: f32 = 0.02;
const VOICED_THR: f32 = 0.4;
// `AUP_PE_PI` upstream — a truncated π, and it feeds the DCT table the
// cepstrum and LPC ride on, so keep the reference's exact constant.
#[allow(clippy::approx_constant)]
const PI: f32 = 3.141_592_6;

/// Band edges in units of a 80-point FFT, widened to the real FFT size.
const BAND_START: [usize; NB_BANDS] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40,
];
const BAND_LPC_COMP: [f32; NB_BANDS] = [
    0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.666_667, 0.5, 0.5, 0.5, 0.333_333, 0.25, 0.25, 0.2,
    0.166_667, 0.173_913,
];

/// Frame result: pitch in Hz (0 when unvoiced) plus the voicing decision.
#[derive(Debug, Clone, Copy, Default)]
pub struct PitchFrame {
    pub freq_hz: f32,
    pub voiced: bool,
}

/// Band-splitting geometry, shared by the analysis and synthesis directions.
/// `Σ size[i]`, the number of (band, tap) pairs — 515 for the current band
/// edges, which is *more* than `NBINS` because adjacent triangles overlap.
/// Asserted in `Bands::new`, so a change to the edges fails loudly rather than
/// truncating the table.
const MAX_TAPS: usize = 515;

struct Bands {
    size: [usize; NB_BANDS - 1],
    offset: [usize; NB_BANDS - 1],
    /// `j as f32 / size[i] as f32` for every (band, tap), in iteration order.
    ///
    /// A soft-float division is ~200 instructions on `rv32imc` and `energy`
    /// did 513 of them per frame, for a quotient that depends only on the band
    /// geometry. Precomputing produces the *same* f32 values, so this is
    /// bit-exact — the frontend parity test still holds.
    frac: [f32; MAX_TAPS],
}

impl Bands {
    fn new() -> Self {
        // `(float)fftSz / 80` in the C, and both roundf() calls see f32 inputs.
        let rate = FFT_SIZE as f32 / 80.0;
        let mut size = [0usize; NB_BANDS - 1];
        let mut offset = [0usize; NB_BANDS - 1];
        for i in 0..NB_BANDS - 1 {
            size[i] = math::round((BAND_START[i + 1] - BAND_START[i]) as f32 * rate) as usize;
            offset[i] = math::round(BAND_START[i] as f32 * rate) as usize;
        }
        let mut frac = [0.0f32; MAX_TAPS];
        let mut t = 0usize;
        for i in 0..NB_BANDS - 1 {
            for j in 0..size[i] {
                assert!(t < MAX_TAPS, "band taps exceed MAX_TAPS");
                frac[t] = j as f32 / size[i] as f32;
                t += 1;
            }
        }
        Self { size, offset, frac }
    }

    /// Triangular analysis: power spectrum → 18 band energies.
    fn energy(&self, bin_pow: &[f32], out: &mut [f32; NB_BANDS]) {
        out.fill(0.0);
        let mut t = 0usize;
        for i in 0..NB_BANDS - 1 {
            for j in 0..self.size[i] {
                let frac = self.frac[t];
                t += 1;
                let acc = (self.offset[i] + j).min(NBINS - 1);
                out[i] += (1.0 - frac) * bin_pow[acc];
                out[i + 1] += frac * bin_pow[acc];
            }
        }
        out[0] *= 2.0;
        out[NB_BANDS - 1] *= 2.0;
    }

    /// Triangular synthesis: 18 band gains → per-bin gain curve.
    ///
    /// Unused under `fast-pitch`, which folds this into [`ac_matrix`] — but it
    /// stays because that matrix is *defined* by replaying this, and the test
    /// comparing the two paths calls it.
    #[cfg_attr(feature = "fast-pitch", allow(dead_code))]
    fn interp_gain(&self, band: &[f32; NB_BANDS], out: &mut [f32]) {
        out.fill(0.0);
        let mut t = 0usize;
        for i in 0..NB_BANDS - 1 {
            for j in 0..self.size[i] {
                let frac = self.frac[t];
                t += 1;
                let acc = (self.offset[i] + j).min(NBINS - 1);
                out[acc] = (1.0 - frac) * band[i] + frac * band[i + 1];
            }
        }
    }
}

/// The 18x17 matrix that replaces `interp_gain` + the 1024-point inverse FFT.
///
/// `real_spectrum_to_autocorrelation` computes, for Ooura's normalisation
/// (measured as exactly ½, pinned by `the_matrix_matches_the_fft`):
///
/// ```text
/// ac[k] = ½·xr[0] + Σ_{b=1}^{511} xr[b]·cos(2πkb/1024)
/// ```
///
/// and `interp_gain` is linear in the 18 band gains, so the composition is a
/// fixed matrix: 306 multiply-adds instead of a 513-bin interpolation and a
/// 1024-point transform. On `rv32imc` that FFT alone is 629 k instructions per
/// frame — 37% of `update_lpc` and 17% of the whole pitch estimator.
///
/// Built once at construction. The cosine table costs 1024 `cos` calls; the
/// matrix itself is ~17 k multiply-adds.
#[cfg(any(feature = "fast-pitch", test))]
fn ac_matrix(bands: &Bands) -> [[f32; LPC_ORDER + 1]; NB_BANDS] {
    let mut cos_tab = [0.0f32; FFT_SIZE];
    for (i, c) in cos_tab.iter_mut().enumerate() {
        *c = math::cos(2.0 * PI * i as f32 / FFT_SIZE as f32);
    }

    // `interp_gain` *assigns* rather than accumulates, and its `.min()` clamp
    // lets two (band, tap) pairs target the same bin — so the last writer wins.
    // Replaying that here is what keeps the matrix faithful; accumulating
    // directly would double-count the overlaps.
    let mut writer = [(usize::MAX, 0.0f32); NBINS];
    let mut t = 0usize;
    for i in 0..NB_BANDS - 1 {
        for j in 0..bands.size[i] {
            let frac = bands.frac[t];
            t += 1;
            let acc = (bands.offset[i] + j).min(NBINS - 1);
            writer[acc] = (i, frac);
        }
    }

    let mut m = [[0.0f32; LPC_ORDER + 1]; NB_BANDS];
    for (acc, &(i, frac)) in writer.iter().enumerate() {
        // Bins nobody writes stay zero (`out.fill(0.0)`), and the caller zeroes
        // Nyquist right after `interp_gain`.
        if i == usize::MAX || acc == NBINS - 1 {
            continue;
        }
        let half = if acc == 0 { 0.5 } else { 1.0 };
        for k in 0..=LPC_ORDER {
            let c = cos_tab[(k * acc) % FFT_SIZE] * half;
            m[i][k] += (1.0 - frac) * c;
            m[i + 1][k] += frac * c;
        }
    }
    m
}

/// `cos((i + ½)·j·π/18)`, with column 0 scaled by `√½` (LPCNet's `dct_table`).
fn dct_table() -> [f32; NB_BANDS * NB_BANDS] {
    let mut t = [0.0f32; NB_BANDS * NB_BANDS];
    for i in 0..NB_BANDS {
        for j in 0..NB_BANDS {
            let v = math::cos((i as f32 + 0.5) * j as f32 * PI / NB_BANDS as f32);
            t[i * NB_BANDS + j] = if j == 0 { v * math::sqrt(0.5) } else { v };
        }
    }
    t
}

/// Levinson-Durbin (`celt_lpc`), bailing out once the residual drops 30 dB.
///
/// `f32` throughout, and the groupings below are the reference's: `(r * r) *
/// error` and `tmp1 + (r * tmp2)` round differently if reassociated.
fn celt_lpc(ac: &[f32], lpc: &mut [f32; LPC_ORDER]) {
    lpc.fill(0.0);
    if ac[0] == 0.0 {
        return;
    }
    let mut error = ac[0];
    for i in 0..LPC_ORDER {
        let mut rr = 0.0f32;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;
        lpc[i] = r;
        for j in 0..(i + 1) >> 1 {
            let (t1, t2) = (lpc[j], lpc[i - 1 - j]);
            lpc[j] = t1 + (r * t2);
            lpc[i - 1 - j] = t2 + (r * t1);
        }
        error -= (r * r) * error;
        if error < 0.001 * ac[0] {
            break;
        }
    }
}

/// Streaming pitch estimator. One instance per VAD session.
pub struct PitchEstimator {
    bands: Bands,
    dct: [f32; NB_BANDS * NB_BANDS],
    /// `√(2/18)`, the DCT/IDCT normalisation. Hoisted out of the 18-iteration
    /// loop in `update_lpc`, which recomputed a soft-float `sqrt` of a constant
    /// on every pass. Same value, so bit-exact.
    dct_scale: f32,
    /// `MAXPATH_W · |j|²` for every transition offset the Viterbi can take.
    ///
    /// The inner loop runs ~3,200 times per frame and recomputed this from
    /// `j` each pass — an int→float conversion and two multiplies, all
    /// soft-float on a core without an FPU. `j` spans `-(DIF_PERIOD-1)..=4`,
    /// so the table is indexed by `|j|`. Same values, so bit-exact.
    path_penalty: [f32; DIF_PERIOD + 1],
    /// Band gains -> autocorrelation lags, replacing the inverse FFT. See
    /// [`ac_matrix`].
    #[cfg(feature = "fast-pitch")]
    ac_mat: [[f32; LPC_ORDER + 1]; NB_BANDS],
    lowpass: Biquad<{ crate::biquad::LOWPASS_SECTIONS }>,

    input_q: [f32; INQ_LEN],
    aligned: [f32; HOP],
    lpc_out: [f32; HOP],
    lpc: [f32; LPC_ORDER],
    /// LPC filter history, newest-first, as a circular buffer: `j` samples ago
    /// lives at `pitch_mem[(pitch_head + j) & (LPC_ORDER - 1)]`.
    ///
    /// It used to be a shift register, which meant a 15-element `copy_within`
    /// per input sample — 3,840 element moves per frame to avoid one add and a
    /// mask. The traversal order is unchanged, so this is bit-exact.
    pitch_mem: [f32; LPC_ORDER],
    pitch_head: usize,
    pitch_filt: f32,

    exc: [f32; EXC_LEN],
    exc_sq: [f32; EXC_LEN],
    xcorr: [[f32; MAX_PERIOD + 1]; N_SUB],
    xcorr_tmp: [[f32; MAX_PERIOD + 1]; N_SUB],
    xcorr_inst: [f32; MAX_PERIOD],
    xcorr_off: usize,
    frm_weight: [f32; N_SUB],
    frm_weight_norm: [f32; N_SUB],

    path: [[f32; MAX_PERIOD]; 2],
    prev: [[u8; MAX_PERIOD]; N_SUB],
    path_all: f32,
    best_period: usize,

    /// Band-gain spectrum. Only the FFT path builds one; `fast-pitch` goes
    /// straight from band gains to autocorrelation, so it needs no buffer.
    #[cfg(not(feature = "fast-pitch"))]
    xr: [f32; NBINS],
}

impl Default for PitchEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl PitchEstimator {
    pub fn new() -> Self {
        Self {
            bands: Bands::new(),
            dct_scale: math::sqrt(2.0f32 / NB_BANDS as f32),
            path_penalty: {
                let mut t = [0.0f32; DIF_PERIOD + 1];
                let mut m = 0usize;
                while m <= DIF_PERIOD {
                    let mag = m as f32;
                    // Grouped exactly as the loop had it.
                    t[m] = MAXPATH_W * mag * mag;
                    m += 1;
                }
                t
            },
            #[cfg(feature = "fast-pitch")]
            ac_mat: ac_matrix(&Bands::new()),
            dct: dct_table(),
            lowpass: Biquad::new(LOWPASS_4KHZ),
            input_q: [0.0; INQ_LEN],
            aligned: [0.0; HOP],
            lpc_out: [0.0; HOP],
            lpc: [0.0; LPC_ORDER],
            pitch_mem: [0.0; LPC_ORDER],
            pitch_head: 0,
            pitch_filt: 0.0,
            exc: [0.0; EXC_LEN],
            exc_sq: [0.0; EXC_LEN],
            xcorr: [[0.0; MAX_PERIOD + 1]; N_SUB],
            xcorr_tmp: [[0.0; MAX_PERIOD + 1]; N_SUB],
            xcorr_inst: [0.0; MAX_PERIOD],
            xcorr_off: 0,
            frm_weight: [0.0; N_SUB],
            frm_weight_norm: [0.0; N_SUB],
            path: [[0.0; MAX_PERIOD]; 2],
            prev: [[0; MAX_PERIOD]; N_SUB],
            path_all: 0.0,
            best_period: 0,
            #[cfg(not(feature = "fast-pitch"))]
            xr: [0.0; NBINS],
        }
    }

    /// Clear every run-time buffer (`AUP_PE_resetVariables`). The band /
    /// DCT / cosine tables are configuration and stay put.
    pub fn reset(&mut self) {
        self.input_q.fill(0.0);
        self.aligned.fill(0.0);
        self.lpc_out.fill(0.0);
        self.lpc = [0.0; LPC_ORDER];
        self.pitch_mem = [0.0; LPC_ORDER];
        self.pitch_head = 0;
        self.pitch_filt = 0.0;
        self.exc.fill(0.0);
        self.exc_sq.fill(0.0);
        self.xcorr.fill([0.0; MAX_PERIOD + 1]);
        self.xcorr_tmp.fill([0.0; MAX_PERIOD + 1]);
        self.xcorr_inst = [0.0; MAX_PERIOD];
        self.xcorr_off = 0;
        self.frm_weight = [0.0; N_SUB];
        self.frm_weight_norm = [0.0; N_SUB];
        self.path = [[0.0; MAX_PERIOD]; 2];
        self.prev.fill([0; MAX_PERIOD]);
        self.path_all = 0.0;
        self.best_period = 0;
        #[cfg(not(feature = "fast-pitch"))]
        self.xr.fill(0.0);
        self.lowpass.reset();
    }

    /// Cepstrum → band gains → autocorrelation → LPC coefficients.
    fn update_lpc(&mut self, cepstrum: &[f32; NB_BANDS]) {
        let mut ex = [0.0f32; NB_BANDS];
        for (i, slot) in ex.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (j, &c) in cepstrum.iter().enumerate() {
                sum += c * self.dct[i * NB_BANDS + j];
            }
            let idct = sum * self.dct_scale;
            *slot = math::pow10(idct) * BAND_LPC_COMP[i];
        }
        // The reference inverse-transforms the (real, symmetric) band-gain
        // spectrum and keeps the first LPC_ORDER+1 lags as the autocorrelation.
        // `fast-pitch` takes the same linear map through a precomputed 18x17
        // matrix instead, skipping both the interpolation and the transform.
        let mut ac = [0.0f32; LPC_ORDER + 1];
        #[cfg(not(feature = "fast-pitch"))]
        {
            self.bands.interp_gain(&ex, &mut self.xr);
            self.xr[NBINS - 1] = 0.0; // remove Nyquist
            ooura::real_spectrum_to_autocorrelation(&self.xr, &mut ac);
        }
        #[cfg(feature = "fast-pitch")]
        for (i, &g) in ex.iter().enumerate() {
            for (k, slot) in ac.iter_mut().enumerate() {
                *slot += g * self.ac_mat[i][k];
            }
        }
        // −40 dB noise floor. `windowSz / 12` is integer division upstream.
        ac[0] += ac[0] * 1e-4 + (WINDOW / 12) as f32 / 38.0;
        // Lag windowing, grouped as the reference has it.
        for (i, slot) in ac.iter_mut().enumerate().skip(1) {
            *slot *= 1.0 - 6e-5 * i as f32 * i as f32;
        }
        celt_lpc(&ac, &mut self.lpc);
    }

    /// Inverse-filter, comb, lowpass and decimate this frame into `exc`.
    fn update_excitation(&mut self, time_signal: &[f32]) {
        self.input_q.copy_within(HOP.., 0);
        self.input_q[INQ_LEN - HOP..].copy_from_slice(time_signal);
        let offset = INQ_LEN - HOP - XCORR_TRAINING_OFFSET;
        self.aligned
            .copy_from_slice(&self.input_q[offset..offset + HOP]);

        // `LPC_ORDER` is a power of two, so the wrap is a mask.
        const MASK: usize = LPC_ORDER - 1;
        let mut head = self.pitch_head;
        for i in 0..HOP {
            let mut sum = self.aligned[i];
            for j in 0..LPC_ORDER {
                sum += self.lpc[j] * self.pitch_mem[(head + j) & MASK];
            }
            // Rotating the head back by one makes the new sample "0 ago" and
            // drops what was "15 ago", exactly as the shift did.
            head = (head + MASK) & MASK;
            self.pitch_mem[head] = self.aligned[i];
            self.lpc_out[i] = sum + 0.7 * self.pitch_filt;
            self.pitch_filt = sum;
        }
        self.pitch_head = head;

        self.lowpass.process(&mut self.lpc_out);
        let n = HOP / RESAMPLE;
        self.exc.copy_within(n.., 0);
        for i in 0..n {
            self.exc[EXC_LEN - n + i] = self.lpc_out[i * RESAMPLE];
        }
        for (dst, &v) in self.exc_sq.iter_mut().zip(&self.exc) {
            *dst = v * v;
        }
    }

    /// Normalized cross-correlation of both half-frames against `exc`.
    fn update_xcorr(&mut self) {
        self.frm_weight.copy_within(2.., 0);
        for sub in 0..2 {
            let acc = 2 * self.xcorr_off + sub;
            let off = sub * CORR_HALF;
            let reference = &self.exc[MAX_PERIOD + off..MAX_PERIOD + off + CORR_HALF];
            for (i, slot) in self.xcorr_inst.iter_mut().enumerate() {
                let moving = &self.exc[off + i..off + i + CORR_HALF];
                // `AUP_PE_MvingXCorr`'s unrolled kernel accumulates each shift
                // in plain j order, so a sequential f32 dot matches it exactly.
                let mut acc = 0.0f32;
                for (&a, &b) in reference.iter().zip(moving) {
                    acc += a * b;
                }
                *slot = acc;
            }

            let e0: f32 = self.exc_sq[MAX_PERIOD + off..MAX_PERIOD + off + CORR_HALF]
                .iter()
                .sum();
            self.frm_weight[2 * (N_FEAT - 1) + sub] = e0;

            let mut window: f32 = self.exc_sq[off..off + CORR_HALF].iter().sum();
            let mut denom = (window + 1.0 + e0).max(1e-12);
            self.xcorr[acc][0] = 2.0 * self.xcorr_inst[0] / denom;
            for i in 1..MAX_PERIOD {
                window = (window - self.exc_sq[off + i - 1]).max(0.0)
                    + self.exc_sq[off + i + CORR_HALF - 1];
                denom = (window + 1.0 + e0).max(1e-12);
                self.xcorr[acc][i] = 2.0 * self.xcorr_inst[i] / denom;
            }

            // Sharpen: damp any lag that is not clearly better than its
            // half-period neighbours (the classic octave-error guard). Every
            // index read here is > `i`, so the in-place update is safe.
            for i in 0..MAX_PERIOD - 2 * MIN_PERIOD {
                let peak = self.xcorr[acc][(MAX_PERIOD + i) / 2]
                    .max(self.xcorr[acc][(MAX_PERIOD + i + 2) / 2])
                    .max(self.xcorr[acc][(MAX_PERIOD + i - 1) / 2]);
                if self.xcorr[acc][i] < peak * 1.1 {
                    self.xcorr[acc][i] *= 0.8;
                }
            }
        }
        self.xcorr_off = (self.xcorr_off + 1) % N_FEAT;
    }

    /// Viterbi over the 6 buffered half-frames, then regress the contour.
    fn estimate(&mut self) -> PitchFrame {
        let total: f32 = 1e-15 + self.frm_weight.iter().sum::<f32>();
        for (dst, &w) in self.frm_weight_norm.iter_mut().zip(&self.frm_weight) {
            *dst = w * (N_SUB as f32 / total);
        }
        self.xcorr_tmp.copy_from_slice(&self.xcorr);
        self.prev.copy_within(2.., 0);

        for sub in N_SUB - 2..N_SUB {
            let xci = (sub + self.xcorr_off * 2) % N_SUB;
            for i in 0..DIF_PERIOD {
                let mut best_score = self.path_all - 1e10;
                let mut best_idx = self.best_period;
                // Transition window follows TEN-VAD: `min(0, 4 − i) ..= 4`,
                // so every state can also jump back to state 4 (at a large
                // quadratic penalty) rather than only ±4 as in LPCNet.
                let lo = 4i32 - i as i32;
                for j in lo.min(0)..=4 {
                    let idx = i as i32 + j;
                    if idx >= DIF_PERIOD as i32 {
                        break;
                    }
                    // Left-associated exactly as the C is, so ties break the same way.
                    let score =
                        self.path[0][idx as usize] - self.path_penalty[j.unsigned_abs() as usize];
                    if score > best_score {
                        best_score = score;
                        best_idx = idx as usize;
                    }
                }
                self.prev[sub][i] = best_idx as u8;
                self.path[1][i] = best_score + self.frm_weight_norm[sub] * self.xcorr_tmp[xci][i];
            }

            let mut max_path = -1e15f32;
            let mut arg = 0usize;
            for i in 0..DIF_PERIOD {
                if self.path[1][i] > max_path {
                    max_path = self.path[1][i];
                    arg = i;
                }
            }
            self.path_all = max_path;
            self.best_period = arg;
            self.path[0] = self.path[1];
            for v in &mut self.path[0][..DIF_PERIOD] {
                *v -= max_path;
            }
        }

        let mut state = self.best_period;
        let mut corr = 0.0f32;
        let mut contour = [0i32; N_SUB];
        for sub in (0..N_SUB).rev() {
            contour[sub] = (MAX_PERIOD - state) as i32;
            let xci = (sub + self.xcorr_off * 2) % N_SUB;
            corr += self.frm_weight_norm[sub] * self.xcorr_tmp[xci][state];
            state = self.prev[sub][state] as usize;
        }
        corr = (corr / N_SUB as f32).max(0.0);
        let voiced = corr >= VOICED_THR;

        let (mut sw, mut sx, mut sxx, mut sxy, mut sy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for sub in 0..N_SUB {
            let w = self.frm_weight_norm[sub];
            let (x, y) = (sub as f32, contour[sub] as f32);
            sw += w;
            sx += w * x;
            sxx += w * x * x;
            sxy += w * x * y;
            sy += w * y;
        }
        let denom = sw * sxx - sx * sx;
        let mut slope = (sw * sxy - sx * sy) / if denom == 0.0 { 1e-15 } else { denom };
        if voiced {
            let lim = (sy / sw) / (4 * 2 * N_FEAT) as f32;
            slope = slope.clamp(-lim, lim);
        } else {
            slope = 0.0;
        }
        let intercept = (sy - slope * sx) / sw;
        let period = intercept + 5.5 * slope;

        PitchFrame {
            freq_hz: if voiced {
                PROC_FS / period.max(1.0)
            } else {
                0.0
            },
            voiced,
        }
    }

    /// Estimate the pitch of one 256-sample frame.
    ///
    /// `time_signal` is the raw (not pre-emphasised) frame in int16 units;
    /// `bin_pow` is `|X[k]|²` of the pre-emphasised, windowed frame.
    pub fn process(&mut self, time_signal: &[f32], bin_pow: &[f32]) -> PitchFrame {
        assert_eq!(time_signal.len(), HOP, "pitch frame must be {HOP} samples");
        assert_eq!(bin_pow.len(), NBINS, "pitch spectrum must be {NBINS} bins");

        let mut band = [0.0f32; NB_BANDS];
        self.bands.energy(bin_pow, &mut band);

        // Log-compress with a decaying floor so one loud band cannot swamp the
        // cepstrum (`logMax − 8`, `follow − 2.5`).
        let mut log_max = -2.0f32;
        let mut follow = -2.0f32;
        let mut ly = [0.0f32; NB_BANDS];
        for i in 0..NB_BANDS {
            let v = math::log10(1e-2 + band[i])
                .max(follow - 2.5)
                .max(log_max - 8.0);
            ly[i] = v;
            log_max = log_max.max(v);
            follow = (follow - 2.5).max(v);
        }

        let ratio = self.dct_scale;
        let mut cepstrum = [0.0f32; NB_BANDS];
        for (i, slot) in cepstrum.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (j, &v) in ly.iter().enumerate() {
                sum += v * self.dct[j * NB_BANDS + i];
            }
            *slot = sum * ratio;
        }

        self.update_lpc(&cepstrum);
        self.update_excitation(time_signal);
        self.update_xcorr();
        self.estimate()
    }
}

/// Per-stage entry points for the MCU instruction-count harness.
///
/// A QEMU TCG plugin reports one instruction total per run, so attributing cost
/// to a stage means running that stage alone. These exist for that and are
/// behind a feature so they are not part of the API.
#[cfg(feature = "bench-internals")]
impl PitchEstimator {
    /// Steps 1 and the log/DCT that follows it — everything `process` does
    /// before `update_lpc`.
    pub fn bench_bands_dct(&mut self, bin_pow: &[f32]) -> [f32; NB_BANDS] {
        let mut band = [0.0f32; NB_BANDS];
        self.bands.energy(bin_pow, &mut band);
        let mut log_max = -2.0f32;
        let mut follow = -2.0f32;
        let mut ly = [0.0f32; NB_BANDS];
        for i in 0..NB_BANDS {
            let v = math::log10(1e-2 + band[i])
                .max(follow - 2.5)
                .max(log_max - 8.0);
            ly[i] = v;
            log_max = log_max.max(v);
            follow = (follow - 2.5).max(v);
        }
        let ratio = self.dct_scale;
        let mut cepstrum = [0.0f32; NB_BANDS];
        for (i, slot) in cepstrum.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (j, &v) in ly.iter().enumerate() {
                sum += v * self.dct[j * NB_BANDS + i];
            }
            *slot = sum * ratio;
        }
        cepstrum
    }

    pub fn bench_update_lpc(&mut self, cepstrum: &[f32; NB_BANDS]) {
        self.update_lpc(cepstrum);
    }

    pub fn bench_update_excitation(&mut self, time_signal: &[f32]) {
        self.update_excitation(time_signal);
    }

    pub fn bench_update_xcorr(&mut self) {
        self.update_xcorr();
    }

    pub fn bench_estimate(&mut self) -> PitchFrame {
        self.estimate()
    }

    /// Just the inverse transform inside `update_lpc`, which is the part that
    /// computes a 1024-point FFT to keep 17 autocorrelation lags. Absent under
    /// `fast-pitch`, which is the configuration that deletes it.
    #[cfg(not(feature = "fast-pitch"))]
    pub fn bench_autocorr_only(&mut self) {
        let mut ac = [0.0f32; LPC_ORDER + 1];
        ooura::real_spectrum_to_autocorrelation(&self.xr, &mut ac);
        core::hint::black_box(&ac);
    }
}

/// Number of analysis bands, exposed for the bench harness.
#[cfg(feature = "bench-internals")]
pub const BANDS: usize = NB_BANDS;

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use alloc::{vec, vec::Vec};

    const _: () = assert!(INQ_LEN == 512 && EXC_LEN == 129 && CORR_HALF == 32);
    const _: () = assert!(MIN_PERIOD == 8 && MAX_PERIOD == 64 && DIF_PERIOD == 56);

    /// The `fast-pitch` matrix must reproduce the FFT it replaces.
    ///
    /// Both paths are built here regardless of the feature, so this runs the
    /// comparison in either configuration rather than only in the one that
    /// happens to be enabled.
    #[test]
    fn the_matrix_matches_the_fft() {
        let bands = Bands::new();
        let m = ac_matrix(&bands);
        let mut worst = 0.0f32;
        for trial in 0..64 {
            let mut ex = [0.0f32; NB_BANDS];
            for (i, slot) in ex.iter_mut().enumerate() {
                let t = (trial * NB_BANDS + i) as f32;
                *slot = (t * 0.29).sin().abs() * 10f32.powf((t * 0.13).cos() * 3.0);
            }
            let mut xr = vec![0.0f32; NBINS];
            bands.interp_gain(&ex, &mut xr);
            xr[NBINS - 1] = 0.0;
            let mut want = [0.0f32; LPC_ORDER + 1];
            ooura::real_spectrum_to_autocorrelation(&xr, &mut want);

            let mut got = [0.0f32; LPC_ORDER + 1];
            for (i, &g) in ex.iter().enumerate() {
                for (k, slot) in got.iter_mut().enumerate() {
                    *slot += g * m[i][k];
                }
            }
            for k in 0..=LPC_ORDER {
                let denom = want[0].abs().max(1e-20);
                worst = worst.max((got[k] - want[k]).abs() / denom);
            }
        }
        assert!(
            worst < 1e-5,
            "matrix path diverges from the FFT by {worst:.3e} relative to lag-0 energy"
        );
    }

    /// `update_lpc` maps 18 band gains to 17 autocorrelation lags through a
    /// 513-bin interpolation and a 1024-point inverse FFT — and every step of
    /// that is **linear**, so the whole chain is an 18x17 matrix.
    ///
    /// If that holds, the FFT (629 k instructions/frame on `rv32imc`, 37% of
    /// `update_lpc`) can be replaced by 306 multiply-adds. This test is the
    /// evidence for the claim before anything is built on it: probe the real
    /// path with unit vectors to get the matrix, then check the matrix
    /// reproduces the real path on inputs that are nothing like unit vectors.
    #[test]
    fn the_band_gain_to_autocorrelation_chain_is_linear() {
        let bands = Bands::new();
        let ac_of = |ex: &[f32; NB_BANDS]| -> [f32; LPC_ORDER + 1] {
            let mut xr = vec![0.0f32; NBINS];
            bands.interp_gain(ex, &mut xr);
            xr[NBINS - 1] = 0.0;
            let mut ac = [0.0f32; LPC_ORDER + 1];
            ooura::real_spectrum_to_autocorrelation(&xr, &mut ac);
            ac
        };

        // Column i is the response to band i alone.
        let mut m = [[0.0f32; LPC_ORDER + 1]; NB_BANDS];
        for i in 0..NB_BANDS {
            let mut unit = [0.0f32; NB_BANDS];
            unit[i] = 1.0;
            m[i] = ac_of(&unit);
        }

        // Inputs shaped like real band gains: positive, wide dynamic range.
        let mut worst = 0.0f32;
        for trial in 0..32 {
            let mut ex = [0.0f32; NB_BANDS];
            for (i, slot) in ex.iter_mut().enumerate() {
                let t = (trial * NB_BANDS + i) as f32;
                *slot = (t * 0.37).sin().abs() * 10f32.powf((t * 0.11).cos() * 3.0);
            }
            let want = ac_of(&ex);
            let mut got = [0.0f32; LPC_ORDER + 1];
            for (i, &g) in ex.iter().enumerate() {
                for (k, slot) in got.iter_mut().enumerate() {
                    *slot += g * m[i][k];
                }
            }
            for k in 0..=LPC_ORDER {
                let denom = want[k].abs().max(want[0].abs()).max(1e-20);
                worst = worst.max((got[k] - want[k]).abs() / denom);
            }
        }
        // Not bit-exact: the matrix sums 18 terms where the FFT sums a
        // butterfly tree. Relative to the lag-0 energy the two agree to float
        // round-off, which is what linearity predicts.
        assert!(
            worst < 1e-5,
            "matrix and FFT disagree by {worst:.3e} relative — the chain is not linear"
        );
    }

    /// Power spectrum of a windowed, pre-emphasised sine — mirrors what the
    /// frontend hands the estimator.
    fn spectrum(f0: f32, phase: &mut f32, frame: &mut [f32], prev: &mut f32) -> Vec<f32> {
        let mut buf = vec![0.0f32; 768];
        for (i, s) in frame.iter_mut().enumerate() {
            *s = (*phase + 2.0 * core::f32::consts::PI * f0 * i as f32 / 16000.0).sin() * 8000.0;
        }
        *phase += 2.0 * core::f32::consts::PI * f0 * HOP as f32 / 16000.0;
        for (i, &s) in frame.iter().enumerate() {
            let e = s - 0.97 * *prev;
            *prev = s;
            let w = 0.5 - 0.5 * (2.0 * core::f32::consts::PI * (512 + i) as f32 / 768.0).cos();
            buf[512 + i] = e * w;
        }
        let mut out = vec![0.0f32; NBINS];
        ooura::power_spectrum(&buf, &mut out);
        out
    }

    #[test]
    fn tracks_a_steady_tone() {
        let mut pe = PitchEstimator::new();
        let mut frame = vec![0.0f32; HOP];
        let (mut phase, mut prev) = (0.0f32, 0.0f32);
        let mut last = PitchFrame::default();
        for _ in 0..40 {
            let spec = spectrum(200.0, &mut phase, &mut frame, &mut prev);
            last = pe.process(&frame, &spec);
        }
        assert!(last.voiced, "steady 200 Hz tone should read as voiced");
        assert!(
            (last.freq_hz - 200.0).abs() < 20.0,
            "pitch {} Hz off 200 Hz",
            last.freq_hz
        );
    }

    #[test]
    fn silence_is_unvoiced() {
        let mut pe = PitchEstimator::new();
        let frame = vec![0.0f32; HOP];
        let spec = vec![0.0f32; NBINS];
        let mut last = PitchFrame::default();
        for _ in 0..20 {
            last = pe.process(&frame, &spec);
        }
        assert!(!last.voiced);
        assert_eq!(last.freq_hz, 0.0);
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut pe = PitchEstimator::new();
        let mut frame = vec![0.0f32; HOP];
        let (mut phase, mut prev) = (0.0f32, 0.0f32);
        let mut first = Vec::new();
        for _ in 0..12 {
            let spec = spectrum(150.0, &mut phase, &mut frame, &mut prev);
            first.push(pe.process(&frame, &spec).freq_hz);
        }
        pe.reset();
        let (mut phase, mut prev) = (0.0f32, 0.0f32);
        let mut second = Vec::new();
        for _ in 0..12 {
            let spec = spectrum(150.0, &mut phase, &mut frame, &mut prev);
            second.push(pe.process(&frame, &spec).freq_hz);
        }
        assert_eq!(first, second);
    }
}
