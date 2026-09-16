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

//! Integer 1024-point FFT and power spectrum, for cores without an FPU.
//!
//! On `rv32imc` the two `f32` transforms in the frontend — the forward one for
//! the spectrum and the inverse one inside the pitch estimator's
//! autocorrelation — cost 1.36 M instructions per frame between them, 29% of
//! the whole frontend, because every butterfly is a soft-float call.
//!
//! This is not a transliteration of [`crate::ooura`] and makes no claim to
//! match it bit for bit; that path stays the reference. It is a separate
//! integer implementation held to a *measured* budget: the network needs the
//! 40 mel features to about 12 bits (finer, and no decision changes on the
//! reference clip), so the spectrum needs roughly that much relative accuracy.
//!
//! # Why no scaling
//!
//! Fixed-point FFTs usually halve the data every stage to prevent overflow,
//! which throws away one bit per stage — ten bits over 1024 points, leaving
//! far less than the twelve needed here.
//!
//! None of that is necessary. The input is i16-valued audio, so |x| < 2^15,
//! and a 1024-point transform grows magnitude by at most 2^10. The result
//! therefore fits in 25 bits, with six to spare in an `i32`. Twiddles are Q30
//! and products accumulate in `i64`, so the only error is twiddle rounding —
//! about `2^-30` per stage, accumulating to roughly `2^-26` over ten. That is
//! fourteen bits clear of the budget.
//!
//! That analysis assumes the requantisation *rounds*. It used to truncate, and
//! an arithmetic shift floors toward −∞, so the per-stage error was −1..0 LSB
//! rather than ±½ — a bias that ten stages accumulate in one direction instead
//! of cancelling. Both fixed-point paths in this crate now share
//! [`fixed_math::cmul_q`], which rounds: **13.3 → 17.6 bits** against the f32
//! transform on the same input, a 20x smaller error for one add per butterfly
//! (+0.3% measured on `rv32imc`).

use crate::fixed_math;
use crate::math;

/// Transform size. Fixed: the frontend windows 768 samples and zero-pads.
pub const N: usize = 1024;
/// `N / 2 + 1` non-redundant bins of a real transform.
pub const BINS: usize = N / 2 + 1;
/// Fractional bits in a twiddle factor.
const TW_FRAC: u32 = 30;

/// Round an `f64` to the nearest `i32`. `f64::round` is `std`-only.
#[inline]
fn libm_round(v: f64) -> i32 {
    (if v < 0.0 { v - 0.5 } else { v + 0.5 }) as i32
}

/// Radix-2 FFT with its twiddle table.
///
/// Build once and reuse: the table costs 1024 transcendental calls, which is
/// several frames' worth of work if repeated.
pub struct FixedFft {
    tw_re: [i32; N / 2],
    tw_im: [i32; N / 2],
}

impl Default for FixedFft {
    fn default() -> Self {
        Self::new()
    }
}

impl FixedFft {
    pub fn new() -> Self {
        let mut f = Self {
            tw_re: [0; N / 2],
            tw_im: [0; N / 2],
        };
        // f64, and rounded. A Q30 twiddle needs 30 significant bits; an f32
        // carries 24, so the old `cos(ang) * 2^30` left the low bits as noise
        // and `as i32` truncated toward zero on top of it. Measured by
        // `twiddles_are_within_one_lsb_of_exact`: **218 LSB off** the exact
        // value, against 0.50 now — the table carried nearly 8 bits of
        // garbage. `math::cos64` already existed for this reason (Ooura's
        // tables use it); this path had not been given it.
        //
        // It does *not* move the end-to-end figure below — requantisation
        // dominates that — so this is a fix to the table, not to the
        // transform. Construction roughly doubles, ~21 ms once at 160 MHz.
        let scale = (1i64 << TW_FRAC) as f64;
        for k in 0..N / 2 {
            let ang = -2.0 * core::f64::consts::PI * k as f64 / N as f64;
            f.tw_re[k] = libm_round(math::cos64(ang) * scale);
            f.tw_im[k] = libm_round(math::sin64(ang) * scale);
        }
        f
    }

    /// `|X[k]|²` for the `BINS` non-redundant bins of a real input.
    ///
    /// `input` is zero-padded to `N`. Values must satisfy `|x| < 2^15`, which
    /// i16-valued audio does by construction — the no-scaling argument above
    /// depends on it.
    ///
    /// The transform is half-length: the `N` real samples are packed as the
    /// real and imaginary parts of an `N/2` complex sequence, and the true
    /// spectrum is recovered from its symmetry. That halves the butterflies,
    /// which is the difference between beating the `f32` path by 1.8x and by
    /// 3.4x.
    pub fn power_spectrum(&self, input: &[i32], out: &mut [i64]) {
        debug_assert!(input.len() <= N);
        debug_assert_eq!(out.len(), BINS);
        const H: usize = N / 2;

        // z[n] = x[2n] + i·x[2n+1]
        let mut re = [0i32; H];
        let mut im = [0i32; H];
        for n in 0..H {
            re[n] = input.get(2 * n).copied().unwrap_or(0);
            im[n] = input.get(2 * n + 1).copied().unwrap_or(0);
        }
        self.transform_half(&mut re, &mut im);

        // X[k] = E[k] + e^{-2πik/N}·O[k], where E and O are the even/odd
        // sub-spectra recovered from Z's conjugate symmetry.
        for k in 0..=H / 2 {
            let j = (H - k) % H;
            let (zr, zi) = (re[k], im[k]);
            let (cr, ci) = (re[j], -im[j]);
            // E = (Z[k] + conj(Z[H-k]))/2,  O = (Z[k] - conj(Z[H-k]))/(2i)
            let (er, ei) = ((zr + cr) / 2, (zi + ci) / 2);
            let (or_, oi) = ((zi - ci) / 2, (cr - zr) / 2);
            let (wr, wi) = (i64::from(self.tw_re[k]), i64::from(self.tw_im[k]));
            let (tr, ti) = fixed_math::cmul_q(or_, oi, wr, wi, TW_FRAC);
            let (xr, xi) = (i64::from(er + tr), i64::from(ei + ti));
            out[k] = xr * xr + xi * xi;
            // The mirrored bin: X[H-k] = conj(E[k]) - conj(w·O[k]).
            if k > 0 && H - k <= H {
                let (yr, yi) = (i64::from(er - tr), i64::from(-(ei - ti)));
                out[H - k] = yr * yr + yi * yi;
            }
        }
        // Nyquist: X[H] = E[0] - O[0] with w = 1.
        let (e0r, o0r) = ((re[0] + re[0]) / 2, (im[0] + im[0]) / 2);
        let nyq = i64::from(e0r - o0r);
        out[H] = nyq * nyq;
    }

    /// In-place complex FFT of length `N/2`, stepping the twiddle table by two.
    fn transform_half(&self, re: &mut [i32; N / 2], im: &mut [i32; N / 2]) {
        const H: usize = N / 2;
        let bits = H.trailing_zeros();
        for i in 0..H {
            let j = ((i as u32).reverse_bits() >> (32 - bits)) as usize;
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= H {
            let step = H / len;
            for base in (0..H).step_by(len) {
                for k in 0..len / 2 {
                    let t = k * step * 2;
                    let (wr, wi) = (i64::from(self.tw_re[t]), i64::from(self.tw_im[t]));
                    let (a, b) = (base + k, base + k + len / 2);
                    let (tr, ti) = fixed_math::cmul_q(re[b], im[b], wr, wi, TW_FRAC);
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len <<= 1;
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// A pure tone must land in one bin, with the right magnitude.
    #[test]
    fn a_single_bin_cosine_transforms_to_that_bin() {
        let fft = FixedFft::new();
        let mut out = [0i64; BINS];
        for bin in [1usize, 7, 64, 300] {
            let x: alloc::vec::Vec<i32> = (0..N)
                .map(|n| {
                    (8000.0
                        * (2.0 * core::f32::consts::PI * bin as f32 * n as f32 / N as f32).cos())
                        as i32
                })
                .collect();
            fft.power_spectrum(&x, &mut out);
            let peak = out.iter().enumerate().max_by_key(|(_, v)| **v).unwrap().0;
            assert_eq!(peak, bin, "tone at bin {bin} peaked at {peak}");
            // A real cosine splits its energy between +f and -f, so the
            // one-sided magnitude is N/2 times the amplitude.
            let want = (8000.0 * N as f32 / 2.0) as i64;
            let got = (out[bin] as f64).sqrt() as i64;
            let err = (got - want).abs() as f64 / want as f64;
            assert!(err < 1e-3, "bin {bin}: |X| {got} vs {want} ({err:.1e})");
        }
    }

    /// Against the f32 reference on a broadband signal — the accuracy that
    /// matters, since the mel bands sum many bins.
    /// How far the twiddle table itself is from the exact Q30 values.
    ///
    /// The end-to-end test above cannot see this — the requantisation error
    /// dominates it — but the table is a fixed artefact and its accuracy is
    /// worth stating on its own. Building a Q30 constant through an `f32`
    /// leaves ~6 bits of noise, because an `f32` mantissa is 24 bits and the
    /// constant needs 30.
    #[test]
    fn twiddles_are_within_one_lsb_of_exact() {
        let f = FixedFft::new();
        let scale = (1i64 << TW_FRAC) as f64;
        let mut worst = 0.0f64;
        for k in 0..N / 2 {
            let ang = -2.0 * core::f64::consts::PI * k as f64 / N as f64;
            worst = worst
                .max((f64::from(f.tw_re[k]) - (ang.cos() * scale)).abs())
                .max((f64::from(f.tw_im[k]) - (ang.sin() * scale)).abs());
        }
        std::println!("worst twiddle error = {worst:.2} LSB of Q30");
        assert!(
            worst <= 0.5,
            "twiddle table is {worst:.1} LSB off exact — built at too low a precision"
        );
    }

    #[test]
    fn tracks_the_f32_transform_to_better_than_twelve_bits() {
        let fft = FixedFft::new();
        let sig: alloc::vec::Vec<f32> = (0..crate::WINDOW_SIZE)
            .map(|n| {
                let t = n as f32;
                6000.0 * (t * 0.031).sin() + 2500.0 * (t * 0.211).cos() + 400.0 * (t * 0.7).sin()
            })
            .collect();
        let qi: alloc::vec::Vec<i32> = sig.iter().map(|&v| v as i32).collect();
        // Feed the f32 reference the *quantised* signal. Handing it `sig`
        // instead measured the input rounding, which is ~1 LSB on an amplitude
        // of 6000 and swamped everything the transform does — the number came
        // out the same whether the butterflies rounded or truncated, which is
        // how the flaw showed itself.
        let sig: alloc::vec::Vec<f32> = qi.iter().map(|&v| v as f32).collect();

        let mut want = [0.0f32; BINS];
        crate::ooura::power_spectrum(&sig, &mut want);
        let mut got = [0i64; BINS];
        fft.power_spectrum(&qi, &mut got);

        // Compare magnitudes: power squares the relative error.
        let mut worst = 0.0f64;
        let peak = want.iter().fold(0.0f32, |m, &v| m.max(v)).sqrt() as f64;
        for k in 0..BINS {
            let a = (got[k] as f64).sqrt();
            let b = f64::from(want[k]).sqrt();
            // Relative to the spectral peak: an empty bin's own relative error
            // is meaningless and not what the mel bands integrate.
            worst = worst.max((a - b).abs() / peak);
        }
        std::println!("worst = {worst:.3e} of peak = {:.1} bits", -worst.log2());
        // Measured 4.87e-6 (17.6 bits). The bound sits just above that rather
        // than at the 12-bit budget: with the transform's own error isolated,
        // a regression to truncation (1.00e-4) should fail here, and against a
        // 12-bit bound it would not.
        assert!(
            worst < 1e-5,
            "worst bin is {worst:.2e} of peak — the transform lost precision"
        );
    }
}
