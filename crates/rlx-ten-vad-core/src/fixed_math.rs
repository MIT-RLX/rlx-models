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

//! Fixed-point primitives, in one place.
//!
//! [`math`](crate::math) is the same seam for floating point: every
//! transcendental goes through it so the `std` / `no_std` split is decided
//! once. This is its integer counterpart, and it exists because the two
//! integer paths in this crate — the network in [`fixed`](crate::fixed) and
//! the transform in [`fft_fixed`](crate::fft_fixed) — had drifted apart.
//!
//! The network rounded its requantisations; the FFT truncated its. Truncation
//! is an arithmetic shift, so it floors toward −∞: the error is not ±½ LSB
//! about zero but −1..0 LSB, a **bias** rather than noise. Ten radix-2 stages
//! accumulate it in one direction. Routing both through [`rsh`] costs the FFT
//! one add per butterfly and buys **4.3 bits**: 13.3 → 17.6 against the f32
//! transform.
//!
//! A second defect surfaced alongside it, in the same spirit: the Q30 twiddle
//! table was built through `f32`, whose 24-bit mantissa cannot hold a 30-bit
//! constant, leaving it 218 LSB off exact. That one is invisible end-to-end —
//! requantisation dominates — so it is pinned by a test on the table itself.
//!
//! Finding the first took fixing the test first. It fed the integer transform a
//! quantised signal and the f32 reference an unquantised one, so it measured
//! input rounding — ~1 LSB on an amplitude of 6000 — and reported the same
//! 13.2 bits whether the butterflies rounded or truncated.

/// Round-to-nearest arithmetic right shift, `i64` → `i32`.
///
/// Truncation biases every requantisation by −½ LSB. In the network the LSTM
/// feeds its own output back, so that bias compounds across frames; in the
/// transform it compounds across stages. Neither can afford it, and rounding
/// costs one add.
///
/// `s == 0` is a plain narrowing, not a shift — `1 << (s - 1)` would underflow.
#[inline]
pub fn rsh(x: i64, s: u32) -> i32 {
    if s == 0 {
        return x as i32;
    }
    ((x + (1 << (s - 1))) >> s) as i32
}

/// One complex multiply-and-requantise: `(a + bi) · (wr + wi·i) >> frac`.
///
/// The butterfly kernel both fixed-point transforms use. Products are formed in
/// `i64` — a Q30 twiddle times a 25-bit datum needs 55 bits — and rounded back
/// through [`rsh`].
#[inline]
pub fn cmul_q(re: i32, im: i32, wr: i64, wi: i64, frac: u32) -> (i32, i32) {
    let (re, im) = (i64::from(re), i64::from(im));
    (rsh(wr * re - wi * im, frac), rsh(wr * im + wi * re, frac))
}

/// Interpolate a Q15 lookup table at Q15 `x`.
///
/// `odd` selects the symmetry: `f(−x) = −f(x)` for tanh, `f(−x) = 1 − f(x)`
/// for sigmoid. `one` is 1.0 in the table's scale, used only by the even case.
///
/// Saturating past the last entry is deliberate — both functions are flat
/// there, and the alternative is a branch on every activation.
#[inline]
pub fn lut_q15<const N: usize>(tab: &[i16; N], x: i32, shift: u32, odd: bool, one: i32) -> i32 {
    let n = N - 1;
    let neg = x < 0;
    let a = i64::from(x).unsigned_abs() as i64;
    let pos = (a >> shift) as usize;
    let v = if pos >= n {
        i32::from(tab[n])
    } else {
        let (lo, hi) = (i32::from(tab[pos]), i32::from(tab[pos + 1]));
        let rem = (a & ((1 << shift) - 1)) as i32;
        lo + (((hi - lo) * rem) >> shift)
    };
    match (odd, neg) {
        (true, true) => -v,
        (true, false) | (false, false) => v,
        (false, true) => one - v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole module exists for: rounding is unbiased where
    /// truncation is not.
    #[test]
    fn rsh_rounds_symmetrically_where_a_shift_would_floor() {
        // −3/2 truncates to −2 and rounds to −1; +3/2 gives +2 both ways.
        assert_eq!(rsh(-3, 1), -1);
        assert_eq!(-3i64 >> 1, -2);
        assert_eq!(rsh(3, 1), 2);

        // Over many values the rounded error averages ~0; the truncated one
        // is a systematic −½ LSB. With a shift of 4 the truncated error is
        // `-(v mod 16)`, uniform on [−15, 0], so it averages −7.5 LSB — over
        // 10,000 samples, about −75,000. Rounding leaves ±½ LSB about zero.
        const N: i64 = 10_000;
        const LSB: i64 = 16;
        let (mut round_err, mut trunc_err) = (0i64, 0i64);
        for x in -5000..5000i64 {
            let v = x * 7;
            round_err += i64::from(rsh(v, 4)) * LSB - v;
            trunc_err += (v >> 4) * LSB - v;
        }
        assert!(
            round_err.abs() <= N,
            "rounding should stay within ±1 LSB per sample on average: {round_err}"
        );
        assert!(
            trunc_err < -(N * LSB / 4),
            "truncation should bias low by several LSB per sample: {trunc_err}"
        );
    }

    #[test]
    fn rsh_of_zero_shift_is_a_narrowing() {
        assert_eq!(rsh(i64::from(i32::MAX), 0), i32::MAX);
        assert_eq!(rsh(-1, 0), -1);
    }

    #[test]
    fn cmul_q_matches_the_scalar_form() {
        let frac = 30;
        let (wr, wi) = (1i64 << 29, -(1i64 << 28));
        for &(re, im) in &[(1000, -2000), (-7, 3), (1 << 20, 1 << 19)] {
            let want = (
                rsh(wr * i64::from(re) - wi * i64::from(im), frac),
                rsh(wr * i64::from(im) + wi * i64::from(re), frac),
            );
            assert_eq!(cmul_q(re, im, wr, wi, frac), want);
        }
    }
}
