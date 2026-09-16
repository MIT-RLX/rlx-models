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

//! Integer-only CRNN forward — no floating point in the whole path.
//!
//! Two jobs. On an MCU without an FPU (ESP32-C3/C6 are `rv32imc`, no `F`) this
//! avoids soft-float in the 79 k MACs per frame. And it is the bit-exact golden
//! model for the FPGA datapath in `rlx-ten-vad-fpga`: the RTL is checked
//! against vectors this module produces, so "the hardware is correct" reduces
//! to "the hardware agrees with this".
//!
//! # Numeric format
//!
//! Chosen by measurement, not taste — see the table in
//! `crates/rlx-ten-vad-fpga/README.md`:
//!
//! | | format | why |
//! |---|---|---|
//! | weights | `i16`, per-tensor power-of-two scale | int8 costs 9 decision flips; ternary costs 20–114 |
//! | activations | `i32`, Q15 | `z1` reaches 177, so 8 integer bits are live |
//! | accumulator | `i64` | 144 terms of (Q15 × Q15) needs 44 bits |
//! | sigmoid/tanh | 1025-point Q15 LUT over [0, 16] | clamping at 8 costs 6.9e-3 |
//!
//! Against the published ONNX model over the 250-frame reference: `max|Δ|`
//! 3e-4, zero decision flips. For scale, the shipped Agora binary sits 9.6e-4
//! from that same model.

use crate::fixed_math::rsh;

/// Right-shift applied to LSTM input activations before the gate matmul.
///
/// The gate accumulator is what forces `i64`: measured partial sums need 36
/// bits and an `i32` datapath allows 31. The width comes from the *product*,
/// not the summation — the conv output reaches 18.8 in Q15 units, so it
/// occupies ~20 bits, and 20 + 16 weight bits is already 36.
///
/// Q15 is the wrong scale for that tensor. It reserves 15 bits below 1.0 and
/// then the values run to 18.8, so five bits are spent representing magnitude
/// above unity. Shifting the input down by `k` narrows every product by `k`
/// bits and costs `k` bits of absolute resolution on a post-ReLU tensor whose
/// useful dynamic range is far less than 20 bits.
///
/// `0` is the original behaviour. See `rlx-ten-vad --example int_ranges`.
#[cfg(feature = "narrow-acc")]
pub const LSTM_IN_SHIFT: u32 = 6;
/// Default: no shift, `i64` accumulate, and the accuracy that goes with it.
#[cfg(not(feature = "narrow-acc"))]
pub const LSTM_IN_SHIFT: u32 = 0;

/// Extra shift applied to the *input* half only.
///
/// The two operands of a gate are not alike. `h` is a gate output, bounded by
/// 1.0 (Q15, 15 bits); the conv output reaches 18.8, so it occupies ~20. A
/// single shift big enough to tame the wide one throws away precision on the
/// narrow one, and `h` is the recurrent path — its resolution is what the LSTM
/// dynamics ride on.
///
/// So the input takes `LSTM_IN_SHIFT + LSTM_X_EXTRA` and `h` takes only
/// `LSTM_IN_SHIFT`, with the requantisation splitting to match.
pub const LSTM_H_EXTRA: u32 = 0;

/// Gate accumulator width.
///
/// This is the whole point of [`LSTM_IN_SHIFT`]. On a 32-bit core an `i64`
/// multiply-accumulate is a widening multiply plus a two-word add with carry;
/// an `i32` one is a multiply and an add. The LSTMs are 88% of the network's
/// multiply-accumulates, so the difference is most of the network's cost.
///
/// `i32` is sound only while the shift keeps partial sums inside 31 bits.
/// Measured over 497,775 frames of the distillation set: shift 0 needs 36 bits,
/// shift 5 needs exactly 31, shift 6 needs 30. Six is the default, for the bit
/// of margin — the *analytical* worst case is 41 bits, so this rests on the
/// measured distribution rather than on construction, which is why it is opt-in.
#[cfg(feature = "narrow-acc")]
type Acc = i32;
#[cfg(not(feature = "narrow-acc"))]
type Acc = i64;

/// Widest values seen while running, for deciding accumulator width.
///
/// The LSTM accumulates `i16 x Q15` products, so a single product needs 31 bits
/// and 144 of them need ~39 — which is why the hot loop is `i64` on a 32-bit
/// core, at roughly twice the instruction count of an `i32` one. Whether that
/// is *necessary* is an empirical question about the trained weights, not a
/// worst-case one, and this answers it.
#[cfg(feature = "range-probe")]
#[derive(Debug, Default, Clone, Copy)]
pub struct Ranges {
    /// Largest |partial sum| at any point inside a gate accumulation.
    pub max_partial: i64,
    /// Largest |gate pre-activation| after the shift back to Q15.
    pub max_pre: i32,
    /// Largest |cell state|.
    pub max_cell: i32,
    /// Largest |activation| fed into a matmul.
    pub max_act: i32,
}

#[cfg(feature = "range-probe")]
impl Ranges {
    #[inline]
    fn note_partial(&mut self, acc: i64, act: i32) {
        self.max_partial = self.max_partial.max(acc.abs());
        self.max_act = self.max_act.max(act.abs());
    }

    /// Bits needed to hold `max_partial` signed, i.e. the accumulator width the
    /// observed data actually demands.
    pub fn partial_bits(&self) -> u32 {
        64 - self.max_partial.unsigned_abs().leading_zeros()
    }
}
use crate::fixed_tables as t;
use crate::{CONTEXT_FRAMES, FEATURE_LEN, HIDDEN};

/// 1.0 in the activation format.
pub const ONE: i32 = 1 << t::ACT_FRAC;

const CH: usize = 16;
const W0: usize = FEATURE_LEN - 2;
const W1: usize = (W0 - 3) / 2 + 1;
const W2: usize = (W1 + 2 - 3) / 2 + 1;
const W3: usize = (W2 + 1 - 3) / 2 + 1;
const FLAT: usize = W3 * CH;
const GATES: usize = 4 * HIDDEN;
const DENSE: usize = 32;

/// Borrow one tensor out of the blob. Free-standing rather than a method: the
/// returned slice must not borrow `self`, or the scratch buffers could not be
/// written in the same scope.
#[inline]
fn tensor(w: &[i16], spec: (usize, usize, u32)) -> (&[i16], u32) {
    let (off, len, frac) = spec;
    (&w[off..off + len], frac)
}

/// Lift a bias from its own scale into an accumulator at `acc_frac`.
#[inline]
fn bias_at(b: i16, b_frac: u32, acc_frac: u32) -> i64 {
    let b = i64::from(b);
    if acc_frac >= b_frac {
        b << (acc_frac - b_frac)
    } else {
        i64::from(rsh(b, b_frac - acc_frac))
    }
}

/// Interpolate a Q15 LUT at Q15 `x`. `odd` selects tanh's `f(−x) = −f(x)` over
/// sigmoid's `f(−x) = 1 − f(x)`.
#[inline]
fn lut(tab: &[i16; t::LUT_N + 1], x: i32, odd: bool) -> i32 {
    crate::fixed_math::lut_q15(tab, x, t::LUT_SHIFT, odd, ONE)
}

#[inline]
fn sigmoid(x: i32) -> i32 {
    lut(&t::SIGMOID_Q15, x, false)
}

#[inline]
fn tanh(x: i32) -> i32 {
    lut(&t::TANH_Q15, x, true)
}

/// The int16 weight blob, aligned so it can be read as `[i16]` in place.
#[repr(align(2))]
struct Align2<T>(T);

static Q15_BLOB: Align2<[u8; 2 * t::BLOB_LEN]> =
    Align2(*include_bytes!("../weights/ten_vad_q15.bin"));

/// The embedded int16 weights, borrowed from flash — never copied to RAM.
pub fn embedded() -> &'static [i16] {
    // SAFETY: `Align2` gives 2-byte alignment, the length is an exact multiple
    // of 2, and `i16` has no invalid bit patterns. Little-endian only, which
    // every target this crate builds for is.
    #[cfg(target_endian = "little")]
    unsafe {
        core::slice::from_raw_parts(Q15_BLOB.0.as_ptr().cast::<i16>(), t::BLOB_LEN)
    }
}

/// Per-frame scratch. Held in the struct rather than on the stack: ~6 kB of
/// locals would overflow a default 4 kB MCU task stack.
struct Scratch {
    c0: [i32; W0],
    a: [[i32; W0]; CH],
    p1: [[i32; W1]; CH],
    s1: [[i32; W2]; CH],
    s2: [[i32; W3]; CH],
    dw: [[i32; W2]; CH],
    flat: [i32; FLAT],
    z: [i32; GATES],
    /// LSTM inputs pre-shifted by [`LSTM_IN_SHIFT`]. Every gate row reads the
    /// same values, so the shift happens once per element rather than once per
    /// (row, element) — 144 shifts a step instead of 36,864.
    xs: [i32; FLAT + HIDDEN],
    d: [i32; DENSE],
}

impl Scratch {
    const fn new() -> Self {
        Self {
            c0: [0; W0],
            a: [[0; W0]; CH],
            p1: [[0; W1]; CH],
            s1: [[0; W2]; CH],
            s2: [[0; W3]; CH],
            dw: [[0; W2]; CH],
            flat: [0; FLAT],
            z: [0; GATES],
            xs: [0; FLAT + HIDDEN],
            d: [0; DENSE],
        }
    }
}

/// Integer-only network, carrying its LSTM state between frames.
pub struct FixedNet<'a> {
    w: &'a [i16],
    h1: [i32; HIDDEN],
    c1: [i32; HIDDEN],
    h2: [i32; HIDDEN],
    c2: [i32; HIDDEN],
    s: Scratch,
    #[cfg(feature = "range-probe")]
    ranges: Ranges,
}

impl FixedNet<'static> {
    /// Borrow the weights compiled into the binary.
    pub fn embedded() -> Self {
        Self::new(embedded())
    }
}

impl<'a> FixedNet<'a> {
    pub fn new(w: &'a [i16]) -> Self {
        debug_assert_eq!(w.len(), t::BLOB_LEN);
        Self {
            w,
            h1: [0; HIDDEN],
            c1: [0; HIDDEN],
            h2: [0; HIDDEN],
            c2: [0; HIDDEN],
            s: Scratch::new(),
            #[cfg(feature = "range-probe")]
            ranges: Ranges::default(),
        }
    }

    /// Widest values seen since construction. See [`Ranges`].
    #[cfg(feature = "range-probe")]
    pub fn ranges(&self) -> Ranges {
        self.ranges
    }

    /// Zero the LSTM state.
    pub fn reset(&mut self) {
        self.h1 = [0; HIDDEN];
        self.c1 = [0; HIDDEN];
        self.h2 = [0; HIDDEN];
        self.c2 = [0; HIDDEN];
    }

    /// Score one `[CONTEXT_FRAMES, FEATURE_LEN]` stack of Q15 features.
    ///
    /// Returns the speech probability in Q15, i.e. `0..=ONE`.
    pub fn forward(&mut self, feat: &[i32]) -> i32 {
        debug_assert_eq!(feat.len(), CONTEXT_FRAMES * FEATURE_LEN);
        let (w, af) = (self.w, t::ACT_FRAC);

        // conv0: 3×3 valid conv to one channel, then a 1×1 projection to CH.
        let (dw, dwf) = tensor(w, t::CONV0_DEPTHWISE_WEIGHT);
        for x in 0..W0 {
            let mut acc = 0i64;
            for ki in 0..CONTEXT_FRAMES {
                for kj in 0..3 {
                    acc += i64::from(feat[ki * FEATURE_LEN + x + kj]) * i64::from(dw[ki * 3 + kj]);
                }
            }
            self.s.c0[x] = rsh(acc, dwf);
        }
        let (pw, pwf) = tensor(w, t::CONV0_POINTWISE_WEIGHT);
        let (b, bf) = tensor(w, t::CONV0_BIAS);
        for ch in 0..CH {
            let bias = bias_at(b[ch], bf, af + pwf);
            for x in 0..W0 {
                let acc = i64::from(self.s.c0[x]) * i64::from(pw[ch]) + bias;
                self.s.a[ch][x] = rsh(acc, pwf).max(0);
            }
        }

        // max-pool k=3 s=2
        for ch in 0..CH {
            for x in 0..W1 {
                let r = &self.s.a[ch];
                self.s.p1[ch][x] = r[x * 2].max(r[x * 2 + 1]).max(r[x * 2 + 2]);
            }
        }

        self.separable::<W1, W2>(
            1,
            t::SEP1_DEPTHWISE_WEIGHT,
            t::SEP1_POINTWISE_WEIGHT,
            t::SEP1_BIAS,
        );
        self.separable::<W2, W3>(
            0,
            t::SEP2_DEPTHWISE_WEIGHT,
            t::SEP2_POINTWISE_WEIGHT,
            t::SEP2_BIAS,
        );

        // `[CH, W3]` → position-major `[W3, CH]`, matching the graph's
        // Squeeze → Transpose(0,2,1) → Reshape.
        for x in 0..W3 {
            for ch in 0..CH {
                self.s.flat[x * CH + ch] = self.s.s2[ch][x];
            }
        }

        let flat = self.s.flat;
        self.step::<FLAT>(
            &flat,
            true,
            t::LSTM1_WEIGHT_IH,
            t::LSTM1_WEIGHT_HH,
            t::LSTM1_BIAS,
        );
        let h1 = self.h1;
        self.step::<HIDDEN>(
            &h1,
            false,
            t::LSTM2_WEIGHT_IH,
            t::LSTM2_WEIGHT_HH,
            t::LSTM2_BIAS,
        );

        // concat(h2, h1) → dense(128→32) → relu → dense(32→1) → sigmoid
        let (w1, w1f) = tensor(w, t::DENSE1_WEIGHT);
        let (b1, b1f) = tensor(w, t::DENSE1_BIAS);
        for j in 0..DENSE {
            let mut acc = bias_at(b1[j], b1f, af + w1f);
            for i in 0..HIDDEN {
                acc += i64::from(self.h2[i]) * i64::from(w1[i * DENSE + j]);
                acc += i64::from(self.h1[i]) * i64::from(w1[(HIDDEN + i) * DENSE + j]);
            }
            self.s.d[j] = rsh(acc, w1f).max(0);
        }
        let (w2, w2f) = tensor(w, t::DENSE2_WEIGHT);
        let (b2, b2f) = tensor(w, t::DENSE2_BIAS);
        let mut acc = bias_at(b2[0], b2f, af + w2f);
        for j in 0..DENSE {
            acc += i64::from(self.s.d[j]) * i64::from(w2[j]);
        }
        sigmoid(rsh(acc, w2f))
    }

    /// depthwise k=3 s=2 with `(pad, …)` left padding → pointwise + bias → ReLU.
    fn separable<const IN: usize, const OUT: usize>(
        &mut self,
        pad: usize,
        dws: (usize, usize, u32),
        pws: (usize, usize, u32),
        bs: (usize, usize, u32),
    ) {
        let (w, af) = (self.w, t::ACT_FRAC);
        let (dw, dwf) = tensor(w, dws);
        for ch in 0..CH {
            for o in 0..OUT {
                let mut acc = 0i64;
                for k in 0..3 {
                    let i = o * 2 + k;
                    if i >= pad && i - pad < IN {
                        let x = if IN == W1 {
                            self.s.p1[ch][i - pad]
                        } else {
                            self.s.s1[ch][i - pad]
                        };
                        acc += i64::from(x) * i64::from(dw[ch * 3 + k]);
                    }
                }
                self.s.dw[ch][o] = rsh(acc, dwf);
            }
        }
        let (pw, pwf) = tensor(w, pws);
        let (b, bf) = tensor(w, bs);
        for oc in 0..CH {
            let bias = bias_at(b[oc], bf, af + pwf);
            for o in 0..OUT {
                let mut acc = bias;
                for ic in 0..CH {
                    acc += i64::from(self.s.dw[ic][o]) * i64::from(pw[oc * CH + ic]);
                }
                let v = rsh(acc, pwf).max(0);
                if OUT == W2 {
                    self.s.s1[oc][o] = v;
                } else {
                    self.s.s2[oc][o] = v;
                }
            }
        }
    }

    /// One LSTM step, gate order `i, f, g, o`. `first` selects which layer's
    /// state to advance.
    fn step<const IN: usize>(
        &mut self,
        x: &[i32; IN],
        first: bool,
        ihs: (usize, usize, u32),
        hhs: (usize, usize, u32),
        bs: (usize, usize, u32),
    ) {
        let (w, af) = (self.w, t::ACT_FRAC);
        let (ih, ihf) = tensor(w, ihs);
        let (hh, hhf) = tensor(w, hhs);
        let (b, bf) = tensor(w, bs);
        let h = if first { &self.h1 } else { &self.h2 };
        // Pre-shift once. `x` is the conv output (first layer) or the previous
        // layer's h; both are read by all `GATES` rows.
        for (dst, &v) in self.s.xs[..IN].iter_mut().zip(x.iter()) {
            *dst = rsh(i64::from(v), LSTM_IN_SHIFT);
        }
        for (dst, &v) in self.s.xs[IN..IN + HIDDEN].iter_mut().zip(h.iter()) {
            *dst = rsh(i64::from(v), LSTM_IN_SHIFT + LSTM_H_EXTRA);
        }
        let xs = &self.s.xs;

        for r in 0..GATES {
            // Both projections share one scale (the generator forces it), so
            // they land in a single accumulator with no rescale between them.
            debug_assert_eq!(ihf, hhf);
            // The accumulator lives `LSTM_IN_SHIFT` bits lower, so the bias is
            // lifted to the matching scale and the final requantisation shifts
            // by that much less.
            let mut acc = bias_at(b[r], bf, af + ihf - LSTM_IN_SHIFT) as Acc;
            for (i, &xi) in xs[..IN].iter().enumerate() {
                // The input half sits `LSTM_X_EXTRA` bits lower, so its
                // products are lifted back before joining `h`'s.
                acc += Acc::from(ih[r * IN + i]) * xi as Acc;
                #[cfg(feature = "range-probe")]
                self.ranges.note_partial(acc as i64, xi);
            }
            for (i, &hi) in xs[IN..IN + HIDDEN].iter().enumerate() {
                // `h` sits `LSTM_H_EXTRA` bits lower; lift its products to the
                // input half's scale before they join.
                acc += (Acc::from(hh[r * HIDDEN + i]) * hi as Acc) << LSTM_H_EXTRA;
                #[cfg(feature = "range-probe")]
                self.ranges.note_partial(acc as i64, hi);
            }
            // `Acc` is `i32` under `narrow-acc` and `i64` otherwise, so this
            // widening is real in one configuration and a no-op in the other.
            #[allow(clippy::useless_conversion)]
            let wide = i64::from(acc);
            self.s.z[r] = rsh(wide, ihf - LSTM_IN_SHIFT);
            #[cfg(feature = "range-probe")]
            {
                self.ranges.max_pre = self.ranges.max_pre.max(self.s.z[r].abs());
            }
        }
        let (h, c) = if first {
            (&mut self.h1, &mut self.c1)
        } else {
            (&mut self.h2, &mut self.c2)
        };
        for k in 0..HIDDEN {
            let ig = sigmoid(self.s.z[k]);
            let fg = sigmoid(self.s.z[HIDDEN + k]);
            let gg = tanh(self.s.z[2 * HIDDEN + k]);
            let og = sigmoid(self.s.z[3 * HIDDEN + k]);
            let nc = rsh(
                i64::from(fg) * i64::from(c[k]) + i64::from(ig) * i64::from(gg),
                af,
            );
            c[k] = nc;
            #[cfg(feature = "range-probe")]
            {
                self.ranges.max_cell = self.ranges.max_cell.max(nc.abs());
            }
            h[k] = rsh(i64::from(og) * i64::from(tanh(nc)), af);
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn blob_length_matches_the_layout() {
        let want: usize = t::LAYOUT.iter().map(|&(_, _, n, _)| n).sum();
        assert_eq!(want, t::BLOB_LEN);
        assert_eq!(embedded().len(), t::BLOB_LEN);
        // Every tensor lands where its accessor constant claims.
        let mut off = 0;
        for &(name, o, n, _) in &t::LAYOUT {
            assert_eq!(o, off, "{name} offset");
            off += n;
        }
    }

    #[test]
    fn luts_track_the_real_functions() {
        for i in -20 * ONE..20 * ONE {
            if i % 97 != 0 {
                continue;
            }
            let x = i as f64 / f64::from(ONE);
            let s = f64::from(sigmoid(i)) / f64::from(ONE);
            let t_ = f64::from(tanh(i)) / f64::from(ONE);
            assert!(
                (s - 1.0 / (1.0 + (-x).exp())).abs() < 2e-4,
                "sigmoid at {x}: {s}"
            );
            assert!((t_ - x.tanh()).abs() < 2e-4, "tanh at {x}: {t_}");
        }
    }

    #[test]
    fn probabilities_are_in_range_and_state_advances() {
        let mut net = FixedNet::embedded();
        let feat: alloc::vec::Vec<i32> = (0..CONTEXT_FRAMES * FEATURE_LEN)
            .map(|i| ((i as f32 * 0.31).sin() * 1.7 * ONE as f32) as i32)
            .collect();
        let a = net.forward(&feat);
        let b = net.forward(&feat);
        assert!((0..=ONE).contains(&a), "probability {a} out of range");
        assert_ne!(a, b, "LSTM state did not advance");
        net.reset();
        assert_eq!(a, net.forward(&feat), "reset did not restore state");
    }
}
