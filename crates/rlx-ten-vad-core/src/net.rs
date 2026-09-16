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

//! Scalar CRNN forward — allocation-free, and the reference the MCU firmware
//! and the FPGA datapath are both checked against.
//!
//! Structurally identical to the rlx graph in `rlx-ten-vad::model`, including
//! the choice to concatenate `x‖h` so the two gate projections are one matmul.
//! Held to that graph bit-for-bit by `net_matches_the_rlx_graph`.

use crate::math;
use crate::weights::NetWeights;
use crate::{CONTEXT_FRAMES, FEATURE_LEN, HIDDEN};

/// Conv channels after the first pointwise projection.
const CH: usize = 16;
/// Widths down the conv stack: 3×3 valid over 41, max-pool, then two stride-2s.
const W0: usize = FEATURE_LEN - 2; // 39
const W1: usize = (W0 - 3) / 2 + 1; // 19
const W2: usize = (W1 + 2 - 3) / 2 + 1; // 10
const W3: usize = (W2 + 1 - 3) / 2 + 1; // 5
/// Flattened conv output feeding the first LSTM.
const FLAT: usize = W3 * CH; // 80
const GATES: usize = 4 * HIDDEN;
const DENSE: usize = 32;

/// One LSTM layer's state. `Default` is hand-written: arrays longer than 32
/// do not implement it.
#[derive(Clone, Copy)]
struct Cell {
    h: [f32; HIDDEN],
    c: [f32; HIDDEN],
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            h: [0.0; HIDDEN],
            c: [0.0; HIDDEN],
        }
    }
}

/// The network, carrying its own LSTM state between frames.
/// Widest activation magnitudes seen, in the f32 net's own units.
///
/// The integer net's accumulator width is set by these: measured over the
/// distillation set, gate pre-activations reach 189.5 and the cell 326.9, while
/// sigmoid and tanh have saturated by ~16. Everything above that is range the
/// deployed LUT discards, paid for in accumulator bits. See
/// `rlx_ten_vad_core::fixed::Ranges` for the integer side.
#[cfg(feature = "range-probe")]
#[derive(Debug, Default, Clone, Copy)]
pub struct FloatRanges {
    pub max_pre: f32,
    pub max_cell: f32,
    pub max_flat: f32,
}

pub struct Net<'a> {
    w: NetWeights<'a>,
    #[cfg(feature = "range-probe")]
    ranges: FloatRanges,
    l1: Cell,
    l2: Cell,
    /// The conv stack's output from the last [`Net::forward`] — what the
    /// recurrent half actually sees. Kept because it is the one activation
    /// whose sparsity is exploitable: it is post-ReLU, and it multiplies the
    /// single largest weight matrix in the model.
    flat: [f32; FLAT],
}

impl<'a> Net<'a> {
    pub fn new(w: NetWeights<'a>) -> Self {
        Self {
            w,
            l1: Cell::default(),
            l2: Cell::default(),
            flat: [0.0; FLAT],
            #[cfg(feature = "range-probe")]
            ranges: FloatRanges::default(),
        }
    }

    /// Widest activations seen since construction. See [`FloatRanges`].
    #[cfg(feature = "range-probe")]
    pub fn ranges(&self) -> FloatRanges {
        self.ranges
    }

    /// The conv stack's output from the last [`Net::forward`], `[FLAT]`.
    pub fn conv_out(&self) -> &[f32] {
        &self.flat
    }

    /// Zero the LSTM state — a new utterance, or the periodic reset.
    pub fn reset(&mut self) {
        self.l1 = Cell::default();
        self.l2 = Cell::default();
    }

    /// Score one `[CONTEXT_FRAMES, FEATURE_LEN]` feature stack, row-major.
    pub fn forward(&mut self, feat: &[f32]) -> f32 {
        debug_assert_eq!(feat.len(), CONTEXT_FRAMES * FEATURE_LEN);
        let w = &self.w;

        // conv0: a genuine 3×3 valid conv over (context, feature) → one channel,
        // then a 1×1 projection to CH with bias, then ReLU.
        let dw = w.conv0_depthwise;
        let mut c0 = [0.0f32; W0];
        for (x, slot) in c0.iter_mut().enumerate() {
            let mut acc = 0.0;
            for ki in 0..CONTEXT_FRAMES {
                for kj in 0..3 {
                    acc += feat[ki * FEATURE_LEN + x + kj] * dw[ki * 3 + kj];
                }
            }
            *slot = acc;
        }
        let mut a = [[0.0f32; W0]; CH];
        for ch in 0..CH {
            let (p, b) = (w.conv0_pointwise[ch], w.conv0_bias[ch]);
            for x in 0..W0 {
                a[ch][x] = (c0[x] * p + b).max(0.0);
            }
        }

        // max-pool k=3 s=2
        let mut p1 = [[0.0f32; W1]; CH];
        for ch in 0..CH {
            for x in 0..W1 {
                let s = &a[ch][x * 2..x * 2 + 3];
                p1[ch][x] = s[0].max(s[1]).max(s[2]);
            }
        }

        // Two separable stages: depthwise k=3 s=2 (padded), pointwise, ReLU.
        let mut s1 = [[0.0f32; W2]; CH];
        separable::<W1, W2>(
            &p1,
            (1, 1),
            w.sep1_depthwise,
            w.sep1_pointwise,
            w.sep1_bias,
            &mut s1,
        );
        let mut s2 = [[0.0f32; W3]; CH];
        separable::<W2, W3>(
            &s1,
            (0, 1),
            w.sep2_depthwise,
            w.sep2_pointwise,
            w.sep2_bias,
            &mut s2,
        );

        // `[CH, W3]` → `[W3, CH]` flattened: position-major, as the graph's
        // `Squeeze → Transpose(0,2,1) → Reshape` produces.
        let flat = &mut self.flat;
        for x in 0..W3 {
            for ch in 0..CH {
                flat[x * CH + ch] = s2[ch][x];
            }
        }
        let flat = *flat;

        #[cfg(feature = "range-probe")]
        for &v in &flat {
            self.ranges.max_flat = self.ranges.max_flat.max(v.abs());
        }
        let h1 = step::<FLAT>(
            &mut self.l1,
            #[cfg(feature = "range-probe")]
            &mut self.ranges,
            &flat,
            w.lstm1_weight_ih,
            w.lstm1_weight_hh,
            w.lstm1_bias,
        );
        let h2 = step::<HIDDEN>(
            &mut self.l2,
            #[cfg(feature = "range-probe")]
            &mut self.ranges,
            &h1,
            w.lstm2_weight_ih,
            w.lstm2_weight_hh,
            w.lstm2_bias,
        );

        // concat(h2, h1) → dense(128→32) → relu → dense(32→1) → sigmoid
        let mut d = [0.0f32; DENSE];
        for (j, slot) in d.iter_mut().enumerate() {
            let mut acc = w.dense1_bias[j];
            for i in 0..HIDDEN {
                acc += h2[i] * w.dense1_weight[i * DENSE + j];
                acc += h1[i] * w.dense1_weight[(HIDDEN + i) * DENSE + j];
            }
            *slot = acc.max(0.0);
        }
        let mut z = w.dense2_bias[0];
        for j in 0..DENSE {
            z += d[j] * w.dense2_weight[j];
        }
        math::sigmoid(z)
    }
}

/// depthwise k=3 s=2 with `(left, right)` zero padding → pointwise + bias → ReLU.
fn separable<const IN: usize, const OUT: usize>(
    x: &[[f32; IN]; CH],
    pad: (usize, usize),
    dw: &[f32],
    pw: &[f32],
    bias: &[f32],
    out: &mut [[f32; OUT]; CH],
) {
    let mut d = [[0.0f32; OUT]; CH];
    for ch in 0..CH {
        for o in 0..OUT {
            let mut acc = 0.0;
            for k in 0..3 {
                // Position in the padded row; outside is zero.
                let i = o * 2 + k;
                if i >= pad.0 && i - pad.0 < IN {
                    acc += x[ch][i - pad.0] * dw[ch * 3 + k];
                }
            }
            d[ch][o] = acc;
        }
    }
    for oc in 0..CH {
        for o in 0..OUT {
            let mut acc = bias[oc];
            for ic in 0..CH {
                acc += d[ic][o] * pw[oc * CH + ic];
            }
            out[oc][o] = acc.max(0.0);
        }
    }
}

/// One LSTM cell step, gate order `i, f, g, o`.
fn step<const IN: usize>(
    cell: &mut Cell,
    #[cfg(feature = "range-probe")] probe: &mut FloatRanges,
    x: &[f32; IN],
    w_ih: &[f32],
    w_hh: &[f32],
    bias: &[f32],
) -> [f32; HIDDEN] {
    let mut z = [0.0f32; GATES];
    for (r, slot) in z.iter_mut().enumerate() {
        let mut acc = bias[r];
        for (i, &xi) in x.iter().enumerate() {
            acc += w_ih[r * IN + i] * xi;
        }
        for (i, &hi) in cell.h.iter().enumerate() {
            acc += w_hh[r * HIDDEN + i] * hi;
        }
        *slot = acc;
        #[cfg(feature = "range-probe")]
        {
            probe.max_pre = probe.max_pre.max(acc.abs());
        }
    }
    for k in 0..HIDDEN {
        let i_g = math::sigmoid(z[k]);
        let f_g = math::sigmoid(z[HIDDEN + k]);
        let g_g = math::tanh(z[2 * HIDDEN + k]);
        let o_g = math::sigmoid(z[3 * HIDDEN + k]);
        cell.c[k] = f_g * cell.c[k] + i_g * g_g;
        cell.h[k] = o_g * math::tanh(cell.c[k]);
        #[cfg(feature = "range-probe")]
        {
            probe.max_cell = probe.max_cell.max(cell.c[k].abs());
        }
    }
    cell.h
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    const _: () = assert!(W0 == 39 && W1 == 19 && W2 == 10 && W3 == 5 && FLAT == 80);

    #[test]
    fn probabilities_are_in_range_and_state_advances() {
        let mut net = Net::new(crate::weights::embedded_net());
        let feat: Vec<f32> = (0..CONTEXT_FRAMES * FEATURE_LEN)
            .map(|i| (i as f32 * 0.31).sin() * 1.7)
            .collect();
        let a = net.forward(&feat);
        let b = net.forward(&feat);
        assert!((0.0..=1.0).contains(&a), "probability {a} out of range");
        assert!((a - b).abs() > 1e-9, "LSTM state did not advance");
        net.reset();
        let c = net.forward(&feat);
        assert!(
            (a - c).abs() < 1e-9,
            "reset did not restore the initial state"
        );
    }
}
