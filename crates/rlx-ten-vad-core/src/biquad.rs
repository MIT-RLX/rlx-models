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

//! Cascaded second-order sections (port of TEN-VAD's `biquad.cc`).
//!
//! Direct-form II per section with a per-section gain applied on the way out:
//! `w = x − a1·w1 − a2·w2`, `y = g·(b0·w + b1·w1 + b2·w2)`.

/// One second-order section: numerator, denominator (`a0 = 1`), and gain.
#[derive(Debug, Clone, Copy)]
pub struct Section {
    pub b: [f32; 3],
    pub a: [f32; 3],
    pub g: f32,
}

/// Anti-alias lowpass used before decimating 16 kHz → 4 kHz for pitch
/// estimation (`AUP_PE_{B,A,G}_4KHZ`).
pub const LOWPASS_4KHZ: [Section; 5] = [
    Section {
        b: [1.0, 1.198_825, 1.0],
        a: [1.0, -1.445_267, 5.463_974e-1],
        g: 2.692_541e-1,
    },
    Section {
        b: [1.0, -5.674_614e-1, 1.0],
        a: [1.0, -1.426_72, 6.820_138e-1],
        g: 2.692_541e-1,
    },
    Section {
        b: [1.0, -1.099_061, 1.0],
        a: [1.0, -1.408_255, 8.286_664e-1],
        g: 2.692_541e-1,
    },
    Section {
        b: [1.0, -1.265_846, 1.0],
        a: [1.0, -1.400_909, 9.240_32e-1],
        g: 2.692_541e-1,
    },
    Section {
        b: [1.0, -1.318_849, 1.0],
        a: [1.0, -1.408_242, 9.789_776e-1],
        g: 2.692_541e-1,
    },
];

/// Sections in [`LOWPASS_4KHZ`].
pub const LOWPASS_SECTIONS: usize = 5;

/// A cascade with its per-section delay registers.
///
/// The section count is a const parameter rather than a `Vec` length: a filter
/// cascade is fixed at design time, and this keeps the whole DSP path free of
/// an allocator.
#[derive(Debug, Clone)]
pub struct Biquad<const N: usize> {
    sections: [Section; N],
    state: [[f32; 2]; N],
}

impl<const N: usize> Biquad<N> {
    pub fn new(sections: [Section; N]) -> Self {
        Self {
            sections,
            state: [[0.0; 2]; N],
        }
    }

    pub fn reset(&mut self) {
        for s in &mut self.state {
            *s = [0.0; 2];
        }
    }

    /// Filter `buf` in place through every section, in order.
    pub fn process(&mut self, buf: &mut [f32]) {
        for (sec, w) in self.sections.iter().zip(&mut self.state) {
            for x in buf.iter_mut() {
                let t = *x - sec.a[1] * w[0] - sec.a[2] * w[1];
                *x = sec.g * (sec.b[0] * t + sec.b[1] * w[0] + sec.b[2] * w[1]);
                w[1] = w[0];
                w[0] = t;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowpass_passes_dc_and_kills_nyquist() {
        let mut lo = Biquad::new(LOWPASS_4KHZ);
        let mut dc = vec![1.0f32; 4096];
        lo.process(&mut dc);
        let dc_tail: f32 = dc[3000..].iter().sum::<f32>() / dc[3000..].len() as f32;

        lo.reset();
        // 8 kHz at fs = 16 kHz — well inside the stopband of a 2 kHz-corner filter.
        let mut nyq: Vec<f32> = (0..4096)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        lo.process(&mut nyq);
        let nyq_peak = nyq[3000..].iter().fold(0.0f32, |m, v| m.max(v.abs()));

        assert!(dc_tail > 0.9, "dc gain {dc_tail}");
        assert!(nyq_peak < 1e-3, "nyquist leak {nyq_peak}");
    }

    #[test]
    fn reset_clears_history() {
        let mut lo = Biquad::new(LOWPASS_4KHZ);
        let mut a = vec![1.0f32; 64];
        lo.process(&mut a);
        lo.reset();
        let mut b = vec![1.0f32; 64];
        lo.process(&mut b);
        assert_eq!(a, b);
    }
}
