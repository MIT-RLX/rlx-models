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

//! Gray-coded frame gaps (`tada.utils.gray_code`).
//!
//! TADA's diffusion head emits the acoustic latent and the token's *duration*
//! in one vector. The duration is carried as two Gray-coded integers — frames
//! before the token and frames after — mapped to ±1 floats. Gray coding means
//! adjacent durations differ in a single bit, so the continuous flow-matching
//! field does not have to cross a multi-bit cliff to move a token by one frame.

/// `gray = v ^ (v >> 1)`.
fn to_gray(v: u32) -> u32 {
    v ^ (v >> 1)
}

/// Inverse of [`to_gray`], over the 32-bit domain upstream uses.
fn from_gray(mut g: i64) -> i64 {
    let mut shift = 1;
    while shift < 32 {
        g ^= g >> shift;
        shift <<= 1;
    }
    g
}

/// Encode `frames` as `num_bits` Gray-code bits in `{-1.0, +1.0}`, most
/// significant bit first. Values are clamped into `[0, 2^num_bits - 1]`.
pub fn encode(frames: u32, num_bits: usize) -> Vec<f32> {
    let max = (1u32 << num_bits) - 1;
    let gray = to_gray(frames.min(max));
    (0..num_bits)
        .map(|i| {
            let bit = (gray >> (num_bits - 1 - i)) & 1;
            bit as f32 * 2.0 - 1.0
        })
        .collect()
}

/// Decode a `±1` (or noisy real-valued) Gray-code vector back to a frame count.
///
/// The flow-matching head does not emit exactly `±1`, so each slot is
/// thresholded with upstream's `((b + 1) / 2).round()`. That expression is
/// **not clamped to `{0, 1}`** there, and it is not clamped here either: an
/// output of, say, `2.6` rounds to `2` and is *added* into the Gray integer
/// rather than contributing a single bit. That looks like a bug, but it is
/// observable behavior of the reference — a solve that drifts out of range
/// lands on a specific wrong duration, not a saturated one — and reproducing it
/// is the difference between a port and a rewrite. Callers are expected to
/// bound the result before using it as an embedding index; upstream instead
/// raises on the out-of-range lookup.
pub fn decode(bits: &[f32], num_bits: usize) -> u32 {
    debug_assert!(bits.len() >= num_bits);
    let mut gray = 0i64;
    for i in 0..num_bits {
        let b = ((bits[i] + 1.0) / 2.0).round() as i64;
        // `+=`, matching upstream — `|=` would silently agree only while every
        // slot really is a bit.
        gray += b << (num_bits - 1 - i);
    }
    from_gray(gray).max(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_value_in_range() {
        for bits in [4usize, 8, 10] {
            for v in 0..(1u32 << bits) {
                assert_eq!(decode(&encode(v, bits), bits), v, "bits={bits} v={v}");
            }
        }
    }

    #[test]
    fn adjacent_values_differ_in_one_bit() {
        let bits = 8;
        for v in 0..255u32 {
            let a = encode(v, bits);
            let b = encode(v + 1, bits);
            let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            assert_eq!(diff, 1, "v={v}");
        }
    }

    #[test]
    fn out_of_range_clamps_to_max() {
        assert_eq!(decode(&encode(9999, 8), 8), 255);
    }

    #[test]
    fn noisy_bits_snap_at_zero() {
        let clean = encode(37, 8);
        let noisy: Vec<f32> = clean.iter().map(|&b| b * 0.61 + 0.2).collect();
        assert_eq!(decode(&noisy, 8), 37);
    }

    /// Reproduces `decode_gray_code_to_time`'s unclamped rounding: a slot that
    /// lands outside `[-1, 1]` contributes a multi-valued term, not a bit.
    #[test]
    fn out_of_range_slots_are_added_not_clamped() {
        // round((2.56722 + 1) / 2) = 2 in the low slot; round((-0.85 + 1) / 2) = 0
        // in the high slot → gray integer 2 → binary 3.
        assert_eq!(decode(&[-0.85147, 2.56722], 2), 3);
        // Clamping each slot to {0, 1} would give gray 1 → binary 1 instead.
        assert_ne!(decode(&[-0.85147, 2.56722], 2), 1);
    }

    /// Deeply out-of-range slots go through signed arithmetic exactly as
    /// torch's `.long()` path does — `round(-4)` twice gives Gray `-12`, whose
    /// 32-bit un-Gray is 13. Nonsense, but the *same* nonsense, and bounded by
    /// the caller before it reaches a duration embedding.
    #[test]
    fn strongly_negative_slots_follow_the_reference_signed_path() {
        assert_eq!(decode(&[-9.0, -9.0], 2), 13);
    }

    #[test]
    fn known_gray_encoding() {
        // 5 = 0b101 → gray 0b111; MSB-first over 4 bits → [0,1,1,1] → -1,1,1,1.
        assert_eq!(encode(5, 4), vec![-1.0, 1.0, 1.0, 1.0]);
    }
}
