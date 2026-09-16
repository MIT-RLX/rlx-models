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

//! The 96-bit NeuralHash value: bit packing, hex formatting, Hamming distance.
//!
//! The reference (`nnhash.py`) derives the hash from the 96 projected scores
//! with a binary step and packs the bit string big-endian:
//!
//! ```python
//! hash_bits = ''.join(['1' if it >= 0 else '0' for it in hash_output])
//! hash_hex  = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)
//! ```
//!
//! `int(bits, 2)` makes `scores[0]` the **most significant** bit, so byte `i`
//! holds scores `8i..8i+8` with score `8i` in bit 7. The threshold is `>= 0`
//! (not `> 0`) — an exactly-zero score sets the bit.

use anyhow::{Result, bail, ensure};
use std::fmt;
use std::str::FromStr;

/// Number of hash bits (rows of the seed matrix).
pub const HASH_BITS: usize = 96;
/// Packed hash width in bytes.
pub const HASH_BYTES: usize = HASH_BITS / 8;

/// A 96-bit NeuralHash, stored big-endian (bit 0 = MSB of byte 0).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NeuralHash([u8; HASH_BYTES]);

impl NeuralHash {
    /// Apply the binary step (`score >= 0`) to the 96 projected scores.
    pub fn from_scores(scores: &[f32]) -> Result<Self> {
        ensure!(
            scores.len() == HASH_BITS,
            "neuralhash: expected {HASH_BITS} scores, got {}",
            scores.len()
        );
        let mut bytes = [0u8; HASH_BYTES];
        for (i, &s) in scores.iter().enumerate() {
            // NaN compares false against every ordering, so it lands on the
            // `< 0` branch here — same as Python's `it >= 0`.
            if s >= 0.0 {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        Ok(Self(bytes))
    }

    /// Build from an explicit bit sequence (`true` = 1), MSB first.
    pub fn from_bits(bits: &[bool]) -> Result<Self> {
        ensure!(
            bits.len() == HASH_BITS,
            "neuralhash: expected {HASH_BITS} bits, got {}",
            bits.len()
        );
        let mut bytes = [0u8; HASH_BYTES];
        for (i, &b) in bits.iter().enumerate() {
            if b {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        Ok(Self(bytes))
    }

    /// Wrap 12 already-packed big-endian bytes.
    pub const fn from_bytes(bytes: [u8; HASH_BYTES]) -> Self {
        Self(bytes)
    }

    /// Parse the 24-character lowercase/uppercase hex form.
    pub fn from_hex(s: &str) -> Result<Self> {
        let s = s.trim();
        ensure!(
            s.len() == HASH_BYTES * 2,
            "neuralhash: expected {} hex chars, got {} ({s:?})",
            HASH_BYTES * 2,
            s.len()
        );
        let mut bytes = [0u8; HASH_BYTES];
        for (i, b) in bytes.iter_mut().enumerate() {
            let hi = hex_nibble(s.as_bytes()[2 * i])?;
            let lo = hex_nibble(s.as_bytes()[2 * i + 1])?;
            *b = (hi << 4) | lo;
        }
        Ok(Self(bytes))
    }

    /// The packed big-endian bytes.
    pub const fn as_bytes(&self) -> &[u8; HASH_BYTES] {
        &self.0
    }

    /// Bit `i` (0 = first score = most significant).
    pub fn bit(&self, i: usize) -> bool {
        assert!(i < HASH_BITS, "bit index {i} out of range");
        self.0[i / 8] >> (7 - (i % 8)) & 1 == 1
    }

    /// All 96 bits, MSB first.
    pub fn bits(&self) -> [bool; HASH_BITS] {
        std::array::from_fn(|i| self.bit(i))
    }

    /// The 24-character lowercase hex form printed by `nnhash.py`.
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(HASH_BYTES * 2);
        for b in self.0 {
            s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
            s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
        }
        s
    }

    /// Number of differing bits (0..=96).
    pub fn hamming(&self, other: &Self) -> u32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Whether two hashes are within `max_distance` bits of each other.
    ///
    /// NeuralHash is a *perceptual* hash: visually similar images produce
    /// nearby, not identical, codes. Apple's own client-side matching used
    /// exact equality after a separate blinding step, so there is no
    /// canonical threshold here — pick one from measured distributions on
    /// your own data rather than assuming a default.
    pub fn matches(&self, other: &Self, max_distance: u32) -> bool {
        self.hamming(other) <= max_distance
    }
}

fn hex_nibble(c: u8) -> Result<u8> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => bail!("neuralhash: invalid hex character {:?}", c as char),
    }
}

impl fmt::Display for NeuralHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for NeuralHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NeuralHash({})", self.to_hex())
    }
}

impl FromStr for NeuralHash {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::from_hex(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msb_first_packing() {
        // Only score 0 positive → the top bit of byte 0 → 0x80...
        let mut scores = vec![-1.0f32; HASH_BITS];
        scores[0] = 1.0;
        let h = NeuralHash::from_scores(&scores).unwrap();
        assert_eq!(h.to_hex(), "800000000000000000000000");

        // Only the last score positive → the low bit of byte 11.
        let mut scores = vec![-1.0f32; HASH_BITS];
        scores[HASH_BITS - 1] = 1.0;
        let h = NeuralHash::from_scores(&scores).unwrap();
        assert_eq!(h.to_hex(), "000000000000000000000001");
    }

    #[test]
    fn zero_score_sets_the_bit() {
        // `nnhash.py` thresholds with `>= 0`, so exact zero is a 1 bit.
        let h = NeuralHash::from_scores(&vec![0.0f32; HASH_BITS]).unwrap();
        assert_eq!(h.to_hex(), "ffffffffffffffffffffffff");
        // -0.0 >= 0.0 is true in IEEE-754 and in Python.
        let h = NeuralHash::from_scores(&vec![-0.0f32; HASH_BITS]).unwrap();
        assert_eq!(h.to_hex(), "ffffffffffffffffffffffff");
    }

    #[test]
    fn nan_scores_clear_the_bit() {
        let h = NeuralHash::from_scores(&vec![f32::NAN; HASH_BITS]).unwrap();
        assert_eq!(h.to_hex(), "000000000000000000000000");
    }

    #[test]
    fn hex_roundtrip() {
        // The example hash from the reference README.
        let h = NeuralHash::from_hex("ab14febaa837b6c1484c35e6").unwrap();
        assert_eq!(h.to_hex(), "ab14febaa837b6c1484c35e6");
        assert_eq!(h.as_bytes()[0], 0xab);
        assert_eq!(h.as_bytes()[11], 0xe6);
        // First nibble 0xa = 1010 → bits 0..4.
        assert_eq!(
            &h.bits()[..4],
            &[true, false, true, false],
            "leading nibble must decode MSB first"
        );
        assert_eq!(NeuralHash::from_bits(&h.bits()).unwrap(), h);
    }

    #[test]
    fn hex_rejects_bad_input() {
        assert!(NeuralHash::from_hex("abc").is_err());
        assert!(NeuralHash::from_hex("zz14febaa837b6c1484c35e6").is_err());
        assert!(NeuralHash::from_hex("AB14FEBAA837B6C1484C35E6").is_ok());
    }

    #[test]
    fn hamming_counts_bits() {
        let a = NeuralHash::from_hex("000000000000000000000000").unwrap();
        let b = NeuralHash::from_hex("ffffffffffffffffffffffff").unwrap();
        assert_eq!(a.hamming(&a), 0);
        assert_eq!(a.hamming(&b), HASH_BITS as u32);
        let c = NeuralHash::from_hex("800000000000000000000001").unwrap();
        assert_eq!(a.hamming(&c), 2);
        assert!(a.matches(&c, 2));
        assert!(!a.matches(&c, 1));
    }

    #[test]
    fn wrong_score_count_is_an_error() {
        assert!(NeuralHash::from_scores(&[0.0; 95]).is_err());
        assert!(NeuralHash::from_scores(&[0.0; 128]).is_err());
    }
}
