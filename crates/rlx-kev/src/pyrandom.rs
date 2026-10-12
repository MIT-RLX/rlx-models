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

//! CPython's `random.Random`, bit-for-bit.
//!
//! The kev recipe augments every record with a per-record seeded
//! `random.Random`: option order is shuffled, a "none of the above" option is
//! sometimes inserted or swapped in, a distractor is sometimes added. Those
//! draws happen in a fixed order, so reproducing the training data means
//! reproducing the stream — not merely its distribution. A different stream
//! gives a different (equally valid) dataset, and then any difference in the
//! final metrics is unattributable.
//!
//! So this is MT19937 plus the exact `random()` / `getrandbits()` /
//! `_randbelow()` / `shuffle()` / `choice()` that CPython layers on top,
//! seeded the way `random.seed(int)` seeds. Verified against CPython golden
//! vectors in `tests/pyrandom_parity.rs`.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// MT19937, as used by CPython's `_randommodule.c`.
#[derive(Clone)]
pub struct Mt19937 {
    state: [u32; N],
    index: usize,
}

impl Mt19937 {
    /// `init_genrand`.
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            state[i] = 1_812_433_253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: N }
    }

    /// `init_by_array` — how CPython seeds from an arbitrary integer.
    pub fn from_key(key: &[u32]) -> Self {
        let mut mt = Self::new(19_650_218);
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = N.max(key.len());
        while k > 0 {
            let prev = mt.state[i - 1];
            mt.state[i] = (mt.state[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt.state[0] = mt.state[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        let mut k = N - 1;
        while k > 0 {
            let prev = mt.state[i - 1];
            mt.state[i] = (mt.state[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_566_083_941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt.state[0] = mt.state[N - 1];
                i = 1;
            }
            k -= 1;
        }
        mt.state[0] = 0x8000_0000;
        mt.index = N;
        mt
    }

    /// `genrand_uint32`.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            for i in 0..N {
                let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
                let mut next = self.state[(i + M) % N] ^ (y >> 1);
                if y & 1 != 0 {
                    next ^= MATRIX_A;
                }
                self.state[i] = next;
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

/// CPython's `random.Random`, restricted to what the recipe uses.
#[derive(Clone)]
pub struct PyRandom {
    mt: Mt19937,
}

impl PyRandom {
    /// `random.Random(n)` for a non-negative integer seed.
    ///
    /// CPython converts `abs(n)` to little-endian 32-bit words and calls
    /// `init_by_array`; zero seeds a single zero word.
    pub fn seed_u64(n: u64) -> Self {
        let mut key = Vec::with_capacity(2);
        let mut v = n;
        if v == 0 {
            key.push(0);
        }
        while v > 0 {
            key.push((v & 0xffff_ffff) as u32);
            v >>= 32;
        }
        Self {
            mt: Mt19937::from_key(&key),
        }
    }

    /// `random.Random(source_seed(seed, tag))` — the recipe's per-record seed,
    /// which is the first 8 bytes of `sha256("{seed}:{tag}")` read big-endian.
    pub fn from_source_seed(seed: u64, tag: &str) -> Self {
        Self::seed_u64(source_seed(seed, tag))
    }

    /// `random()` — a 53-bit double from two 32-bit words.
    pub fn random(&mut self) -> f64 {
        let a = self.mt.next_u32() >> 5; // 27 bits
        let b = self.mt.next_u32() >> 6; // 26 bits
        (a as f64 * 67_108_864.0 + b as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `getrandbits(k)` for `1 <= k <= 64`.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        assert!((1..=64).contains(&k), "getrandbits({k}) out of range");
        if k <= 32 {
            return (self.mt.next_u32() >> (32 - k)) as u64;
        }
        // CPython fills 32-bit words low-to-high, masking the top word.
        let lo = self.mt.next_u32() as u64;
        let hi_bits = k - 32;
        let hi = (self.mt.next_u32() >> (32 - hi_bits)) as u64;
        lo | (hi << 32)
    }

    /// `_randbelow(n)` — rejection sampling on `getrandbits(n.bit_length())`.
    ///
    /// The rejection loop is why the stream cannot be reconstructed from a
    /// count of draws: how many words a shuffle consumes depends on the values.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        // `k = n.bit_length()`, NOT `(n-1).bit_length()`: CPython draws
        // `bit_length(n)` bits and rejects `r >= n`, so a power of two still
        // costs an extra bit and the rejection loop still runs.
        let k = 64 - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `shuffle(x)` — Fisher-Yates walking down from the end.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        if x.len() < 2 {
            return;
        }
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i as u64 + 1) as usize;
            x.swap(i, j);
        }
    }

    /// `choice(seq)` — index only, so the caller keeps ownership.
    pub fn choice_index(&mut self, len: usize) -> usize {
        assert!(len > 0, "choice from an empty sequence");
        self.randbelow(len as u64) as usize
    }
}

/// `kev.data.source_seed`: the first 8 bytes of `sha256("{seed}:{tag}")`,
/// big-endian.
pub fn source_seed(seed: u64, tag: &str) -> u64 {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(format!("{seed}:{tag}").as_bytes());
    let d = h.finalize();
    u64::from_be_bytes(d[..8].try_into().expect("32-byte digest"))
}
