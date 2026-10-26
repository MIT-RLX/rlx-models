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

//! Host-side attention masks, RoPE tables and the key/value slot ledger.
//!
//! Every attention layer scores `[cache slots | this step's tokens]`, and what
//! each query may see is decided here, on the host, from absolute positions:
//!
//! * a **full** layer sees every earlier position;
//! * a **sliding** layer sees the current token and `sliding_window - 1`
//!   before it (`q - k <= window - 1`).
//!
//! Because the rule reads positions rather than slot order, a sliding layer's
//! cache can be a ring of exactly `window - 1` slots: once a position falls out
//! of every future query's window its slot is reused. Keys are rotated before
//! they are stored, so where a key sits in the cache carries no meaning —
//! attention is permutation-invariant over keys once the mask is applied.

use crate::attention::MASK_PENALTY;

/// `[seq, head_dim / 2]` NeoX RoPE tables for absolute `positions`.
///
/// Angles in f64, rounded to f32 once. `scripts/flint_reference.py` does the
/// same, so the port and the reference read identical tables.
pub fn rope_tables(positions: &[usize], head_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let inv: Vec<f64> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect();
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);
    for &p in positions {
        for &f in &inv {
            let a = p as f64 * f;
            cos.push(a.cos() as f32);
            sin.push(a.sin() as f32);
        }
    }
    (cos, sin)
}

/// The additive bias `[queries, slots.len() + queries]` for one step.
///
/// `slots[j]` is the absolute position cached in slot `j` (`None` = empty);
/// `queries` are the absolute positions of this step's tokens, in order, and
/// are also its keys. `history` is how many earlier positions a query may see
/// (`None` = unbounded, a full layer).
pub fn step_bias(slots: &[Option<usize>], queries: &[usize], history: Option<usize>) -> Vec<f32> {
    let keys = slots.len() + queries.len();
    let mut out = vec![MASK_PENALTY; queries.len() * keys];
    let visible = |q: usize, k: usize| k <= q && history.is_none_or(|h| q - k <= h);
    for (qi, &q) in queries.iter().enumerate() {
        let row = &mut out[qi * keys..(qi + 1) * keys];
        for (j, slot) in slots.iter().enumerate() {
            if let Some(k) = *slot
                && visible(q, k)
            {
                row[j] = 0.0;
            }
        }
        for (ti, &k) in queries.iter().enumerate() {
            if ti <= qi && visible(q, k) {
                row[slots.len() + ti] = 0.0;
            }
        }
    }
    out
}

/// Which absolute position each cache slot of one layer kind holds.
///
/// Full layers fill slots in position order and never reuse one; sliding
/// layers keep a ring of `history` slots, so their memory stays at the window
/// however long the sequence grows.
#[derive(Debug, Clone)]
pub struct SlotLedger {
    pos: Vec<Option<usize>>,
    ring: bool,
}

impl SlotLedger {
    /// A full layer's ledger: `cap` linear slots.
    pub fn linear(cap: usize) -> Self {
        SlotLedger {
            pos: vec![None; cap],
            ring: false,
        }
    }

    /// A sliding layer's ledger: a ring of `history` slots.
    pub fn ring(history: usize) -> Self {
        SlotLedger {
            pos: vec![None; history],
            ring: true,
        }
    }

    pub fn cap(&self) -> usize {
        self.pos.len()
    }

    pub fn positions(&self) -> &[Option<usize>] {
        &self.pos
    }

    /// The slot absolute position `p` is written to, or `None` when a linear
    /// ledger is full.
    pub fn slot_for(&self, p: usize) -> Option<usize> {
        if self.pos.is_empty() {
            return None;
        }
        if self.ring {
            Some(p % self.pos.len())
        } else {
            (p < self.pos.len()).then_some(p)
        }
    }

    /// Record that position `p` now occupies its slot.
    pub fn commit(&mut self, p: usize) {
        if let Some(s) = self.slot_for(p) {
            self.pos[s] = Some(p);
        }
    }

    pub fn clear(&mut self) {
        self.pos.iter_mut().for_each(|s| *s = None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn visible(b: &[f32]) -> Vec<bool> {
        b.iter().map(|&x| x == 0.0).collect()
    }

    #[test]
    fn prefill_bias_is_causal_and_windowed() {
        let q: Vec<usize> = (0..6).collect();
        let full = visible(&step_bias(&[], &q, None));
        let win = visible(&step_bias(&[], &q, Some(2)));
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(full[i * 6 + j], j <= i);
                assert_eq!(win[i * 6 + j], j <= i && i - j <= 2, "q{i} k{j}");
            }
        }
    }

    #[test]
    fn ring_slots_expire_with_the_window() {
        // history 3: after positions 0..5, the ring holds 3, 4, 5.
        let mut ring = SlotLedger::ring(3);
        for p in 0..6 {
            ring.commit(p);
        }
        let mut held: Vec<usize> = ring.positions().iter().flatten().copied().collect();
        held.sort();
        assert_eq!(held, vec![3, 4, 5]);
        // Query at 6 sees 3, 4, 5 and itself; at 7 (same chunk) it loses 3.
        let b = visible(&step_bias(ring.positions(), &[6, 7], Some(3)));
        let keys = 3 + 2;
        let row = |r: usize| -> Vec<usize> {
            let mut v: Vec<usize> = ring
                .positions()
                .iter()
                .enumerate()
                .filter(|(j, _)| b[r * keys + j])
                .map(|(_, p)| p.unwrap())
                .collect();
            v.sort();
            v
        };
        assert_eq!(row(0), vec![3, 4, 5]);
        assert_eq!(row(1), vec![4, 5]);
        assert!(b[keys + 3] && b[keys + 4], "position 7 sees 6 and itself");
        assert!(!b[4], "position 6 cannot see 7");
    }

    #[test]
    fn a_full_linear_ledger_refuses_the_next_position() {
        let l = SlotLedger::linear(2);
        assert_eq!(l.slot_for(1), Some(1));
        assert_eq!(l.slot_for(2), None);
    }
}
