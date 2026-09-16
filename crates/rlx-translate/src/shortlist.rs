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

//! Reader for `MT/shortlists/all-<lang>.shortlist`.
//!
//! The `PDecTranslatorBlock` sets `enable_shortlist`, `shortlist-cond-n: 100`
//! and `shortlist-lang-pair: all-fr`; this is the file that names it. For each
//! **source** token it lists up to 100 candidate **target** tokens, and the
//! decoder scores only their union instead of the whole 168 000-entry
//! vocabulary. That is both a quality constraint and the reason on-device
//! decoding is fast.
//!
//! # Layout
//!
//! ```text
//!   0x00 'B' <0x04 u32=1> …zero padding…        16-byte header
//!   u32 × (N+1)                                 CSR offsets, N = 167 969
//!   u32 × offsets[N]                            entries, id = value >> 8
//! ```
//!
//! Each entry's low byte is a zero tag — the same tag-byte container style as
//! `pyespresso.mdl.bin` — so the token id occupies the upper 24 bits.
//!
//! Verified against the shipped `all-fr` table: `▁the` lists `▁le`/`▁la`/`▁les`,
//! `▁dog` lists `▁chien`, `▁cat` lists `▁chat`, `▁water` lists `▁eau`. The
//! entries are **not** ordered by relevance, so eyeballing the first few is
//! misleading — check membership.
//!
//! The file has a ~584 KB tail beyond `offsets[N]` entries that is not accounted
//! for here; it is likely the `shortlist-freq-n` table. Reading stops at the CSR
//! extent rather than guessing.

use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;
use std::path::Path;

/// Header size before the CSR offsets.
const HEADER: usize = 16;

/// A loaded per-source-token candidate table.
#[derive(Debug)]
pub struct Shortlist {
    offsets: Vec<u32>,
    entries: Vec<u32>,
}

impl Shortlist {
    /// Reads a `.shortlist`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&bytes).with_context(|| format!("parsing {}", path.display()))
    }

    /// Parses shortlist bytes.
    pub fn parse(b: &[u8]) -> Result<Self> {
        ensure!(b.len() > HEADER + 8, "shortlist is too short");
        // The offsets run until the point where the CSR closes on itself: the
        // table length is not stated, so it is solved from the file size.
        // entries_start = HEADER + (n+1)*4 and entries_len = offsets[n].
        let u32_at =
            |i: usize| -> u32 { u32::from_le_bytes(b[i..i + 4].try_into().expect("four bytes")) };
        // The fourth header word is the number of CSR offsets, so the table
        // covers `n = that - 1` source tokens — which is exactly the model's
        // vocabulary size on every shipped table: 168 000 for the French-family
        // bundle, 96 000 for the en-zh-ja-ko one, 48 000 for `all-en`.
        //
        // This used to probe a hard-coded window around 167 969, the French
        // number. That was wrong twice over: it missed French's real length by
        // 31 source tokens, and it failed outright on every other vocabulary —
        // and because the caller loaded the table with `.ok()`, the failure was
        // silent and every CJK and into-English direction ran with no shortlist
        // at all, scoring the whole vocabulary instead of ~700 candidates.
        let n = u32_at(12) as usize;
        ensure!(n > 0, "shortlist header declares no offsets");
        let n = n - 1;
        ensure!(
            HEADER + (n + 1) * 4 <= b.len(),
            "shortlist declares {n} source tokens but is only {} bytes",
            b.len()
        );
        ensure!(u32_at(HEADER) == 0, "shortlist does not start at offset 0");

        let offsets: Vec<u32> = (0..=n).map(|i| u32_at(HEADER + i * 4)).collect();
        ensure!(
            offsets.windows(2).all(|w| w[0] <= w[1]),
            "shortlist offsets are not monotonic"
        );
        let start = HEADER + (n + 1) * 4;
        let count = *offsets.last().expect("non-empty") as usize;
        ensure!(
            start + count * 4 <= b.len(),
            "shortlist entries overrun the file"
        );
        let entries: Vec<u32> = (0..count).map(|i| u32_at(start + i * 4) >> 8).collect();
        Ok(Self { offsets, entries })
    }

    /// Number of source tokens covered.
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Candidate target tokens for one source token.
    pub fn candidates(&self, token: u32) -> &[u32] {
        let i = token as usize;
        if i + 1 >= self.offsets.len() {
            return &[];
        }
        let (s, e) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
        if e <= s || e > self.entries.len() {
            return &[];
        }
        &self.entries[s..e]
    }

    /// Union of the candidates for every source token, plus `always`.
    ///
    /// `always` carries the tokens the decoder must always be able to emit —
    /// end-of-sequence and any control tokens — which the per-token lists do
    /// not necessarily contain.
    pub fn union(&self, source: &[u32], always: &[u32]) -> Vec<u32> {
        let mut set: BTreeSet<u32> = always.iter().copied().collect();
        for t in source {
            set.extend(self.candidates(*t).iter().copied());
        }
        set.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a table with the shipped layout: 16-byte header, `n+1` offsets,
    /// then `id << 8` entries.
    fn build(lists: &[&[u32]]) -> Vec<u8> {
        let n = lists.len();
        let mut out = vec![0u8; HEADER];
        out[1] = b'B';
        // The fourth header word is the offset count. These fixtures used to
        // leave it zero and pad to 167 969 entries so the old size *probe*
        // would land — which meant they asserted the parser's mistake rather
        // than the format. Real tables put the count here.
        out[12..16].copy_from_slice(&((n + 1) as u32).to_le_bytes());
        let mut off = 0u32;
        let mut offsets = Vec::with_capacity(n + 1);
        for l in lists {
            offsets.push(off);
            off += l.len() as u32;
        }
        offsets.push(off);
        for o in &offsets {
            out.extend_from_slice(&o.to_le_bytes());
        }
        for l in lists {
            for id in *l {
                out.extend_from_slice(&(id << 8).to_le_bytes());
            }
        }
        out
    }

    /// The table length now comes from the header, so nothing needs padding.
    fn padded(lists: Vec<&[u32]>) -> Vec<u8> {
        build(&lists)
    }

    #[test]
    fn reads_candidate_lists_and_unshifts_ids() {
        let bytes = padded(vec![&[7, 9], &[], &[3]]);
        let s = Shortlist::parse(&bytes).expect("parses");
        assert_eq!(s.len(), 3);
        assert_eq!(s.candidates(0), &[7, 9]);
        assert_eq!(s.candidates(1), &[] as &[u32]);
        assert_eq!(s.candidates(2), &[3]);
    }

    #[test]
    fn out_of_range_tokens_yield_nothing_rather_than_panicking() {
        let bytes = padded(vec![&[1]]);
        let s = Shortlist::parse(&bytes).expect("parses");
        assert!(s.candidates(999_999).is_empty());
    }

    #[test]
    fn union_merges_and_always_includes_the_control_tokens() {
        let bytes = padded(vec![&[5, 6], &[6, 7], &[]]);
        let s = Shortlist::parse(&bytes).expect("parses");
        let u = s.union(&[0, 1], &[2]);
        assert_eq!(u, vec![2, 5, 6, 7], "sorted, deduplicated, control kept");
        // Even with no source tokens the control set survives.
        assert_eq!(s.union(&[], &[2, 1]), vec![1, 2]);
    }

    #[test]
    fn non_monotonic_offsets_are_rejected() {
        let mut bytes = padded(vec![&[1, 2], &[3]]);
        // Corrupt offsets[1] to be larger than offsets[2].
        let at = HEADER + 4;
        bytes[at..at + 4].copy_from_slice(&99u32.to_le_bytes());
        assert!(Shortlist::parse(&bytes).is_err());
    }

    #[test]
    fn a_short_file_is_rejected() {
        assert!(Shortlist::parse(&[0u8; 8]).is_err());
    }
}
