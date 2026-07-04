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

//! Prompt KV-cache reuse: an in-memory, size-bounded LRU keyed by token
//! prefix, plus disk persistence. Mirrors mlx-lm's `LRUPromptCache`.
//!
//! A server holds the KV cache of recently-seen prompt prefixes (e.g. a
//! stable system prompt). A new request looks up the **longest cached
//! prefix** of its prompt, restores that KV, and prefills only the suffix —
//! turning a repeated multi-turn prompt's prefill cost to ~0.
//!
//! This module is **host-side**: a [`KvCacheState`] is plain f32 produced
//! identically on every backend, so a cache is device-neutral (but only
//! valid for the exact model that produced it — hence the `fingerprint`).

use crate::autoregressive::KvCacheState;
use std::io::{Read, Write};
use std::path::Path;

/// One cached prefix: the tokens it covers and their KV. Invariant:
/// `kv.past_len == tokens.len()`.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub tokens: Vec<u32>,
    pub kv: KvCacheState,
}

impl CacheEntry {
    /// Approximate heap footprint of the KV tensors in bytes.
    pub fn kv_bytes(&self) -> usize {
        let f = |v: &Vec<Vec<f32>>| v.iter().map(|l| l.len()).sum::<usize>() * 4;
        f(&self.kv.layers_k) + f(&self.kv.layers_v)
    }
}

/// Length of the longest common prefix of two token slices.
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Size-bounded LRU of prompt-prefix KV caches.
pub struct PromptCache {
    entries: Vec<CacheEntry>,
    /// Per-entry recency tick (parallel to `entries`).
    ticks: Vec<u64>,
    clock: u64,
    bytes: usize,
    cap_bytes: usize,
}

impl PromptCache {
    /// New cache bounded to `cap_bytes` of KV data (0 ⇒ effectively disabled).
    pub fn new(cap_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            ticks: Vec::new(),
            clock: 0,
            bytes: 0,
            cap_bytes,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Find the cached entry that is the **longest prefix** of `query`.
    /// Returns `(matched_len, entry_clone)`; `matched_len == entry.tokens.len()`
    /// (a cached entry only matches when it is fully a prefix of `query`).
    /// Touches LRU recency on a hit.
    pub fn longest_prefix(&mut self, query: &[u32]) -> Option<(usize, CacheEntry)> {
        let mut best: Option<usize> = None;
        let mut best_len = 0usize;
        for (i, e) in self.entries.iter().enumerate() {
            if e.tokens.len() > query.len() {
                continue;
            }
            let cpl = common_prefix_len(&e.tokens, query);
            // Only a *full* entry-prefix match is reusable.
            if cpl == e.tokens.len() && cpl > best_len {
                best_len = cpl;
                best = Some(i);
            }
        }
        let i = best?;
        self.clock += 1;
        self.ticks[i] = self.clock;
        Some((best_len, self.entries[i].clone()))
    }

    /// Insert (or refresh) the prefix `tokens → kv`. Evicts LRU entries to
    /// stay under `cap_bytes`. Entries whose tokens duplicate an existing
    /// one are replaced.
    pub fn insert(&mut self, tokens: Vec<u32>, kv: KvCacheState) {
        debug_assert_eq!(
            kv.past_len,
            tokens.len(),
            "kv.past_len must equal tokens.len()"
        );
        let entry = CacheEntry { tokens, kv };
        let add = entry.kv_bytes();

        // Replace an identical-prefix entry if present.
        if let Some(i) = self.entries.iter().position(|e| e.tokens == entry.tokens) {
            self.bytes -= self.entries[i].kv_bytes();
            self.bytes += add;
            self.entries[i] = entry;
            self.clock += 1;
            self.ticks[i] = self.clock;
            self.evict_to_cap();
            return;
        }

        self.clock += 1;
        self.entries.push(entry);
        self.ticks.push(self.clock);
        self.bytes += add;
        self.evict_to_cap();
    }

    /// Drop least-recently-used entries until within `cap_bytes` (always
    /// keeps at least one entry if it alone exceeds the cap).
    fn evict_to_cap(&mut self) {
        while self.bytes > self.cap_bytes && self.entries.len() > 1 {
            // Find the LRU index.
            let lru = self
                .ticks
                .iter()
                .enumerate()
                .min_by_key(|(_, t)| **t)
                .map(|(i, _)| i)
                .unwrap();
            self.bytes -= self.entries[lru].kv_bytes();
            self.entries.remove(lru);
            self.ticks.remove(lru);
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.ticks.clear();
        self.bytes = 0;
    }
}

// ─── disk persistence ────────────────────────────────────────────────
//
// A self-describing little-endian binary blob: magic + fingerprint + tokens
// + per-layer k/v f32 data. Used to warm-start a stable system prompt across
// restarts. The fingerprint (a model/config hash) guards against restoring a
// cache built for different weights.

const MAGIC: &[u8; 8] = b"RLXKV001";

fn write_u64<W: Write>(w: &mut W, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn write_f32_block<W: Write>(w: &mut W, data: &[f32]) -> std::io::Result<()> {
    write_u64(w, data.len() as u64)?;
    for &x in data {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}
fn read_f32_block<R: Read>(r: &mut R) -> std::io::Result<Vec<f32>> {
    let n = read_u64(r)? as usize;
    let mut out = Vec::with_capacity(n);
    let mut buf = [0u8; 4];
    for _ in 0..n {
        r.read_exact(&mut buf)?;
        out.push(f32::from_le_bytes(buf));
    }
    Ok(out)
}

/// Serialize one [`CacheEntry`] to `path`, tagged with `fingerprint`.
pub fn save_entry(path: &Path, entry: &CacheEntry, fingerprint: u64) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(MAGIC)?;
    write_u64(&mut f, fingerprint)?;
    write_u64(&mut f, entry.kv.past_len as u64)?;
    write_u64(&mut f, entry.tokens.len() as u64)?;
    for &t in &entry.tokens {
        write_u64(&mut f, t as u64)?;
    }
    let layers = entry.kv.layers_k.len() as u64;
    write_u64(&mut f, layers)?;
    for k in &entry.kv.layers_k {
        write_f32_block(&mut f, k)?;
    }
    for v in &entry.kv.layers_v {
        write_f32_block(&mut f, v)?;
    }
    f.flush()
}

/// Load a [`CacheEntry`] from `path`, verifying the magic and `fingerprint`.
/// Returns `Ok(None)` on a fingerprint/magic mismatch (cache is stale).
pub fn load_entry(path: &Path, fingerprint: u64) -> std::io::Result<Option<CacheEntry>> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC || read_u64(&mut f)? != fingerprint {
        return Ok(None);
    }
    let past_len = read_u64(&mut f)? as usize;
    let n_tokens = read_u64(&mut f)? as usize;
    let mut tokens = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        tokens.push(read_u64(&mut f)? as u32);
    }
    let layers = read_u64(&mut f)? as usize;
    let mut layers_k = Vec::with_capacity(layers);
    for _ in 0..layers {
        layers_k.push(read_f32_block(&mut f)?);
    }
    let mut layers_v = Vec::with_capacity(layers);
    for _ in 0..layers {
        layers_v.push(read_f32_block(&mut f)?);
    }
    Ok(Some(CacheEntry {
        tokens,
        kv:         KvCacheState {
            past_len,
            layers_kv_base: vec![0; layers],
            layers_k,
            layers_v,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(past: usize, fill: f32) -> KvCacheState {
        KvCacheState {
            past_len: past,
            layers_kv_base: vec![0; 1],
            layers_k: vec![vec![fill; past * 2]],
            layers_v: vec![vec![fill; past * 2]],
        }
    }

    #[test]
    fn longest_prefix_picks_deepest_match() {
        let mut c = PromptCache::new(1 << 20);
        c.insert(vec![1, 2], kv(2, 0.1));
        c.insert(vec![1, 2, 3, 4], kv(4, 0.2));
        let (len, e) = c.longest_prefix(&[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(len, 4);
        assert_eq!(e.tokens, vec![1, 2, 3, 4]);
        // A query that only shares the shorter prefix falls back to it.
        let (len, _) = c.longest_prefix(&[1, 2, 9]).unwrap();
        assert_eq!(len, 2);
    }

    #[test]
    fn no_match_returns_none() {
        let mut c = PromptCache::new(1 << 20);
        c.insert(vec![5, 6], kv(2, 0.0));
        assert!(c.longest_prefix(&[1, 2, 3]).is_none());
        // Entry longer than the query can't be a prefix of it.
        assert!(c.longest_prefix(&[5]).is_none());
    }

    #[test]
    fn lru_evicts_when_over_cap() {
        // Each entry of past=4 is 1 layer * (8+8) f32 = 64 bytes.
        let mut c = PromptCache::new(100); // room for one entry, not two
        c.insert(vec![1, 2, 3, 4], kv(4, 0.1));
        c.insert(vec![5, 6, 7, 8], kv(4, 0.2)); // evicts the first
        assert_eq!(c.len(), 1);
        assert!(c.longest_prefix(&[1, 2, 3, 4]).is_none());
        assert!(c.longest_prefix(&[5, 6, 7, 8]).is_some());
    }

    #[test]
    fn lru_keeps_recently_used() {
        let mut c = PromptCache::new(140); // two entries fit (~128), third evicts
        c.insert(vec![1, 1, 1, 1], kv(4, 0.1));
        c.insert(vec![2, 2, 2, 2], kv(4, 0.2));
        let _ = c.longest_prefix(&[1, 1, 1, 1]); // touch entry 1 → entry 2 is LRU
        c.insert(vec![3, 3, 3, 3], kv(4, 0.3)); // evicts entry 2
        assert!(c.longest_prefix(&[1, 1, 1, 1]).is_some());
        assert!(c.longest_prefix(&[2, 2, 2, 2]).is_none());
        assert!(c.longest_prefix(&[3, 3, 3, 3]).is_some());
    }

    #[test]
    fn disk_round_trip_with_fingerprint() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rlx_pc_test_{}.bin", std::process::id()));
        let entry = CacheEntry {
            tokens: vec![7, 8, 9],
            kv: kv(3, 0.42),
        };
        save_entry(&path, &entry, 0xABCD).unwrap();
        let back = load_entry(&path, 0xABCD)
            .unwrap()
            .expect("fingerprint matches");
        assert_eq!(back.tokens, entry.tokens);
        assert_eq!(back.kv.past_len, 3);
        assert_eq!(back.kv.layers_k[0], entry.kv.layers_k[0]);
        // Wrong fingerprint → treated as stale.
        assert!(load_entry(&path, 0x9999).unwrap().is_none());
        let _ = std::fs::remove_file(&path);
    }
}
