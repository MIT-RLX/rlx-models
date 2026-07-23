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

//! Native n-gram model — a self-contained, **memory-mapped** model over a small
//! token vocabulary (~86 punctuation/class tokens, order 6, fully interpolated).
//!
//! The packed file is `mmap`'d and the n-gram table is viewed *zero-copy* as a
//! sorted `&[NgramRecord]` slice (via `bytemuck`), so loading is O(1) and the
//! ~7 MB table stays out of the heap — pages are demand-faulted by the OS and
//! shared between processes. Lookups are binary searches over the sorted records
//! (longest-suffix match), with no per-query allocation. This mirrors rlx's own
//! mmap weight loading (rlx-gguf).
//!
//! File layout (little-endian; this spec is the source of truth for the packer):
//! ```text
//!   Header (64 B): magic "RLXNGRM1", order, ctx_width, vocab_count,
//!                  ngram_count, vocab_str_bytes, reserved[9]
//!   VocabEntry { tid, str_off, str_len } [vocab_count]
//!   string blob [vocab_str_bytes]                 (padded to 4)
//!   NgramRecord { ctx[CTX_WIDTH] u16, tok u16, ctx_len u8, _pad,
//!                 logprob f32 } [ngram_count]      (sorted by (ctx_len, ctx, tok))
//! ```
//! Token ids are `u16` (the vocab tops out well under 65 536). `CTX_WIDTH` bounds
//! the stored context depth; the header records the actual depth used.

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::Path;

const MAGIC: &[u8; 8] = b"RLXNGRM1";

/// Max context depth the record layout can hold (order-4 storage); the current
/// model stores order ≤ 3, i.e. context length ≤ 2.
const CTX_WIDTH: usize = 2;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Header {
    magic: [u8; 8],
    order: u32,
    ctx_width: u32,
    vocab_count: u32,
    ngram_count: u32,
    vocab_str_bytes: u32,
    _reserved: [u32; 9],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VocabEntry {
    tid: u32,
    str_off: u32,
    str_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NgramRecord {
    ctx: [u16; CTX_WIDTH],
    tok: u16,
    ctx_len: u8,
    _pad: u8,
    logprob: f32,
}

/// Binary-search key: records are sorted by `(ctx_len, ctx, tok)`.
#[inline]
fn key(ctx_len: u8, ctx: [u16; CTX_WIDTH], tok: u16) -> (u8, [u16; CTX_WIDTH], u16) {
    (ctx_len, ctx, tok)
}

pub struct NgramModel {
    mmap: Mmap,
    pub order: usize,
    max_ctx: usize,
    ngram_off: usize,
    ngram_count: usize,
    tok2str: HashMap<u32, String>,
    str2tok: HashMap<String, u32>,
}

impl NgramModel {
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        // Safety: the model file is read-only and not mutated for the map's lifetime.
        let mmap = unsafe { Mmap::map(&file)? };

        let hdr_size = std::mem::size_of::<Header>();
        if mmap.len() < hdr_size || &mmap[0..8] != MAGIC {
            bail!("bad n-gram model magic");
        }
        let header: Header = bytemuck::pod_read_unaligned(&mmap[0..hdr_size]);
        let ctx_width = header.ctx_width as usize;
        if ctx_width > CTX_WIDTH {
            bail!("n-gram model ctx_width {ctx_width} exceeds supported {CTX_WIDTH}");
        }
        let vocab_count = header.vocab_count as usize;
        let ngram_count = header.ngram_count as usize;
        let blob_len = header.vocab_str_bytes as usize;

        // Section offsets (mirror of the writer; each start is 4-aligned, and the
        // mmap base is page-aligned, so the bytemuck casts below are sound).
        let vocab_off = hdr_size;
        let vocab_bytes = vocab_count * std::mem::size_of::<VocabEntry>();
        let blob_off = vocab_off + vocab_bytes;
        let ngram_off = (blob_off + blob_len + 3) & !3;
        let rec_size = std::mem::size_of::<NgramRecord>();
        if ngram_off + ngram_count * rec_size > mmap.len() {
            bail!("n-gram model truncated");
        }

        // Vocab is tiny (~86 entries) — resolve the string table into small maps
        // once; the big n-gram table stays mmap-resident.
        let vents: &[VocabEntry] = bytemuck::cast_slice(&mmap[vocab_off..vocab_off + vocab_bytes]);
        let blob = &mmap[blob_off..blob_off + blob_len];
        let mut tok2str = HashMap::with_capacity(vocab_count);
        let mut str2tok = HashMap::with_capacity(vocab_count);
        for e in vents {
            let (a, b) = (e.str_off as usize, (e.str_off + e.str_len) as usize);
            let s = String::from_utf8_lossy(&blob[a..b]).into_owned();
            str2tok.insert(s.clone(), e.tid);
            tok2str.insert(e.tid, s);
        }

        Ok(Self {
            mmap,
            order: header.order as usize,
            max_ctx: ctx_width,
            ngram_off,
            ngram_count,
            tok2str,
            str2tok,
        })
    }

    /// Zero-copy view of the sorted n-gram records straight from the mmap.
    #[inline]
    fn records(&self) -> &[NgramRecord] {
        let n = self.ngram_count * std::mem::size_of::<NgramRecord>();
        bytemuck::cast_slice(&self.mmap[self.ngram_off..self.ngram_off + n])
    }

    pub fn token_for(&self, s: &str) -> Option<u32> {
        self.str2tok.get(s).copied()
    }
    pub fn string_for(&self, tid: u32) -> Option<&str> {
        self.tok2str.get(&tid).map(String::as_str)
    }

    /// log P(tok | ctx) via longest-suffix n-gram match (ctx truncated to stored depth).
    pub fn cond(&self, ctx: &[u32], tok: u32) -> f32 {
        let recs = self.records();
        let tok16 = tok as u16;
        let use_len = ctx.len().min(self.max_ctx);
        for l in (0..=use_len).rev() {
            let mut q = [0u16; CTX_WIDTH];
            for (i, &t) in ctx[ctx.len() - l..].iter().enumerate() {
                q[i] = t as u16;
            }
            let target = key(l as u8, q, tok16);
            if let Ok(idx) =
                recs.binary_search_by(|r| key(r.ctx_len, r.ctx, r.tok).cmp(&target))
            {
                return recs[idx].logprob;
            }
        }
        -99.0
    }

    /// Joint log-prob of a token sequence.
    pub fn joint(&self, seq: &[u32]) -> f32 {
        let mut s = 0.0;
        for i in 0..seq.len() {
            let cs = i.min(self.max_ctx);
            s += self.cond(&seq[i - cs..i], seq[i]);
        }
        s
    }
}
