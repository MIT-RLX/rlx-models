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

//! PLE — `Qwen4ExpTextPLELayer`, hashed n-gram features injected into every
//! hyper-connection stream.
//!
//! Qwen3.8-Flash-Next puts **one** of these on layer 1 (zero-based), and it is
//! where the model's parameters mostly live: `per_layer_token_embd` is
//! `[160, 320001536]` — 51 billion weights, more than the rest of the
//! checkpoint put together. Nothing is learned per token id; the table is
//! addressed by a hash of the trailing token window, so a bigram or trigram
//! that never occurred in training still lands on a row.
//!
//! ```text
//!   emb   = concat_h ngram_table[row_h(window)]          [s, 16·160]
//!   key   = GroupedRMSNorm(emb @ ple_keyᵀ)               [s, hc, D]
//!   query = GroupedRMSNorm(hidden)                       [s, hc, D]
//!   dot   = Σ_D key·query / √D                           [s, hc, 1]
//!   gate  = σ(sign(dot) · √max(|dot|, 1e-6))
//!   gated = gate · (emb @ ple_valueᵀ)                    [s, hc, D]
//!   out   = hidden + gated + silu(dilated_conv(GroupedRMSNorm(gated)))
//! ```
//!
//! ## The table is not a graph parameter
//!
//! 28 GB packed is not something to put in an arena, and the gather touches at
//! most `16 · seq` of its 320 million rows. So [`emit_ple`] takes the
//! embeddings as a graph **input** and the row addressing happens on the host —
//! [`ngram_rows`] computes the indices, [`ngram_embeddings`] reads them through
//! a caller-supplied row reader. llama.cpp reaches the same arrangement from
//! the other side, marking the tensor `TENSOR_READ_LAZY` so its rows are read
//! straight off disk.
//!
//! ## The convolution is dilated by the n-gram size
//!
//! `kernel_size = 4`, `dilation = ngram_size = 3`, so it reaches
//! `(4 − 1) · 3 = 9` positions back and each tap lands on a window that shares
//! no tokens with its neighbour. A plain kernel-4 conv is a different operator;
//! so is a kernel-4 dilated conv with only `3` left-padding.
//!
//! ## The EOS reset is asymmetric
//!
//! Walking back from a token, the first predecessor that is EOS (or missing)
//! truncates that position *and every older one* to EOS. The token's **own**
//! EOS does not cut its context — the reference's `_shift_right_ignore_eos`
//! takes the last EOS *strictly before* `i`. Getting this backwards changes
//! every row index at a document boundary.

use anyhow::{Result, bail};
use rlx_flow::Emit;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{Activation, BinaryOp, Op};
use rlx_ir::{DType, HirGraphExt, HirNodeId, Shape};

use crate::common::{ChunkInputs, act, carried_conv1d, grouped_rms_norm, linear, scalar};
use crate::config::{PleHash, Qwen4ExpConfig};

/// Graph-input prefix for the host-gathered n-gram embeddings.
///
/// The actual input is [`ple_embed_input`], one per PLE module, because the
/// hash constants are per-module and so are the rows they address.
pub const PLE_EMBED_INPUT: &str = "ple_embeddings";

/// Graph-input name for PLE module `ple_layer_index`'s embeddings.
pub fn ple_embed_input(ple_layer_index: usize) -> String {
    format!("{PLE_EMBED_INPUT}.{ple_layer_index}")
}

/// The `(ngram_size − 1)`-token window preceding a batch, oldest first.
///
/// Shorter than `ngram_size - 1` (or empty) is fine: the missing entries read
/// as EOS, which is what the reference's zero-state does.
pub type PrevContext<'a> = &'a [u32];

/// Hash a batch of token ids into `[seq, n_heads]` table row indices,
/// token-major, starting a fresh sequence.
pub fn ngram_rows(hash: &PleHash, tokens: &[u32]) -> Vec<u32> {
    ngram_rows_with_history(hash, tokens, &[])
}

/// [`ngram_rows`] continuing from an earlier batch.
///
/// `prev` carries the tokens immediately before `tokens[0]`, oldest first — the
/// reference keeps them in a conv-state slot of the cache so a chunked prefill
/// hashes the same windows a single-shot one would.
pub fn ngram_rows_with_history(hash: &PleHash, tokens: &[u32], prev: PrevContext<'_>) -> Vec<u32> {
    let n = hash.ngram_size;
    let eos = hash.eos_token_id as u64;
    let mut out = Vec::with_capacity(tokens.len() * hash.n_heads);
    let mut window = vec![0u64; n];
    for i in 0..tokens.len() {
        // The token itself is never truncated, even when it IS the EOS.
        window[0] = tokens[i] as u64;
        let mut cut = false;
        for s in 1..n {
            let t = if cut {
                None
            } else if s <= i {
                Some(tokens[i - s])
            } else {
                // `prev` is oldest-first, so the token `s` back from
                // `tokens[0]` is `prev[len - (s - i)]`.
                let back = s - i;
                prev.len().checked_sub(back).map(|p| prev[p])
            };
            cut = cut || t.is_none_or(|t| t as u64 == eos);
            window[s] = if cut { eos } else { t.unwrap() as u64 };
        }
        out.extend(hash.rows(&window));
    }
    out
}

/// Gather the n-gram embeddings for `tokens` into the flat
/// `[seq · n_heads · head_dim]` buffer [`emit_ple`]'s graph input expects.
///
/// `read_row` is handed a table row index and must return `head_dim` floats.
/// Keeping the read behind a closure is what lets a 28 GB quantized table stay
/// on disk: the caller does a `pread` per row (at `head_dim = 160` and
/// `IQ4_NL`, 90 bytes) instead of materializing anything.
pub fn ngram_embeddings(
    hash: &PleHash,
    tokens: &[u32],
    prev: PrevContext<'_>,
    head_dim: usize,
    mut read_row: impl FnMut(u32) -> Result<Vec<f32>>,
) -> Result<Vec<f32>> {
    let rows = ngram_rows_with_history(hash, tokens, prev);
    let mut out = Vec::with_capacity(rows.len() * head_dim);
    for r in rows {
        let v = read_row(r)?;
        if v.len() != head_dim {
            bail!(
                "qwen4exp: PLE row {r} returned {} values, expected {head_dim}",
                v.len()
            );
        }
        out.extend_from_slice(&v);
    }
    Ok(out)
}

/// Where [`prepare_ple_embeddings`] reads table rows from.
///
/// The row read is behind a trait because the real table is 51 billion weights:
/// a serving path implements this over a `pread` into the GGUF (at
/// `head_dim = 160` and `IQ4_NL`, 90 bytes per row), while a test uses
/// [`DensePleTable`]. `&mut self` so an implementation can keep a file handle
/// or a cache.
pub trait PleRowSource {
    /// `head_dim` floats for table row `index`.
    fn row(&mut self, index: u32) -> Result<Vec<f32>>;
}

/// A whole n-gram table held in memory, row-major `[rows, head_dim]`.
///
/// Only usable at synthetic sizes — the published table is 28 GB packed.
#[derive(Debug, Clone)]
pub struct DensePleTable {
    pub data: Vec<f32>,
    pub rows: usize,
    pub head_dim: usize,
}

/// The published n-gram table name.
pub const PLE_TABLE_TENSOR: &str = "per_layer_token_embd.weight";

/// A [`PleRowSource`] that reads rows out of a GGUF without holding the table.
///
/// This is what makes the real checkpoint runnable. `per_layer_token_embd` is
/// `[160, 320001536]` — 51 billion parameters, ~28 GB as the published IQ4_NL —
/// and a decode step needs **nine** of its 320 million rows. Holding it is not
/// an option, and neither is mapping it: borrowing tensor bytes from an mmap
/// makes every page it touches resident, and on macOS that memory cannot be
/// handed back without `munmap`. So this seeks and reads.
///
/// A row costs the quantization blocks it overlaps, not the tensor. For the
/// published type that is exact — IQ4_NL blocks 32 elements, a row is 160, so
/// every row is 5 whole blocks of 18 bytes: **90 bytes read per row**. Nothing
/// here is specialized to that type, though; the block geometry comes from
/// [`rlx_gguf::bytes_for_public`], so an f32, k-quant or i-quant table all work
/// and a row that straddles a block boundary simply reads the two blocks.
pub struct GgufPleRows {
    file: std::fs::File,
    /// Absolute file offset of the tensor's first byte.
    base: u64,
    dtype: rlx_gguf::GgmlType,
    head_dim: usize,
    rows: u64,
    blk_elems: usize,
    blk_bytes: usize,
    buf: Vec<u8>,
    /// `first_block` → that window, dequantized.
    cache: std::collections::HashMap<u64, Vec<f32>>,
    order: std::collections::VecDeque<u64>,
    cache_cap: usize,
    bytes_read: u64,
    hits: u64,
}

impl GgufPleRows {
    /// Open [`PLE_TABLE_TENSOR`] in `path`.
    pub fn open(path: impl AsRef<std::path::Path>, head_dim: usize) -> Result<Self> {
        Self::open_tensor(path, PLE_TABLE_TENSOR, head_dim)
    }

    /// Open a named tensor as the row table.
    pub fn open_tensor(
        path: impl AsRef<std::path::Path>,
        tensor: &str,
        head_dim: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        if head_dim == 0 {
            bail!("qwen4exp: ple head_dim must be > 0");
        }
        // Header only: the point of this type is never to map the data.
        let header = rlx_gguf::GgufFile::header_from_path(path)
            .map_err(|e| anyhow::anyhow!("qwen4exp: reading {path:?} header: {e}"))?;
        let t = header
            .get(tensor)
            .ok_or_else(|| anyhow::anyhow!("qwen4exp: {path:?} has no tensor {tensor:?}"))?;
        // GGML order: the innermost dimension is the row width.
        if t.shape.len() != 2 || t.shape[0] != head_dim {
            bail!(
                "qwen4exp: {tensor} is {:?}, expected [{head_dim}, rows]",
                t.shape
            );
        }
        let rows = t.shape[1] as u64;
        let blk_elems = block_elements(t.dtype)
            .ok_or_else(|| anyhow::anyhow!("qwen4exp: no block geometry for {:?}", t.dtype))?;
        let blk_bytes = rlx_gguf::bytes_for_public(t.dtype, blk_elems)
            .ok_or_else(|| anyhow::anyhow!("qwen4exp: no block size for {:?}", t.dtype))?;
        let base = header
            .data_offset()
            .checked_add(t.offset)
            .ok_or_else(|| anyhow::anyhow!("qwen4exp: {tensor} offset overflows"))?;
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("qwen4exp: opening {path:?}: {e}"))?;
        Ok(GgufPleRows {
            file,
            base,
            dtype: t.dtype,
            head_dim,
            rows,
            blk_elems,
            blk_bytes,
            buf: Vec::new(),
            cache: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
            cache_cap: 256,
            bytes_read: 0,
            hits: 0,
        })
    }

    /// How many dequantized block windows to keep. `0` disables the cache.
    pub fn with_cache_blocks(mut self, n: usize) -> Self {
        self.cache_cap = n;
        self.cache.clear();
        self.order.clear();
        self
    }

    /// Rows in the table.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Bytes actually read from the file so far — the number the whole design
    /// exists to keep small, so it is observable rather than asserted.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Rows served from the window cache without touching the file.
    pub fn cache_hits(&self) -> u64 {
        self.hits
    }
}

/// Elements per quantization block for `dtype`, by probing the one public
/// function that knows: it returns `None` for an unaligned count.
///
/// Derived rather than tabulated so a new ggml type needs no change here, and
/// public so nothing else has to write `32` or `256` and hope.
pub fn block_elements(dtype: rlx_gguf::GgmlType) -> Option<usize> {
    [1usize, 32, 64, 128, 256]
        .into_iter()
        .find(|&qk| rlx_gguf::bytes_for_public(dtype, qk).is_some())
}

impl PleRowSource for GgufPleRows {
    fn row(&mut self, index: u32) -> Result<Vec<f32>> {
        let idx = index as u64;
        if idx >= self.rows {
            bail!(
                "qwen4exp: n-gram row {idx} is past the table's {} rows",
                self.rows
            );
        }
        let first_el = idx * self.head_dim as u64;
        let blk = self.blk_elems as u64;
        let first_blk = first_el / blk;
        let last_blk = (first_el + self.head_dim as u64 - 1) / blk;
        let n_blk = (last_blk - first_blk + 1) as usize;
        let start = (first_el - first_blk * blk) as usize;

        if self.cache_cap > 0
            && let Some(w) = self.cache.get(&first_blk)
            && w.len() >= start + self.head_dim
        {
            self.hits += 1;
            return Ok(w[start..start + self.head_dim].to_vec());
        }

        let n_bytes = n_blk * self.blk_bytes;
        let off = self.base + first_blk * self.blk_bytes as u64;
        self.buf.resize(n_bytes, 0);
        {
            use std::io::{Read, Seek, SeekFrom};
            self.file
                .seek(SeekFrom::Start(off))
                .map_err(|e| anyhow::anyhow!("qwen4exp: seeking to n-gram row {idx}: {e}"))?;
            self.file
                .read_exact(&mut self.buf)
                .map_err(|e| anyhow::anyhow!("qwen4exp: reading n-gram row {idx}: {e}"))?;
        }
        self.bytes_read += n_bytes as u64;
        let window = rlx_gguf::dequant_typed(
            self.dtype,
            &self.buf,
            n_blk * self.blk_elems,
            PLE_TABLE_TENSOR,
        )
        .map_err(|e| anyhow::anyhow!("qwen4exp: dequantizing n-gram row {idx}: {e}"))?;
        let out = window[start..start + self.head_dim].to_vec();
        if self.cache_cap > 0 {
            if self.order.len() == self.cache_cap
                && let Some(old) = self.order.pop_front()
            {
                self.cache.remove(&old);
            }
            self.order.push_back(first_blk);
            self.cache.insert(first_blk, window);
        }
        Ok(out)
    }
}

impl PleRowSource for DensePleTable {
    fn row(&mut self, index: u32) -> Result<Vec<f32>> {
        let r = index as usize;
        if r >= self.rows {
            bail!(
                "qwen4exp: PLE row {r} is out of range for a {}-row table",
                self.rows
            );
        }
        let off = r * self.head_dim;
        Ok(self.data[off..off + self.head_dim].to_vec())
    }
}

/// Build the `ple_embeddings.{i}` graph input for `tokens`.
///
/// Derives module `ple_layer_index`'s hash from `cfg`, hashes the token
/// windows, and gathers the rows — the whole host side of PLE in one call. The
/// result is `[seq · ple_embed_dim]`, which is what
/// [`crate::flow::build_qwen4exp_text_flow`] declares.
pub fn prepare_ple_embeddings(
    cfg: &crate::config::Qwen4ExpConfig,
    ple_layer_index: usize,
    tokens: &[u32],
    table: &mut dyn PleRowSource,
) -> Result<Vec<f32>> {
    let hash = PleHash::derive(cfg, ple_layer_index);
    ngram_embeddings(&hash, tokens, &[], cfg.ple_head_dim, |r| table.row(r))
}

/// Static shape/hyper-parameters of one PLE module.
#[derive(Debug, Clone, Copy)]
pub struct PleDims {
    pub hidden: usize,
    /// `hc_count`.
    pub streams: usize,
    /// `ple_embed_dim` — the concatenated n-gram width.
    pub embed_dim: usize,
    pub conv_kernel: usize,
    /// Convolution dilation — `ngram_size`, not 1.
    pub dilation: usize,
    pub eps: f32,
    pub norm_offset: bool,
    pub seq: usize,
}

impl PleDims {
    /// Derive from the model config, for a step of `seq` positions.
    ///
    /// There is exactly one statement of how a config becomes these dimensions, so
    /// prefill, chunked prefill, the decode step, the MTP block and the tests
    /// cannot disagree about what model they are building. They used to hold four
    /// copies of it.
    pub fn from_config(cfg: &Qwen4ExpConfig, seq: usize) -> Self {
        PleDims {
            hidden: cfg.hidden_size,
            streams: cfg.hc_count,
            embed_dim: cfg.ple_embed_dim(),
            conv_kernel: cfg.ple_conv_kernel,
            dilation: cfg.ple_ngram_size,
            eps: cfg.rms_norm_eps,
            norm_offset: cfg.rms_norm_offset,
            seq,
        }
    }
    /// `hc_count · hidden_size`.
    pub fn width(&self) -> usize {
        self.streams * self.hidden
    }
}

/// Dilated depthwise causal convolution over `[1, seq, channels]`, SiLU'd.
///
/// `out[c, t] = silu(Σ_k w[k, c] · x[c, t − (K − 1 − k)·dilation])`, with the
/// `(K − 1)·dilation` left positions zero — a left-pad plus one grouped
/// `Op::Conv` with `dilation > 1`. `ple_conv1d.weight` arrives as
/// `[channels, kernel]`, which is `[C, 1, k, 1]` reshaped, same as
/// `ssm_conv1d`.
fn dilated_causal_conv1d(
    emit: &mut Emit<'_>,
    key: &str,
    x: HirNodeId,
    channels: usize,
    seq: usize,
    k: usize,
    dilation: usize,
) -> Result<HirNodeId> {
    let w = emit.load_param(key, false)?;
    let hist = (k - 1) * dilation;
    let pad = emit.synth_zeros(&format!("{key}.causal_pad"), hist * channels);
    let mut gb = HirMut::new(emit.hir());

    let pad = gb.reshape_(pad, vec![1, hist as i64, channels as i64]);
    let padded = gb.concat_(vec![pad, x], 1);
    let width = seq + hist;

    let bcw = gb.transpose_(padded, vec![0, 2, 1]);
    let nchw = gb.reshape_(bcw, vec![1, channels as i64, width as i64, 1]);
    let w4 = gb.reshape_(w, vec![channels as i64, 1, k as i64, 1]);
    let conv = gb.add_node(
        Op::Conv {
            kernel_size: vec![k, 1],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![dilation, 1],
            groups: channels,
        },
        vec![nchw, w4],
        Shape::new(&[1, channels, seq, 1], DType::F32),
    );
    let bcs = gb.reshape_(conv, vec![1, channels as i64, seq as i64]);
    let bsc = gb.transpose_(bcs, vec![0, 2, 1]);
    Ok(gb.silu(bsc))
}

/// Emit one PLE module for GGUF block `prefix`.
///
/// * `hidden` — `[1, seq, hc, hidden]`, the stream state entering the layer.
/// * `emb` — `[seq, ple_embed_dim]`, the host-gathered n-gram embeddings.
///
/// Returns the updated stream state, `[1, seq, hc, hidden]`. PLE runs *before*
/// the layer's first hyper-connection, and its output replaces the residual
/// rather than being injected through one.
pub fn emit_ple(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    emb: HirNodeId,
    d: PleDims,
) -> Result<HirNodeId> {
    let (out, _) = emit_ple_with_state(emit, prefix, hidden, emb, d, None, ChunkInputs::default())?;
    Ok(out)
}

/// [`emit_ple`] carrying the dilated convolution's history across calls.
///
/// `None` zero-pads on the left, which is the whole-prompt prefill.
/// `Some(state)` prepends `(kernel − 1) · ngram_size` carried positions and
/// returns the new trailing window — the second of the three cache rows the
/// reference keeps for a PLE layer. (The third is the token history the hash
/// needs, which lives on the host: see
/// [`ngram_rows_with_history`].)
#[allow(clippy::too_many_arguments)]
pub fn emit_ple_with_state(
    emit: &mut Emit<'_>,
    prefix: &str,
    hidden: HirNodeId,
    emb: HirNodeId,
    d: PleDims,
    conv_state: Option<HirNodeId>,
    chunk: ChunkInputs,
) -> Result<(HirNodeId, Option<HirNodeId>)> {
    let f = DType::F32;
    let (s, si) = (d.seq, d.seq as i64);
    let (h, hi) = (d.streams, d.streams as i64);
    let di = d.hidden as i64;
    let width = d.width();

    // The embeddings come in as a graph input, so their width is the caller's
    // claim rather than a weight's shape — check it, because a wrong
    // `ngram_size` or `head_dim` gives a plausible-looking buffer that the
    // projections would happily contract against.
    {
        let gb = HirMut::new(emit.hir());
        let got = gb.shape(emb).dim(1).unwrap_static();
        if got != d.embed_dim {
            bail!(
                "qwen4exp: {prefix} got {got}-wide n-gram embeddings, but \
                 ple_embed_dim is {}",
                d.embed_dim
            );
        }
    }

    let key = linear(emit, &format!("{prefix}.ple_key.weight"), emb)?; // [s, hc·D]
    let value = linear(emit, &format!("{prefix}.ple_value.weight"), emb)?; // [s, D]

    let hidden_flat = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(hidden, vec![si, width as i64])
    };

    let key_n = grouped_rms_norm(
        emit,
        &format!("{prefix}.ple_norm_key"),
        key,
        h,
        d.hidden,
        d.eps,
        d.norm_offset,
    )?;
    let query_n = grouped_rms_norm(
        emit,
        &format!("{prefix}.ple_norm_query"),
        hidden_flat,
        h,
        d.hidden,
        d.eps,
        d.norm_offset,
    )?;

    let inv_sqrt_d = scalar(
        emit,
        &format!("{prefix}.ple.inv_sqrt_d"),
        1.0 / (d.hidden as f32).sqrt(),
    );
    let floor = scalar(emit, &format!("{prefix}.ple.floor"), 1e-6);

    // ── per-stream dot product, signed square root, sigmoid ──
    let gated_flat = {
        let mut gb = HirMut::new(emit.hir());
        let k3 = gb.reshape_(key_n, vec![si, hi, di]);
        let q3 = gb.reshape_(query_n, vec![si, hi, di]);
        let prod = gb.mul(k3, q3);
        let dot = gb.sum(prod, vec![2], true); // [s, hc, 1]
        let dot = gb.mul(dot, inv_sqrt_d);

        let mag = act(&mut gb, Activation::Abs, dot);
        let mag = gb.add_node(
            Op::Binary(BinaryOp::Max),
            vec![mag, floor],
            Shape::new(&[s, h, 1], f),
        );
        let mag = gb.sqrt(mag);
        let sgn = act(&mut gb, Activation::Sign, dot);
        let signed = gb.mul(sgn, mag);
        let gate = act(&mut gb, Activation::Sigmoid, signed);

        // One value broadcast across the streams, scaled per stream.
        let v3 = gb.reshape_(value, vec![si, 1, di]);
        let gated = gb.mul(v3, gate);
        gb.reshape_(gated, vec![si, width as i64])
    };

    let normed = grouped_rms_norm(
        emit,
        &format!("{prefix}.ple_norm_conv"),
        gated_flat,
        h,
        d.hidden,
        d.eps,
        d.norm_offset,
    )?;

    // `apply_mask_to_padding_states` — the reference zeroes BOTH the gated
    // value and its normed copy before the convolution. With right padding the
    // causal conv never reads a pad from a real position, so this only matters
    // for the window this chunk exports.
    let (gated_flat, normed) = match chunk.pad_mask {
        Some(keep) => {
            let mut gb = HirMut::new(emit.hir());
            let keep = gb.reshape_(keep, vec![si, 1]);
            (gb.mul(gated_flat, keep), gb.mul(normed, keep))
        }
        None => (gated_flat, normed),
    };

    let conv_in = {
        let mut gb = HirMut::new(emit.hir());
        gb.reshape_(normed, vec![1, si, width as i64])
    };
    let (conv, next_conv) = match conv_state {
        Some(st) => {
            let (c, next) = carried_conv1d(
                emit,
                &format!("{prefix}.ple_conv1d.weight"),
                conv_in,
                st,
                s,
                width,
                d.conv_kernel,
                d.dilation,
                chunk.conv_shift,
            )?;
            // `carried_conv1d` is the raw convolution; the SiLU that
            // `dilated_causal_conv1d` folds in has to be applied here.
            let mut gb = HirMut::new(emit.hir());
            (gb.silu(c), Some(next))
        }
        None => (
            dilated_causal_conv1d(
                emit,
                &format!("{prefix}.ple_conv1d.weight"),
                conv_in,
                width,
                s,
                d.conv_kernel,
                d.dilation,
            )?,
            None,
        ),
    };

    let mut gb = HirMut::new(emit.hir());
    let gated4 = gb.reshape_(gated_flat, vec![1, si, hi, di]);
    let conv4 = gb.reshape_(conv, vec![1, si, hi, di]);
    let sum = gb.add(gated4, conv4);
    Ok((gb.add(hidden, sum), next_conv))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash() -> PleHash {
        PleHash {
            ngram_size: 3,
            heads_per_ngram: 2,
            n_heads: 4,
            layer_multipliers: vec![3, 5, 7],
            head_offsets: vec![0, 11, 22, 33],
            head_vocab_sizes: vec![11, 11, 11, 11],
            padded_rows: 44,
            eos_token_id: 9,
        }
    }

    /// Bigram heads see two tokens, trigram heads three — and both fold from
    /// the same first term, so they agree exactly when the third token is
    /// XOR-neutral.
    #[test]
    fn bigram_and_trigram_heads_hash_different_windows() {
        let h = hash();
        let w = [1u64, 2, 3];
        let m = [3u64, 5, 7]; // == hash().layer_multipliers
        let rows = h.rows(&w);
        let bi = (w[0] * m[0]) ^ (w[1] * m[1]);
        let tri = bi ^ (w[2] * m[2]);
        assert_eq!(rows[0] as u64, bi % 11);
        assert_eq!(rows[1] as u64, bi % 11 + 11);
        assert_eq!(rows[2] as u64, tri % 11 + 22);
        assert_eq!(rows[3] as u64, tri % 11 + 33);
    }

    /// The window at the very start of a sequence has no predecessors, so they
    /// read as EOS.
    #[test]
    fn missing_history_reads_as_eos() {
        let h = hash();
        let got = ngram_rows(&h, &[1]);
        let want = h.rows(&[1, 9, 9]);
        assert_eq!(got, want);
    }

    /// An EOS truncates that position and every older one — but never the
    /// token's own position.
    #[test]
    fn eos_truncates_backwards_only() {
        let h = hash();
        // tokens: [4, 9, 7] — token 2's predecessor is EOS.
        let got = ngram_rows(&h, &[4, 9, 7]);
        let want: Vec<u32> = [
            h.rows(&[4, 9, 9]), // nothing before it
            h.rows(&[9, 4, 9]), // IS the eos, but its own position stands
            h.rows(&[7, 9, 9]), // predecessor is eos → it and older become eos
        ]
        .concat();
        assert_eq!(got, want);
    }

    /// A chunked prefill must hash the same windows as a single-shot one; that
    /// is what `prev` is for.
    #[test]
    fn previous_context_reproduces_a_single_shot_batch() {
        let h = hash();
        let all = [1u32, 2, 3, 4, 5];
        let whole = ngram_rows(&h, &all);
        let first = ngram_rows(&h, &all[..3]);
        let rest = ngram_rows_with_history(&h, &all[3..], &all[1..3]);
        assert_eq!(whole, [first, rest].concat());
    }

    #[test]
    fn embeddings_are_laid_out_token_major() {
        let h = hash();
        let emb =
            ngram_embeddings(&h, &[1, 2], &[], 2, |r| Ok(vec![r as f32, -(r as f32)])).unwrap();
        let rows = ngram_rows(&h, &[1, 2]);
        assert_eq!(emb.len(), 2 * 4 * 2);
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(emb[i * 2], *r as f32);
            assert_eq!(emb[i * 2 + 1], -(*r as f32));
        }
    }
}
