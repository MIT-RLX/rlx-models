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

//! Routed experts on the host, computed straight from the checkpoint's bytes.
//!
//! ```text
//!   y[r] = Σ_k w[r,k] · down_e(silu(gate_e(x[r])) · up_e(x[r]))     e = ids[r,k]
//! ```
//!
//! Each projection is read in blocks of [`ROW_BLOCK`] output rows — HF stores
//! a linear as `[out, in]`, so a block of outputs is one contiguous `pread` —
//! and every output is a dot product over that block, in the stored dtype. A
//! task is one (expert, projection, block): a decode step's six experts give
//! ~100 tasks per projection pass, enough for every core.
//!
//! This replaced decoding each fired expert to f32, transposing it into a bank
//! and running a grouped-GEMM graph over the bank. Per token that path wrote
//! ~3.5 GB of f32 banks from 1.8 GB of bf16 and copied them again into the
//! graph's arena; here the 1.8 GB is read once and nothing else is written.
//! The values are the same — bf16 widens to f32 exactly and accumulation is
//! f32 — but the summation order differs, so results move at f32 rounding.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::cell::RefCell;

use crate::checkpoint::{Checkpoint, StoredDtype, decode_into};
use crate::config::FlintConfig;
use crate::moe::expert_key;
use crate::quant::{ExpertScheme, QuantStore};

/// Where the routed experts are read from.
#[derive(Clone, Copy)]
pub enum ExpertWeights<'a> {
    /// The checkpoint itself, in its stored dtype.
    Checkpoint(&'a Checkpoint),
    /// A block-quantized copy ([`crate::quant`]).
    Quantized(&'a QuantStore),
}

impl ExpertWeights<'_> {
    /// Start reading one projection into the page cache.
    pub fn advise(&self, layer: usize, expert: usize, proj: &str) {
        let name = expert_key(layer, expert, proj);
        match self {
            Self::Checkpoint(c) => {
                if let Ok(e) = c.entry(&name) {
                    c.advise_read(e);
                }
            }
            Self::Quantized(q) => {
                if let Ok(e) = q.entry(&name) {
                    q.advise_read(e);
                }
            }
        }
    }
}

/// Output rows one task reads and computes.
pub const ROW_BLOCK: usize = 64;

/// One expert this step fired, and the rows that chose it.
struct Fired {
    expert: usize,
    /// Rows of `x`, in first-use order.
    rows: Vec<usize>,
}

thread_local! {
    /// Raw bytes of the block being read, kept 4-byte aligned so they can be
    /// viewed as bf16 or f32 in place.
    static RAW: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    /// The block widened to f32, for dtypes that cannot be read in place.
    static WIDE: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// A block of weight rows, `[rows, cols]`, as stored.
enum Block<'a> {
    Bf16(&'a [u16]),
    F32(&'a [f32]),
    /// Whole quantized rows.
    Quant(ExpertScheme, &'a [u8]),
}

/// Read rows `row0..row0 + nrows` of expert `expert`'s `proj` (`[out, inp]`)
/// and hand them to `f`.
#[allow(clippy::too_many_arguments)]
fn with_block<T>(
    src: ExpertWeights<'_>,
    layer: usize,
    expert: usize,
    proj: &str,
    (out, inp): (usize, usize),
    row0: usize,
    nrows: usize,
    f: impl FnOnce(Block<'_>) -> T,
) -> Result<T> {
    let name = expert_key(layer, expert, proj);
    let ckpt = match src {
        ExpertWeights::Checkpoint(c) => c,
        ExpertWeights::Quantized(q) => {
            let e = q.entry(&name)?;
            if (e.rows, e.cols) != (out, inp) {
                bail!(
                    "flint: {name} is [{}, {}], expected [{out}, {inp}]",
                    e.rows,
                    e.cols
                );
            }
            let bytes = nrows * q.scheme.row_bytes(inp);
            return RAW.with(|raw| {
                let mut raw = raw.borrow_mut();
                raw.resize(bytes.div_ceil(4), 0);
                // SAFETY: a u32 buffer of at least `bytes` bytes, viewed as bytes.
                let buf =
                    unsafe { std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u8, bytes) };
                q.read_rows(e, row0, buf)?;
                Ok(f(Block::Quant(q.scheme, buf)))
            });
        }
    };
    let e = ckpt.entry(&name)?;
    if e.shape != [out, inp] {
        bail!("flint: {name} is {:?}, expected [{out}, {inp}]", e.shape);
    }
    let n = nrows * inp;
    let bytes = n * e.dtype.bytes();
    RAW.with(|raw| {
        let mut raw = raw.borrow_mut();
        raw.resize(bytes.div_ceil(4), 0);
        // SAFETY: a u32 buffer of at least `bytes` bytes, viewed as bytes.
        let buf = unsafe { std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u8, bytes) };
        ckpt.read_rows(e, row0, buf)?;
        // Stored little-endian: read in place where that matches the target.
        let in_place = cfg!(target_endian = "little");
        Ok(match e.dtype {
            StoredDtype::BF16 if in_place => {
                // SAFETY: 4-byte aligned, `n` u16s long.
                f(Block::Bf16(unsafe {
                    std::slice::from_raw_parts(raw.as_ptr() as *const u16, n)
                }))
            }
            StoredDtype::F32 if in_place => {
                // SAFETY: 4-byte aligned, `n` f32s long.
                f(Block::F32(unsafe {
                    std::slice::from_raw_parts(raw.as_ptr() as *const f32, n)
                }))
            }
            dtype => WIDE.with(|wide| {
                let mut wide = wide.borrow_mut();
                wide.resize(n, 0.0);
                decode_into(dtype, buf, &mut wide);
                f(Block::F32(&wide))
            }),
        })
    })
}

/// `Σ x[i] · bf16(w[i])`, sixteen independent lanes so the loop vectorizes.
#[inline]
fn dot_bf16(x: &[f32], w: &[u16]) -> f32 {
    const L: usize = 16;
    let mut acc = [0f32; L];
    let ((xc, xr), (wc, wr)) = (x.as_chunks::<L>(), w.as_chunks::<L>());
    let tail: f32 = xr
        .iter()
        .zip(wr)
        .map(|(&a, &b)| a * f32::from_bits((b as u32) << 16))
        .sum();
    for (xs, ws) in xc.iter().zip(wc) {
        for l in 0..L {
            acc[l] = xs[l].mul_add(f32::from_bits((ws[l] as u32) << 16), acc[l]);
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// `Σ x[i] · w[i]`, as [`dot_bf16`].
#[inline]
pub(crate) fn dot_f32(x: &[f32], w: &[f32]) -> f32 {
    const L: usize = 16;
    let mut acc = [0f32; L];
    let ((xc, xr), (wc, wr)) = (x.as_chunks::<L>(), w.as_chunks::<L>());
    let tail: f32 = xr.iter().zip(wr).map(|(&a, &b)| a * b).sum();
    for (xs, ws) in xc.iter().zip(wc) {
        for l in 0..L {
            acc[l] = xs[l].mul_add(ws[l], acc[l]);
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// `out[r, j] = inputs[r] · w[j]` for every input row and every weight row of
/// `block` (`[nrows, cols]`): `[inputs.len(), nrows]`.
fn block_outputs(block: Block<'_>, cols: usize, inputs: &[&[f32]]) -> Vec<f32> {
    let mut out = Vec::new();
    match block {
        Block::Quant(scheme, w) => {
            for x in inputs {
                out.extend(
                    w.chunks_exact(scheme.row_bytes(cols))
                        .map(|row| scheme.dot(x, row)),
                );
            }
        }
        Block::Bf16(w) => {
            for x in inputs {
                out.extend(w.chunks_exact(cols).map(|row| dot_bf16(x, row)));
            }
        }
        Block::F32(w) => {
            for x in inputs {
                out.extend(w.chunks_exact(cols).map(|row| dot_f32(x, row)));
            }
        }
    }
    out
}

/// One projection over every fired expert: for expert `f` and each of its
/// rows `r`, `inputs(f, r) · W_fᵀ` with `W_f` `[out, inp]`. Returns, per
/// expert, `[rows_f, out]` row-major.
fn project<'a>(
    src: ExpertWeights<'_>,
    layer: usize,
    fired: &[Fired],
    proj: &str,
    (out, inp): (usize, usize),
    inputs: &(dyn Fn(usize, usize) -> &'a [f32] + Sync),
) -> Result<Vec<Vec<f32>>> {
    let blocks = out.div_ceil(ROW_BLOCK);
    let tasks: Vec<(usize, usize)> = (0..fired.len())
        .flat_map(|f| (0..blocks).map(move |b| (f, b)))
        .collect();
    let parts: Vec<Vec<f32>> = tasks
        .par_iter()
        .map(|&(f, b)| -> Result<Vec<f32>> {
            let (row0, nrows) = (b * ROW_BLOCK, ROW_BLOCK.min(out - b * ROW_BLOCK));
            let xs: Vec<&[f32]> = (0..fired[f].rows.len()).map(|r| inputs(f, r)).collect();
            with_block(
                src,
                layer,
                fired[f].expert,
                proj,
                (out, inp),
                row0,
                nrows,
                |blk| block_outputs(blk, inp, &xs),
            )
        })
        .collect::<Result<_>>()
        .with_context(|| format!("flint: layer {layer} experts' {proj}"))?;
    // Scatter each task's `[rows_f, nrows]` into its expert's `[rows_f, out]`.
    let mut res: Vec<Vec<f32>> = fired
        .iter()
        .map(|f| vec![0.0; f.rows.len() * out])
        .collect();
    for (&(f, b), part) in tasks.iter().zip(&parts) {
        let (row0, nrows) = (b * ROW_BLOCK, ROW_BLOCK.min(out - b * ROW_BLOCK));
        for (r, vals) in part.chunks_exact(nrows).enumerate() {
            res[f][r * out + row0..r * out + row0 + nrows].copy_from_slice(vals);
        }
    }
    Ok(res)
}

/// `out[r] += Σ_k w[r,k] · expert_{ids[r,k]}(x[r])` for the first `real` rows
/// of `x` (`[rows, hidden]`); `ids` / `w` are `[real, top_k]`, `ids`
/// f32-encoded. Returns how many distinct experts were read.
#[allow(clippy::too_many_arguments)]
pub fn add_routed(
    src: ExpertWeights<'_>,
    cfg: &FlintConfig,
    layer: usize,
    x: &[f32],
    ids: &[f32],
    w: &[f32],
    real: usize,
    out: &mut [f32],
) -> Result<usize> {
    let (hsz, inter, k) = (
        cfg.hidden_size,
        cfg.moe_intermediate_size,
        cfg.num_experts_per_tok,
    );
    if ids.len() != real * k || w.len() != real * k || x.len() < real * hsz {
        bail!("flint: layer {layer} routing does not cover {real} rows");
    }
    // Group the (row, k) picks by expert; `pick[r * k + j]` = (expert slot,
    // position of row r among that expert's rows).
    let mut fired: Vec<Fired> = Vec::new();
    let mut slot_of = vec![usize::MAX; cfg.num_experts];
    let mut pick = Vec::with_capacity(real * k);
    for r in 0..real {
        for j in 0..k {
            let e = ids[r * k + j];
            if !(e >= 0.0 && (e as usize) < cfg.num_experts) {
                bail!("flint: layer {layer} routed row {r} to expert {e}");
            }
            let e = e as usize;
            if slot_of[e] == usize::MAX {
                slot_of[e] = fired.len();
                fired.push(Fired {
                    expert: e,
                    rows: Vec::new(),
                });
            }
            let f = slot_of[e];
            pick.push((f, fired[f].rows.len()));
            fired[f].rows.push(r);
        }
    }

    let row_of = |f: usize, i: usize| {
        let r = fired[f].rows[i];
        &x[r * hsz..(r + 1) * hsz]
    };
    let gate = project(src, layer, &fired, "gate_proj", (inter, hsz), &row_of)?;
    let up = project(src, layer, &fired, "up_proj", (inter, hsz), &row_of)?;
    let act: Vec<Vec<f32>> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| {
            g.iter()
                .zip(u)
                .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                .collect()
        })
        .collect();
    let act_of = |f: usize, i: usize| &act[f][i * inter..(i + 1) * inter];
    let down = project(src, layer, &fired, "down_proj", (hsz, inter), &act_of)?;

    // Add in (row, k) order, the routed graph's order.
    for r in 0..real {
        let dst = &mut out[r * hsz..(r + 1) * hsz];
        for j in 0..k {
            let (f, i) = pick[r * k + j];
            let wk = w[r * k + j];
            for (d, &y) in dst.iter_mut().zip(&down[f][i * hsz..(i + 1) * hsz]) {
                *d += wk * y;
            }
        }
    }
    Ok(fired.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dots_match_a_plain_sum() {
        for len in [1usize, 15, 16, 17, 100, 2048] {
            let x: Vec<f32> = (0..len)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1)
                .collect();
            let wf: Vec<f32> = (0..len)
                .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1)
                .collect();
            let wb: Vec<u16> = wf.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let plain_f: f64 = x
                .iter()
                .zip(&wf)
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            let plain_b: f64 = x
                .iter()
                .zip(&wb)
                .map(|(a, b)| (*a as f64) * (f32::from_bits((*b as u32) << 16) as f64))
                .sum();
            assert!(
                (dot_f32(&x, &wf) as f64 - plain_f).abs() < 1e-4,
                "f32 len {len}"
            );
            assert!(
                (dot_bf16(&x, &wb) as f64 - plain_b).abs() < 1e-4,
                "bf16 len {len}"
            );
        }
    }
}
