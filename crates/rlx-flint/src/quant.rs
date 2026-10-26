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

//! Block-quantized routed experts, written once from the checkpoint.
//!
//! The routed experts are 74 of the checkpoint's 80 GB and every decode step
//! reads ~1.7 GB of them, so on a machine whose RAM cannot cache them all,
//! decode waits on the disk. Fewer bytes per weight is fewer bytes to wait
//! for — and at 4 bits the experts fit in the page cache of a 64 GB machine.
//!
//! The blocks are llama.cpp's, 32 weights each, one f16 scale `d` per block:
//!
//! | scheme | bytes / block | value | bytes / weight |
//! |---|--:|---|--:|
//! | [`ExpertScheme::Q8_0`] | 34 | `d · q`, `q ∈ [-127, 127]` | 1.0625 |
//! | [`ExpertScheme::Q4_0`] | 18 | `d · (q - 8)`, `q ∈ [0, 15]`; byte `j` holds weight `j` (low nibble) and `j + 16` (high) | 0.5625 |
//!
//! Each projection keeps its `[out, in]` row-major layout, so a block of
//! output rows is still one contiguous read. The store is one file,
//! `experts-<scheme>.bin`, plus `experts.json` mapping each tensor name to its
//! offset. Only routed experts are quantized; everything else is read from the
//! checkpoint as before.

use anyhow::{Context, Result, anyhow, bail};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::checkpoint::Checkpoint;
use crate::config::FlintConfig;
use crate::moe::{EXPERT_PROJS, expert_key};

/// Weights per block.
pub const QK: usize = 32;

/// How the routed experts are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertScheme {
    Q8_0,
    Q4_0,
}

impl ExpertScheme {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "q8_0" | "q8" => Ok(Self::Q8_0),
            "q4_0" | "q4" => Ok(Self::Q4_0),
            other => bail!("flint: unknown expert scheme `{other}` (q8_0, q4_0)"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q8_0 => 2 + QK,
            Self::Q4_0 => 2 + QK / 2,
        }
    }

    /// Bytes of one row of `cols` weights (`cols` a multiple of [`QK`]).
    pub fn row_bytes(self, cols: usize) -> usize {
        cols / QK * self.block_bytes()
    }

    /// Quantize one row (`src.len()` a multiple of [`QK`]) into `dst`.
    pub fn quantize_row(self, src: &[f32], dst: &mut [u8]) {
        for (x, out) in src
            .as_chunks::<QK>()
            .0
            .iter()
            .zip(dst.chunks_exact_mut(self.block_bytes()))
        {
            match self {
                Self::Q8_0 => {
                    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let d = amax / 127.0;
                    let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
                    out[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
                    for (q, v) in out[2..].iter_mut().zip(x) {
                        *q = ((v * inv).round().clamp(-127.0, 127.0) as i8) as u8;
                    }
                }
                Self::Q4_0 => {
                    // llama.cpp: the scale carries the sign of the largest
                    // magnitude, so that value lands exactly on -8.
                    let max = x
                        .iter()
                        .fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
                    let d = max / -8.0;
                    let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
                    out[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
                    let q = |v: f32| ((v * inv + 8.5) as i32).clamp(0, 15) as u8;
                    for j in 0..QK / 2 {
                        out[2 + j] = q(x[j]) | (q(x[j + QK / 2]) << 4);
                    }
                }
            }
        }
    }

    /// The weights one stored row decodes to.
    pub fn dequantize_row(self, row: &[u8], out: &mut [f32]) {
        for (b, o) in row
            .chunks_exact(self.block_bytes())
            .zip(out.as_chunks_mut::<QK>().0)
        {
            let d = scale(b);
            match self {
                Self::Q8_0 => {
                    for (v, &q) in o.iter_mut().zip(&b[2..]) {
                        *v = d * (q as i8) as f32;
                    }
                }
                Self::Q4_0 => {
                    for j in 0..QK / 2 {
                        o[j] = d * ((b[2 + j] & 0x0f) as i32 - 8) as f32;
                        o[j + QK / 2] = d * ((b[2 + j] >> 4) as i32 - 8) as f32;
                    }
                }
            }
        }
    }

    /// `Σ x[i] · w[i]` over one stored row.
    #[inline]
    pub fn dot(self, x: &[f32], row: &[u8]) -> f32 {
        match self {
            Self::Q8_0 => dot_q8_0(x, row),
            Self::Q4_0 => dot_q4_0(x, row),
        }
    }
}

#[inline]
fn scale(b: &[u8]) -> f32 {
    half::f16::from_le_bytes([b[0], b[1]]).to_f32()
}

#[inline]
fn dot_q8_0(x: &[f32], row: &[u8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; the slices are whole blocks.
    unsafe {
        neon::dot_q8_0(x, row)
    }
    #[cfg(not(target_arch = "aarch64"))]
    dot_q8_0_scalar(x, row)
}

#[inline]
fn dot_q4_0(x: &[f32], row: &[u8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: as above.
    unsafe {
        neon::dot_q4_0(x, row)
    }
    #[cfg(not(target_arch = "aarch64"))]
    dot_q4_0_scalar(x, row)
}

/// The block dots with explicit widening: the compiler did not vectorize the
/// scalar loops below, which ran 4-5x slower per weight than the bf16 dot.
/// Activations stay f32, so these add no error over the dequantized weights.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{QK, scale};
    use std::arch::aarch64::*;

    /// `Σ x[i] · q[i]` over 32 int8 in two registers, into `acc`.
    #[inline(always)]
    unsafe fn block(x: *const f32, q0: int8x16_t, q1: int8x16_t) -> float32x4_t {
        unsafe {
            let w = [
                vmovl_s8(vget_low_s8(q0)),
                vmovl_high_s8(q0),
                vmovl_s8(vget_low_s8(q1)),
                vmovl_high_s8(q1),
            ];
            let (mut a, mut b) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            for (j, w) in w.iter().enumerate() {
                let lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(*w)));
                let hi = vcvtq_f32_s32(vmovl_high_s16(*w));
                a = vfmaq_f32(a, lo, vld1q_f32(x.add(8 * j)));
                b = vfmaq_f32(b, hi, vld1q_f32(x.add(8 * j + 4)));
            }
            vaddq_f32(a, b)
        }
    }

    pub(super) unsafe fn dot_q8_0(x: &[f32], row: &[u8]) -> f32 {
        unsafe {
            let mut total = vdupq_n_f32(0.0);
            for (xs, b) in x
                .as_chunks::<QK>()
                .0
                .iter()
                .zip(row.as_chunks::<{ 2 + QK }>().0.iter())
            {
                let q = b.as_ptr().add(2) as *const i8;
                let s = block(xs.as_ptr(), vld1q_s8(q), vld1q_s8(q.add(16)));
                total = vfmaq_n_f32(total, s, scale(b));
            }
            vaddvq_f32(total)
        }
    }

    pub(super) unsafe fn dot_q4_0(x: &[f32], row: &[u8]) -> f32 {
        unsafe {
            let (mask, eight) = (vdupq_n_u8(0x0f), vdupq_n_s8(8));
            let mut total = vdupq_n_f32(0.0);
            for (xs, b) in x
                .as_chunks::<QK>()
                .0
                .iter()
                .zip(row.as_chunks::<{ 2 + QK / 2 }>().0.iter())
            {
                let v = vld1q_u8(b.as_ptr().add(2));
                // Low nibbles are weights 0..16, high nibbles 16..32.
                let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask)), eight);
                let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), eight);
                let s = block(xs.as_ptr(), lo, hi);
                total = vfmaq_n_f32(total, s, scale(b));
            }
            vaddvq_f32(total)
        }
    }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn dot_q8_0_scalar(x: &[f32], row: &[u8]) -> f32 {
    let mut sum = 0f32;
    for (xs, b) in x
        .as_chunks::<QK>()
        .0
        .iter()
        .zip(row.as_chunks::<{ 2 + QK }>().0.iter())
    {
        let q = &b[2..];
        let mut acc = [0f32; 8];
        for l in 0..QK / 8 {
            for i in 0..8 {
                acc[i] = xs[l * 8 + i].mul_add(q[l * 8 + i] as i8 as f32, acc[i]);
            }
        }
        sum = scale(b).mul_add(acc.iter().sum::<f32>(), sum);
    }
    sum
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn dot_q4_0_scalar(x: &[f32], row: &[u8]) -> f32 {
    let mut sum = 0f32;
    for (xs, b) in x
        .as_chunks::<QK>()
        .0
        .iter()
        .zip(row.as_chunks::<{ 2 + QK / 2 }>().0.iter())
    {
        let q = &b[2..];
        let (lo_x, hi_x) = xs.split_at(QK / 2);
        let mut acc = [0f32; 16];
        for j in 0..QK / 2 {
            let lo = (q[j] & 0x0f) as i32 - 8;
            let hi = (q[j] >> 4) as i32 - 8;
            acc[j] = lo_x[j].mul_add(lo as f32, acc[j]);
            acc[j] = hi_x[j].mul_add(hi as f32, acc[j]);
        }
        sum = scale(b).mul_add(acc.iter().sum::<f32>(), sum);
    }
    sum
}

/// Where one quantized tensor sits in the store.
#[derive(Debug, Clone)]
pub struct QuantEntry {
    pub offset: u64,
    pub rows: usize,
    pub cols: usize,
}

/// An opened `experts.json` + `experts-<scheme>.bin`.
pub struct QuantStore {
    pub scheme: ExpertScheme,
    dir: PathBuf,
    file: File,
    index: HashMap<String, QuantEntry>,
}

const INDEX: &str = "experts.json";

impl QuantStore {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let meta: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join(INDEX))
                .with_context(|| format!("flint: reading {dir:?}/{INDEX}"))?,
        )?;
        let scheme = ExpertScheme::parse(meta["scheme"].as_str().unwrap_or_default())?;
        let file_name = meta["file"]
            .as_str()
            .ok_or_else(|| anyhow!("flint: {INDEX} has no `file`"))?;
        let file = File::open(dir.join(file_name))
            .with_context(|| format!("flint: opening {dir:?}/{file_name}"))?;
        let mut index = HashMap::new();
        for (name, v) in meta["tensors"]
            .as_object()
            .ok_or_else(|| anyhow!("flint: {INDEX} has no `tensors`"))?
        {
            let a = v.as_array().filter(|a| a.len() == 3);
            let get = |i: usize| a.and_then(|a| a[i].as_u64());
            let (Some(offset), Some(rows), Some(cols)) = (get(0), get(1), get(2)) else {
                bail!("flint: {INDEX}: bad entry for {name}");
            };
            index.insert(
                name.clone(),
                QuantEntry {
                    offset,
                    rows: rows as usize,
                    cols: cols as usize,
                },
            );
        }
        Ok(QuantStore {
            scheme,
            dir,
            file,
            index,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn entry(&self, name: &str) -> Result<&QuantEntry> {
        self.index
            .get(name)
            .ok_or_else(|| anyhow!("flint: `{name}` is not in the {} store", self.scheme.name()))
    }

    /// Stored rows `row0..` of `e` into `buf` (whole rows), with `pread`.
    pub fn read_rows(&self, e: &QuantEntry, row0: usize, buf: &mut [u8]) -> Result<()> {
        let rb = self.scheme.row_bytes(e.cols);
        if !buf.len().is_multiple_of(rb) || row0 * rb + buf.len() > e.rows * rb {
            bail!(
                "flint: {} bytes from row {row0} do not fit the tensor",
                buf.len()
            );
        }
        read_exact_at(&self.file, buf, e.offset + (row0 * rb) as u64)
    }

    /// Start reading `e` into the page cache; see [`Checkpoint::advise_read`].
    pub fn advise_read(&self, e: &QuantEntry) {
        crate::checkpoint::advise(&self.file, e.offset, e.rows * self.scheme.row_bytes(e.cols));
    }
}

fn read_exact_at(file: &File, buf: &mut [u8], at: u64) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, at)
            .context("flint: reading the expert store")
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            let n = file.seek_read(&mut buf[done..], at + done as u64)?;
            if n == 0 {
                bail!("flint: the expert store ended early");
            }
            done += n;
        }
        Ok(())
    }
}

/// Quantize every routed expert of `ckpt` into `out_dir`, in parallel over
/// tensors. Returns the store's size in bytes.
pub fn write_store(
    ckpt: &Checkpoint,
    cfg: &FlintConfig,
    scheme: ExpertScheme,
    out_dir: &Path,
) -> Result<u64> {
    std::fs::create_dir_all(out_dir)?;
    if !cfg.hidden_size.is_multiple_of(QK) || !cfg.moe_intermediate_size.is_multiple_of(QK) {
        bail!("flint: expert widths must be multiples of {QK} to quantize");
    }
    // Lay every tensor out first, so the writes can run in any order.
    let mut layout: Vec<(String, u64, usize, usize)> = Vec::new();
    let mut at = 0u64;
    for l in cfg.moe_layers() {
        for e in 0..cfg.num_experts {
            for proj in EXPERT_PROJS {
                let name = expert_key(l, e, proj);
                let t = ckpt.entry(&name)?;
                let (rows, cols) = (t.shape[0], t.shape[1]);
                layout.push((name, at, rows, cols));
                at += (rows * scheme.row_bytes(cols)) as u64;
            }
        }
    }
    let file_name = format!("experts-{}.bin", scheme.name());
    let file = File::create(out_dir.join(&file_name))?;
    file.set_len(at)?;
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total = layout.len();
    layout
        .par_iter()
        .try_for_each(|(name, off, rows, cols)| -> Result<()> {
            let (w, _) = ckpt.load_f32(name)?;
            let rb = scheme.row_bytes(*cols);
            let mut out = vec![0u8; rows * rb];
            for (src, dst) in w.chunks_exact(*cols).zip(out.chunks_exact_mut(rb)) {
                scheme.quantize_row(src, dst);
            }
            write_all_at(&file, &out, *off)?;
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_multiple_of(3000) {
                eprintln!("[rlx-flint] quantized {n}/{total} expert matrices");
            }
            Ok(())
        })?;
    let tensors: serde_json::Map<String, serde_json::Value> = layout
        .iter()
        .map(|(n, off, r, c)| (n.clone(), serde_json::json!([off, r, c])))
        .collect();
    let meta = serde_json::json!({
        "scheme": scheme.name(),
        "file": file_name,
        "block": QK,
        "source": ckpt.dir(),
        "tensors": tensors,
    });
    std::fs::write(out_dir.join(INDEX), serde_json::to_vec(&meta)?)?;
    Ok(at)
}

fn write_all_at(file: &File, buf: &[u8], at: u64) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_all_at(buf, at)
            .context("flint: writing the expert store")
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            done += file.seek_write(&buf[done..], at + done as u64)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013 + if i % 29 == 0 { 0.9 } else { 0.0 })
            .collect()
    }

    #[test]
    fn quantized_dots_track_the_f32_dot() {
        let (w, x) = (row(2048), row(2048).into_iter().rev().collect::<Vec<_>>());
        let exact: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        let norm: f32 = w.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
        for (scheme, tol) in [(ExpertScheme::Q8_0, 0.005), (ExpertScheme::Q4_0, 0.08)] {
            let mut q = vec![0u8; scheme.row_bytes(w.len())];
            scheme.quantize_row(&w, &mut q);
            let got = scheme.dot(&x, &q);
            let rel = (got - exact).abs() / norm;
            assert!(rel < tol, "{scheme:?}: {got} vs {exact} (rel {rel:.4})");
        }
    }

    #[test]
    fn vector_dots_match_the_scalar_dots() {
        let (w, x) = (row(4096), row(4096).into_iter().rev().collect::<Vec<_>>());
        for scheme in [ExpertScheme::Q8_0, ExpertScheme::Q4_0] {
            let mut q = vec![0u8; scheme.row_bytes(w.len())];
            scheme.quantize_row(&w, &mut q);
            let scalar = match scheme {
                ExpertScheme::Q8_0 => dot_q8_0_scalar(&x, &q),
                ExpertScheme::Q4_0 => dot_q4_0_scalar(&x, &q),
            };
            let got = scheme.dot(&x, &q);
            assert!(
                (got - scalar).abs() <= 1e-4 * scalar.abs().max(1.0),
                "{scheme:?}"
            );
        }
    }

    #[test]
    fn q4_0_round_trips_a_representable_block() {
        // Values that are exact multiples of the scale come back exactly.
        let d = 0.25f32;
        let x: Vec<f32> = (0..QK).map(|i| d * ((i % 16) as f32 - 8.0)).collect();
        let mut q = vec![0u8; ExpertScheme::Q4_0.row_bytes(QK)];
        ExpertScheme::Q4_0.quantize_row(&x, &mut q);
        for i in 0..QK {
            let mut onehot = vec![0f32; QK];
            onehot[i] = 1.0;
            assert_eq!(ExpertScheme::Q4_0.dot(&onehot, &q), x[i], "weight {i}");
        }
    }
}
