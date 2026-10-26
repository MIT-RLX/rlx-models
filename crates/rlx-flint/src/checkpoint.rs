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

//! A sharded safetensors checkpoint, mapped once and indexed once.
//!
//! The published model is 15 shards and 36 734 tensors, 36 096 of them routed
//! experts that are read a few at a time for as long as the process runs. So
//! every shard header is parsed **once**, at open, into a name → (shard,
//! offset, dtype, shape) table, and an expert read is one `pread` at a known
//! offset ([`Checkpoint::read_rows`]) — no header parse, and the OS page
//! cache decides which experts stay warm. (`rlx_core::safetensors_checkpoint` re-parses a
//! shard's ~230 KB header per tensor, which is fine for loading a model once
//! and ruinous at 850 expert reads per token.)
//!
//! [`CheckpointRef`] makes it a [`WeightSource`], read non-destructively, so
//! the per-layer graphs are built from it directly and more than once.

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;
use rlx_flow::WeightSource;
use rlx_ir::DType;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Storage dtype of a tensor. Only the float types a Flint checkpoint uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredDtype {
    F32,
    F16,
    BF16,
}

impl StoredDtype {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::BF16),
            _ => None,
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

/// Where one tensor's bytes are.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    shard: usize,
    /// Absolute byte offset in the shard file.
    offset: usize,
    pub dtype: StoredDtype,
    pub shape: Vec<usize>,
}

impl TensorEntry {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct Checkpoint {
    dir: PathBuf,
    shards: Vec<Arc<Mmap>>,
    /// The same files, for `pread` — see [`Checkpoint::read_rows`].
    files: Vec<File>,
    index: HashMap<String, TensorEntry>,
}

impl Checkpoint {
    /// Open a directory (`model.safetensors.index.json`, or every
    /// `*.safetensors` in it) or a single `.safetensors` file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let (dir, files): (PathBuf, Vec<PathBuf>) = if path.is_file() {
            (
                path.parent().unwrap_or(Path::new(".")).to_path_buf(),
                vec![path.to_path_buf()],
            )
        } else {
            let idx = path.join("model.safetensors.index.json");
            let mut files: Vec<PathBuf> = if idx.is_file() {
                let v: serde_json::Value = serde_json::from_str(
                    &std::fs::read_to_string(&idx).with_context(|| format!("reading {idx:?}"))?,
                )?;
                let mut names: Vec<String> = v
                    .get("weight_map")
                    .and_then(|m| m.as_object())
                    .ok_or_else(|| anyhow!("{idx:?} has no weight_map"))?
                    .values()
                    .filter_map(|s| s.as_str().map(str::to_string))
                    .collect();
                names.sort();
                names.dedup();
                names.into_iter().map(|n| path.join(n)).collect()
            } else {
                std::fs::read_dir(path)
                    .with_context(|| format!("listing {path:?}"))?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                    .collect()
            };
            files.sort();
            (path.to_path_buf(), files)
        };
        if files.is_empty() {
            bail!("flint: no safetensors files under {path:?}");
        }

        let mut shards = Vec::with_capacity(files.len());
        let mut handles = Vec::with_capacity(files.len());
        let mut index = HashMap::new();
        for (si, file) in files.iter().enumerate() {
            let f = File::open(file).with_context(|| format!("opening {file:?}"))?;
            // SAFETY: the checkpoint is treated as read-only for the life of the
            // map; a concurrent writer would be a caller error.
            let mmap = unsafe { Mmap::map(&f) }.with_context(|| format!("mapping {file:?}"))?;
            if mmap.len() < 8 {
                bail!("flint: {file:?} is too short to be safetensors");
            }
            let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(
                mmap.get(8..8 + n)
                    .ok_or_else(|| anyhow!("{file:?}: truncated header"))?,
            )
            .with_context(|| format!("parsing {file:?} header"))?;
            let base = 8 + n;
            for (name, meta) in header {
                if name == "__metadata__" {
                    continue;
                }
                let dtype_s = meta.get("dtype").and_then(|d| d.as_str()).unwrap_or("");
                let dtype = StoredDtype::parse(dtype_s)
                    .ok_or_else(|| anyhow!("flint: {name} has unsupported dtype {dtype_s}"))?;
                let shape: Vec<usize> = meta
                    .get("shape")
                    .and_then(|s| s.as_array())
                    .ok_or_else(|| anyhow!("{name}: no shape"))?
                    .iter()
                    .map(|x| x.as_u64().unwrap_or(0) as usize)
                    .collect();
                let offs = meta
                    .get("data_offsets")
                    .and_then(|s| s.as_array())
                    .ok_or_else(|| anyhow!("{name}: no data_offsets"))?;
                let (a, b) = (
                    offs[0].as_u64().unwrap_or(0) as usize,
                    offs[1].as_u64().unwrap_or(0) as usize,
                );
                let entry = TensorEntry {
                    shard: si,
                    offset: base + a,
                    dtype,
                    shape,
                };
                if b - a != entry.numel() * dtype.bytes() || base + b > mmap.len() {
                    bail!("flint: {name} in {file:?} has inconsistent offsets");
                }
                index.insert(name, entry);
            }
            shards.push(Arc::new(mmap));
            handles.push(f);
        }
        Ok(Checkpoint {
            dir,
            shards,
            files: handles,
            index,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn entry(&self, name: &str) -> Result<&TensorEntry> {
        self.index
            .get(name)
            .ok_or_else(|| anyhow!("flint: tensor `{name}` is not in {:?}", self.dir))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    fn bytes(&self, e: &TensorEntry) -> &[u8] {
        let len = e.numel() * e.dtype.bytes();
        &self.shards[e.shard][e.offset..e.offset + len]
    }

    /// A tensor's stored bytes, borrowed from the map.
    pub fn raw(&self, name: &str) -> Result<&[u8]> {
        Ok(self.bytes(self.entry(name)?))
    }

    /// The whole tensor as f32, with its shape.
    pub fn load_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let e = self.entry(name)?;
        let mut out = vec![0.0f32; e.numel()];
        decode_into(e.dtype, self.bytes(e), &mut out);
        Ok((out, e.shape.clone()))
    }

    /// Rows of a rank-2 tensor as f32, concatenated — the embedding lookup.
    pub fn rows_f32(&self, name: &str, rows: &[u32]) -> Result<Vec<f32>> {
        let e = self.entry(name)?;
        if e.shape.len() != 2 {
            bail!("flint: {name} is {:?}, not a matrix", e.shape);
        }
        let (n, cols) = (e.shape[0], e.shape[1]);
        let bytes = self.bytes(e);
        let w = cols * e.dtype.bytes();
        let mut out = vec![0.0f32; rows.len() * cols];
        for (i, &r) in rows.iter().enumerate() {
            let r = r as usize;
            if r >= n {
                bail!("flint: row {r} is past {name}'s {n} rows");
            }
            decode_into(
                e.dtype,
                &bytes[r * w..(r + 1) * w],
                &mut out[i * cols..(i + 1) * cols],
            );
        }
        Ok(out)
    }

    /// Rows `row0..row0 + buf.len() / row_bytes` of a rank-2 tensor, as
    /// stored, into `buf`. Rows are contiguous in safetensors' row-major
    /// layout, so a row block is one read.
    ///
    /// This is the expert hot path, and it uses `pread` rather than the memory
    /// map: first touch of a mapped page is a fault even when the page is
    /// already in the page cache, and a 2 MB expert is ~128 of them — ~100 000
    /// a token. `pread` copies straight out of the page cache instead.
    pub fn read_rows(&self, e: &TensorEntry, row0: usize, buf: &mut [u8]) -> Result<()> {
        if e.shape.len() != 2 {
            bail!("flint: a {:?} tensor has no rows to read", e.shape);
        }
        let row_bytes = e.shape[1] * e.dtype.bytes();
        let end = row0 * row_bytes + buf.len();
        if !buf.len().is_multiple_of(row_bytes) || end > e.numel() * e.dtype.bytes() {
            bail!(
                "flint: {} bytes from row {row0} do not fit a {:?} tensor",
                buf.len(),
                e.shape
            );
        }
        self.read_range(e, row0 * row_bytes, buf)
    }

    /// Ask the OS to start reading `e` into the page cache, and return at once
    /// (`F_RDADVISE` on macOS, `posix_fadvise(WILLNEED)` on Linux; nothing
    /// elsewhere). A later [`Checkpoint::read_rows`] then copies from memory
    /// instead of waiting on the disk. Best effort: an error is ignored.
    pub fn advise_read(&self, e: &TensorEntry) {
        advise(
            &self.files[e.shard],
            e.offset as u64,
            e.numel() * e.dtype.bytes(),
        );
    }

    /// `pread` `buf.len()` bytes starting `start` bytes into a tensor; through
    /// the map where `pread` is unavailable.
    fn read_range(&self, e: &TensorEntry, start: usize, buf: &mut [u8]) -> Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.files[e.shard]
                .read_exact_at(buf, (e.offset + start) as u64)
                .context("flint: reading checkpoint bytes")
        }
        #[cfg(not(unix))]
        {
            buf.copy_from_slice(&self.bytes(e)[start..start + buf.len()]);
            Ok(())
        }
    }
}

/// Ask the OS to start reading `len` bytes of `file` at `offset` into the page
/// cache, without waiting: `F_RDADVISE` on macOS, `posix_fadvise(WILLNEED)`
/// on Linux, nothing elsewhere. Best effort: an error is ignored.
pub(crate) fn advise(file: &File, offset: u64, len: usize) {
    #[cfg(target_vendor = "apple")]
    {
        use std::os::fd::AsRawFd;
        let ra = libc::radvisory {
            ra_offset: offset as libc::off_t,
            ra_count: len.min(i32::MAX as usize) as libc::c_int,
        };
        // SAFETY: a valid descriptor and a live `radvisory`.
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_RDADVISE, &ra) };
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: a valid descriptor; the call only advises.
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                offset as libc::off_t,
                len as libc::off_t,
                libc::POSIX_FADV_WILLNEED,
            )
        };
    }
    #[cfg(not(any(target_vendor = "apple", target_os = "linux", target_os = "android")))]
    let _ = (file, offset, len);
}

/// Decode `bytes` of `dtype` into `out` (little-endian storage).
pub fn decode_into(dtype: StoredDtype, bytes: &[u8], out: &mut [f32]) {
    match dtype {
        StoredDtype::F32 => {
            for (o, c) in out.iter_mut().zip(bytes.as_chunks::<4>().0.iter()) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        StoredDtype::BF16 => {
            for (o, c) in out.iter_mut().zip(bytes.as_chunks::<2>().0.iter()) {
                *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
        StoredDtype::F16 => {
            for (o, c) in out.iter_mut().zip(bytes.as_chunks::<2>().0.iter()) {
                *o = half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
            }
        }
    }
}

/// `&Checkpoint` as a [`WeightSource`]. Non-destructive, so one checkpoint
/// builds any number of graphs.
pub struct CheckpointRef<'a>(pub &'a Checkpoint);

impl WeightSource for CheckpointRef<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.0.load_f32(key)?;
        if !transpose {
            return Ok((data, shape));
        }
        if shape.len() != 2 {
            bail!("flint: transpose requires a matrix, {key} is {shape:?}");
        }
        let mut t = vec![0.0f32; data.len()];
        crate::moe::transpose_into(&data, shape[0], shape[1], &mut t);
        Ok((t, vec![shape[1], shape[0]]))
    }

    fn has(&self, key: &str) -> bool {
        self.0.contains(key)
    }

    fn stored_dtype(&self, key: &str) -> Option<DType> {
        match self.0.entry(key).ok()?.dtype {
            StoredDtype::F32 => Some(DType::F32),
            StoredDtype::F16 => Some(DType::F16),
            StoredDtype::BF16 => Some(DType::BF16),
        }
    }

    /// A half-precision matrix as its own bytes (transposed on request), so a
    /// bf16 weight reaches the arena without an f32 round trip.
    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        let e = self.0.entry(key)?;
        if e.dtype == StoredDtype::F32 || e.shape.len() != 2 {
            return Ok(None);
        }
        let raw = self.0.raw(key)?;
        let (r, c) = (e.shape[0], e.shape[1]);
        if !transpose {
            return Ok(Some((raw.to_vec(), vec![r, c])));
        }
        let mut out = vec![0u8; raw.len()];
        for i in 0..r {
            for j in 0..c {
                let (src, dst) = ((i * c + j) * 2, (j * r + i) * 2);
                out[dst..dst + 2].copy_from_slice(&raw[src..src + 2]);
            }
        }
        Ok(Some((out, vec![c, r])))
    }
}

/// [`CheckpointRef`] that reports no stored dtype, so every weight loads as
/// f32 whatever the flow's storage setting. The reference the bf16-resident
/// graphs are compared against.
pub struct CheckpointF32<'a>(pub &'a Checkpoint);

impl WeightSource for CheckpointF32<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        CheckpointRef(self.0).take(key, transpose)
    }

    fn has(&self, key: &str) -> bool {
        self.0.contains(key)
    }
}
