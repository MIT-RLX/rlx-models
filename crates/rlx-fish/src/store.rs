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

//! A sharded safetensors checkpoint mapped once, with every tensor's byte
//! range indexed up front.
//!
//! Drama 3 has 62 k tensors, 61 k of them expert slices read a few at a time
//! for every token. Re-parsing a shard header per read (what a general loader
//! does) would dominate the expert path; here a read is a slice of the map.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use safetensors::Dtype;

use crate::nn::TensorSource;

/// Where one tensor lives.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    shard: usize,
    start: usize,
    end: usize,
}

/// The mapped checkpoint.
pub struct ShardStore {
    dir: PathBuf,
    maps: Vec<Mmap>,
    index: HashMap<String, TensorInfo>,
}

impl ShardStore {
    /// Map every `*.safetensors` shard of `dir`.
    pub fn open(dir: &Path) -> Result<Self> {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("read {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        shards.sort();
        ensure!(
            !shards.is_empty(),
            "no safetensors shards in {}",
            dir.display()
        );
        let mut maps = Vec::with_capacity(shards.len());
        let mut index = HashMap::new();
        for (si, path) in shards.iter().enumerate() {
            let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
            // SAFETY: read-only map of a checkpoint nobody writes while we run.
            let map =
                unsafe { Mmap::map(&file) }.with_context(|| format!("map {}", path.display()))?;
            let (header_len, meta) = safetensors::SafeTensors::read_metadata(&map)
                .with_context(|| format!("parse {}", path.display()))?;
            let base = 8 + header_len;
            for (name, info) in meta.tensors() {
                let (s, e) = info.data_offsets;
                index.insert(
                    name.clone(),
                    TensorInfo {
                        dtype: info.dtype,
                        shape: info.shape.clone(),
                        shard: si,
                        start: base + s,
                        end: base + e,
                    },
                );
            }
            maps.push(map);
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            maps,
            index,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.index
            .get(name)
            .with_context(|| format!("tensor {name} not in checkpoint"))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    /// The stored bytes, borrowed from the map.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let i = self.info(name)?;
        Ok(&self.maps[i.shard][i.start..i.end])
    }

    /// `rows` of a rank-2 tensor as f32 (`[rows.len(), cols]`).
    pub fn rows_f32(&self, name: &str, rows: &[u32]) -> Result<Vec<f32>> {
        let i = self.info(name)?;
        ensure!(i.shape.len() == 2, "{name}: rows of a {:?} tensor", i.shape);
        let (n, cols) = (i.shape[0], i.shape[1]);
        let width = dtype_size(i.dtype)?;
        let data = self.bytes(name)?;
        let mut out = Vec::with_capacity(rows.len() * cols);
        for &r in rows {
            ensure!((r as usize) < n, "{name}: row {r} of {n}");
            let off = r as usize * cols * width;
            out.extend(to_f32(name, i.dtype, &data[off..off + cols * width])?);
        }
        Ok(out)
    }
}

fn dtype_size(d: Dtype) -> Result<usize> {
    Ok(match d {
        Dtype::F32 => 4,
        Dtype::BF16 | Dtype::F16 => 2,
        Dtype::F8_E4M3 | Dtype::U8 | Dtype::I8 => 1,
        other => bail!("unsupported dtype {other:?}"),
    })
}

/// Decode stored bytes to f32.
pub fn to_f32(name: &str, dtype: Dtype, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::BF16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::F16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::F8_E4M3 => b.iter().map(|&v| e4m3(v)).collect(),
        other => bail!("{name}: cannot read {other:?} as f32"),
    })
}

/// The 256-entry `e4m3fn` → f32 table.
pub fn e4m3_table() -> &'static [f32; 256] {
    &E4M3
}

/// `e4m3fn` → f32 (no infinities; `0x7f`/`0xff` are NaN).
pub fn e4m3(b: u8) -> f32 {
    E4M3[b as usize]
}

static E4M3: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
    let mut t = [0f32; 256];
    for (i, slot) in t.iter_mut().enumerate() {
        *slot = rlx_core::dsv4_quant::e4m3fn_to_f32(i as u8);
    }
    t
});

impl TensorSource for ShardStore {
    fn tensor(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let i = self.info(key)?;
        Ok((to_f32(key, i.dtype, self.bytes(key)?)?, i.shape.clone()))
    }

    fn half_bytes(&self, key: &str) -> Option<(Vec<u8>, rlx_ir::DType, Vec<usize>)> {
        let i = self.info(key).ok()?;
        let dt = match i.dtype {
            Dtype::BF16 => rlx_ir::DType::BF16,
            Dtype::F16 => rlx_ir::DType::F16,
            _ => return None,
        };
        Some((self.bytes(key).ok()?.to_vec(), dt, i.shape.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_table() {
        assert_eq!(e4m3(0x00), 0.0);
        assert_eq!(e4m3(0x38), 1.0); // exp 7 = bias, mantissa 0
        assert_eq!(e4m3(0xb8), -1.0);
        assert_eq!(e4m3(0x7e), 448.0); // largest finite
        assert!(e4m3(0x7f).is_nan());
        assert_eq!(e4m3(0x01), 2f32.powi(-9)); // smallest subnormal
    }
}
