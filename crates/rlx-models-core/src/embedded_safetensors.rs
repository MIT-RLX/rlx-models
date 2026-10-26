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

//! Parse HuggingFace safetensors blobs from memory (`include_bytes!`, etc.).
//!
//! Used by the `rlx-vad` crate to embed `silero_vad_16k.safetensors` without
//! mmap or disk I/O. For sharded on-disk checkpoints see
//! [`SafetensorsCheckpoint`](crate::safetensors_checkpoint::SafetensorsCheckpoint).

use anyhow::{Context, Result};
use safetensors::SafeTensors;

use crate::safetensors_checkpoint::tensor_view_to_f32;

/// Zero-copy view over an in-memory safetensors file.
pub struct EmbeddedSafetensors<'a> {
    inner: SafeTensors<'a>,
}

impl<'a> EmbeddedSafetensors<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let inner = SafeTensors::deserialize(bytes).context("parse embedded safetensors")?;
        Ok(Self { inner })
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let view = self
            .inner
            .tensor(name)
            .with_context(|| format!("tensor {name}"))?;
        tensor_view_to_f32(name, view)
    }
}

/// [`WeightLoader`](crate::weight_loader::WeightLoader) over a safetensors
/// file held in memory — for targets without a filesystem (wasm32) or bytes
/// fetched over the network. Mirrors [`SafetensorsMmapLoader`](crate::SafetensorsMmapLoader):
/// tensors convert to F32 on `take`, each at most once.
///
/// Reads the header itself rather than through `SafeTensors::deserialize`,
/// whose validation multiplies element count by dtype width in *bits* in a
/// `usize` — on wasm32 that overflows for any tensor over 512 MB
/// (EmbeddingGemma's 805 MB token table).
pub struct SafetensorsBytesLoader {
    bytes: std::sync::Arc<[u8]>,
    /// name → (dtype, shape, absolute byte range).
    index: std::collections::BTreeMap<
        String,
        (safetensors::Dtype, Vec<usize>, std::ops::Range<usize>),
    >,
    taken: std::collections::HashSet<String>,
}

impl SafetensorsBytesLoader {
    pub fn new(bytes: impl Into<std::sync::Arc<[u8]>>) -> Result<Self> {
        #[derive(serde::Deserialize)]
        struct Entry {
            dtype: safetensors::Dtype,
            shape: Vec<usize>,
            data_offsets: [u64; 2],
        }
        let bytes = bytes.into();
        let head_len = bytes
            .get(..8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
            .context("safetensors bytes: shorter than the header length")?;
        let data_start = usize::try_from(head_len)
            .ok()
            .and_then(|n| n.checked_add(8))
            .filter(|&n| n <= bytes.len())
            .context("safetensors bytes: header length past the end")?;
        let header: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..data_start]).context("safetensors bytes: header")?;
        let mut index = std::collections::BTreeMap::new();
        for (name, v) in header {
            if name == "__metadata__" {
                continue;
            }
            let e: Entry =
                serde_json::from_value(v).with_context(|| format!("header entry {name}"))?;
            let range = usize::try_from(e.data_offsets[0])
                .ok()
                .zip(usize::try_from(e.data_offsets[1]).ok())
                .and_then(|(a, b)| Some(data_start.checked_add(a)?..data_start.checked_add(b)?))
                .filter(|r| r.start <= r.end && r.end <= bytes.len())
                .with_context(|| format!("{name}: data offsets out of range"))?;
            index.insert(name, (e.dtype, e.shape, range));
        }
        Ok(Self {
            bytes,
            index,
            taken: Default::default(),
        })
    }

    pub fn has(&self, key: &str) -> bool {
        self.index.contains_key(key) && !self.taken.contains(key)
    }

    /// Rows of a 2-D tensor as F32, converting only those rows — for
    /// gathering embeddings without materializing the table. Does not count
    /// as a `take`.
    pub fn rows(&self, key: &str, rows: &[usize]) -> Result<Vec<f32>> {
        let (dtype, shape, range) = self
            .index
            .get(key)
            .with_context(|| format!("tensor {key}"))?;
        let [n, width] = shape[..] else {
            anyhow::bail!("{key}: rows of a {}-D tensor", shape.len());
        };
        let row_bytes = (range.end - range.start) / n.max(1);
        let mut out = Vec::with_capacity(rows.len() * width);
        for &r in rows {
            if r >= n {
                anyhow::bail!("{key}: row {r} out of {n}");
            }
            let start = range.start + r * row_bytes;
            out.extend(crate::safetensors_checkpoint::raw_to_f32(
                key,
                *dtype,
                &self.bytes[start..start + row_bytes],
            )?);
        }
        Ok(out)
    }
}

/// Row gather over a 2-D tensor (an embedding table) that never puts the
/// table in a graph. From in-memory bytes only the requested rows are
/// converted; from a file the table is loaded once.
pub enum TensorRows {
    Dense {
        data: Vec<f32>,
        width: usize,
    },
    Bytes {
        loader: SafetensorsBytesLoader,
        key: String,
    },
}

impl TensorRows {
    pub fn open(src: &SafetensorsSource, key: &str) -> Result<Self> {
        Ok(match src {
            SafetensorsSource::Bytes(b) => Self::Bytes {
                loader: SafetensorsBytesLoader::new(b.clone())?,
                key: key.to_string(),
            },
            SafetensorsSource::File(_) => {
                let (data, shape) = src.loader()?.take(key)?;
                let [_, width] = shape[..] else {
                    anyhow::bail!("{key}: expected a 2-D table, got {shape:?}");
                };
                Self::Dense { data, width }
            }
        })
    }

    /// `[ids.len() · width]`, row `i` = table row `ids[i]`.
    pub fn gather(&self, ids: &[u32]) -> Result<Vec<f32>> {
        match self {
            Self::Dense { data, width } => {
                let mut out = Vec::with_capacity(ids.len() * width);
                for &i in ids {
                    let start = i as usize * width;
                    out.extend_from_slice(
                        data.get(start..start + width)
                            .with_context(|| format!("row {i} out of range"))?,
                    );
                }
                Ok(out)
            }
            Self::Bytes { loader, key } => {
                let rows: Vec<usize> = ids.iter().map(|&i| i as usize).collect();
                loader.rows(key, &rows)
            }
        }
    }
}

impl crate::weight_loader::WeightLoader for SafetensorsBytesLoader {
    fn format_id(&self) -> &'static str {
        "safetensors"
    }

    fn len(&self) -> usize {
        self.index.len() - self.taken.len()
    }

    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        if self.taken.contains(key) {
            anyhow::bail!("weight already taken: {key}");
        }
        let (dtype, shape, range) = self
            .index
            .get(key)
            .with_context(|| format!("tensor {key}"))?;
        let data =
            crate::safetensors_checkpoint::raw_to_f32(key, *dtype, &self.bytes[range.clone()])?;
        let shape = shape.clone();
        self.taken.insert(key.to_string());
        Ok((data, shape))
    }

    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.take(key)?;
        let [rows, cols] = shape[..] else {
            anyhow::bail!("transpose requires 2D, got {shape:?}");
        };
        let mut t = vec![0f32; data.len()];
        for i in 0..rows {
            for j in 0..cols {
                t[j * rows + i] = data[i * cols + j];
            }
        }
        Ok((t, vec![cols, rows]))
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.index
            .keys()
            .filter(|k| !self.taken.contains(*k))
            .cloned()
            .collect()
    }
}

/// Where a safetensors file comes from: a path (memory-mapped on take) or one
/// shared in-memory copy. Models that rebuild graphs — one per sequence-length
/// bucket — keep this and call [`loader`](Self::loader) per build, so an
/// in-memory checkpoint is never copied again.
#[derive(Clone)]
pub enum SafetensorsSource {
    File(std::path::PathBuf),
    Bytes(std::sync::Arc<[u8]>),
}

impl SafetensorsSource {
    /// `name` inside `src`: the file itself for a directory source, otherwise
    /// its bytes (read once).
    pub fn from_asset(src: &crate::AssetSource, name: &str) -> Result<Self> {
        Ok(match src.dir_path() {
            Some(dir) => Self::File(dir.join(name)),
            None => Self::Bytes(src.read_vec(name)?.into()),
        })
    }

    /// A fresh loader with nothing taken yet.
    pub fn loader(&self) -> Result<Box<dyn crate::weight_loader::WeightLoader>> {
        match self {
            Self::File(p) => crate::weight_registry::open_weight_loader(p)
                .with_context(|| format!("open {}", p.display())),
            Self::Bytes(b) => Ok(Box::new(SafetensorsBytesLoader::new(b.clone())?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_loader::WeightLoader;

    #[test]
    fn rows_from_bytes_match_the_dense_table() {
        let vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4, 3], &data)
            .unwrap();
        let bytes: std::sync::Arc<[u8]> =
            safetensors::serialize([("e", view)], None).unwrap().into();
        let gathered = TensorRows::open(&SafetensorsSource::Bytes(bytes), "e")
            .unwrap()
            .gather(&[3, 0, 3])
            .unwrap();
        assert_eq!(gathered, vec![9., 10., 11., 0., 1., 2., 9., 10., 11.]);
    }

    #[test]
    fn bytes_loader_reads_and_transposes() {
        let data: Vec<u8> = [1f32, 2., 3., 4., 5., 6.]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 3], &data)
            .unwrap();
        let bytes = safetensors::serialize([("w", view)], None).unwrap();
        let mut l = SafetensorsBytesLoader::new(bytes).unwrap();
        assert!(l.has("w"));
        assert_eq!(l.remaining_keys(), vec!["w".to_string()]);
        let (t, s) = l.take_transposed("w").unwrap();
        assert_eq!((t, s), (vec![1., 4., 2., 5., 3., 6.], vec![3, 2]));
        assert!(l.take("w").is_err(), "a tensor is taken at most once");
    }
}
