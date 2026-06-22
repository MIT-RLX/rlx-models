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

//! Mmap-backed HuggingFace sharded safetensors checkpoints.
//!
//! Keeps shard files mapped once and reuses the mapping across selective
//! [`WeightMap`] loads (prefix subsets, single keys, etc.).

use crate::weight_map::WeightMap;
use anyhow::{Context, Result, bail};
use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

type F32TensorLoad = (String, Vec<f32>, Vec<usize>);

/// Cached mmap views of a sharded (or flat) safetensors directory.
pub struct SafetensorsCheckpoint {
    dir: PathBuf,
    /// Tensor name → shard filename (relative to `dir`).
    index: HashMap<String, String>,
    shards: Mutex<HashMap<String, Arc<Mmap>>>,
}

impl SafetensorsCheckpoint {
    pub fn open(dir: &Path) -> Result<Self> {
        let dir = dir.to_path_buf();
        let index_path = dir.join("model.safetensors.index.json");
        let index = if index_path.is_file() {
            let raw = std::fs::read(&index_path).context("read model.safetensors.index.json")?;
            let parsed: serde_json::Value =
                serde_json::from_slice(&raw).context("parse model.safetensors.index.json")?;
            let weight_map = parsed
                .get("weight_map")
                .and_then(|m| m.as_object())
                .context("weight_map in index")?;
            weight_map
                .iter()
                .filter_map(|(k, v)| v.as_str().map(|shard| (k.clone(), shard.to_string())))
                .collect()
        } else {
            Self::index_from_dir(&dir)?
        };
        if index.is_empty() {
            bail!("no safetensors tensors found under {dir:?}");
        }
        Ok(Self {
            dir,
            index,
            shards: Mutex::new(HashMap::new()),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(|s| s.as_str())
    }

    /// Whether a tensor name is present in the checkpoint index.
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Raw storage bytes + dtype + shape for one tensor, copied out of the
    /// mmap. For custom codecs (e.g. integer-quant) that need the native
    /// bytes rather than an F32 conversion. Bytes are in the file's storage
    /// order (row-major, dtype-native).
    pub fn tensor_raw(&self, name: &str) -> Result<(Vec<u8>, safetensors::Dtype, Vec<usize>)> {
        let shard = self
            .index
            .get(name)
            .with_context(|| format!("tensor {name} not in checkpoint index"))?;
        let mmap = self.mmap_shard(shard)?;
        let st = SafeTensors::deserialize(mmap.as_ref()).context("parse safetensors")?;
        let view = st
            .tensor(name)
            .with_context(|| format!("tensor {name} missing in shard {shard}"))?;
        Ok((view.data().to_vec(), view.dtype(), view.shape().to_vec()))
    }

    /// Load selected rows from a rank-2 tensor without materializing the full matrix.
    pub fn load_tensor_rows_f32(
        &self,
        name: &str,
        rows: &[u32],
        cols: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let shard = self
            .index
            .get(name)
            .with_context(|| format!("tensor {name} not in checkpoint index"))?;
        let mmap = self.mmap_shard(shard)?;
        let st = SafeTensors::deserialize(mmap.as_ref()).context("parse safetensors")?;
        let view = st
            .tensor(name)
            .with_context(|| format!("tensor {name} missing in shard {shard}"))?;
        let shape: Vec<usize> = view.shape().to_vec();
        anyhow::ensure!(
            shape.len() == 2,
            "{name}: expected rank-2 embed, got shape {shape:?}"
        );
        anyhow::ensure!(
            shape[1] == cols,
            "{name}: cols {} != expected {cols}",
            shape[1]
        );
        let vocab = shape[0];
        let mut out = Vec::with_capacity(rows.len());
        for &row in rows {
            let r = row as usize;
            anyhow::ensure!(r < vocab, "{name}: row {row} >= vocab {vocab}");
            out.push(read_tensor_row_f32(&view, r, cols)?);
        }
        Ok(out)
    }

    pub fn load_selected(&self, want: &HashSet<String>) -> Result<WeightMap> {
        if want.is_empty() {
            bail!("load_selected: empty key set");
        }
        let mut shard_files: HashSet<String> = HashSet::new();
        for key in want {
            if let Some(shard) = self.index.get(key) {
                shard_files.insert(shard.clone());
            }
        }
        if shard_files.is_empty() {
            bail!("no requested tensors found under {:?}", self.dir);
        }

        let mut tensors = HashMap::with_capacity(want.len());
        if shard_files.len() == 1 {
            let shard = shard_files.into_iter().next().unwrap();
            let mmap = self.mmap_shard(&shard)?;
            ingest_selected_from_bytes(mmap.as_ref(), want, &mut tensors)?;
        } else {
            let mut handles = Vec::with_capacity(shard_files.len());
            for shard in shard_files {
                let mmap = self.mmap_shard(&shard)?;
                let want = want.clone();
                handles.push(std::thread::spawn(move || {
                    let mut local = HashMap::new();
                    ingest_selected_from_bytes(mmap.as_ref(), &want, &mut local)?;
                    Ok::<_, anyhow::Error>(local)
                }));
            }
            for handle in handles {
                let part = handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("shard load thread panicked"))??;
                tensors.extend(part);
            }
        }
        if tensors.is_empty() {
            bail!("no requested tensors loaded from {:?}", self.dir);
        }
        Ok(WeightMap::from_tensors(tensors))
    }

    fn index_from_dir(dir: &Path) -> Result<HashMap<String, String>> {
        let mut index = HashMap::new();
        for entry in std::fs::read_dir(dir).with_context(|| format!("read_dir {dir:?}"))? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .map(str::to_string)
                .context("non-utf8 shard filename")?;
            let mmap = map_file(&path)?;
            let st = SafeTensors::deserialize(mmap.as_ref()).context("parse safetensors")?;
            for key in st.names() {
                index.insert(key.to_string(), name.clone());
            }
        }
        Ok(index)
    }

    fn mmap_shard(&self, shard: &str) -> Result<Arc<Mmap>> {
        let mut guard = self
            .shards
            .lock()
            .map_err(|_| anyhow::anyhow!("safetensors shard cache lock poisoned"))?;
        if let Some(m) = guard.get(shard) {
            return Ok(Arc::clone(m));
        }
        let path = self.dir.join(shard);
        let mmap = map_file(&path)?;
        guard.insert(shard.to_string(), Arc::clone(&mmap));
        Ok(mmap)
    }
}

fn map_file(path: &Path) -> Result<Arc<Mmap>> {
    let file = File::open(path).with_context(|| format!("open {path:?}"))?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    Ok(Arc::new(mmap))
}

fn ingest_selected_from_bytes(
    data: &[u8],
    want: &HashSet<String>,
    tensors: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<()> {
    use rayon::prelude::*;
    let st = SafeTensors::deserialize(data).context("parsing safetensors")?;
    // Collect wanted (name, view) pairs first so we can parallelize the
    // per-tensor BF16/F16 → F32 conversion (the dominant cost on bf16 ckpts).
    let selected: Vec<(String, safetensors::tensor::TensorView<'_>)> = st
        .tensors()
        .into_iter()
        .filter(|(name, _)| want.contains(name.as_str()))
        .map(|(name, view)| (name.to_string(), view))
        .collect();
    let converted: Vec<Result<F32TensorLoad>> = selected
        .into_par_iter()
        .map(|(name, view)| {
            let shape: Vec<usize> = view.shape().to_vec();
            let f32_data = tensor_bytes_to_f32(name.as_str(), view)?;
            Ok((name, f32_data, shape))
        })
        .collect();
    for r in converted {
        let (name, f32_data, shape) = r?;
        if f32_data.is_empty() {
            continue;
        }
        tensors.insert(name, (f32_data, shape));
    }
    Ok(())
}

fn read_tensor_row_f32(
    view: &safetensors::tensor::TensorView<'_>,
    row: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    let bytes = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let row_bytes = cols * 4;
            let off = row * row_bytes;
            Ok(bytes[off..off + row_bytes]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        safetensors::Dtype::F16 => {
            let row_bytes = cols * 2;
            let off = row * row_bytes;
            Ok(bytes[off..off + row_bytes]
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        safetensors::Dtype::BF16 => {
            let row_bytes = cols * 2;
            let off = row * row_bytes;
            Ok(bytes[off..off + row_bytes]
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        other => anyhow::bail!("row slice: unsupported dtype {other:?}"),
    }
}

fn tensor_bytes_to_f32(name: &str, view: safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    tensor_view_to_f32(name, view)
}

/// Decode a safetensors tensor view into a contiguous f32 vector (C order).
pub fn tensor_view_to_f32(
    name: &str,
    view: safetensors::tensor::TensorView<'_>,
) -> Result<Vec<f32>> {
    let bytes = view.data();
    Ok(match view.dtype() {
        safetensors::Dtype::F32 => super::weight_map::bytes_to_f32_vec(bytes),
        safetensors::Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        safetensors::Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        safetensors::Dtype::I64 => bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        safetensors::Dtype::I32 => bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        safetensors::Dtype::C64 => return Ok(vec![]),
        other => anyhow::bail!("{name}: unsupported dtype {other:?}"),
    })
}
