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

//! Weight sources the NLLB graphs are built from.
//!
//! Graphs are rebuilt per length bucket, so a source must hand out the same
//! tensor more than once. [`Checkpoint`] keeps the weights **on disk** —
//! an mmapped safetensors file, or the zip index of a `pytorch_model.bin`
//! read one tensor at a time — so the host never holds a second full copy
//! next to the compiled graphs (each of which owns its weights on CPU).
//! [`ShapeRecorder`] lets a graph be built and compiled before any weight
//! is read, so they can then be streamed in one tensor at a time.
//! [`CloningWeightSource`] serves an in-memory [`WeightMap`] (tests).

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::weight_map::WeightMap;
use rlx_flow::WeightSource;
use rlx_flow::formats::SafetensorsSource;
use rlx_runtime::CompiledGraph;
use std::path::Path;

/// Where the model's weights live between graph builds.
pub enum Checkpoint {
    Memory(WeightMap),
    Safetensors(SafetensorsSource),
    Torch(rlx_torch_ckpt::PtModel),
}

impl Checkpoint {
    /// A weights directory (`*.safetensors`, else `pytorch_model.bin`) or file.
    pub fn open(path: &Path) -> Result<Self> {
        let file = if path.is_dir() {
            let has_safetensors = std::fs::read_dir(path)?
                .filter_map(|e| e.ok())
                .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors"));
            if has_safetensors {
                return Ok(Self::Safetensors(SafetensorsSource::open(path)?));
            }
            let bin = path.join("pytorch_model.bin");
            if !bin.is_file() {
                bail!(
                    "nllb: no *.safetensors or pytorch_model.bin in {}",
                    path.display()
                );
            }
            bin
        } else {
            path.to_path_buf()
        };
        match file.extension().and_then(|e| e.to_str()) {
            Some("bin" | "pt" | "pth") => Ok(Self::Torch(
                rlx_torch_ckpt::PtModel::open(&file).with_context(|| {
                    format!("nllb: reading {} as a torch checkpoint", file.display())
                })?,
            )),
            _ => Ok(Self::Safetensors(SafetensorsSource::open(&file)?)),
        }
    }

    /// Read one tensor (row-major f32 + shape), leaving it in place.
    pub fn read(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        match self {
            Self::Memory(m) => m
                .get(key)
                .map(|(d, s)| (d.to_vec(), s.to_vec()))
                .ok_or_else(|| anyhow!("missing weight `{key}`")),
            Self::Safetensors(s) => s.take(key, false),
            Self::Torch(m) => {
                let t = m.tensor(key)?;
                Ok((t.data, t.shape))
            }
        }
    }
}

impl Checkpoint {
    /// A tensor's shape as stored, without reading it.
    pub fn shape(&self, key: &str) -> Option<Vec<usize>> {
        match self {
            Self::Memory(m) => m.get(key).map(|(_, s)| s.to_vec()),
            Self::Safetensors(s) => s.info(key).map(|(_, s)| s.to_vec()),
            Self::Torch(m) => m.shape_of(key).map(<[usize]>::to_vec),
        }
    }

    /// Read every tensor a [`ShapeRecorder`] build took and bind it into
    /// `compiled`, one at a time — so a graph's weights never exist twice
    /// (a full param map next to the freshly allocated arena).
    pub fn stream_into(
        &mut self,
        compiled: &mut CompiledGraph,
        taken: &[(String, bool)],
    ) -> Result<()> {
        for (key, transpose) in taken {
            // Already on pages shared with another graph of the same
            // param-sharing scope: nothing to read.
            if compiled.param_already_set(key) {
                continue;
            }
            let (data, _) = self.take(key, *transpose)?;
            compiled.set_param(key, &data);
        }
        Ok(())
    }
}

/// A [`WeightSource`] that answers with shapes only and records what was
/// taken: build a graph against it, compile, then
/// [`Checkpoint::stream_into`] the weights.
pub struct ShapeRecorder<'a> {
    src: &'a Checkpoint,
    taken: Vec<(String, bool)>,
}

impl<'a> ShapeRecorder<'a> {
    pub fn new(src: &'a Checkpoint) -> Self {
        Self {
            src,
            taken: Vec::new(),
        }
    }

    /// `(key, transpose)` of every tensor taken, first take of each key.
    pub fn into_taken(self) -> Vec<(String, bool)> {
        self.taken
    }
}

impl WeightSource for ShapeRecorder<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let mut shape = self
            .src
            .shape(key)
            .ok_or_else(|| anyhow!("missing weight `{key}`"))?;
        if transpose {
            let &[r, c] = shape.as_slice() else {
                bail!("transpose requested for non-2D tensor `{key}` (shape {shape:?})");
            };
            shape = vec![c, r];
        }
        if !self.taken.iter().any(|(k, _)| k == key) {
            self.taken.push((key.to_string(), transpose));
        }
        Ok((Vec::new(), shape))
    }

    fn has(&self, key: &str) -> bool {
        self.src.has(key)
    }
}

impl WeightSource for Checkpoint {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        if let Self::Memory(m) = self {
            return CloningWeightSource(m).take(key, transpose);
        }
        let (data, shape) = self.read(key)?;
        if transpose {
            Ok(transpose_2d(&data, &shape, key)?)
        } else {
            Ok((data, shape))
        }
    }

    fn has(&self, key: &str) -> bool {
        match self {
            Self::Memory(m) => m.has(key),
            Self::Safetensors(s) => s.has(key),
            Self::Torch(m) => m.shape_of(key).is_some(),
        }
    }
}

/// HF `[out, in]` → `[in, out]`, with the cache-blocked parallel transpose
/// (a naive loop is several times slower on the 4096×1024 FFN weights and
/// dominated graph builds).
fn transpose_2d(data: &[f32], shape: &[usize], key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let &[r, c] = shape else {
        return Err(anyhow!(
            "transpose requested for non-2D tensor `{key}` (shape {shape:?})"
        ));
    };
    Ok((rlx_core::weight_map::transpose_2d(data, r, c), vec![c, r]))
}

/// Clones tensors out of a borrowed [`WeightMap`] instead of consuming them.
pub struct CloningWeightSource<'a>(pub &'a WeightMap);

impl WeightSource for CloningWeightSource<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self
            .0
            .get(key)
            .ok_or_else(|| anyhow!("missing weight `{key}`"))?;
        if transpose {
            transpose_2d(data, shape, key)
        } else {
            Ok((data.to_vec(), shape.to_vec()))
        }
    }

    fn has(&self, key: &str) -> bool {
        self.0.has(key)
    }
}
