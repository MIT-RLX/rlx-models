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

//! Named twiddle parameters for training and compiled inference.

use crate::config::TransformDir;
use crate::twiddle::{TwiddleSet, twiddle_index, twiddle_name_set};
use anyhow::{Context, Result, ensure};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct WeightStore(pub HashMap<String, Vec<f32>>);

impl WeightStore {
    pub fn from_twiddles(twiddles: &[f32], n_fft: usize) -> Self {
        Self::from_twiddles_dir(twiddles, n_fft, TransformDir::Forward)
    }

    pub fn from_twiddles_dir(twiddles: &[f32], n_fft: usize, dir: TransformDir) -> Self {
        let _ = dir;
        Self::from_twiddles_set(twiddles, n_fft, TwiddleSet::Shared)
    }

    pub fn from_twiddles_set(twiddles: &[f32], n_fft: usize, set: TwiddleSet) -> Self {
        let half = n_fft / 2;
        let stages = n_fft.trailing_zeros() as usize;
        let mut store = Self::default();
        for s in 0..stages {
            for b in 0..half {
                let base = twiddle_index(s, b, half, 0);
                store
                    .0
                    .insert(twiddle_name_set(set, s, b, "re"), vec![twiddles[base]]);
                store
                    .0
                    .insert(twiddle_name_set(set, s, b, "im"), vec![twiddles[base + 1]]);
            }
        }
        store
    }

    pub fn to_twiddles(&self, n_fft: usize) -> Result<Vec<f32>> {
        self.to_twiddles_dir(n_fft, TransformDir::Forward)
    }

    pub fn to_twiddles_dir(&self, n_fft: usize, dir: TransformDir) -> Result<Vec<f32>> {
        let _ = dir;
        self.to_twiddles_set(n_fft, TwiddleSet::Shared)
    }

    pub fn to_twiddles_set(&self, n_fft: usize, set: TwiddleSet) -> Result<Vec<f32>> {
        let half = n_fft / 2;
        let stages = n_fft.trailing_zeros() as usize;
        let mut out = vec![0f32; stages * half * 2];
        for s in 0..stages {
            for b in 0..half {
                let base = twiddle_index(s, b, half, 0);
                let re_name = twiddle_name_set(set, s, b, "re");
                let im_name = twiddle_name_set(set, s, b, "im");
                out[base] = *self
                    .0
                    .get(&re_name)
                    .with_context(|| format!("missing twiddle param {re_name}"))?
                    .first()
                    .context("empty twiddle re")?;
                out[base + 1] = *self
                    .0
                    .get(&im_name)
                    .with_context(|| format!("missing twiddle param {im_name}"))?
                    .first()
                    .context("empty twiddle im")?;
            }
        }
        Ok(out)
    }

    pub fn merge(&self, other: &Self) -> Self {
        let mut out = self.clone();
        for (k, v) in &other.0 {
            out.0.insert(k.clone(), v.clone());
        }
        out
    }

    pub fn apply(&self, exec: &mut rlx_runtime::CompiledGraph) {
        for (name, data) in &self.0 {
            exec.set_param(name, data);
        }
    }

    /// Bind twiddles for a compiled butterfly graph.
    pub fn apply_butterfly(
        &self,
        exec: &mut rlx_runtime::CompiledGraph,
        _batch: usize,
        _n_fft: usize,
    ) {
        self.apply(exec);
    }

    /// Bind twiddles only for non-skip ternary butterflies present in a pruned graph.
    pub fn apply_butterfly_for_gates(
        &self,
        exec: &mut rlx_runtime::CompiledGraph,
        n_fft: usize,
        gates: &[i8],
    ) {
        use crate::pruned::{gate_count, gate_index};
        use crate::ternary_gates::GateMode;
        let half = n_fft / 2;
        let stages = n_fft.trailing_zeros() as usize;
        if gates.len() < gate_count(n_fft) {
            return;
        }
        for s in 0..stages {
            for b in 0..half {
                let gi = gate_index(s, b, half);
                if GateMode::from_i8(gates[gi]) == GateMode::Skip {
                    continue;
                }
                let re_name = twiddle_name_set(TwiddleSet::Shared, s, b, "re");
                let im_name = twiddle_name_set(TwiddleSet::Shared, s, b, "im");
                if let Some(v) = self.0.get(&re_name) {
                    exec.set_param(&re_name, v);
                }
                if let Some(v) = self.0.get(&im_name) {
                    exec.set_param(&im_name, v);
                }
            }
        }
    }
}

/// Encoder + decoder twiddle checkpoints.
#[derive(Debug, Clone, Default)]
pub struct EncDecWeights {
    pub encoder: WeightStore,
    pub decoder: WeightStore,
}

impl EncDecWeights {
    pub fn from_twiddles(encoder: &[f32], decoder: &[f32], n_fft: usize) -> Self {
        Self {
            encoder: WeightStore::from_twiddles_set(encoder, n_fft, TwiddleSet::Encoder),
            decoder: WeightStore::from_twiddles_set(decoder, n_fft, TwiddleSet::Decoder),
        }
    }

    pub fn merged(&self) -> WeightStore {
        self.encoder.merge(&self.decoder)
    }

    pub fn encoder_twiddles(&self, n_fft: usize) -> Result<Vec<f32>> {
        self.encoder.to_twiddles_set(n_fft, TwiddleSet::Encoder)
    }

    pub fn decoder_twiddles(&self, n_fft: usize) -> Result<Vec<f32>> {
        self.decoder.to_twiddles_set(n_fft, TwiddleSet::Decoder)
    }

    pub fn from_merged(store: &WeightStore, n_fft: usize) -> Result<Self> {
        Ok(Self {
            encoder: {
                let tw = store.to_twiddles_set(n_fft, TwiddleSet::Encoder)?;
                WeightStore::from_twiddles_set(&tw, n_fft, TwiddleSet::Encoder)
            },
            decoder: {
                let tw = store.to_twiddles_set(n_fft, TwiddleSet::Decoder)?;
                WeightStore::from_twiddles_set(&tw, n_fft, TwiddleSet::Decoder)
            },
        })
    }
}

pub fn export_safetensors(path: &Path, weights: &WeightStore) -> Result<()> {
    ensure!(!weights.0.is_empty(), "no weights to export");
    let mut storages: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
    for (name, data) in &weights.0 {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        storages.push((name.clone(), bytes, vec![data.len()]));
    }
    let mut views: HashMap<String, safetensors::tensor::TensorView> = HashMap::new();
    for (name, bytes, shape) in &storages {
        views.insert(
            name.clone(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                .context("tensor view")?,
        );
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    safetensors::serialize_to_file(&views, None, path)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

pub fn load_safetensors(path: &Path) -> Result<WeightStore> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let mut store = WeightStore::default();
    for name in st.names() {
        let view = st.tensor(name)?;
        ensure!(
            view.dtype() == safetensors::Dtype::F32,
            "expected f32 weights in {path:?}, got {:?} for {name}",
            view.dtype()
        );
        let data: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        store.0.insert(name.to_string(), data);
    }
    Ok(store)
}
