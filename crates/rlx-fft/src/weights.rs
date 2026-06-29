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
use crate::twiddle::{TwiddleSet, twiddle_index, twiddle_name_set, twiddle_packed_name};
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

    /// Bind twiddles for a compiled butterfly graph. The dense butterfly / gated
    /// / stockham graphs consume a single packed `[stages, half, 2]` twiddle
    /// tensor per set, so reconstruct and bind that; any non-twiddle params in
    /// the store (rare) pass through by name.
    pub fn apply_butterfly(
        &self,
        exec: &mut rlx_runtime::CompiledGraph,
        _batch: usize,
        n_fft: usize,
    ) {
        for set in [TwiddleSet::Shared, TwiddleSet::Encoder, TwiddleSet::Decoder] {
            if let Ok(buf) = self.to_twiddles_set(n_fft, set) {
                exec.set_param(&twiddle_packed_name(set), &buf);
            }
        }
        for (name, data) in &self.0 {
            if parse_twiddle_name(name).is_none() {
                exec.set_param(name, data);
            }
        }
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

/// Twiddle sets and their per-scalar param-name prefixes (see `twiddle_name_set`).
const TWIDDLE_SETS: [(TwiddleSet, &str); 3] = [
    (TwiddleSet::Shared, "twiddle"),
    (TwiddleSet::Encoder, "encoder.twiddle"),
    (TwiddleSet::Decoder, "decoder.twiddle"),
];

/// Suffix marking a packed `[stages, half, 2]` twiddle tensor on disk.
const PACKED_SUFFIX: &str = ".packed";

/// Env knob: when `1`, write packed twiddles as f16 (≈2× smaller, lossy).
const WEIGHTS_F16_ENV: &str = "RLX_FFT_WEIGHTS_F16";

/// Parse a per-scalar twiddle name `{prefix}.s{S}.b{B}.{re|im}` → (set, stage, butterfly, part).
fn parse_twiddle_name(name: &str) -> Option<(TwiddleSet, usize, usize, usize)> {
    for (set, prefix) in TWIDDLE_SETS {
        let Some(rest) = name.strip_prefix(prefix).and_then(|r| r.strip_prefix(".s")) else {
            continue;
        };
        let (s_str, rest) = rest.split_once(".b")?;
        let (b_str, part) = rest.split_once('.')?;
        let part = match part {
            "re" => 0,
            "im" => 1,
            _ => return None,
        };
        return Some((set, s_str.parse().ok()?, b_str.parse().ok()?, part));
    }
    None
}

/// Read a tensor view as f32, widening f16 on the fly.
fn view_to_f32(view: &safetensors::tensor::TensorView) -> Result<Vec<f32>> {
    let raw = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        safetensors::Dtype::F16 => Ok(raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        other => anyhow::bail!("unsupported weight dtype {other:?}"),
    }
}

/// Serialize a weight store. Contiguous per-scalar twiddles are collapsed into a
/// single packed `[stages, half, 2]` tensor per set (one header entry instead of
/// `stages·half·2`), optionally f16 (`RLX_FFT_WEIGHTS_F16=1`). Non-twiddle params
/// and any incomplete twiddle set are written individually (f32), so the file
/// stays loadable by [`load_safetensors`] either way.
pub fn export_safetensors(path: &Path, weights: &WeightStore) -> Result<()> {
    ensure!(!weights.0.is_empty(), "no weights to export");
    let f16 = rlx_ir::env::var(WEIGHTS_F16_ENV).as_deref() == Some("1");

    // Bucket twiddle scalars by set; everything else passes through verbatim.
    let mut sets: HashMap<
        &str,
        (
            TwiddleSet,
            usize,
            usize,
            HashMap<(usize, usize, usize), f32>,
        ),
    > = HashMap::new();
    let mut others: Vec<(String, Vec<f32>)> = Vec::new();
    for (name, data) in &weights.0 {
        match parse_twiddle_name(name) {
            Some((set, s, b, part)) if data.len() == 1 => {
                let prefix = TWIDDLE_SETS.iter().find(|(s, _)| *s == set).unwrap().1;
                let entry = sets.entry(prefix).or_insert((set, 0, 0, HashMap::new()));
                entry.1 = entry.1.max(s);
                entry.2 = entry.2.max(b);
                entry.3.insert((s, b, part), data[0]);
            }
            _ => others.push((name.clone(), data.clone())),
        }
    }

    // (name, bytes, shape, dtype) — owns bytes so the views can borrow them.
    let mut storages: Vec<(String, Vec<u8>, Vec<usize>, safetensors::Dtype)> = Vec::new();
    for (prefix, (_set, max_s, max_b, values)) in &sets {
        let stages = max_s + 1;
        let half = max_b + 1;
        let expected = stages * half * 2;
        if values.len() != expected {
            // Incomplete set — fall back to per-scalar so nothing is dropped.
            for ((s, b, part), v) in values {
                let part_str = if *part == 0 { "re" } else { "im" };
                let entry = sets_name(prefix, *s, *b, part_str);
                storages.push((
                    entry,
                    v.to_le_bytes().to_vec(),
                    vec![1],
                    safetensors::Dtype::F32,
                ));
            }
            continue;
        }
        let mut buf = vec![0f32; expected];
        for ((s, b, part), v) in values {
            buf[(s * half + b) * 2 + part] = *v;
        }
        let (bytes, dtype) = if f16 {
            (
                buf.iter()
                    .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                    .collect(),
                safetensors::Dtype::F16,
            )
        } else {
            (
                buf.iter().flat_map(|&x| x.to_le_bytes()).collect(),
                safetensors::Dtype::F32,
            )
        };
        storages.push((
            format!("{prefix}{PACKED_SUFFIX}"),
            bytes,
            vec![stages, half, 2],
            dtype,
        ));
    }
    for (name, data) in &others {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        storages.push((
            name.clone(),
            bytes,
            vec![data.len()],
            safetensors::Dtype::F32,
        ));
    }

    let mut views: HashMap<String, safetensors::tensor::TensorView> = HashMap::new();
    for (name, bytes, shape, dtype) in &storages {
        views.insert(
            name.clone(),
            safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes)
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

/// Load a weight store. Understands both the packed twiddle layout and the
/// legacy per-scalar layout, and f32 or f16 storage — the returned store is
/// always the per-scalar map the bind path expects.
pub fn load_safetensors(path: &Path) -> Result<WeightStore> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let mut store = WeightStore::default();
    for name in st.names() {
        let view = st.tensor(name)?;
        let data = view_to_f32(&view)?;
        if let Some(prefix) = name.strip_suffix(PACKED_SUFFIX) {
            // Packed [stages, half, 2] → expand to per-scalar twiddle params.
            let shape = view.shape();
            ensure!(
                shape.len() == 3 && shape[2] == 2,
                "packed twiddles {name} expected [stages, half, 2], got {shape:?}"
            );
            let (stages, half) = (shape[0], shape[1]);
            ensure!(
                data.len() == stages * half * 2,
                "packed twiddles {name} size mismatch"
            );
            for s in 0..stages {
                for b in 0..half {
                    let base = (s * half + b) * 2;
                    store
                        .0
                        .insert(sets_name(prefix, s, b, "re"), vec![data[base]]);
                    store
                        .0
                        .insert(sets_name(prefix, s, b, "im"), vec![data[base + 1]]);
                }
            }
        } else {
            store.0.insert(name.to_string(), data);
        }
    }
    Ok(store)
}

/// `{prefix}.s{stage}.b{butterfly}.{part}` — the per-scalar twiddle name for a
/// given set prefix (matches [`twiddle_name_set`]).
fn sets_name(prefix: &str, stage: usize, butterfly: usize, part: &str) -> String {
    format!("{prefix}.s{stage}.b{butterfly}.{part}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_twiddles(stages: usize, half: usize) -> Vec<f32> {
        (0..stages * half * 2)
            .map(|i| i as f32 * 0.25 - 3.0)
            .collect()
    }

    /// New packed format collapses to one tensor and round-trips exactly (f32).
    #[test]
    fn packed_roundtrip_is_single_tensor_and_exact() {
        let (n_fft, stages, half) = (8usize, 3usize, 4usize);
        let tw = sample_twiddles(stages, half);
        let store = WeightStore::from_twiddles(&tw, n_fft);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tw.safetensors");
        export_safetensors(&path, &store).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(st.names().len(), 1, "all twiddles pack into one tensor");
        assert!(st.names()[0].ends_with(PACKED_SUFFIX));

        let back = load_safetensors(&path).unwrap().to_twiddles(n_fft).unwrap();
        assert_eq!(back, tw);
    }

    /// Legacy per-scalar checkpoints still load (backward compatibility).
    #[test]
    fn legacy_per_scalar_still_loads() {
        let (n_fft, stages, half) = (8usize, 3usize, 4usize);
        let tw = sample_twiddles(stages, half);
        let store = WeightStore::from_twiddles(&tw, n_fft);

        // Emulate the old layout: one [1]-shaped f32 tensor per scalar.
        let storages: Vec<(String, Vec<u8>)> = store
            .0
            .iter()
            .map(|(name, data)| (name.clone(), data[0].to_le_bytes().to_vec()))
            .collect();
        let views: HashMap<String, safetensors::tensor::TensorView> = storages
            .iter()
            .map(|(name, bytes)| {
                (
                    name.clone(),
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], bytes)
                        .unwrap(),
                )
            })
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.safetensors");
        safetensors::serialize_to_file(&views, None, &path).unwrap();

        let back = load_safetensors(&path).unwrap().to_twiddles(n_fft).unwrap();
        assert_eq!(back, tw);
    }

    /// Encoder + decoder sets pack independently and survive a round-trip.
    #[test]
    fn packed_roundtrip_encdec() {
        let (n_fft, stages, half) = (8usize, 3usize, 4usize);
        let enc = sample_twiddles(stages, half);
        let dec: Vec<f32> = enc.iter().map(|x| -x).collect();
        let store = EncDecWeights::from_twiddles(&enc, &dec, n_fft).merged();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("encdec.safetensors");
        export_safetensors(&path, &store).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(st.names().len(), 2, "encoder + decoder packed tensors");

        let loaded = load_safetensors(&path).unwrap();
        assert_eq!(
            loaded.to_twiddles_set(n_fft, TwiddleSet::Encoder).unwrap(),
            enc
        );
        assert_eq!(
            loaded.to_twiddles_set(n_fft, TwiddleSet::Decoder).unwrap(),
            dec
        );
    }
}
