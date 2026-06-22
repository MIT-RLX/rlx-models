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

//! Load a FunASR checkpoint directory into an in-memory [`WeightMap`], from
//! either `*.safetensors` or a native PyTorch `model.pt` ([`crate::pt`]). The
//! map is consumed by the graph builders through `rlx_core`'s tested
//! [`WeightMapSource`] adapter (which also performs lazy 2-D transposes).

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::weight_map::WeightMap;
use rlx_flow::WeightSource;

pub use rlx_core::flow_util::WeightMapSource;

use crate::pt::StateDict;

/// A non-consuming [`WeightSource`] over a borrowed [`WeightMap`]: each `take`
/// clones the tensor (with an optional 2-D transpose), so the same map backs
/// repeated graph builds at different sequence lengths.
pub struct RefSource<'a>(pub &'a WeightMap);

impl WeightSource for RefSource<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self
            .0
            .get(key)
            .ok_or_else(|| anyhow!("missing weight {key:?}"))?;
        if !transpose {
            return Ok((data.to_vec(), shape.to_vec()));
        }
        if shape.len() != 2 {
            bail!("transpose requires a rank-2 weight: {key} has shape {shape:?}");
        }
        let (rows, cols) = (shape[0], shape[1]);
        let mut out = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        Ok((out, vec![cols, rows]))
    }

    fn has(&self, key: &str) -> bool {
        self.0.has(key)
    }
}

/// Load all weights from a model directory.
pub fn load_dir(dir: &Path) -> Result<WeightMap> {
    // Prefer safetensors (memory-mapped, robust).
    if has_safetensors(dir) {
        return WeightMap::from_safetensors_dir(dir)
            .with_context(|| format!("safetensors in {}", dir.display()));
    }
    // Native PyTorch checkpoints.
    for cand in ["model.pt", "model.pth", "pytorch_model.bin", "model.pb"] {
        let p = dir.join(cand);
        if p.is_file() {
            return load_pt(&p);
        }
    }
    // Last resort: first *.pt / *.pth / *.bin in the directory (e.g. CAM++'s
    // `campplus_cn_common.bin`).
    if let Some(p) = first_with_ext(dir, &["pt", "pth", "bin"])? {
        return load_pt(&p);
    }
    bail!(
        "no weights (*.safetensors or model.pt) found in {}",
        dir.display()
    )
}

/// Load a single native PyTorch checkpoint file.
///
/// The modern (ZIP) format goes through upstream [`rlx_nemo::PtModel`] (lazy,
/// memory-mapped); the legacy (pre-1.6, non-ZIP) format — still common for
/// FunASR checkpoints — falls back to the in-crate [`StateDict`] reader.
pub fn load_pt(path: &Path) -> Result<WeightMap> {
    match rlx_nemo::PtModel::open(path) {
        Ok(model) => {
            let mut tensors = std::collections::HashMap::new();
            for name in model.names() {
                let t = model
                    .tensor(&name)
                    .with_context(|| format!("read tensor {name} from {}", path.display()))?;
                tensors.insert(name, (t.data, t.shape));
            }
            Ok(WeightMap::from_tensors(tensors))
        }
        Err(_) => {
            let sd = StateDict::load(path).with_context(|| format!("load {}", path.display()))?;
            Ok(WeightMap::from_tensors(sd.tensors))
        }
    }
}

fn has_safetensors(dir: &Path) -> bool {
    if dir.join("model.safetensors").is_file() || dir.join("model.safetensors.index.json").is_file()
    {
        return true;
    }
    first_with_ext(dir, &["safetensors"])
        .ok()
        .flatten()
        .is_some()
}

fn first_with_ext(dir: &Path, exts: &[&str]) -> Result<Option<std::path::PathBuf>> {
    if !dir.is_dir() {
        return Ok(None);
    }
    for entry in std::fs::read_dir(dir)? {
        let p = entry?.path();
        if let Some(e) = p.extension().and_then(|s| s.to_str()) {
            if exts.contains(&e) {
                return Ok(Some(p));
            }
        }
    }
    Ok(None)
}
