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

//! Checkpoint loading. Microsoft's TrOCR repos ship `model.safetensors`
//! (base) or only `pytorch_model.bin` (small, large); both load into a
//! [`WeightMap`](rlx_core::weight_map::WeightMap) under the HF tensor
//! names, which the graph builders take from and check off.

use anyhow::{Context, Result, bail};
use rlx_core::weight_map::WeightMap;
use std::collections::HashMap;
use std::path::Path;

/// The checkpoint file in a model directory (or the path itself).
pub fn checkpoint_path(path: &Path) -> Result<std::path::PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    for name in ["model.safetensors", "pytorch_model.bin"] {
        let p = path.join(name);
        if p.is_file() {
            return Ok(p);
        }
    }
    bail!(
        "{}: no model.safetensors or pytorch_model.bin",
        path.display()
    )
}

/// Load every tensor of a TrOCR checkpoint as f32.
pub fn load(path: &Path) -> Result<WeightMap> {
    let file = checkpoint_path(path)?;
    if file.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        return rlx_core::load_weight_map(&file, &[]);
    }
    let pt = rlx_torch_ckpt::PtModel::open(&file)
        .with_context(|| format!("opening {}", file.display()))?;
    let mut t = HashMap::new();
    for name in pt.names() {
        let x = pt.tensor(&name)?;
        t.insert(name, (x.data, x.shape));
    }
    Ok(WeightMap::from_tensors(t))
}
