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

//! Minimal safetensors f32 map helpers.

use anyhow::{Context, Result, bail};
use safetensors::tensor::{SafeTensors, TensorView};
use safetensors::{Dtype, serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub fn save_f32_map(path: &Path, tensors: &HashMap<String, Vec<f32>>) -> Result<()> {
    let owned: HashMap<String, Vec<u8>> = tensors
        .iter()
        .map(|(k, v)| (k.clone(), f32_slice_to_bytes(v)))
        .collect();
    let mut views = HashMap::new();
    for (k, bytes) in &owned {
        let shape = vec![tensors[k].len()];
        views.insert(k.clone(), TensorView::new(Dtype::F32, shape, bytes)?);
    }
    let serialized = serialize(views, None)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serialized).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

pub fn load_f32_map(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    load_f32_map_bytes(&bytes)
}

pub fn load_f32_map_bytes(bytes: &[u8]) -> Result<HashMap<String, Vec<f32>>> {
    let st = SafeTensors::deserialize(bytes)?;
    let mut out = HashMap::new();
    for name in st.names() {
        let t = st.tensor(name)?;
        if t.dtype() != Dtype::F32 {
            bail!("tensor {name} is {:?}, expected F32", t.dtype());
        }
        let data = t.data();
        if data.len() % 4 != 0 {
            bail!("tensor {name} byte length not multiple of 4");
        }
        let mut v = Vec::with_capacity(data.len() / 4);
        for chunk in data.chunks_exact(4) {
            v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        out.insert(name.to_string(), v);
    }
    Ok(out)
}

fn f32_slice_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
