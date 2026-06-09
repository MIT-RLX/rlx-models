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

//! K-quant / legacy GGUF ViT weights for DINOv2 (`DequantMatMul` via [`rlx_flow::GgufPackedParams`]).

use anyhow::{Result, ensure};
use rlx_core::gguf_support::load_gguf_file;
use rlx_core::weight_loader::{GgufLoader, WeightLoader};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{GgufPackedLinear, GgufPackedParams};
use rlx_gguf::GgmlType;
use std::collections::HashMap;
use std::path::Path;

/// True when the GGUF file stores quant matmul weights (K-quant or Q4_0/Q8_0).
pub fn gguf_has_packed_linears(path: &Path) -> Result<bool> {
    let raw = load_gguf_file(path)?;
    Ok(raw.tensors.values().any(|t| {
        matches!(
            t.dtype,
            GgmlType::Q4K
                | GgmlType::Q5K
                | GgmlType::Q6K
                | GgmlType::Q8K
                | GgmlType::Q2K
                | GgmlType::Q3K
                | GgmlType::Q4_0
                | GgmlType::Q8_0
        )
    }))
}

/// Load DINOv2 from GGUF: packed linears + F32 norms/embeddings in a [`WeightMap`].
pub fn load_dinov2_from_gguf(path: &Path) -> Result<(WeightMap, GgufPackedParams)> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-utf8 path {:?}", path))?;
    let mut loader = GgufLoader::from_file(path_str)?;
    let keys = loader.remaining_keys();

    let mut linears = HashMap::new();
    let mut f32_tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    for key in &keys {
        if let Some(prefix) = key.strip_suffix(".weight") {
            if let Some((bytes, scheme, shape)) = loader.take_packed(key)? {
                ensure!(shape.len() == 2, "{key}: expected 2D weight, got {shape:?}");
                let in_dim = shape[0];
                let out_dim = shape[1];
                let bias_key = format!("{prefix}.bias");
                let bias = if keys.iter().any(|k| k == &bias_key) {
                    let (b, bshape) = loader.take(&bias_key)?;
                    ensure!(bshape == vec![out_dim], "{bias_key}: shape mismatch");
                    b
                } else {
                    vec![0.0f32; out_dim]
                };
                linears.insert(
                    prefix.to_string(),
                    GgufPackedLinear {
                        w_q: bytes,
                        scheme,
                        in_dim,
                        out_dim,
                        bias,
                    },
                );
                continue;
            }
        }
        let (data, shape) = loader.take(key)?;
        f32_tensors.insert(key.clone(), (data, shape));
    }

    Ok((
        WeightMap::from_tensors(f32_tensors),
        GgufPackedParams { linears },
    ))
}
