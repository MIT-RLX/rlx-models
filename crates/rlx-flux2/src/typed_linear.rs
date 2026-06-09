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

//! BF16/F16 linear weights kept in native dtype for GPU upload (avoids f32 RAM doubling).

use anyhow::{Context, Result, ensure};
use rlx_ir::DType;
use safetensors::SafeTensors;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// One transposed linear weight matrix `[in_dim, out_dim]` in bf16/f16 bytes + f32 bias.
#[derive(Debug, Clone)]
pub struct TypedLinear {
    pub weight_bytes: Vec<u8>,
    pub bias: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
    pub dtype: DType,
}

/// Prefix → typed linear (e.g. `"x_embedder"` for `x_embedder.weight`).
#[derive(Debug, Clone, Default)]
pub struct TypedLinearStore {
    pub linears: HashMap<String, TypedLinear>,
}

impl TypedLinearStore {
    pub fn get(&self, prefix: &str) -> Option<&TypedLinear> {
        self.linears.get(prefix)
    }

    pub fn skip_keys(&self) -> HashSet<String> {
        self.linears
            .keys()
            .flat_map(|p| [format!("{p}.weight"), format!("{p}.bias")])
            .collect()
    }
}

/// Load 2D `.weight` tensors as BF16/F16 without f32 widen. Keys are linear prefixes.
pub fn load_typed_linears_from_file(
    path: &Path,
    exclude_prefixes: &HashSet<String>,
) -> Result<TypedLinearStore> {
    let data = std::fs::read(path).with_context(|| format!("reading {path:?}"))?;
    load_typed_linears_from_bytes(&data, exclude_prefixes)
}

pub fn load_typed_linears_from_bytes(
    data: &[u8],
    exclude_prefixes: &HashSet<String>,
) -> Result<TypedLinearStore> {
    let st = SafeTensors::deserialize(data).context("parsing safetensors")?;
    let mut linears = HashMap::new();
    let mut biases: HashMap<String, Vec<f32>> = HashMap::new();

    for (name, view) in st.tensors() {
        if name.ends_with(".bias") {
            let prefix = name.strip_suffix(".bias").unwrap().to_string();
            if exclude_prefixes.contains(&prefix) {
                continue;
            }
            let bytes = view.data();
            let bias = match view.dtype() {
                safetensors::Dtype::F32 => decode_f32(bytes),
                safetensors::Dtype::F16 => bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::BF16 => bytes
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                other => anyhow::bail!("unsupported bias dtype {other:?} for {name}"),
            };
            biases.insert(prefix, bias);
            continue;
        }
        if !name.ends_with(".weight") {
            continue;
        }
        let prefix = name.strip_suffix(".weight").unwrap().to_string();
        if exclude_prefixes.contains(&prefix) {
            continue;
        }
        let shape: Vec<usize> = view.shape().to_vec();
        if shape.len() != 2 {
            continue;
        }
        let (out_dim, in_dim) = (shape[0], shape[1]);
        let (dtype, weight_bytes) = match view.dtype() {
            safetensors::Dtype::BF16 => (
                DType::BF16,
                transpose_half_bytes(view.data(), out_dim, in_dim),
            ),
            safetensors::Dtype::F16 => (
                DType::F16,
                transpose_half_bytes(view.data(), out_dim, in_dim),
            ),
            _ => continue,
        };
        linears.insert(
            prefix.clone(),
            TypedLinear {
                weight_bytes,
                bias: Vec::new(),
                in_dim,
                out_dim,
                dtype,
            },
        );
    }

    for (prefix, tl) in linears.iter_mut() {
        if let Some(b) = biases.remove(prefix) {
            ensure!(
                b.len() == tl.out_dim,
                "bias len {} != out_dim {} for {prefix}",
                b.len(),
                tl.out_dim
            );
            tl.bias = b;
        }
    }

    Ok(TypedLinearStore { linears })
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn transpose_half_bytes(data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let elem = 2usize;
    assert_eq!(
        data.len(),
        rows * cols * elem,
        "transpose_half_bytes size mismatch"
    );
    let mut out = vec![0u8; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            let src = (r * cols + c) * elem;
            let dst = (c * rows + r) * elem;
            out[dst..dst + elem].copy_from_slice(&data[src..src + elem]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_bf16_2x3() {
        let mut data = Vec::new();
        for i in 0..6u16 {
            data.extend_from_slice(&half::bf16::from_f32(i as f32).to_le_bytes());
        }
        let t = transpose_half_bytes(&data, 2, 3);
        assert_eq!(t.len(), 12);
    }
}
