// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Load Zonos `model.safetensors` (BF16) into f32 maps.
//!
//! Compiled Metal graphs store Linear weights as F16 in the arena (see
//! [`crate::flow::use_f16_linear_weights`]); host maps stay f32 and Metal
//! narrows on `set_param`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

pub struct WeightMap {
    pub tensors: HashMap<String, Vec<f32>>,
    pub shapes: HashMap<String, Vec<usize>>,
}

impl WeightMap {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())
            .with_context(|| format!("read {}", path.as_ref().display()))?;
        let st = SafeTensors::deserialize(&bytes).context("deserialize safetensors")?;
        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        for name in st.names() {
            let tv = st.tensor(name)?;
            let shape: Vec<usize> = tv.shape().to_vec();
            let data = tensor_to_f32(&tv)?;
            shapes.insert(name.to_string(), shape);
            tensors.insert(name.to_string(), data);
        }
        Ok(Self { tensors, shapes })
    }

    pub fn get(&self, name: &str) -> Result<&[f32]> {
        self.tensors
            .get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| anyhow!("missing weight {name}"))
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        self.shapes
            .get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| anyhow!("missing shape {name}"))
    }
}

fn tensor_to_f32(tv: &TensorView<'_>) -> Result<Vec<f32>> {
    let n: usize = tv.shape().iter().product();
    match tv.dtype() {
        safetensors::Dtype::F32 => {
            let mut out = vec![0f32; n];
            for (i, chunk) in tv.data().chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(out)
        }
        safetensors::Dtype::BF16 => {
            let mut out = vec![0f32; n];
            for (i, chunk) in tv.data().chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out[i] = bf16_to_f32(bits);
            }
            Ok(out)
        }
        other => Err(anyhow!("unsupported dtype {other:?}")),
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
