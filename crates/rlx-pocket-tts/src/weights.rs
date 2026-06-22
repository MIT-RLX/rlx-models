// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Safetensors → ndarray loader with BF16 → F32 conversion.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use half::bf16;
use ndarray::{Array, Array1, Array2, Array3, ArrayD, IxDyn};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};

/// Owns the safetensors file bytes plus a `SafeTensors` view.
pub struct WeightFile {
    bytes: Vec<u8>,
}

impl WeightFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        // Validate header.
        let _st = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parse safetensors {}", path.display()))?;
        Ok(Self { bytes })
    }

    pub fn view(&self) -> Result<SafeTensors<'_>> {
        SafeTensors::deserialize(&self.bytes).map_err(|e| anyhow!("safetensors parse: {e}"))
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let view = self.view()?;
        view.tensor(name)
            .map_err(|e| anyhow!("missing tensor `{name}`: {e}"))
    }

    pub fn names(&self) -> Result<Vec<String>> {
        Ok(self
            .view()?
            .names()
            .into_iter()
            .map(|s| s.to_string())
            .collect())
    }

    /// Get a tensor as `Vec<f32>` regardless of stored dtype (F32, BF16, F16).
    pub fn get_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let t = self.tensor(name)?;
        let shape: Vec<usize> = t.shape().to_vec();
        let raw = t.data();
        let data = match t.dtype() {
            Dtype::F32 => bytes_to_f32(raw),
            Dtype::BF16 => bf16_bytes_to_f32(raw),
            Dtype::F16 => f16_bytes_to_f32(raw),
            other => return Err(anyhow!("unsupported dtype {other:?} for {name}")),
        };
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(anyhow!(
                "tensor {name}: decoded {} f32s, expected {} (shape {:?})",
                data.len(),
                expected,
                shape
            ));
        }
        Ok((data, shape))
    }

    pub fn get_1d(&self, name: &str) -> Result<Array1<f32>> {
        let (data, shape) = self.get_f32(name)?;
        if shape.len() != 1 {
            return Err(anyhow!("{name}: expected 1D, got {:?}", shape));
        }
        Ok(Array1::from(data))
    }

    pub fn get_2d(&self, name: &str) -> Result<Array2<f32>> {
        let (data, shape) = self.get_f32(name)?;
        if shape.len() != 2 {
            return Err(anyhow!("{name}: expected 2D, got {:?}", shape));
        }
        Array2::from_shape_vec((shape[0], shape[1]), data)
            .map_err(|e| anyhow!("{name}: reshape: {e}"))
    }

    pub fn get_3d(&self, name: &str) -> Result<Array3<f32>> {
        let (data, shape) = self.get_f32(name)?;
        if shape.len() != 3 {
            return Err(anyhow!("{name}: expected 3D, got {:?}", shape));
        }
        Array3::from_shape_vec((shape[0], shape[1], shape[2]), data)
            .map_err(|e| anyhow!("{name}: reshape: {e}"))
    }

    pub fn get_dyn(&self, name: &str) -> Result<ArrayD<f32>> {
        let (data, shape) = self.get_f32(name)?;
        Array::from_shape_vec(IxDyn(&shape), data).map_err(|e| anyhow!("{name}: reshape: {e}"))
    }

    /// Optional 1D weight. Returns `None` if the tensor is absent.
    pub fn opt_1d(&self, name: &str) -> Result<Option<Array1<f32>>> {
        if self.tensor(name).is_ok() {
            Ok(Some(self.get_1d(name)?))
        } else {
            Ok(None)
        }
    }
}

fn bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(raw.len() / 4);
    for chunk in raw.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

fn bf16_bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(raw.len() / 2);
    for chunk in raw.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(bf16::from_bits(bits).to_f32());
    }
    out
}

fn f16_bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(raw.len() / 2);
    for chunk in raw.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(half::f16::from_bits(bits).to_f32());
    }
    out
}
