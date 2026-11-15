// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! YuNet weights: `NAME.weight` / `NAME.bias` (f32) for every [`crate::model::CONVS`] entry, in
//! a safetensors file. Made from the ONNX export once by `rlx-yunet-convert` (feature `onnx`).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use safetensors::tensor::{Dtype, TensorView};

use crate::model::CONVS;

/// Tensors by name, checked against the architecture.
#[derive(Debug, Clone, Default)]
pub struct Weights {
    pub tensors: HashMap<String, Vec<f32>>,
}

impl Weights {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_safetensors(&bytes)
    }

    pub fn from_safetensors(bytes: &[u8]) -> Result<Self> {
        let st = safetensors::SafeTensors::deserialize(bytes).context("parse safetensors")?;
        let mut tensors = HashMap::new();
        for (name, t) in st.tensors() {
            ensure!(t.dtype() == Dtype::F32, "{name}: {:?}, want f32", t.dtype());
            tensors.insert(
                name,
                t.data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect(),
            );
        }
        let w = Self { tensors };
        w.check()?;
        Ok(w)
    }

    /// Every convolution's weight and bias present, at the architecture's sizes.
    pub fn check(&self) -> Result<()> {
        for c in &CONVS {
            let n: usize = c.weight_shape().iter().product();
            for (suffix, len) in [("weight", n), ("bias", c.out_c)] {
                let key = format!("{}.{suffix}", c.name);
                match self.tensors.get(&key) {
                    Some(v) if v.len() == len => {}
                    Some(v) => bail!("{key}: {} values, want {len}", v.len()),
                    None => bail!("missing {key}"),
                }
            }
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.check()?;
        let bytes: Vec<(String, Vec<u8>, Vec<usize>)> = CONVS
            .iter()
            .flat_map(|c| {
                [
                    ("weight", c.weight_shape().to_vec()),
                    ("bias", vec![c.out_c]),
                ]
                .map(|(s, shape)| {
                    let key = format!("{}.{s}", c.name);
                    let data = self.tensors[&key]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    (key, data, shape)
                })
            })
            .collect();
        let views: Vec<(String, TensorView<'_>)> = bytes
            .iter()
            .map(|(k, d, s)| Ok((k.clone(), TensorView::new(Dtype::F32, s.clone(), d)?)))
            .collect::<Result<_>>()?;
        safetensors::serialize_to_file(views, None, path)
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }
}
