//! Safetensors weight store: name → (f32 data, shape). All tensors are f32.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use safetensors::SafeTensors;

#[derive(Default)]
pub struct Weights {
    map: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl Weights {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let st = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parse safetensors {}", path.display()))?;
        let mut map = HashMap::new();
        for name in st.names() {
            let view = st.tensor(name)?;
            let shape = view.shape().to_vec();
            use safetensors::tensor::Dtype;
            let data: Vec<f32> = match view.dtype() {
                Dtype::F32 => view
                    .data()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                other => anyhow::bail!("unexpected dtype {other:?} for tensor {name}"),
            };
            map.insert(name.to_string(), (data, shape));
        }
        Ok(Self { map })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Result<&(Vec<f32>, Vec<usize>)> {
        self.map
            .get(name)
            .with_context(|| format!("missing weight tensor: {name}"))
    }

    /// Raw data slice.
    pub fn data(&self, name: &str) -> Result<&[f32]> {
        Ok(self.get(name)?.0.as_slice())
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(self.get(name)?.1.as_slice())
    }

    /// A 1-D vector (e.g. bias / norm weight).
    pub fn vec1(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self.get(name)?.0.clone())
    }
}
