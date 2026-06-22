use anyhow::{Context, Result, bail};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

pub struct WeightStore {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightStore {
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes).context("parse safetensors")?;
        let mut tensors = HashMap::new();
        for name in st.names() {
            let view = st.tensor(name).with_context(|| format!("tensor {name}"))?;
            let data = tensor_f32(&view, name)?;
            let shape = view.shape().to_vec();
            let key = normalize_key(name);
            tensors.insert(key, (data, shape));
        }
        Ok(Self { tensors })
    }

    pub fn get(&self, key: &str) -> Result<(&[f32], &[usize])> {
        self.tensors
            .get(key)
            .map(|(d, s)| (d.as_slice(), s.as_slice()))
            .with_context(|| format!("missing weight {key}"))
    }

    pub fn get_optional(&self, key: &str) -> Result<Option<(Vec<f32>, Vec<usize>)>> {
        Ok(self.tensors.get(key).cloned())
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}

fn normalize_key(name: &str) -> String {
    name.strip_prefix("dac.").unwrap_or(name).to_string()
}

fn tensor_f32(view: &safetensors::tensor::TensorView<'_>, name: &str) -> Result<Vec<f32>> {
    use safetensors::tensor::Dtype;
    let raw = view.data();
    Ok(match view.dtype() {
        Dtype::F32 => {
            let n = raw.len() / 4;
            let mut out = Vec::with_capacity(n);
            #[cfg(target_endian = "little")]
            unsafe {
                std::ptr::copy_nonoverlapping(raw.as_ptr(), out.as_mut_ptr() as *mut u8, raw.len());
                out.set_len(n);
            }
            #[cfg(not(target_endian = "little"))]
            {
                out.extend(
                    raw.chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                );
            }
            out
        }
        dt => bail!("tensor {name}: unsupported dtype {dt:?} (expected F32)"),
    })
}
