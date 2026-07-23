//!
//! Loads tensors one-at-a-time from the file so peak RSS stays near the owned
//! f32 map size instead of (file bytes + f32 map) during open.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

#[derive(Default)]
pub struct Weights {
    map: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl Weights {
    pub fn load(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut hdr_len_buf = [0u8; 8];
        file.read_exact(&mut hdr_len_buf)
            .with_context(|| format!("read header length {}", path.display()))?;
        let hdr_len = u64::from_le_bytes(hdr_len_buf) as usize;
        ensure!(
            hdr_len > 0 && hdr_len < 64 * 1024 * 1024,
            "implausible safetensors header length {hdr_len} in {}",
            path.display()
        );
        let mut hdr_bytes = vec![0u8; hdr_len];
        file.read_exact(&mut hdr_bytes)
            .with_context(|| format!("read header {}", path.display()))?;
        let header: Value = serde_json::from_slice(&hdr_bytes)
            .with_context(|| format!("parse safetensors header {}", path.display()))?;
        let obj = header
            .as_object()
            .with_context(|| format!("safetensors header is not an object {}", path.display()))?;
        let data_base = 8u64 + hdr_len as u64;

        let mut map = HashMap::with_capacity(obj.len().saturating_sub(1));
        // Scratch reused across tensors so we never hold (full file + all f32).
        let mut raw = Vec::new();
        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = info
                .get("dtype")
                .and_then(|v| v.as_str())
                .with_context(|| format!("tensor {name} missing dtype"))?;
            let shape: Vec<usize> = info
                .get("shape")
                .and_then(|v| v.as_array())
                .with_context(|| format!("tensor {name} missing shape"))?
                .iter()
                .map(|x| {
                    x.as_u64()
                        .map(|n| n as usize)
                        .context("non-integer shape dim")
                })
                .collect::<Result<_>>()?;
            let offsets = info
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .with_context(|| format!("tensor {name} missing data_offsets"))?;
            ensure!(offsets.len() == 2, "tensor {name} data_offsets len != 2");
            let start = offsets[0]
                .as_u64()
                .context("data_offsets[0]")?
                .saturating_add(data_base);
            let end = offsets[1]
                .as_u64()
                .context("data_offsets[1]")?
                .saturating_add(data_base);
            ensure!(end >= start, "tensor {name} inverted data_offsets");
            let nbytes = (end - start) as usize;
            raw.resize(nbytes, 0);
            file.seek(SeekFrom::Start(start))
                .with_context(|| format!("seek {name} in {}", path.display()))?;
            file.read_exact(&mut raw)
                .with_context(|| format!("read tensor {name} from {}", path.display()))?;
            let data = decode_f32(dtype, &raw)
                .with_context(|| format!("decode tensor {name} ({dtype})"))?;
            map.insert(name.clone(), (data, shape));
        }
        Ok(Self { map })
    }

    /// Build from an already-decoded f32 map (e.g. GGUF dequant).
    pub fn from_map(map: HashMap<String, (Vec<f32>, Vec<usize>)>) -> Self {
        Self { map }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Result<&(Vec<f32>, Vec<usize>)> {
        self.map
            .get(name)
            .with_context(|| format!("missing weight tensor: {name}"))
    }

    pub fn data(&self, name: &str) -> Result<&[f32]> {
        Ok(self.get(name)?.0.as_slice())
    }

    pub fn f16_round_params(&mut self) {
        for (data, _) in self.map.values_mut() {
            for v in data.iter_mut() {
                *v = half::f16::from_f32(*v).to_f32();
            }
        }
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(self.get(name)?.1.as_slice())
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

fn decode_f32(dtype: &str, raw: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        "F32" => {
            ensure!(raw.len() % 4 == 0, "F32 byte length not multiple of 4");
            Ok(raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        "F16" => {
            ensure!(raw.len() % 2 == 0, "F16 byte length not multiple of 2");
            Ok(raw
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect())
        }
        other => bail!("unexpected dtype {other}"),
    }
}
