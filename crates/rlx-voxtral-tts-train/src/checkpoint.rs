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

//! Safetensors export + merge into `consolidated.safetensors`.

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use rlx_voxtral_tts::config::{CONSOLIDATED_WEIGHTS, CodecArgs, resolve_model_dir};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::weights::{
    WeightStore, graph_param_shape, graph_param_to_hf_key, hf_key_to_graph_param,
    hf_lora_key_to_graph_param, is_encoder_param, lora_param_shape, lora_param_to_hf_key,
    tensor_view_to_f32,
};

pub fn export_encoder_weights(weights: &WeightStore, path: &Path, codec: &CodecArgs) -> Result<()> {
    let mut filtered = WeightStore::default();
    for (name, data) in &weights.0 {
        if is_encoder_param(name) {
            filtered.0.insert(name.clone(), data.clone());
        }
    }
    export_named_weights(&filtered, path, graph_param_to_hf_key, codec)
}

pub fn export_lora_weights(weights: &WeightStore, path: &Path) -> Result<()> {
    let mut storages: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
    for (name, data) in &weights.0 {
        let key = lora_param_to_hf_key(name);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shape = lora_param_shape(name, data.len())?;
        storages.push((key, bytes, shape));
    }
    write_storages(path, &storages)
}

pub fn load_encoder_weights(path: &Path) -> Result<WeightStore> {
    load_named_weights(path, |key| {
        if key.starts_with(crate::config::PREFIX_CODEC) {
            hf_key_to_graph_param(key)
        } else {
            Some(key.to_string())
        }
    })
}

pub fn load_lora_weights(path: &Path) -> Result<WeightStore> {
    load_named_weights(path, |key| {
        if key.starts_with("lora.") {
            Some(key.to_string())
        } else {
            hf_lora_key_to_graph_param(key)
        }
    })
}

fn load_named_weights(
    path: &Path,
    key_to_param: impl Fn(&str) -> Option<String>,
) -> Result<WeightStore> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)?;
    let mut out = WeightStore::default();
    for key in st.names() {
        let Some(param) = key_to_param(key) else {
            continue;
        };
        let view = st.tensor(key)?;
        let data = tensor_view_to_f32(view.data(), view.dtype())?;
        out.0.insert(param, data);
    }
    if out.0.is_empty() {
        bail!("no tensors loaded from {}", path.display());
    }
    Ok(out)
}

fn write_storages(path: &Path, storages: &[(String, Vec<u8>, Vec<usize>)]) -> Result<()> {
    let mut views: HashMap<String, safetensors::tensor::TensorView> = HashMap::new();
    for (key, bytes, shape) in storages {
        views.insert(
            key.clone(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                .context("tensor view")?,
        );
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    safetensors::serialize_to_file(&views, None, path)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn export_named_weights(
    weights: &WeightStore,
    path: &Path,
    key_fn: impl Fn(&str) -> String,
    codec: &rlx_voxtral_tts::config::CodecArgs,
) -> Result<()> {
    let mut storages: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
    for (name, data) in &weights.0 {
        let key = key_fn(name);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shape = graph_param_shape(name, data.len(), codec);
        storages.push((key, bytes, shape));
    }
    write_storages(path, &storages)
}

pub fn inject_weights(
    model_dir: &Path,
    encoder_weights: Option<&Path>,
    lora_weights: Option<&Path>,
) -> Result<PathBuf> {
    let dir = resolve_model_dir(model_dir)?;
    let consolidated = dir.join(CONSOLIDATED_WEIGHTS);
    if !consolidated.is_file() {
        bail!("missing {}", consolidated.display());
    }
    backup_once(&consolidated)?;

    let bytes = fs::read(&consolidated)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let mut merged: HashMap<String, (Vec<u8>, Vec<usize>, safetensors::Dtype)> = HashMap::new();
    for key in st.names() {
        let view = st.tensor(key)?;
        merged.insert(
            key.to_string(),
            (view.data().to_vec(), view.shape().to_vec(), view.dtype()),
        );
    }

    if let Some(enc) = encoder_weights {
        merge_file_into(&mut merged, enc, true)?;
    }
    if let Some(lora) = lora_weights {
        merge_file_into(&mut merged, lora, false)?;
    }

    let storages: Vec<(String, Vec<u8>, Vec<usize>, safetensors::Dtype)> = merged
        .into_iter()
        .map(|(k, (d, s, ty))| (k, d, s, ty))
        .collect();
    let mut views: HashMap<String, safetensors::tensor::TensorView> = HashMap::new();
    for (key, data, shape, dtype) in &storages {
        views.insert(
            key.clone(),
            safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)
                .with_context(|| format!("view {key}"))?,
        );
    }
    safetensors::serialize_to_file(&views, None, &consolidated)
        .with_context(|| format!("write {}", consolidated.display()))?;
    Ok(consolidated)
}

fn merge_file_into(
    merged: &mut HashMap<String, (Vec<u8>, Vec<usize>, safetensors::Dtype)>,
    path: &Path,
    bf16_out: bool,
) -> Result<()> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)?;
    for key in st.names() {
        let view = st.tensor(key)?;
        let shape: Vec<usize> = view.shape().to_vec();
        let raw = view.data();
        let (data, dtype) = if bf16_out {
            (
                tensor_to_bf16_bytes(raw, view.dtype())?,
                safetensors::Dtype::BF16,
            )
        } else {
            (raw.to_vec(), view.dtype())
        };
        merged.insert(key.to_string(), (data, shape, dtype));
    }
    Ok(())
}

fn tensor_to_bf16_bytes(raw: &[u8], dtype: safetensors::Dtype) -> Result<Vec<u8>> {
    match dtype {
        safetensors::Dtype::F32 => {
            ensure!(raw.len().is_multiple_of(4));
            Ok(raw
                .chunks_exact(4)
                .flat_map(|chunk| {
                    bf16::from_f32(f32::from_le_bytes(chunk.try_into().unwrap())).to_le_bytes()
                })
                .collect())
        }
        safetensors::Dtype::BF16 => Ok(raw.to_vec()),
        other => bail!("unsupported dtype for inject: {other:?}"),
    }
}

fn backup_once(path: &Path) -> Result<()> {
    let backup = path.with_extension("safetensors.backup");
    if backup.exists() {
        return Ok(());
    }
    fs::copy(path, &backup).with_context(|| format!("backup {}", path.display()))?;
    Ok(())
}
