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

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use safetensors::serialize_to_file;
use safetensors::tensor::{Dtype, TensorView};

use crate::plan::WeightsFormat;

pub fn export_weights(
    params: &HashMap<String, Vec<f32>>,
    i64_params: &HashMap<String, Vec<i64>>,
    shapes: &HashMap<String, Vec<usize>>,
    out_dir: &Path,
    format: WeightsFormat,
) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir)?;
    let st_path = out_dir.join("model.safetensors");
    write_safetensors(params, i64_params, shapes, &st_path)?;
    match format {
        WeightsFormat::Safetensors => Ok(st_path),
        WeightsFormat::Gguf => {
            let gguf_path = out_dir.join("model.gguf");
            write_gguf_via_python(&st_path, &gguf_path)?;
            Ok(gguf_path)
        }
    }
}

fn write_safetensors(
    params: &HashMap<String, Vec<f32>>,
    i64_params: &HashMap<String, Vec<i64>>,
    shapes: &HashMap<String, Vec<usize>>,
    path: &Path,
) -> Result<()> {
    let mut names: Vec<_> = params.keys().chain(i64_params.keys()).collect();
    names.sort();
    names.dedup();
    let mut packed: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::with_capacity(names.len());
    for name in names {
        let shape = shapes
            .get(name.as_str())
            .cloned()
            .unwrap_or_else(|| vec![1]);
        let elems = shape.iter().product::<usize>().max(1);
        if let Some(data) = params.get(name) {
            let data = if data.is_empty() {
                vec![0.0f32; elems]
            } else {
                data.clone()
            };
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            packed.push((name.clone(), Dtype::F32, shape, bytes));
        } else if let Some(data) = i64_params.get(name) {
            let data = if data.is_empty() {
                vec![0i64; elems]
            } else {
                data.clone()
            };
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            packed.push((name.clone(), Dtype::I64, shape, bytes));
        }
    }
    let mut tensors: Vec<(String, TensorView<'_>)> = Vec::with_capacity(packed.len());
    for (name, dtype, shape, bytes) in &packed {
        let view =
            TensorView::new(*dtype, shape.clone(), bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
        tensors.push((name.clone(), view));
    }
    let meta = HashMap::from([("source".to_string(), "rlx-onnx-decompose".to_string())]);
    serialize_to_file(tensors, Some(meta), path).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

fn write_gguf_via_python(safetensors_path: &Path, gguf_path: &Path) -> Result<()> {
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/onnx_decompose_to_gguf.py");
    if !script.is_file() {
        bail!(
            "GGUF export script missing at {}; install: pip install gguf safetensors numpy",
            script.display()
        );
    }
    let status = Command::new("python3")
        .arg(&script)
        .arg(safetensors_path)
        .arg(gguf_path)
        .status()
        .context("spawn python3 for GGUF export")?;
    if !status.success() {
        bail!("GGUF export failed (exit {status})");
    }
    Ok(())
}
