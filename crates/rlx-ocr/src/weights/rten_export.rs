// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Export `.rten` graph constants to a single `.safetensors` file.

use anyhow::{Context, Result, bail};
use rlx_core::weight_map::WeightMap;
use rten_model_file::header::{Header, HeaderError};
use rten_model_file::schema::root_as_model;
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use std::collections::HashMap;
use std::path::Path;

/// Drain all f32 constants from an `.rten` file into a [`WeightMap`].
pub fn weight_map_from_rten(path: &Path) -> Result<WeightMap> {
    let data = std::fs::read(path).with_context(|| format!("read {:?}", path))?;
    let tensors = extract_f32_tensors(&data)?;
    Ok(WeightMap::from_tensors(tensors))
}

/// Write every f32 constant in an `.rten` file to `.safetensors`.
pub fn export_rten_to_safetensors(rten_path: &Path, out_path: &Path) -> Result<()> {
    let data = std::fs::read(rten_path).with_context(|| format!("read {:?}", rten_path))?;
    let tensors = extract_f32_tensors(&data)?;
    let views: HashMap<String, TensorView<'_>> = tensors
        .iter()
        .map(|(name, (data, shape))| {
            let bytes = bytemuck::cast_slice(data.as_slice());
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                .with_context(|| format!("tensor view for {name}"))?;
            Ok((name.clone(), view))
        })
        .collect::<Result<_>>()?;
    safetensors::serialize_to_file(&views, None, out_path)
        .with_context(|| format!("write {:?}", out_path))?;
    eprintln!("wrote {} tensors to {}", tensors.len(), out_path.display());
    Ok(())
}

fn extract_f32_tensors(file_data: &[u8]) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    // v2+ files start with `RTEN`; legacy ocrs checkpoints are v1 flatbuffers (no header).
    let (model_bytes, tensor_base) = match Header::from_buf(file_data) {
        Ok(header) => {
            let model_off = header.model_offset as usize;
            let model_len = header.model_len as usize;
            let model_bytes = file_data
                .get(model_off..model_off + model_len)
                .context("model segment out of range")?;
            (model_bytes, Some(header.tensor_data_offset as usize))
        }
        Err(HeaderError::InvalidMagic) => (file_data, None),
        Err(err) => bail!("invalid RTEN header: {err}"),
    };
    let model = root_as_model(model_bytes).context("parse RTEN flatbuffer")?;
    let graph = model.graph();

    let mut out = HashMap::new();
    let Some(nodes) = graph.nodes() else {
        return Ok(out);
    };

    for node in nodes.iter() {
        let Some(constant) = node.data_as_constant_node() else {
            continue;
        };
        let name = node
            .name()
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("const_{}", out.len()));
        let shape: Vec<usize> = constant.shape().iter().map(|d| d as usize).collect();
        let n_elem: usize = shape.iter().product();
        if n_elem == 0 {
            continue;
        }

        let data: Vec<f32> = if let Some(offset) = constant.data_offset() {
            let Some(tensor_base) = tensor_base else {
                eprintln!("skip {name}: external tensor offset without RTEN v2 header");
                continue;
            };
            let start = tensor_base + offset as usize;
            let end = start + n_elem * 4;
            let bytes = file_data
                .get(start..end)
                .with_context(|| format!("tensor data for {name} offset {offset}"))?;
            if bytes.len() != n_elem * 4 {
                bail!(
                    "{name}: expected {} f32 bytes, got {}",
                    n_elem * 4,
                    bytes.len()
                );
            }
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        } else if let Some(fd) = constant.data_as_float_data() {
            let vec = fd.data();
            (0..vec.len()).map(|i| vec.get(i)).collect()
        } else if let Some(id) = constant.data_as_int_32_data() {
            let vec = id.data();
            (0..vec.len()).map(|i| vec.get(i) as f32).collect()
        } else {
            eprintln!("skip {name}: unsupported constant layout");
            continue;
        };
        out.insert(name, (data, shape));
    }
    if out.is_empty() {
        bail!("no f32 constants found in RTEN file");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn export_ocrs_detection_rten() -> Result<()> {
        let path = match env::var("OCR_DETECTION_RTEN") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                eprintln!("skip export_ocrs_detection_rten: set OCR_DETECTION_RTEN");
                return Ok(());
            }
        };
        let map = weight_map_from_rten(&path)?;
        assert!(map.keys().next().is_some());
        Ok(())
    }
}
