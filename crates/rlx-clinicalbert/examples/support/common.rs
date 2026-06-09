// Shared helpers for rlx-clinicalbert examples.
#![allow(dead_code)]

use anyhow::{Context, Result, bail};
use rlx_runtime::Device;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct ParityInputs {
    pub input_ids: Vec<u64>,
    pub attention_mask: Vec<u64>,
    pub token_type_ids: Vec<u64>,
}

#[derive(Debug, Deserialize)]
pub struct ParityMeta {
    pub seq: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

pub fn parse_flag(args: &[String], name: &str) -> Result<Option<String>> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == name {
            return Ok(Some(
                args.get(i + 1)
                    .with_context(|| format!("{name} needs a value"))?
                    .clone(),
            ));
        }
        i += 1;
    }
    Ok(None)
}

pub fn require_flag(args: &[String], name: &str) -> Result<String> {
    parse_flag(args, name)?.ok_or_else(|| anyhow::anyhow!("missing required flag {name}"))
}

pub fn parse_device(s: &str) -> Result<Device> {
    match s.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" => Ok(Device::Metal),
        "mlx" => Ok(Device::Mlx),
        "cuda" => Ok(Device::Cuda),
        "rocm" => Ok(Device::Rocm),
        "wgpu" | "vulkan" => Ok(Device::Vulkan),
        other => bail!("unknown device {other:?}"),
    }
}

pub fn weights_dir(weights: &Path) -> PathBuf {
    if weights.is_dir() {
        weights.to_path_buf()
    } else {
        weights
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

pub fn load_parity_inputs(dir: &Path) -> Result<(ParityMeta, ParityInputs)> {
    let meta: ParityMeta =
        serde_json::from_str(&fs::read_to_string(dir.join("meta.json"))?).context("meta.json")?;
    let inputs: ParityInputs = serde_json::from_str(&fs::read_to_string(dir.join("inputs.json"))?)
        .context("inputs.json")?;
    Ok((meta, inputs))
}

pub fn f32_vec_from_u64(v: &[u64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

pub fn position_ids(seq: usize) -> Vec<f32> {
    (0..seq).map(|i| i as f32).collect()
}

pub fn write_f32_bin(path: &Path, data: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}
