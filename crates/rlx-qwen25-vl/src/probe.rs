// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Load / run HF AIF token-dynamics probes for paper decode.

use crate::aif::AifProbe;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Sanitize sample ids for on-disk probe filenames.
pub fn sanitize_sample_id(id: &str) -> String {
    id.replace('/', "_")
}

/// Load `{sample_id}_vision_dynamics.npy` + meta written by `scripts/aif_export_probes.py`.
pub fn load_probe_sample(probe_dir: &Path, sample_id: &str) -> Result<AifProbe> {
    let sid = sanitize_sample_id(sample_id);
    let dynamics_path = probe_dir.join(format!("{sid}_vision_dynamics.npy"));
    if !dynamics_path.is_file() {
        bail!("missing probe dynamics at {}", dynamics_path.display());
    }
    let meta_path = probe_dir.join(format!("{sid}_meta.json"));
    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&meta_path).with_context(|| format!("read {}", meta_path.display()))?,
    )
    .with_context(|| format!("parse {}", meta_path.display()))?;
    load_probe_from_dynamics_npy(&dynamics_path, &meta)
}

/// Load parity-reference layout (`vision_dynamics.npy` + `meta.json` in one dir).
pub fn load_probe_reference_dump(dump_dir: &Path) -> Result<AifProbe> {
    let dynamics_path = dump_dir.join("vision_dynamics.npy");
    let meta_path = dump_dir.join("meta.json");
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&meta_path).context("read meta.json")?)
            .context("parse meta.json")?;
    load_probe_from_dynamics_npy(&dynamics_path, &meta)
}

fn load_probe_from_dynamics_npy(
    dynamics_path: &Path,
    meta: &serde_json::Value,
) -> Result<AifProbe> {
    let flat = read_npy_f32(dynamics_path)?;
    let n_vision = meta["n_vision_tokens"].as_u64().unwrap_or(0) as usize;
    let n_layers = meta
        .get("aif_n_layers")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or_else(|| {
            if n_vision > 0 && flat.len().is_multiple_of(n_vision) {
                flat.len() / n_vision
            } else {
                0
            }
        });
    anyhow::ensure!(
        n_vision > 0 && n_layers > 0 && flat.len() == n_vision * n_layers,
        "dynamics shape mismatch: flat={} n_vision={n_vision} n_layers={n_layers}",
        flat.len()
    );
    let dynamics: Vec<Vec<f32>> = flat.chunks(n_layers).map(|c| c.to_vec()).collect();
    Ok(AifProbe::build(dynamics))
}

/// Run `scripts/aif_probe_single.py` (HF attentions) into a temp dir and load the probe.
pub fn run_hf_python_probe(
    image: &Path,
    question: &str,
    sample_id: &str,
    out_dir: &Path,
    vlmevalkit: bool,
) -> Result<AifProbe> {
    let script = probe_single_script()?;
    let python = std::env::var("RLX_QWEN25_VL_PYTHON").unwrap_or_else(|_| "python3".into());
    std::fs::create_dir_all(out_dir)?;
    let status = Command::new(&python)
        .arg(&script)
        .env("RLX_QWEN25_VL_IMAGE", image)
        .env("RLX_QWEN25_VL_OUT_DIR", out_dir)
        .env("RLX_QWEN25_VL_PROMPT", question)
        .env("RLX_QWEN25_VL_SAMPLE_ID", sanitize_sample_id(sample_id))
        .env(
            "RLX_QWEN25_VL_VLMEVALKIT",
            if vlmevalkit { "1" } else { "0" },
        )
        .envs(forward_hf_env())
        .status()
        .with_context(|| format!("run {script:?}"))?;
    anyhow::ensure!(status.success(), "HF AIF probe script failed");
    load_probe_sample(out_dir, sample_id)
}

fn probe_single_script() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("RLX_QWEN25_VL_AIF_PROBE_SCRIPT") {
        return Ok(PathBuf::from(p));
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = manifest
        .join("../../scripts/aif_probe_single.py")
        .canonicalize()
        .or_else(|_| {
            // Facade / workspace root invocation.
            PathBuf::from("scripts/aif_probe_single.py").canonicalize()
        })
        .context("locate scripts/aif_probe_single.py (set RLX_QWEN25_VL_AIF_PROBE_SCRIPT)")?;
    Ok(script)
}

fn forward_hf_env() -> Vec<(String, String)> {
    [
        "RLX_QWEN25_VL_HF_DIR",
        "RLX_QWEN25_VL_DOWNLOAD",
        "RLX_QWEN25_VL_DEVICE",
        "RLX_QWEN25_VL_VLMEVALKIT",
        "HF_TOKEN",
        "HF_HOME",
    ]
    .into_iter()
    .filter_map(|name| std::env::var(name).ok().map(|v| (name.to_string(), v)))
    .collect()
}

fn read_npy_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    parse_npy_f32(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn parse_npy_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    anyhow::ensure!(bytes.starts_with(b"\x93NUMPY"), "not npy");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let data = &bytes[10 + header_len..];
    anyhow::ensure!(data.len().is_multiple_of(4), "npy f32 payload size");
    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_sample_id_replaces_slashes() {
        assert_eq!(sanitize_sample_id("a/b"), "a_b");
    }
}
