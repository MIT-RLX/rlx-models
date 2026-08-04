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

//! Weight preparation: convert a VLASH checkpoint to a ready-to-load bundle in
//! **GGUF** (`.gguf`) or the **RLX package** (`.rlxp`) format, with the RLX
//! canonical key names ([`crate::weights::canonical_key`]) baked in so the
//! runtime loads them with no remap.
//!
//! GGUF stores shapes in GGML order (innermost dim first — reversed from
//! row-major), while the payload bytes are identical to the row-major flatten;
//! we reverse the shape on write and on read so a `[out,in]` Linear round-trips
//! byte-for-byte at F32. Rank-1 tensors (norms/biases) are kept at F16 even
//! under a quantizing scheme (32 numbers aren't worth the accuracy hit).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rlx_core::weight_map::WeightMap;
use rlx_gguf::{GgmlType, GgufFile, GgufWriter, MetaValue, quantize};

use crate::config::VlashVariant;

/// Quantization scheme applied to the large (rank-≥2) weights when writing a
/// GGUF bundle. Norms/biases stay F16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantScheme {
    /// Lossless f32 (largest; used for exact round-trip checks).
    F32,
    /// Half precision (≈2× smaller, negligible loss). **Default.**
    F16,
    /// 8-bit blocks of 32 for rank-≥2 weights (≈4× smaller); others F16.
    Q8_0,
    /// 4-bit K-quant (superblocks of 256) for rank-≥2 weights (≈8× smaller);
    /// norms/biases and any non-divisible tensor stay F16.
    Q4K,
}

impl QuantScheme {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" => Some(QuantScheme::F32),
            "f16" | "fp16" | "half" => Some(QuantScheme::F16),
            "q8_0" | "q8" | "q8-0" => Some(QuantScheme::Q8_0),
            "q4_k" | "q4k" | "q4" | "q4_k_m" => Some(QuantScheme::Q4K),
            _ => None,
        }
    }

    fn dtype_for(self, shape: &[usize], n: usize) -> GgmlType {
        match self {
            QuantScheme::F32 => GgmlType::F32,
            QuantScheme::F16 => GgmlType::F16,
            // Q8_0 blocks are 32 wide → only quantize when divisible.
            QuantScheme::Q8_0 if shape.len() >= 2 && n.is_multiple_of(32) => GgmlType::Q8_0,
            QuantScheme::Q8_0 => GgmlType::F16,
            // Q4_K superblocks are 256 wide → only quantize when divisible.
            QuantScheme::Q4K if shape.len() >= 2 && n.is_multiple_of(256) => GgmlType::Q4K,
            QuantScheme::Q4K => GgmlType::F16,
        }
    }
}

/// Write `wm` (canonical keys) to a GGUF v3 bundle at `path`.
pub fn write_gguf(
    wm: &WeightMap,
    path: &Path,
    scheme: QuantScheme,
    variant: VlashVariant,
) -> Result<()> {
    let mut w = GgufWriter::new();
    w.set_arch("vlash");
    w.set_meta(
        "general.name",
        MetaValue::String(format!("vlash-{}", variant.as_str())),
    );
    w.set_meta(
        "vlash.variant",
        MetaValue::String(variant.as_str().to_string()),
    );

    let mut keys: Vec<String> = wm.keys().map(|s| s.to_string()).collect();
    keys.sort();
    for k in &keys {
        let (data, shape) = wm.get(k).ok_or_else(|| anyhow!("missing tensor {k}"))?;
        let n: usize = shape.iter().product();
        let dt = scheme.dtype_for(shape, n);
        let bytes = quantize(data, dt).with_context(|| format!("quantize {k} as {dt:?}"))?;
        // GGML order = row-major reversed.
        let ggml_shape: Vec<usize> = shape.iter().rev().copied().collect();
        w.add_tensor_bytes(k.clone(), ggml_shape, dt, bytes)
            .with_context(|| format!("add tensor {k}"))?;
    }
    w.write_to_path(path)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Read a GGUF bundle into a [`WeightMap`] (keys verbatim, shapes de-reversed to
/// row-major, all tensors dequantized to f32).
pub fn read_gguf(path: &Path) -> Result<WeightMap> {
    let f = GgufFile::from_path(path).with_context(|| format!("open {}", path.display()))?;
    let names: Vec<String> = f.keys().map(|s| s.to_string()).collect();
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for name in names {
        let (data, ggml_shape) = f.dequant_f32(&name)?;
        let shape: Vec<usize> = ggml_shape.iter().rev().copied().collect();
        t.insert(name, (data, shape));
    }
    Ok(WeightMap::from_tensors(t))
}

/// Load prepared or raw weights into a canonical-keyed [`WeightMap`], dispatching
/// on the path extension:
/// - `*.gguf`  → [`read_gguf`] (canonical keys already baked in),
/// - `*.rlxp`  → [`read_rlxp`],
/// - otherwise → [`crate::weights::load_remapped`] (safetensors file/dir + remap).
pub fn load_prepped(path: &str) -> Result<WeightMap> {
    let p = Path::new(path);
    match p.extension().and_then(|e| e.to_str()) {
        Some("gguf") => read_gguf(p),
        Some("rlxp") | Some("rlxpack") => read_rlxp(p),
        _ => crate::weights::load_remapped(path),
    }
}

// ---------------------------------------------------------------- rlxp --------

/// The rlx-pkg `scheme` string + encoded bytes for one tensor under `scheme`.
/// rlxp stores shapes verbatim (row-major), so `[out,in]` round-trips as
/// identity; only the payload encoding changes with the scheme.
fn rlxp_scheme_bytes(
    data: &[f32],
    shape: &[usize],
    scheme: QuantScheme,
) -> Result<(&'static str, Vec<u8>)> {
    let n: usize = shape.iter().product();
    let dt = scheme.dtype_for(shape, n);
    let bytes = quantize(data, dt).with_context(|| format!("quantize as {dt:?}"))?;
    let label = match dt {
        GgmlType::F32 => "f32",
        GgmlType::F16 => "f16",
        GgmlType::Q8_0 => "gguf_q8_0",
        GgmlType::Q4K => "gguf_q4_k",
        other => anyhow::bail!("unsupported rlxp scheme {other:?}"),
    };
    Ok((label, bytes))
}

/// Write `wm` (canonical keys) to a flat `.rlxp` package under `scheme`
/// (F32/F16/Q8_0; norms/biases stay F16 under Q8_0). The variant is embedded as
/// a `config.json` sidecar.
pub fn write_rlxp(
    wm: &WeightMap,
    path: &Path,
    variant: VlashVariant,
    scheme: QuantScheme,
) -> Result<()> {
    let mut weights: Vec<rlx_pkg::PackedWeight> = Vec::with_capacity(wm.len());
    let mut keys: Vec<String> = wm.keys().map(|s| s.to_string()).collect();
    keys.sort();
    for k in &keys {
        let (data, shape) = wm.get(k).ok_or_else(|| anyhow!("missing tensor {k}"))?;
        let (label, bytes) = rlxp_scheme_bytes(data, shape, scheme)?;
        weights.push(rlx_pkg::PackedWeight::hot(
            k.clone(),
            shape.to_vec(),
            label,
            "row_major",
            bytes,
        ));
    }
    let config = format!(
        "{{\"model_type\":\"vlash\",\"variant\":\"{}\"}}",
        variant.as_str()
    );
    let opts = rlx_pkg::WriteOptions {
        name: format!("vlash-{}", variant.as_str()),
        producer: Some("rlx-vlash".to_string()),
        container: rlx_pkg::ContainerKind::Flat,
        include_graph: false, // weight-only pack
        sidecars: vec![(
            "config.json".to_string(),
            "application/json".to_string(),
            config.into_bytes(),
        )],
        ..Default::default()
    };
    // A weight-only pack needs no executable graph.
    let graph = rlx_ir::Graph::new(format!("vlash-{}", variant.as_str()));
    rlx_pkg::write_package(path, &graph, &weights, &opts)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Read a `.rlxp` weight pack into a [`WeightMap`] (keys + row-major shapes
/// verbatim), dequantizing each tensor to f32 by its stored scheme
/// (`f32` / `f16` / `bf16` / `gguf_q8_0`).
pub fn read_rlxp(path: &Path) -> Result<WeightMap> {
    let pack = rlx_pkg::Package::open(path).with_context(|| format!("open {}", path.display()))?;
    let idx = pack
        .weights_index()
        .ok_or_else(|| anyhow!("{}: no weights index", path.display()))?;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for e in &idx.tensors {
        let n: usize = e.shape.iter().product();
        let data = match e.scheme.as_str() {
            s if s == "f32" || s.starts_with("f32") => pack.tensor_f32(&e.name)?,
            "f16" => {
                let raw = pack.tensor_bytes(&e.name)?;
                (0..n)
                    .map(|i| half::f16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]).to_f32())
                    .collect()
            }
            "bf16" => {
                let raw = pack.tensor_bytes(&e.name)?;
                (0..n)
                    .map(|i| half::bf16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]).to_f32())
                    .collect()
            }
            "gguf_q8_0" => {
                let raw = pack.tensor_bytes(&e.name)?;
                rlx_gguf::dequant_q8_0(&raw, n)
                    .with_context(|| format!("dequant q8_0 {}", e.name))?
            }
            "gguf_q4_k" => {
                let raw = pack.tensor_bytes(&e.name)?;
                rlx_gguf::dequant_q4_k(&raw, n)
                    .with_context(|| format!("dequant q4_k {}", e.name))?
            }
            other => anyhow::bail!("{}: unsupported rlxp scheme {other:?}", e.name),
        };
        t.insert(e.name.clone(), (data, e.shape.clone()));
    }
    Ok(WeightMap::from_tensors(t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{synth_weights, tiny_config};

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("rlx_vlash_prep_{name}_{}", std::process::id()));
        p
    }

    #[test]
    fn gguf_f32_roundtrip_exact() {
        let cfg = tiny_config(VlashVariant::Pi05);
        let wm = synth_weights(&cfg);
        let path = tmp("f32.gguf");
        write_gguf(&wm, &path, QuantScheme::F32, VlashVariant::Pi05).unwrap();
        let back = read_gguf(&path).unwrap();

        let mut n_checked = 0;
        for k in wm.keys() {
            let (a, sa) = wm.get(k).unwrap();
            let (b, sb) = back.get(k).expect("key present after roundtrip");
            assert_eq!(sa, sb, "shape mismatch for {k}");
            assert_eq!(a, b, "F32 data mismatch for {k}");
            n_checked += 1;
        }
        assert!(n_checked > 50, "expected many tensors, got {n_checked}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn gguf_f16_roundtrip_close() {
        let cfg = tiny_config(VlashVariant::Pi0);
        let wm = synth_weights(&cfg);
        let path = tmp("f16.gguf");
        write_gguf(&wm, &path, QuantScheme::F16, VlashVariant::Pi0).unwrap();
        let back = read_gguf(&path).unwrap();
        for k in wm.keys() {
            let (a, _) = wm.get(k).unwrap();
            let (b, _) = back.get(k).unwrap();
            let maxd = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(maxd < 1e-2, "{k}: F16 roundtrip max|Δ|={maxd}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rlxp_f32_roundtrip_exact() {
        let cfg = tiny_config(VlashVariant::Pi05);
        let wm = synth_weights(&cfg);
        let path = tmp("bundle-f32.rlxp");
        write_rlxp(&wm, &path, VlashVariant::Pi05, QuantScheme::F32).unwrap();
        let back = read_rlxp(&path).unwrap();
        let mut n = 0;
        for k in wm.keys() {
            let (a, sa) = wm.get(k).unwrap();
            let (b, sb) = back.get(k).expect("key present after rlxp roundtrip");
            assert_eq!(sa, sb, "shape mismatch for {k}");
            assert_eq!(a, b, "rlxp f32 data mismatch for {k}");
            n += 1;
        }
        assert!(n > 50, "expected many tensors, got {n}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rlxp_f16_and_q8_roundtrip_close() {
        let cfg = tiny_config(VlashVariant::Pi0);
        let wm = synth_weights(&cfg);
        for (name, scheme, tol) in [
            ("bundle-f16.rlxp", QuantScheme::F16, 1e-2f32),
            ("bundle-q8.rlxp", QuantScheme::Q8_0, 5e-2f32),
            ("bundle-q4.rlxp", QuantScheme::Q4K, 5e-2f32),
        ] {
            let path = tmp(name);
            write_rlxp(&wm, &path, VlashVariant::Pi0, scheme).unwrap();
            let back = read_rlxp(&path).unwrap();
            for k in wm.keys() {
                let (a, sa) = wm.get(k).unwrap();
                let (b, sb) = back.get(k).expect("key present");
                assert_eq!(sa, sb, "{name}: shape mismatch {k}");
                let maxd = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                assert!(maxd < tol, "{name} {k}: max|Δ|={maxd} (tol {tol})");
            }
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn gguf_q8_and_q4_roundtrip_close() {
        let cfg = tiny_config(VlashVariant::Pi0);
        let wm = synth_weights(&cfg);
        // Values are small (≈[-0.06,0.06]); Q8_0 ~1e-3, Q4_K ~1e-2 worst-case.
        for (name, scheme, tol) in [
            ("q8.gguf", QuantScheme::Q8_0, 5e-2f32),
            ("q4.gguf", QuantScheme::Q4K, 5e-2f32),
        ] {
            let path = tmp(name);
            write_gguf(&wm, &path, scheme, VlashVariant::Pi0).unwrap();
            let back = read_gguf(&path).unwrap();
            for k in wm.keys() {
                let (a, _) = wm.get(k).unwrap();
                let (b, _) = back.get(k).unwrap();
                let maxd = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                assert!(maxd < tol, "{name} {k}: roundtrip max|Δ|={maxd}");
            }
            let _ = std::fs::remove_file(&path);
        }
    }
}
