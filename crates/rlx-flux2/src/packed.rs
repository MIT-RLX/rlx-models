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

//! NVFP4 packed linear weights for FLUX.2 (GPU `DequantMatMul` / `Nvfp4Block`).

use anyhow::{Context, Result, ensure};
use rlx_ir::nvfp4::{nvfp4_scale_bytes, nvfp4_weight_bytes};
use rlx_ir::quant::QuantScheme;
use std::collections::HashMap;
use std::path::Path;

/// One GGUF-quant linear (K-quant or Q4_0/Q8_0) for `Op::DequantMatMul`.
#[derive(Debug, Clone)]
pub struct Flux2GgufLinearPacked {
    pub w_q: Vec<u8>,
    pub scheme: QuantScheme,
    pub in_dim: usize,
    pub out_dim: usize,
    pub bias: Vec<f32>,
}

/// One NVFP4 linear: packed E2M1 weights + FP8 E4M3 block scales along K.
#[derive(Debug, Clone)]
pub struct Nvfp4LinearPacked {
    pub w_q: Vec<u8>,
    pub scale: Vec<u8>,
    pub in_dim: usize,
    pub out_dim: usize,
    pub bias: Vec<f32>,
    /// Per-tensor global scale (DequantMatMul `zp` input slot).
    pub global_scale: f32,
}

/// Param prefix → NVFP4 payload (HIR param names `{prefix}.weight` / `.scale`).
#[derive(Debug, Clone, Default)]
pub struct Flux2PackedParams {
    pub linears: HashMap<String, Nvfp4LinearPacked>,
    pub gguf_linears: HashMap<String, Flux2GgufLinearPacked>,
}

impl Flux2PackedParams {
    pub fn get_nvfp4(&self, linear_name: &str) -> Option<&Nvfp4LinearPacked> {
        self.linears.get(linear_name)
    }

    pub fn get_gguf(&self, linear_name: &str) -> Option<&Flux2GgufLinearPacked> {
        self.gguf_linears.get(linear_name)
    }

    pub fn has_packed_linear(&self, linear_name: &str) -> bool {
        self.linears.contains_key(linear_name) || self.gguf_linears.contains_key(linear_name)
    }

    /// Safetensors keys to skip when loading f32 weights (packed linears loaded separately).
    pub fn exclude_f32_keys(&self) -> std::collections::HashSet<String> {
        let mut keys = std::collections::HashSet::new();
        for prefix in self.linears.keys() {
            keys.insert(format!("{prefix}.weight"));
            keys.insert(format!("{prefix}.bias"));
            keys.insert(format!("{prefix}.weight_scale"));
            keys.insert(format!("{prefix}.global_scale"));
        }
        for prefix in self.gguf_linears.keys() {
            keys.insert(format!("{prefix}.weight"));
            keys.insert(format!("{prefix}.bias"));
        }
        keys
    }
}

/// True when a safetensors file contains U8 weight + F8_E4M3 scale pairs.
pub fn safetensors_has_nvfp4(path: &Path) -> Result<bool> {
    if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        return Ok(false);
    }
    let data = std::fs::read(path).with_context(|| format!("reading {path:?}"))?;
    let st = safetensors::SafeTensors::deserialize(&data)?;
    let tensors: Vec<_> = st.tensors();
    Ok(tensors.iter().any(|(name, view)| {
        name.ends_with(".weight")
            && matches!(view.dtype(), safetensors::Dtype::U8)
            && tensors.iter().any(|(sn, sv)| {
                sn.as_str() == format!("{name}_scale")
                    && matches!(sv.dtype(), safetensors::Dtype::F8_E4M3)
            })
    }))
}

/// Load NVFP4 linears from a safetensors checkpoint (BFL / diffusers naming).
pub fn load_flux2_nvfp4_from_file(path: &Path) -> Result<Flux2PackedParams> {
    let data = std::fs::read(path).with_context(|| format!("reading {path:?}"))?;
    let st = safetensors::SafeTensors::deserialize(&data)?;
    let tensors: Vec<_> = st.tensors();
    let mut linears = HashMap::new();

    for (name, view) in &tensors {
        if !name.ends_with(".weight") || !matches!(view.dtype(), safetensors::Dtype::U8) {
            continue;
        }
        let scale_name = format!("{name}_scale");
        let Some((_, scale_view)) = tensors.iter().find(|(n, _)| n.as_str() == scale_name) else {
            continue;
        };
        if !matches!(scale_view.dtype(), safetensors::Dtype::F8_E4M3) {
            continue;
        }
        let shape: Vec<usize> = view.shape().to_vec();
        ensure!(
            shape.len() == 2,
            "NVFP4 weight {name} must be 2D, got {shape:?}"
        );
        let (out_dim, in_dim) = (shape[0], shape[1]);
        let w_q = view.data().to_vec();
        let scale = scale_view.data().to_vec();
        let expected_w = nvfp4_weight_bytes(in_dim, out_dim);
        let expected_s = nvfp4_scale_bytes(in_dim, out_dim);
        ensure!(
            w_q.len() == expected_w,
            "{name}: weight bytes {} != expected {expected_w}",
            w_q.len()
        );
        ensure!(
            scale.len() == expected_s,
            "{name}: scale bytes {} != expected {expected_s}",
            scale.len()
        );
        let prefix = name.strip_suffix(".weight").unwrap().to_string();
        let bias_name = format!("{prefix}.bias");
        let bias = tensors
            .iter()
            .find(|(n, _)| n.as_str() == bias_name)
            .map(|(_, v)| {
                v.data()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
            .unwrap_or_else(|| vec![0.0f32; out_dim]);
        let gs_name = format!("{prefix}.global_scale");
        let global_scale = tensors
            .iter()
            .find(|(n, _)| n.as_str() == gs_name)
            .and_then(|(_, v)| {
                if v.data().len() >= 4 {
                    Some(f32::from_le_bytes([
                        v.data()[0],
                        v.data()[1],
                        v.data()[2],
                        v.data()[3],
                    ]))
                } else {
                    None
                }
            })
            .unwrap_or(1.0f32);

        linears.insert(
            prefix,
            Nvfp4LinearPacked {
                w_q,
                scale,
                in_dim,
                out_dim,
                bias,
                global_scale,
            },
        );
    }

    Ok(Flux2PackedParams {
        linears,
        gguf_linears: HashMap::new(),
    })
}

/// Build synthetic NVFP4 weights for tests (identity-ish small matmul).
pub fn synthetic_nvfp4_linear(
    in_dim: usize,
    out_dim: usize,
    tag: &str,
) -> (String, Nvfp4LinearPacked) {
    let w_q = vec![0x11u8; nvfp4_weight_bytes(in_dim, out_dim)];
    let scale = vec![0x38u8; nvfp4_scale_bytes(in_dim, out_dim)]; // ~1.0 fp8
    (
        tag.to_string(),
        Nvfp4LinearPacked {
            w_q,
            scale,
            in_dim,
            out_dim,
            bias: vec![0.0f32; out_dim],
            global_scale: 1.0f32,
        },
    )
}

/// Tiny packed params covering `x_embedder` for quick-check tests.
pub fn synthetic_flux2_packed_tiny(cfg: &super::Flux2Config) -> Flux2PackedParams {
    let mut linears = HashMap::new();
    let (n, p) = synthetic_nvfp4_linear(cfg.in_channels, cfg.inner_dim(), "x_embedder");
    linears.insert(n, p);
    Flux2PackedParams {
        linears,
        gguf_linears: HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::nvfp4::NVFP4_GROUP_SIZE;

    #[test]
    fn nvfp4_byte_counts() {
        assert_eq!(nvfp4_weight_bytes(16, 8), 64);
        assert_eq!(nvfp4_scale_bytes(16, 8), 8);
        assert_eq!(NVFP4_GROUP_SIZE, 16);
    }
}
