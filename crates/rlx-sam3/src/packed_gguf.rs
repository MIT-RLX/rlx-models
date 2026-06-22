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

//! K-quant / legacy GGUF weights for SAM3 (native ViT matmul + extract fallbacks).

use anyhow::{Result, ensure};
use rlx_core::gguf_support::load_gguf_file;
use rlx_core::weight_loader::{GgufLoader, WeightLoader};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{GgufPackedLinear, GgufPackedParams};
use rlx_gguf::GgmlType;
use rlx_ir::quant::QuantScheme;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// True when the GGUF file stores quant matmul weights (K-quant or Q4_0/Q8_0).
pub fn gguf_has_packed_linears(path: &Path) -> Result<bool> {
    let raw = load_gguf_file(path)?;
    Ok(raw.tensors.values().any(|t| {
        matches!(
            t.dtype,
            GgmlType::Q4K
                | GgmlType::Q5K
                | GgmlType::Q6K
                | GgmlType::Q8K
                | GgmlType::Q2K
                | GgmlType::Q3K
                | GgmlType::Q4_0
                | GgmlType::Q8_0
        )
    }))
}

fn dequant_gguf_bytes(bytes: &[u8], n: usize, scheme: QuantScheme) -> Result<Vec<f32>> {
    let raw = match scheme {
        QuantScheme::GgufQ4_0 => rlx_gguf::dequant_q4_0(bytes, n)?,
        QuantScheme::GgufQ8_0 => rlx_gguf::dequant_q8_0(bytes, n)?,
        QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k(bytes, n)?,
        QuantScheme::GgufQ5K => rlx_gguf::dequant_q5_k(bytes, n)?,
        QuantScheme::GgufQ6K => rlx_gguf::dequant_q6_k(bytes, n)?,
        QuantScheme::GgufQ8K => rlx_gguf::dequant_q8_k(bytes, n)?,
        QuantScheme::GgufQ2K => rlx_gguf::dequant_q2_k(bytes, n)?,
        QuantScheme::GgufQ3K => rlx_gguf::dequant_q3_k(bytes, n)?,
        other => anyhow::bail!("dequant_gguf_bytes: unsupported scheme {other:?}"),
    };
    ensure!(raw.len() == n, "dequant length {} vs {n}", raw.len());
    Ok(raw)
}

/// Load SAM3 from GGUF: 2D quant linears stay packed; other quant tensors dequant to F32 in the map.
///
/// Honors an optional `<path>.tensor_map.json` sidecar produced by sam3.cpp's
/// `convert_mlx_sam3_to_gguf.py`. The script hashes every tensor name to a
/// 12-char SHA-1 prefix (`t_xxxxxxxxxxxx`) and stores the GGUF tensor under
/// that opaque key, while the sidecar maps it back to the friendly name SAM3
/// expects (`backbone.vision_backbone.trunk…`). When the sidecar is present
/// we translate on the fly so the rest of the loader sees friendly names.
pub fn load_sam3_from_gguf(path: &Path) -> Result<(WeightMap, GgufPackedParams)> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-utf8 path {:?}", path))?;
    let mut loader = GgufLoader::from_file(path_str)?;
    let opaque_keys = loader.remaining_keys();

    let name_map = load_sidecar_tensor_map(path)?;
    // sam3.cpp's `convert_mlx_sam3_to_gguf.py` writes tensors in PyTorch
    // convention (`weight.shape == [out, in]`); the legacy rlx-sam3 GGUFs
    // wrote them in `[in, out]` order. The sidecar JSON is the converter's
    // own marker, so its presence picks the layout.
    let weight_is_out_in = name_map.is_some();
    let resolve = |k: &str| -> String {
        name_map
            .as_ref()
            .and_then(|m| m.get(k))
            .cloned()
            .unwrap_or_else(|| k.to_string())
    };
    let resolved: Vec<(String, String)> = opaque_keys
        .iter()
        .map(|k| (k.clone(), resolve(k)))
        .collect();

    let mut linears = HashMap::new();
    let mut f32_tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    for (opaque_key, friendly_key) in &resolved {
        if friendly_key.strip_suffix(".weight").is_none() {
            let (data, shape) = loader.take(opaque_key)?;
            f32_tensors.insert(friendly_key.clone(), (data, shape));
            continue;
        }

        let Some((bytes, scheme, shape)) = loader.take_packed(opaque_key)? else {
            let (data, shape) = loader.take(opaque_key)?;
            f32_tensors.insert(friendly_key.clone(), (data, shape));
            continue;
        };

        let packed_dims = if shape.len() == 2 {
            if weight_is_out_in {
                Some((shape[1], shape[0])) // [out, in] → (in, out)
            } else {
                Some((shape[0], shape[1])) // [in, out]
            }
        } else if shape.len() == 4 && shape[2] == 1 && shape[3] == 1 {
            // 1×1 conv [out_c, in_c, 1, 1] → packed 2D matmul.
            Some((shape[1], shape[0]))
        } else if shape.len() == 4 && shape[2] == 3 && shape[3] == 3 {
            // 3×3 conv [out_c, in_c, 3, 3] — stay packed (dequant per forward).
            Some((shape[1] * 9, shape[0]))
        } else {
            None
        };

        if let Some((in_dim, out_dim)) = packed_dims {
            // The bias is taken on its own iteration (so the WeightMap also
            // sees it for the few code paths that look it up directly); here
            // we only embed an empty vector and let the lookup path resolve
            // bias from the WeightMap via `take_linear_b`.
            let bias = vec![0.0f32; out_dim];
            linears.insert(
                friendly_key.clone(),
                GgufPackedLinear {
                    w_q: bytes,
                    scheme,
                    in_dim,
                    out_dim,
                    bias,
                },
            );
        } else {
            let n: usize = shape.iter().product();
            let data = dequant_gguf_bytes(&bytes, n, scheme)?;
            f32_tensors.insert(friendly_key.clone(), (data, shape));
        }
    }

    Ok((
        WeightMap::from_tensors(f32_tensors),
        GgufPackedParams { linears },
    ))
}

/// Read the sidecar `<gguf>.tensor_map.json` if present. Returns `Ok(None)`
/// when the sidecar is absent (the GGUF likely uses friendly names directly),
/// and `Err` only on a malformed file.
fn load_sidecar_tensor_map(path: &Path) -> Result<Option<HashMap<String, String>>> {
    let mut s = path.as_os_str().to_owned();
    s.push(".tensor_map.json");
    let sidecar = PathBuf::from(s);
    if !sidecar.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&sidecar)?;
    let map: HashMap<String, String> = serde_json::from_str(&text)?;
    Ok(Some(map))
}

/// Dequant a packed 2D GGUF linear to F32 row-major (`in_dim` × `out_dim`).
pub fn gguf_packed_to_f32(p: &GgufPackedLinear) -> Result<Vec<f32>> {
    dequant_gguf_bytes(&p.w_q, p.in_dim * p.out_dim, p.scheme)
}

/// Dequant a packed GGUF linear to F32, transposed like [`WeightMap::take_transposed`].
pub fn gguf_packed_to_transposed(p: &GgufPackedLinear) -> Result<Vec<f32>> {
    let raw = gguf_packed_to_f32(p)?;
    let mut out = vec![0f32; raw.len()];
    for r in 0..p.in_dim {
        for c in 0..p.out_dim {
            out[c * p.in_dim + r] = raw[r * p.out_dim + c];
        }
    }
    Ok(out)
}

/// Lookup packed linears for HF `*_weight` keys or `.weight` GGUF stems.
pub fn packed_linear<'a>(m: &'a GgufPackedParams, key: &str) -> Option<&'a GgufPackedLinear> {
    m.get_linear(key).or_else(|| {
        key.strip_suffix("_weight")
            .map(|stem| format!("{stem}.weight"))
            .and_then(|alt| m.get_linear(&alt))
    })
}

/// [`WeightMap::take`] with GGUF packed 2D weight fallback.
pub fn take_or_gguf(
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
    key: &str,
) -> Result<(Vec<f32>, Vec<usize>)> {
    if weights.has(key) {
        return weights.take(key);
    }
    if let Some(p) = gguf_packed.and_then(|m| packed_linear(m, key)) {
        let data = gguf_packed_to_f32(p)?;
        return Ok((data, vec![p.in_dim, p.out_dim]));
    }
    anyhow::bail!("missing weight: {key}")
}

/// Like [`take_transposed_or_gguf`], but also returns the GGUF key when weights stay packed.
pub fn take_transposed_with_gguf_key(
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
    key: &str,
) -> Result<(Vec<f32>, Option<String>)> {
    if weights.has(key) {
        return Ok((weights.take_transposed(key)?.0, None));
    }
    if gguf_packed.and_then(|m| packed_linear(m, key)).is_some() {
        return Ok((Vec::new(), Some(key.to_string())));
    }
    anyhow::bail!("missing weight: {key}")
}

/// [`WeightMap::take_transposed`] with GGUF packed 2D weight fallback.
pub fn take_transposed_or_gguf(
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
    key: &str,
) -> Result<Vec<f32>> {
    if weights.has(key) {
        return Ok(weights.take_transposed(key)?.0);
    }
    if let Some(p) = gguf_packed.and_then(|m| packed_linear(m, key)) {
        return gguf_packed_to_transposed(p);
    }
    anyhow::bail!("missing weight: {key}")
}

/// Dequant a packed 3×3 conv to F32 NCHW `[out_c, in_c, 3, 3]`.
pub fn gguf_packed_conv3_to_f32(
    p: &GgufPackedLinear,
    out_c: usize,
    in_c: usize,
) -> Result<Vec<f32>> {
    ensure!(
        p.in_dim == in_c * 9 && p.out_dim == out_c,
        "packed conv3 shape {}x{} vs {in_c}x{out_c}×3×3",
        p.in_dim,
        p.out_dim
    );
    dequant_gguf_bytes(&p.w_q, out_c * in_c * 9, p.scheme)
}

/// 3×3 conv weight (`[out_c, in_c, 3, 3]`) with optional GGUF packed fallback.
pub fn take_conv3x3_with_gguf_key(
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
    key: &str,
) -> Result<(Vec<f32>, Vec<usize>, Option<String>)> {
    if weights.has(key) {
        let (data, shape) = weights.take(key)?;
        return Ok((data, shape, None));
    }
    if let Some(p) = gguf_packed.and_then(|m| packed_linear(m, key)) {
        return Ok((
            Vec::new(),
            vec![p.out_dim, p.in_dim / 9, 3, 3],
            Some(key.to_string()),
        ));
    }
    anyhow::bail!("missing weight: {key}")
}

/// 1×1 conv weight (`[out_c, in_c, 1, 1]`) with optional GGUF packed 2D fallback.
pub fn take_conv1x1_with_gguf_key(
    weights: &mut WeightMap,
    gguf_packed: Option<&GgufPackedParams>,
    key: &str,
) -> Result<(Vec<f32>, Vec<usize>, Option<String>)> {
    if weights.has(key) {
        let (data, shape) = weights.take(key)?;
        return Ok((data, shape, None));
    }
    if let Some(p) = gguf_packed.and_then(|m| packed_linear(m, key)) {
        return Ok((
            Vec::new(),
            vec![p.out_dim, p.in_dim, 1, 1],
            Some(key.to_string()),
        ));
    }
    anyhow::bail!("missing weight: {key}")
}

/// F32 [`rlx_core::host_kernels::linear`] or fused [`rlx_cpu::gguf_matmul::gguf_matmul_bt`].
pub fn linear_maybe_gguf(
    x: &[f32],
    m: usize,
    k: usize,
    w_t: &[f32],
    gguf_key: Option<&str>,
    gguf_packed: Option<&GgufPackedParams>,
    n: usize,
    b: &[f32],
) -> Result<Vec<f32>> {
    use rlx_core::host_kernels::linear;

    let gguf = gguf_key.and_then(|key| gguf_packed.and_then(|p| packed_linear(p, key)));
    let mut out = vec![0f32; m * n];
    if let Some(p) = gguf {
        ensure!(
            p.in_dim == k && p.out_dim == n,
            "packed linear shape {k}x{n} vs gguf {}x{}",
            p.in_dim,
            p.out_dim
        );
        rlx_cpu::gguf_matmul::gguf_matmul_bt(x, &p.w_q, &mut out, m, k, n, p.scheme);
    } else {
        ensure!(
            !w_t.is_empty(),
            "linear: missing F32 weights and no GGUF packed entry"
        );
        return linear(x, m, k, w_t, n, b);
    }
    for row in 0..m {
        for col in 0..n {
            out[row * n + col] += b[col];
        }
    }
    Ok(out)
}

/// 3×3 NCHW conv, stride 1, pad 1 (same as PyTorch `padding=1`).
pub(crate) fn conv2d_3x3_nchw_pad1(
    input: &[f32],
    c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut out = vec![0f32; c * h * w];
    for oc in 0..c {
        let b = bias[oc];
        let oup = &mut out[oc * h * w..(oc + 1) * h * w];
        for v in oup.iter_mut() {
            *v = b;
        }
    }
    for oc in 0..c {
        for ic in 0..c {
            let w_oi = &weight[((oc * c + ic) * 9)..((oc * c + ic) * 9 + 9)];
            let inp = &input[ic * h * w..(ic + 1) * h * w];
            let oup = &mut out[oc * h * w..(oc + 1) * h * w];
            for oy in 0..h {
                for ox in 0..w {
                    let mut acc = 0.0f32;
                    for ky in 0..3 {
                        let iy = oy as isize + ky as isize - 1;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..3 {
                            let ix = ox as isize + kx as isize - 1;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            acc += inp[iy as usize * w + ix as usize] * w_oi[ky * 3 + kx];
                        }
                    }
                    oup[oy * w + ox] += acc;
                }
            }
        }
    }
    out
}

/// Packed 3×3 conv: one-time dequant to NCHW, then native pad-1 conv.
pub fn conv2d_3x3_nchw_gguf(
    input: &[f32],
    c: usize,
    h: usize,
    w: usize,
    p: &GgufPackedLinear,
    bias: &[f32],
    nchw_cache: &mut Option<Vec<f32>>,
) -> Result<Vec<f32>> {
    if nchw_cache.is_none() {
        *nchw_cache = Some(gguf_packed_conv3_to_f32(p, c, c)?);
    }
    let weight_nchw = nchw_cache.as_ref().expect("conv3 nchw cache");
    Ok(conv2d_3x3_nchw_pad1(input, c, h, w, weight_nchw, bias))
}

/// Materialize packed 1×1 conv to NCHW F32 `[out_c, in_c, 1, 1]` (same layout as safetensors).
pub fn gguf_packed_conv1_to_nchw(
    gguf_packed: &GgufPackedParams,
    key: &str,
    out_c: usize,
    in_c: usize,
) -> Result<Vec<f32>> {
    let p = packed_linear(gguf_packed, key)
        .ok_or_else(|| anyhow::anyhow!("missing packed conv1: {key}"))?;
    ensure!(
        p.in_dim == in_c && p.out_dim == out_c,
        "packed conv1 {key}: {}x{} vs {in_c}x{out_c}",
        p.in_dim,
        p.out_dim
    );
    gguf_packed_to_transposed(p)
}
