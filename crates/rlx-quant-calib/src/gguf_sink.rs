// RLX models — calibration quantization.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! GGUF sink — turn calibrated weights into a **loadable** quantized
//! checkpoint.
//!
//! The calibration algorithms ([`crate::awq`], [`crate::gptq`],
//! [`crate::dynamic`]) compute better-quantized weights but produce only
//! in-memory tensors. This module serializes a set of (optionally quantized)
//! tensors to a GGUF v3 file via [`rlx_gguf::GgufWriter`], so the result loads
//! through the existing rlx GGUF path and runs on the first-class
//! `QMatMul` / `DequantMatMul` ops — backend-portable, no extra glue.
//!
//! Two encodings matter:
//!   - **Pass-through** (`F32` / `F16`) for tensors that shouldn't be quantized
//!     (norms, 1-D biases, embeddings) — written verbatim.
//!   - **GGML block-quant** (`Q4_0`, `Q8_0`, …) for 2-D linear weights. AWQ /
//!     GPTQ improve the f32 values that get block-quantized (AWQ in particular
//!     reshapes the weight to be *quantization-friendly*), so the calibration
//!     benefit carries through the GGML re-encoding.
//!
//! GGUF stores tensor shapes in **`ne` order** — the fastest-varying
//! (contiguous) dimension first. For a row-major `[out, inn]` linear weight,
//! pass shape `[inn, out]`; the [`linear`] helper does this for you.

use crate::quant::{GroupQuant, dequantize};
use anyhow::{Result, bail};
use rlx_gguf::{GgmlType, GgufWriter, MetaValue, quantize};
use std::path::Path;

/// How to encode one tensor in the output file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// Verbatim 32-bit floats.
    F32,
    /// Half precision (2 bytes/elem) — near-lossless pass-through.
    F16,
    /// GGML block quantization (`Q4_0`, `Q8_0`, …). The tensor's contiguous
    /// dimension must be a multiple of the format's block size.
    Quant(GgmlType),
}

/// Encode f32 `data` into GGUF tensor bytes per `encoding`, returning the
/// on-disk [`GgmlType`] and block bytes.
pub fn encode_bytes(data: &[f32], encoding: Encoding) -> Result<(GgmlType, Vec<u8>)> {
    match encoding {
        Encoding::F32 => {
            let mut b = Vec::with_capacity(data.len() * 4);
            for &x in data {
                b.extend_from_slice(&x.to_le_bytes());
            }
            Ok((GgmlType::F32, b))
        }
        Encoding::F16 => {
            let mut b = Vec::with_capacity(data.len() * 2);
            for &x in data {
                b.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
            }
            Ok((GgmlType::F16, b))
        }
        Encoding::Quant(ty) => Ok((ty, quantize(data, ty)?)),
    }
}

/// One tensor to write into the GGUF file.
pub struct SinkTensor<'a> {
    pub name: String,
    /// Shape in GGUF `ne` order (contiguous dim first). For a row-major
    /// `[out, inn]` weight this is `[inn, out]`; see [`linear`].
    pub shape: Vec<usize>,
    pub data: &'a [f32],
    pub encoding: Encoding,
}

impl<'a> SinkTensor<'a> {
    /// A row-major `[out, inn]` linear weight, written in GGUF `ne` order
    /// (`shape = [inn, out]`) with the given encoding.
    pub fn linear(
        name: impl Into<String>,
        data: &'a [f32],
        out: usize,
        inn: usize,
        encoding: Encoding,
    ) -> Self {
        Self {
            name: name.into(),
            shape: vec![inn, out],
            data,
            encoding,
        }
    }

    /// A 1-D tensor (norm weight, bias) written verbatim as f32.
    pub fn vector(name: impl Into<String>, data: &'a [f32]) -> Self {
        Self {
            name: name.into(),
            shape: vec![data.len()],
            data,
            encoding: Encoding::F32,
        }
    }
}

/// Write a GGUF v3 file from a set of tensors + metadata.
///
/// `arch` populates `general.architecture` (e.g. `"llama"`, `"qwen3"`) so the
/// loader can dispatch the right model builder.
pub fn write_gguf_file(
    path: &Path,
    arch: &str,
    meta: &[(String, MetaValue)],
    tensors: &[SinkTensor<'_>],
) -> Result<()> {
    let mut w = GgufWriter::new();
    w.set_arch(arch);
    for (k, v) in meta {
        w.set_meta(k.clone(), v.clone());
    }
    for t in tensors {
        let n: usize = t.shape.iter().product();
        if n != t.data.len() {
            bail!(
                "tensor {}: shape product {n} != data len {}",
                t.name,
                t.data.len()
            );
        }
        let (ty, bytes) = encode_bytes(t.data, t.encoding)?;
        w.add_tensor_bytes(t.name.clone(), t.shape.clone(), ty, bytes)?;
    }
    w.write_to_path(path)?;
    Ok(())
}

/// Serialize a calibrated [`GroupQuant`] linear weight to GGML `ty` block
/// bytes. The calibrated weight is dequantized to f32 and re-encoded in the
/// GGML format; AWQ/GPTQ leave the f32 better-conditioned for that re-encoding,
/// so the calibration gain carries into the deployed file.
pub fn group_quant_bytes(qd: &GroupQuant, ty: GgmlType) -> Result<Vec<u8>> {
    quantize(&dequantize(qd), ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::quantize_rtn;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
        for (x, y) in a.iter().zip(b) {
            d += x * y;
            na += x * x;
            nb += y * y;
        }
        d / (na.sqrt() * nb.sqrt() + 1e-12)
    }

    #[test]
    fn writes_loadable_gguf_with_quant_and_passthrough() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");

        // A quantized 2-D linear weight + a verbatim norm vector.
        let (out, inn) = (4usize, 64usize);
        let w: Vec<f32> = (0..out * inn).map(|i| (i as f32 * 0.013).sin()).collect();
        let norm: Vec<f32> = (0..16).map(|i| 1.0 + i as f32 * 0.1).collect();

        write_gguf_file(
            &path,
            "llama",
            &[(
                "general.name".into(),
                MetaValue::String("calib-test".into()),
            )],
            &[
                SinkTensor::linear(
                    "blk.0.attn_q.weight",
                    &w,
                    out,
                    inn,
                    Encoding::Quant(GgmlType::Q8_0),
                ),
                SinkTensor::vector("blk.0.attn_norm.weight", &norm),
            ],
        )
        .unwrap();

        // Read back through rlx_gguf's own loader → proves it's a valid file.
        let f = rlx_gguf::GgufFile::from_path(&path).unwrap();
        let (wq, wshape) = f.dequant_f32("blk.0.attn_q.weight").unwrap();
        assert_eq!(wq.len(), out * inn, "weight element count");
        assert_eq!(wshape.iter().product::<usize>(), out * inn);
        assert!(cosine(&wq, &w) > 0.999, "Q8_0 weight round-trip cosine");

        let (n, _) = f.dequant_f32("blk.0.attn_norm.weight").unwrap();
        for (a, b) in n.iter().zip(&norm) {
            assert!((a - b).abs() < 1e-6, "f32 norm verbatim");
        }
    }

    #[test]
    fn group_quant_serializes_to_gguf_blocks() {
        // Calibrated weight (group RTN) → GGUF Q8_0 bytes → loadable.
        let (out, inn) = (2usize, 64usize);
        let w: Vec<f32> = (0..out * inn).map(|i| (i as f32 * 0.02).cos()).collect();
        let qd = quantize_rtn(&w, out, inn, 8, 32);
        let bytes = group_quant_bytes(&qd, GgmlType::Q8_0).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.gguf");
        let mut writer = GgufWriter::new();
        writer.set_arch("llama");
        writer
            .add_tensor_bytes(
                "blk.0.ffn_down.weight",
                vec![inn, out],
                GgmlType::Q8_0,
                bytes,
            )
            .unwrap();
        writer.write_to_path(&path).unwrap();

        let f = rlx_gguf::GgufFile::from_path(&path).unwrap();
        let (back, _) = f.dequant_f32("blk.0.ffn_down.weight").unwrap();
        assert!(cosine(&back, &w) > 0.99, "calibrated → GGUF round-trip");
    }
}
