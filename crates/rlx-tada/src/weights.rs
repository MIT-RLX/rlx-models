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

//! Streaming safetensors access for the TADA checkpoints.
//!
//! `HumeAI/tada-1b` is a single 3.9 GB bf16 file holding three models at once
//! (the Llama backbone, the diffusion head, and a copy of the codec decoder
//! that nothing uses). Materializing it as f32 in one go costs ~8.6 GB, most of
//! it never touched by whichever stage is running — so this reads the header
//! once and then pulls one tensor at a time, on demand — each stage asks only
//! for the keys it needs.
//!
//! Reads go through `pread`, not a mapping. A mapping is tidier to write, but
//! reading a 3.9 GB checkpoint through one makes every page resident and those
//! pages are charged to the process at exactly the moment the graph's arena is
//! being allocated — measured at ~2.6 GB of the peak, and `madvise(DONTNEED)`
//! did not reclaim it on macOS. `pread` into a caller-sized buffer never adds
//! to the footprint.
//!
//! It also drops the `_precomputed_mask` buffers on sight: each is an
//! 8192×8192 all-true bool (67 MB, six per attention stack) that upstream only
//! consults when no explicit mask is supplied, which never happens on the
//! inference path.

use anyhow::{Context, Result, bail};
use half::{bf16, f16};
use ndarray::{Array1, Array3, ArrayView3};
use rlx_dac::layers::{
    Decoder, DecoderBlock, Encoder, EncoderBlock, ResidualUnit, WnConv1d, WnConvTranspose1d,
};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Element types this reader understands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dt {
    F32,
    Bf16,
    F16,
    I64,
}

impl Dt {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "F32" => Dt::F32,
            "BF16" => Dt::Bf16,
            "F16" => Dt::F16,
            "I64" => Dt::I64,
            _ => return None,
        })
    }

    fn width(self) -> usize {
        match self {
            Dt::I64 => 8,
            Dt::F32 => 4,
            Dt::Bf16 | Dt::F16 => 2,
        }
    }
}

/// A safetensors file read on demand, with dtype conversion to f32.
pub struct TensorStore {
    file: File,
    /// `name → (absolute byte range, dtype, shape)`.
    index: HashMap<String, TensorMeta>,
}

#[derive(Clone, Debug)]
struct TensorMeta {
    start: u64,
    end: u64,
    dtype: Dt,
    shape: Vec<usize>,
}

/// Read `len` bytes at `offset` without moving the file cursor (thread-safe).
fn pread(file: &File, offset: u64, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    read_exact_at(file, &mut buf, offset)?;
    Ok(buf)
}

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
            .with_context(|| format!("read {} bytes at {offset}", buf.len()))
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = file.try_clone()?;
        f.seek(SeekFrom::Start(offset))?;
        f.read_exact(buf)
            .with_context(|| format!("read {} bytes at {offset}", buf.len()))
    }
}

impl TensorStore {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let n = u64::from_le_bytes(
            pread(&file, 0, 8)?
                .try_into()
                .map_err(|_| anyhow::anyhow!("{}: truncated header", path.display()))?,
        );
        let header: serde_json::Value = serde_json::from_slice(&pread(&file, 8, n as usize)?)
            .with_context(|| format!("parse safetensors header of {}", path.display()))?;
        let data_start = 8 + n;

        let obj = header
            .as_object()
            .with_context(|| format!("{}: header is not an object", path.display()))?;
        let mut index = HashMap::new();
        for (name, v) in obj {
            if name == "__metadata__" || name.ends_with("_precomputed_mask") {
                continue;
            }
            let Some(dtype) = v["dtype"].as_str().and_then(Dt::parse) else {
                // BOOL and the quantized types are simply not part of any TADA
                // inference path; skipping keeps the reader honest about what
                // it can actually serve.
                continue;
            };
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .with_context(|| format!("{name}: missing shape"))?
                .iter()
                .map(|d| d.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = v["data_offsets"]
                .as_array()
                .with_context(|| format!("{name}: missing data_offsets"))?;
            let (a, b) = (off[0].as_u64().unwrap_or(0), off[1].as_u64().unwrap_or(0));
            let want = shape.iter().product::<usize>() * dtype.width();
            if (b - a) as usize != want {
                bail!("{name}: {} bytes on disk, shape implies {want}", b - a);
            }
            index.insert(
                name.clone(),
                TensorMeta {
                    start: data_start + a,
                    end: data_start + b,
                    dtype,
                    shape,
                },
            );
        }
        if index.is_empty() {
            bail!("{}: no readable tensors", path.display());
        }
        Ok(Self { file, index })
    }

    pub fn has(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    pub fn shape(&self, key: &str) -> Result<&[usize]> {
        self.index
            .get(key)
            .map(|m| m.shape.as_slice())
            .with_context(|| format!("missing tensor {key}"))
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    /// Decode one tensor to f32.
    pub fn get(&self, key: &str) -> Result<Vec<f32>> {
        let meta = self
            .index
            .get(key)
            .with_context(|| format!("missing tensor {key}"))?
            .clone();
        let raw = pread(&self.file, meta.start, (meta.end - meta.start) as usize)?;
        let n = raw.len() / meta.dtype.width();
        let mut out = vec![0f32; n];
        decode_into(&raw, meta.dtype, &mut out, key)?;
        Ok(out)
    }

    pub fn get_opt(&self, key: &str) -> Result<Option<Vec<f32>>> {
        if self.has(key) {
            self.get(key).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Decode one row of a 2-D tensor into `out`, touching nothing else.
    ///
    /// The token embedding table is 128 256 × hidden — 1.05 GB as f32 — and an
    /// utterance reads a few dozen rows of it. Reading rows straight out of the
    /// mapping keeps that table off the heap entirely.
    pub fn row(&self, key: &str, index: usize, out: &mut [f32]) -> Result<()> {
        let meta = self
            .index
            .get(key)
            .with_context(|| format!("missing tensor {key}"))?
            .clone();
        if meta.shape.len() != 2 {
            bail!("{key}: row() needs a 2-D tensor, got {:?}", meta.shape);
        }
        let (rows, cols) = (meta.shape[0], meta.shape[1]);
        if index >= rows {
            bail!("{key}: row {index} is out of range ({rows} rows)");
        }
        if out.len() != cols {
            bail!("{key}: row buffer is {} wide, tensor is {cols}", out.len());
        }
        let w = meta.dtype.width();
        let start = meta.start + (index * cols * w) as u64;
        let raw = pread(&self.file, start, cols * w)?;
        decode_into(&raw, meta.dtype, out, key)
    }

    /// Decode a 2-D `[out, in]` linear weight into row-major `[out, in]`.
    pub fn linear(&self, key: &str) -> Result<Linear> {
        let shape = self.shape(key)?.to_vec();
        if shape.len() != 2 {
            bail!("{key}: expected a 2-D weight, got {shape:?}");
        }
        Ok(Linear {
            weight: self.get(key)?,
            out_dim: shape[0],
            in_dim: shape[1],
            bias: self.get_opt(&key.replace(".weight", ".bias"))?,
        })
    }
}

fn decode_into(raw: &[u8], dtype: Dt, out: &mut [f32], key: &str) -> Result<()> {
    let _ = key;
    match dtype {
        Dt::F32 => {
            for (o, c) in out.iter_mut().zip(raw.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        Dt::Bf16 => {
            for (o, c) in out.iter_mut().zip(raw.chunks_exact(2)) {
                *o = bf16::from_le_bytes([c[0], c[1]]).to_f32();
            }
        }
        Dt::F16 => {
            for (o, c) in out.iter_mut().zip(raw.chunks_exact(2)) {
                *o = f16::from_le_bytes([c[0], c[1]]).to_f32();
            }
        }
        Dt::I64 => {
            for (o, c) in out.iter_mut().zip(raw.chunks_exact(8)) {
                *o = i64::from_le_bytes(c.try_into().unwrap()) as f32;
            }
        }
    }
    Ok(())
}

/// A dense `y = x·Wᵀ + b` layer held row-major as `[out_dim, in_dim]`.
#[derive(Clone, Debug)]
pub struct Linear {
    pub weight: Vec<f32>,
    pub out_dim: usize,
    pub in_dim: usize,
    pub bias: Option<Vec<f32>>,
}

impl Linear {
    /// `[out, in]` → `[in, out]`, the orientation rlx matmul params want.
    pub fn transposed(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.weight.len()];
        for o in 0..self.out_dim {
            for i in 0..self.in_dim {
                out[i * self.out_dim + o] = self.weight[o * self.in_dim + i];
            }
        }
        out
    }
}

/// Fuse PyTorch weight-norm parameters into a dense kernel.
///
/// `torch.nn.utils.parametrizations.weight_norm(conv)` (default `dim=0`)
/// stores `original0 = g` shaped `[dim0, 1, 1]` and `original1 = v` shaped like
/// the kernel; the effective weight is `g · v / ‖v‖` with the norm taken over
/// every axis but the first. Applies unchanged to `ConvTranspose1d`, whose
/// axis 0 is the *input* channel.
fn fuse_weight_norm(g: &[f32], v: ArrayView3<f32>) -> Array3<f32> {
    let (d0, d1, d2) = v.dim();
    let mut out = Array3::<f32>::zeros((d0, d1, d2));
    for c in 0..d0 {
        let mut sq = 0f64;
        for a in 0..d1 {
            for b in 0..d2 {
                sq += (v[[c, a, b]] as f64).powi(2);
            }
        }
        let scale = (g[c] as f64) / sq.sqrt().max(1e-12);
        for a in 0..d1 {
            for b in 0..d2 {
                out[[c, a, b]] = (v[[c, a, b]] as f64 * scale) as f32;
            }
        }
    }
    out
}

/// Weight-norm fused along **dim 2** (the kernel axis), as
/// `nn.utils.weight_norm(conv, dim=2)` parameterizes wav2vec2's positional
/// convolution: `w[:, :, j] = g[0, 0, j] · v[:, :, j] / ‖v[:, :, j]‖`.
///
/// Distinct from `fuse_weight_norm`'s dim-0 form, and the two are not
/// interchangeable — a dim-0 fusion here normalizes across the wrong axis and
/// yields a positional embedding that is wrong everywhere but silently finite.
pub fn fuse_weight_norm_dim2(g: &[f32], v: ArrayView3<f32>) -> Vec<f32> {
    let (d0, d1, d2) = v.dim();
    let mut out = vec![0f32; d0 * d1 * d2];
    for j in 0..d2 {
        let mut sq = 0f64;
        for a in 0..d0 {
            for b in 0..d1 {
                sq += (v[[a, b, j]] as f64).powi(2);
            }
        }
        let scale = (g[j] as f64) / sq.sqrt().max(1e-12);
        for a in 0..d0 {
            for b in 0..d1 {
                out[(a * d1 + b) * d2 + j] = (v[[a, b, j]] as f64 * scale) as f32;
            }
        }
    }
    out
}

fn load_wn(store: &TensorStore, prefix: &str) -> Result<(Array3<f32>, Option<Array1<f32>>)> {
    let gk = format!("{prefix}.parametrizations.weight.original0");
    let vk = format!("{prefix}.parametrizations.weight.original1");
    let g = store
        .get(&gk)
        .with_context(|| format!("{prefix}: weight-norm g"))?;
    let v = store
        .get(&vk)
        .with_context(|| format!("{prefix}: weight-norm v"))?;
    let vs = store.shape(&vk)?.to_vec();
    if vs.len() != 3 {
        bail!("{prefix}: expected a 3-D conv kernel, got {vs:?}");
    }
    if g.len() != vs[0] {
        bail!(
            "{prefix}: weight-norm g has {} entries, kernel dim0 is {}",
            g.len(),
            vs[0]
        );
    }
    let view = ArrayView3::from_shape((vs[0], vs[1], vs[2]), &v)?;
    let bias = store
        .get_opt(&format!("{prefix}.bias"))?
        .map(Array1::from_vec);
    Ok((fuse_weight_norm(&g, view), bias))
}

fn load_conv(
    store: &TensorStore,
    prefix: &str,
    stride: usize,
    pad: usize,
    dilation: usize,
    same_length: bool,
) -> Result<WnConv1d> {
    let (weight, bias) = load_wn(store, prefix)?;
    Ok(WnConv1d {
        weight,
        bias,
        stride,
        pad,
        dilation,
        same_length,
    })
}

fn load_conv_transpose(
    store: &TensorStore,
    prefix: &str,
    stride: usize,
    pad: usize,
) -> Result<WnConvTranspose1d> {
    let (weight, bias) = load_wn(store, prefix)?;
    Ok(WnConvTranspose1d {
        weight,
        bias,
        stride,
        pad,
        out_offset: 0,
    })
}

fn load_alpha(store: &TensorStore, key: &str) -> Result<Array1<f32>> {
    let shape = store.shape(key)?.to_vec();
    if shape.len() != 3 || shape[0] != 1 || shape[2] != 1 {
        bail!("{key}: expected a [1, C, 1] Snake alpha, got {shape:?}");
    }
    Ok(Array1::from_vec(store.get(key)?))
}

/// `ResidualUnit`: `Snake → conv(k=7, dilated) → Snake → conv(k=1)`.
fn load_residual(store: &TensorStore, prefix: &str, dilation: usize) -> Result<ResidualUnit> {
    let pad = ((7 - 1) * dilation) / 2;
    Ok(ResidualUnit::from_parts(
        load_alpha(store, &format!("{prefix}.0.alpha"))?,
        load_conv(store, &format!("{prefix}.1"), 1, pad, dilation, true)?,
        load_alpha(store, &format!("{prefix}.2.alpha"))?,
        load_conv(store, &format!("{prefix}.3"), 1, 0, 1, true)?,
    ))
}

/// Load TADA's `WavEncoder` as an [`Encoder`] the `rlx-dac` graph builder
/// understands — the module is DAC's encoder verbatim, only the key prefix and
/// the weight-norm parameter naming differ.
pub fn load_wav_encoder(store: &TensorStore, prefix: &str, strides: &[usize]) -> Result<Encoder> {
    let stem = load_conv(store, &format!("{prefix}.block.0"), 1, 3, 1, true)?;
    let mut blocks = Vec::with_capacity(strides.len());
    for (i, &stride) in strides.iter().enumerate() {
        let p = format!("{prefix}.block.{}", i + 1);
        blocks.push(EncoderBlock::from_parts(
            [
                load_residual(store, &format!("{p}.block.0.block"), 1)?,
                load_residual(store, &format!("{p}.block.1.block"), 3)?,
                load_residual(store, &format!("{p}.block.2.block"), 9)?,
            ],
            load_alpha(store, &format!("{p}.block.3.alpha"))?,
            load_conv(
                store,
                &format!("{p}.block.4"),
                stride,
                stride.div_ceil(2),
                1,
                false,
            )?,
        ));
    }
    let tail = strides.len() + 1;
    Ok(Encoder::from_parts(
        stem,
        blocks,
        load_alpha(store, &format!("{prefix}.block.{tail}.alpha"))?,
        // WavEncoder's head is WNConv1d(d_model, d_latent, k=3, padding=1).
        load_conv(
            store,
            &format!("{prefix}.block.{}", tail + 1),
            1,
            1,
            1,
            true,
        )?,
    ))
}

/// Load TADA's `DACDecoder` as a [`Decoder`].
pub fn load_wav_decoder(store: &TensorStore, prefix: &str, strides: &[usize]) -> Result<Decoder> {
    let stem = load_conv(store, &format!("{prefix}.model.0"), 1, 3, 1, true)?;
    let mut blocks = Vec::with_capacity(strides.len());
    for (i, &stride) in strides.iter().enumerate() {
        let p = format!("{prefix}.model.{}", i + 1);
        blocks.push(DecoderBlock::from_parts(
            load_alpha(store, &format!("{p}.block.0.alpha"))?,
            load_conv_transpose(store, &format!("{p}.block.1"), stride, stride.div_ceil(2))?,
            [
                load_residual(store, &format!("{p}.block.2.block"), 1)?,
                load_residual(store, &format!("{p}.block.3.block"), 3)?,
                load_residual(store, &format!("{p}.block.4.block"), 9)?,
            ],
        ));
    }
    let tail = strides.len() + 1;
    Ok(Decoder::from_parts(
        stem,
        blocks,
        load_alpha(store, &format!("{prefix}.model.{tail}.alpha"))?,
        load_conv(
            store,
            &format!("{prefix}.model.{}", tail + 1),
            1,
            3,
            1,
            true,
        )?,
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_norm_fusion_reproduces_the_scale() {
        // v with a known norm: ‖[3,4]‖ = 5, so g=10 must yield [6, 8].
        let v = Array3::from_shape_vec((1, 1, 2), vec![3.0, 4.0]).unwrap();
        let w = fuse_weight_norm(&[10.0], v.view());
        assert!((w[[0, 0, 0]] - 6.0).abs() < 1e-5);
        assert!((w[[0, 0, 1]] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn weight_norm_normalizes_each_output_channel_independently() {
        let v = Array3::from_shape_vec((2, 1, 2), vec![3.0, 4.0, 1.0, 0.0]).unwrap();
        let w = fuse_weight_norm(&[5.0, 2.0], v.view());
        assert!((w[[0, 0, 0]] - 3.0).abs() < 1e-5);
        assert!((w[[1, 0, 0]] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn zero_kernel_does_not_divide_by_zero() {
        let v = Array3::<f32>::zeros((1, 1, 3));
        let w = fuse_weight_norm(&[1.0], v.view());
        assert!(w.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn linear_transpose_round_trips() {
        let l = Linear {
            weight: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            out_dim: 2,
            in_dim: 3,
            bias: None,
        };
        // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
        assert_eq!(l.transposed(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
