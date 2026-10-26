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

//! Lazy, per-tensor weight access for the FLUX.1 components.
//!
//! The denoiser is 12B parameters; materializing it as f32 would need ~48 GB
//! of host memory before a single byte reached the device. [`TensorFile`]
//! instead maps the checkpoint and widens one tensor at a time, and every
//! graph is built over a [`rlx_flow::WeightSource`] that reports each tensor's
//! on-disk dtype — so with [`rlx_flow::WeightStorage::Auto`] a bf16 matrix is
//! stored on device at 2 B/elem, exactly as shipped.
//!
//! [`DitSource`] is the denoiser's view: it de-interleaves the query/key rows
//! (see [`deinterleave_rows`]) so the graph can use the half-split RoPE
//! kernel, and serves each LoRA target's low-rank pair under
//! `<weight>@lora_a` / `<weight>@lora_b` for the graph's adapter branch.
//! Adapters stay unmerged: folding `B·A` into a bf16 matrix and storing the
//! sum at bf16 rounds away most of the update's precision (measured: 6.6e-4
//! relative error on the truncated denoiser vs 1.0e-6 unmerged).

use anyhow::{Context, Result, bail};
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_core::weight_map::transpose_2d;
use rlx_flow::WeightSource;
use rlx_ir::DType;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::config::Flux1Config;
use crate::lora::LoraSet;

/// A safetensors file (or directory of shards), mapped lazily, with its
/// headers parsed up front so shapes and dtypes are free.
pub struct TensorFile {
    ckpt: SafetensorsCheckpoint,
    /// Name → (dtype string, shape), from the headers.
    header: HashMap<String, (String, Vec<usize>)>,
    /// Prefix stripped from on-disk names (`model.diffusion_model.`), if every
    /// tensor carries one.
    prefix: String,
    path: PathBuf,
}

impl TensorFile {
    /// Open a `.safetensors` file or a directory holding one or more.
    pub fn open(path: &Path) -> Result<Self> {
        let (ckpt, files) = if path.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                .collect();
            files.sort();
            if files.is_empty() {
                bail!("no .safetensors files under {}", path.display());
            }
            (SafetensorsCheckpoint::open(path)?, files)
        } else {
            (
                SafetensorsCheckpoint::open_file(path)?,
                vec![path.to_path_buf()],
            )
        };
        let mut header = HashMap::new();
        for f in &files {
            header.extend(read_header(f)?);
        }
        Ok(Self {
            ckpt,
            header,
            prefix: String::new(),
            path: path.to_path_buf(),
        })
    }

    /// Strip `prefix` from every name when all of them carry it (Comfy-style
    /// `model.diffusion_model.` single files).
    pub fn with_common_prefix(mut self, prefix: &str) -> Self {
        if !self.header.is_empty() && self.header.keys().all(|k| k.starts_with(prefix)) {
            self.prefix = prefix.to_string();
        }
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn real(&self, key: &str) -> String {
        format!("{}{key}", self.prefix)
    }

    /// Every tensor name (prefix stripped).
    pub fn keys(&self) -> Vec<String> {
        let n = self.prefix.len();
        self.header.keys().map(|k| k[n..].to_string()).collect()
    }

    pub fn contains(&self, key: &str) -> bool {
        self.header.contains_key(&self.real(key))
    }

    pub fn shape(&self, key: &str) -> Option<Vec<usize>> {
        self.header.get(&self.real(key)).map(|(_, s)| s.clone())
    }

    /// The on-disk float dtype of `key`, if it is a float tensor.
    pub fn stored_dtype(&self, key: &str) -> Option<DType> {
        Some(match self.header.get(&self.real(key))?.0.as_str() {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            _ => return None,
        })
    }

    /// One tensor, widened to f32.
    pub fn load(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.ckpt
            .load_tensor_f32(&self.real(key))
            .with_context(|| format!("{}: `{key}`", self.path.display()))
    }

    /// One half-precision tensor's raw little-endian bytes (the file's own
    /// width), when it is stored as F16 / BF16.
    pub fn load_half_bytes(&self, key: &str) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        if !matches!(self.stored_dtype(key), Some(DType::F16 | DType::BF16)) {
            return Ok(None);
        }
        let (bytes, _, shape) = self
            .ckpt
            .tensor_raw(&self.real(key))
            .with_context(|| format!("{}: `{key}`", self.path.display()))?;
        Ok(Some((bytes, shape)))
    }

    /// Selected rows of a rank-2 tensor (an embedding lookup).
    pub fn load_rows(&self, key: &str, rows: &[u32]) -> Result<Vec<f32>> {
        let cols = self.shape(key).context("missing tensor")?[1];
        Ok(self
            .ckpt
            .load_tensor_rows_f32(&self.real(key), rows, cols)?
            .concat())
    }
}

/// Parse a safetensors header: `name → (dtype, shape)`.
fn read_header(path: &Path) -> Result<HashMap<String, (String, Vec<usize>)>> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len) as usize;
    if n > 512 << 20 {
        bail!(
            "{}: implausible safetensors header length {n}",
            path.display()
        );
    }
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf)?;
    let v: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&buf)?;
    let mut out = HashMap::new();
    for (k, t) in v {
        if k == "__metadata__" {
            continue;
        }
        let dtype = t["dtype"].as_str().unwrap_or("").to_string();
        let shape = t["shape"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|d| d.as_u64().map(|d| d as usize))
                    .collect()
            })
            .unwrap_or_default();
        out.insert(k, (dtype, shape));
    }
    Ok(out)
}

/// Reorder each head's rows from interleaved RoPE pairs `(x0, x1), (x2, x3), …`
/// to the half-split layout `x0, x2, … | x1, x3, …`.
///
/// FLUX.1 rotates adjacent pairs; the RoPE kernels rotate `x[i]` with
/// `x[i + d/2]`. Permuting the q and k projections' output rows the same way
/// turns one into the other, and leaves attention unchanged — `q·k` is a sum
/// over the head dimension, indifferent to its order — while the per-head
/// RMSNorm only needs its scale permuted along with it. `rows` covers whole
/// heads; `cols` is 1 for a bias or norm scale.
pub fn deinterleave_rows<T: Copy + Default>(data: &mut [T], head_dim: usize, cols: usize) {
    let half = head_dim / 2;
    let head_len = head_dim * cols;
    let mut tmp = vec![T::default(); head_len];
    for head in data.chunks_mut(head_len) {
        for j in 0..half {
            let (even, odd) = (2 * j * cols, (2 * j + 1) * cols);
            tmp[j * cols..(j + 1) * cols].copy_from_slice(&head[even..even + cols]);
            tmp[(half + j) * cols..(half + j + 1) * cols].copy_from_slice(&head[odd..odd + cols]);
        }
        head.copy_from_slice(&tmp);
    }
}

/// Suffixes of the synthetic LoRA keys [`DitSource`] serves.
pub const LORA_A: &str = "@lora_a";
pub const LORA_B: &str = "@lora_b";

/// Load-time quantization of the denoiser's projection weights
/// (`--weight-quant`): each bf16 matrix is encoded once into a GGUF block
/// format and multiplied by a fused `DequantMatMul`, so it stays resident at
/// that width instead of 2 B/elem. LoRA branches, norms and biases stay f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightQuant {
    /// 8.5 bits/weight.
    Q8_0,
    /// 6.56 bits/weight.
    Q6K,
    /// 5.5 bits/weight.
    Q5K,
    /// 4.5 bits/weight.
    Q4K,
}

impl WeightQuant {
    /// `q8_0`, `q6_k`, `q5_k`, `q4_k`; `none` / `bf16` = keep the checkpoint's width.
    pub fn parse(s: &str) -> Result<Option<Self>> {
        Ok(Some(
            match s.to_ascii_lowercase().replace('-', "_").as_str() {
                "none" | "bf16" | "off" => return Ok(None),
                "q8_0" | "q8" => Self::Q8_0,
                "q6_k" | "q6k" => Self::Q6K,
                "q5_k" | "q5k" => Self::Q5K,
                "q4_k" | "q4k" => Self::Q4K,
                o => bail!("--weight-quant {o}: expected q8_0, q6_k, q5_k, q4_k or none"),
            },
        ))
    }

    fn ggml(self) -> rlx_gguf::GgmlType {
        match self {
            Self::Q8_0 => rlx_gguf::GgmlType::Q8_0,
            Self::Q6K => rlx_gguf::GgmlType::Q6K,
            Self::Q5K => rlx_gguf::GgmlType::Q5K,
            Self::Q4K => rlx_gguf::GgmlType::Q4K,
        }
    }

    pub fn scheme(self) -> rlx_ir::quant::QuantScheme {
        use rlx_ir::quant::QuantScheme as Q;
        match self {
            Self::Q8_0 => Q::GgufQ8_0,
            Self::Q6K => Q::GgufQ6K,
            Self::Q5K => Q::GgufQ5K,
            Self::Q4K => Q::GgufQ4K,
        }
    }

    /// Elements per quant block (a row must be a whole number of them).
    pub fn block(self) -> usize {
        match self {
            Self::Q8_0 => 32,
            _ => 256,
        }
    }

    /// Encode `[rows, cols]` half-precision bytes (`dtype`), rows in parallel.
    pub fn encode_half(
        self,
        bytes: &[u8],
        dtype: DType,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<u8>> {
        use rayon::prelude::*;
        let encoded: Vec<Vec<u8>> = bytes
            .par_chunks(2 * cols)
            .map(|row| {
                let f: Vec<f32> = row
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        match dtype {
                            DType::F16 => half::f16::from_bits(bits).to_f32(),
                            _ => half::bf16::from_bits(bits).to_f32(),
                        }
                    })
                    .collect();
                rlx_gguf::quantize(&f, self.ggml())
            })
            .collect::<Result<_>>()?;
        debug_assert_eq!(encoded.len(), rows);
        Ok(encoded.concat())
    }

    /// Encode `[rows, cols]` f32, rows in parallel.
    pub fn encode_f32(self, data: &[f32], rows: usize, cols: usize) -> Result<Vec<u8>> {
        use rayon::prelude::*;
        let encoded: Vec<Vec<u8>> = data
            .par_chunks(cols)
            .map(|row| rlx_gguf::quantize(row, self.ggml()))
            .collect::<Result<_>>()?;
        debug_assert_eq!(encoded.len(), rows);
        Ok(encoded.concat())
    }
}

/// What the denoiser's builder needs beyond [`WeightSource`]: the weights
/// carrying a LoRA graph branch, and host-side weights with their patches
/// merged (the conditioning MLPs, [`crate::embed::VecEmbedder`]). Implemented by
/// [`DitSource`] (a checkpoint) and by the web export's record / replay
/// sources (`crate::web`), so all three build the same graph.
pub trait DitWeights: WeightSource {
    fn lora_targets(&self) -> std::collections::HashSet<String>;
    fn load_merged(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)>;
}

impl DitWeights for DitSource<'_> {
    fn lora_targets(&self) -> std::collections::HashSet<String> {
        self.lora.targets().map(str::to_string).collect()
    }

    fn load_merged(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        DitSource::load_merged(self, key)
    }
}

/// The denoiser's weights: [`TensorFile`] + q/k row permutation + LoRA pairs.
pub struct DitSource<'a> {
    pub file: &'a TensorFile,
    pub lora: &'a LoraSet,
    pub cfg: &'a Flux1Config,
    /// Serve projection weights packed (see [`WeightQuant`]).
    pub quant: Option<WeightQuant>,
}

impl DitSource<'_> {
    /// `key` with its LoRA patches merged in f32 — for the few weights the
    /// host consumes directly (the conditioning MLPs).
    pub fn load_merged(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (mut data, shape) = self.file.load(key)?;
        self.lora.apply(key, &mut data, &shape)?;
        Ok((data, shape))
    }

    /// Whether `weight_key` has a LoRA branch.
    pub fn has_lora(&self, weight_key: &str) -> bool {
        self.lora.targets().any(|t| t == weight_key)
    }

    /// The `(A [R, in], B [rows, R])` pair for `weight_key`, rows permuted like the weight's.
    fn lora_pair(&self, weight_key: &str) -> Result<(Vec<f32>, Vec<usize>, Vec<f32>, Vec<usize>)> {
        let shape = self
            .file
            .shape(weight_key)
            .with_context(|| format!("LoRA target `{weight_key}` missing"))?;
        let (a, mut b, r) = self
            .lora
            .stacked(weight_key, shape[0])
            .with_context(|| format!("no LoRA on `{weight_key}`"))?;
        if let Some(rows) = self.qk_rows(weight_key) {
            deinterleave_rows(&mut b[..rows * r], self.cfg.head_dim, r);
        }
        Ok((a, vec![r, shape[1]], b, vec![shape[0], r]))
    }

    /// Rows `[0, n)` of `key` that hold q and k (to be de-interleaved), if any.
    fn qk_rows(&self, key: &str) -> Option<usize> {
        let qk = 2 * self.cfg.hidden;
        let base = key
            .strip_suffix(".weight")
            .or_else(|| key.strip_suffix(".bias"))?;
        (base.ends_with("_attn.qkv")
            || (base.starts_with("single_blocks.") && base.ends_with(".linear1")))
        .then_some(qk)
    }

    /// One tensor exactly as the graph wants it, before any transpose.
    pub fn load(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (mut data, shape) = self.file.load(key)?;
        let hd = self.cfg.head_dim;
        if let Some(rows) = self.qk_rows(key) {
            let cols = if shape.len() == 2 { shape[1] } else { 1 };
            deinterleave_rows(&mut data[..rows * cols], hd, cols);
        } else if key.ends_with("_norm.scale") {
            deinterleave_rows(&mut data, hd, 1);
        }
        Ok((data, shape))
    }
}

/// `[rows, cols]` of 2-byte elements → `[cols, rows]`: 64×64 tiles, output
/// column bands across threads, each tile written row by row.
pub fn transpose_half(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    use rayon::prelude::*;
    const T: usize = 64;
    let mut out = vec![0u8; src.len()];
    out.par_chunks_mut(2 * T * rows)
        .enumerate()
        .for_each(|(cb, dst)| {
            let c0 = cb * T;
            let nc = dst.len() / (2 * rows);
            for r0 in (0..rows).step_by(T) {
                let r1 = (r0 + T).min(rows);
                for c in 0..nc {
                    let d = &mut dst[2 * (c * rows + r0)..2 * (c * rows + r1)];
                    for (k, r) in (r0..r1).enumerate() {
                        let s = 2 * (r * cols + c0 + c);
                        d[2 * k..2 * k + 2].copy_from_slice(&src[s..s + 2]);
                    }
                }
            }
        });
    out
}

impl WeightSource for DitSource<'_> {
    /// Projection weights quantized at load, when [`DitSource::quant`] is set:
    /// `[out, in]` rows (GGUF's own order, so no transpose) with the q/k rows
    /// permuted first.
    fn take_packed(&mut self, key: &str) -> Result<Option<rlx_flow::GgufPackedLinear>> {
        let Some(q) = self.quant else {
            return Ok(None);
        };
        if key.ends_with(LORA_A) || key.ends_with(LORA_B) || !key.ends_with(".weight") {
            return Ok(None);
        }
        let Some(dtype) = self.file.stored_dtype(key) else {
            return Ok(None);
        };
        let Some((mut bytes, shape)) = self.file.load_half_bytes(key)? else {
            return Ok(None);
        };
        if shape.len() != 2 || shape[1] % q.block() != 0 {
            return Ok(None);
        }
        let (rows, cols) = (shape[0], shape[1]);
        // LoRA patches are merged before quantizing, so the projection needs
        // no low-rank branch in the graph: with two adapters (emoji + a
        // few-step one) those branches were ~10 % of every denoising step.
        if let Some((a, b, r)) = self.lora.stacked(key, rows) {
            use rayon::prelude::*;
            let mut w: Vec<f32> = bytes
                .par_chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    match dtype {
                        DType::F16 => half::f16::from_bits(bits).to_f32(),
                        _ => half::bf16::from_bits(bits).to_f32(),
                    }
                })
                .collect();
            rlx_core::host_kernels::matmul_accumulate(&b, &a, &mut w, rows, r, cols);
            if let Some(qk) = self.qk_rows(key) {
                deinterleave_rows(&mut w[..qk * cols], self.cfg.head_dim, cols);
            }
            return Ok(Some(rlx_flow::GgufPackedLinear {
                w_q: q.encode_f32(&w, rows, cols)?,
                scheme: q.scheme(),
                in_dim: cols,
                out_dim: rows,
                bias: Vec::new(),
            }));
        }
        if let Some(qk) = self.qk_rows(key) {
            deinterleave_rows(&mut bytes[..2 * qk * cols], self.cfg.head_dim, 2 * cols);
        }
        Ok(Some(rlx_flow::GgufPackedLinear {
            w_q: q.encode_half(&bytes, dtype, rows, cols)?,
            scheme: q.scheme(),
            in_dim: cols,
            out_dim: rows,
            bias: Vec::new(),
        }))
    }

    /// Rank-2 checkpoint weights straight from their half-precision bits:
    /// q/k rows permuted and the matrix transposed in u16, never widened.
    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        if key.ends_with(LORA_A) || key.ends_with(LORA_B) {
            return Ok(None);
        }
        let Some((mut bytes, shape)) = self.file.load_half_bytes(key)? else {
            return Ok(None);
        };
        if shape.len() != 2 {
            return Ok(None);
        }
        let (rows, cols) = (shape[0], shape[1]);
        if let Some(qk) = self.qk_rows(key) {
            // Whole rows move, so a row is just `2 * cols` bytes.
            deinterleave_rows(&mut bytes[..2 * qk * cols], self.cfg.head_dim, 2 * cols);
        }
        Ok(Some(if transpose {
            (transpose_half(&bytes, rows, cols), vec![cols, rows])
        } else {
            (bytes, shape)
        }))
    }

    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        if let Some(base) = key.strip_suffix(LORA_A) {
            let (a, a_shape, _, _) = self.lora_pair(base)?;
            return Ok(maybe_transpose(a, a_shape, transpose));
        }
        if let Some(base) = key.strip_suffix(LORA_B) {
            let (_, _, b, b_shape) = self.lora_pair(base)?;
            return Ok(maybe_transpose(b, b_shape, transpose));
        }
        let (data, shape) = self.load(key)?;
        Ok(maybe_transpose(data, shape, transpose))
    }

    fn has(&self, key: &str) -> bool {
        match key
            .strip_suffix(LORA_A)
            .or_else(|| key.strip_suffix(LORA_B))
        {
            Some(base) => self.has_lora(base),
            None => self.file.contains(key),
        }
    }

    fn keys(&self) -> Vec<String> {
        self.file.keys()
    }

    fn stored_dtype(&self, key: &str) -> Option<DType> {
        if key.ends_with(LORA_A) || key.ends_with(LORA_B) {
            return Some(DType::F32);
        }
        self.file.stored_dtype(key)
    }
}

/// A [`TensorFile`] served as-is (the text encoders), under an optional name prefix.
pub struct PlainSource<'a> {
    pub file: &'a TensorFile,
    pub prefix: &'a str,
    /// Serve 2-D `.weight` matrices packed (see [`WeightQuant`]).
    pub quant: Option<WeightQuant>,
}

impl WeightSource for PlainSource<'_> {
    /// `[out, in]` rows, quantized at load when [`PlainSource::quant`] is set.
    fn take_packed(&mut self, key: &str) -> Result<Option<rlx_flow::GgufPackedLinear>> {
        let Some(q) = self.quant else {
            return Ok(None);
        };
        let key = format!("{}{key}", self.prefix);
        if !key.ends_with(".weight") {
            return Ok(None);
        }
        let Some(dtype) = self.file.stored_dtype(&key) else {
            return Ok(None);
        };
        let Some((bytes, shape)) = self.file.load_half_bytes(&key)? else {
            return Ok(None);
        };
        if shape.len() != 2 || shape[1] % q.block() != 0 {
            return Ok(None);
        }
        let (rows, cols) = (shape[0], shape[1]);
        Ok(Some(rlx_flow::GgufPackedLinear {
            w_q: q.encode_half(&bytes, dtype, rows, cols)?,
            scheme: q.scheme(),
            in_dim: cols,
            out_dim: rows,
            bias: Vec::new(),
        }))
    }

    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.file.load(&format!("{}{key}", self.prefix))?;
        Ok(maybe_transpose(data, shape, transpose))
    }

    fn has(&self, key: &str) -> bool {
        self.file.contains(&format!("{}{key}", self.prefix))
    }

    fn stored_dtype(&self, key: &str) -> Option<DType> {
        self.file.stored_dtype(&format!("{}{key}", self.prefix))
    }
}

fn maybe_transpose(data: Vec<f32>, shape: Vec<usize>, transpose: bool) -> (Vec<f32>, Vec<usize>) {
    if transpose && shape.len() == 2 {
        (
            transpose_2d(&data, shape[0], shape[1]),
            vec![shape[1], shape[0]],
        )
    } else {
        (data, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_half_matches_f32_transpose() {
        let (r, c) = (70, 131);
        let v: Vec<u16> = (0..r * c).map(|i| i as u16).collect();
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        let want: Vec<u8> = transpose_2d(&f, r, c)
            .iter()
            .flat_map(|&x| (x as u16).to_le_bytes())
            .collect();
        assert_eq!(transpose_half(&bytes, r, c), want);
    }

    #[test]
    fn deinterleave_moves_pairs_to_halves() {
        let mut v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        deinterleave_rows(&mut v, 4, 1);
        assert_eq!(v, [0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
        let mut m: Vec<f32> = (0..8).map(|i| i as f32).collect(); // 4 rows × 2 cols
        deinterleave_rows(&mut m, 4, 2);
        assert_eq!(m, [0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
    }
}
