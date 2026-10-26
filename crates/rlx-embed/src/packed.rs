//! int8 storage for encoder checkpoints: a quarter of the f32 download, run
//! as int8 — the projections never become f32 at all.
//!
//! An ordinary safetensors file in which:
//! - every projection weight (`*.weight`, 2-D, inner dim a multiple of 32) is
//!   `U8` GGUF **Q8_0** blocks (an f16 scale and 32 int8 values per block,
//!   `[out, in/32·34]`), which the BERT flow runs as fused `DequantMatMul`s;
//! - every other 2-D `*.weight` (the embedding tables) is `I8`, one f32 scale
//!   per row beside it as `<name>.qscale` (symmetric, round-to-nearest) — a
//!   caller gathers rows of these on the host ([`PackedWeights::row`]);
//! - everything else (biases, LayerNorms) stays as it was.
//!
//! Version 1 files (every 2-D weight `I8` + `.qscale`) still load: their
//! projections are dequantized to f32.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{GgufPackedLinear, WeightSource};
use rlx_ir::quant::QuantScheme;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

/// The suffix of an int8 table's per-row scales.
pub const SCALE_SUFFIX: &str = ".qscale";
/// Values per Q8_0 block, and its bytes (f16 scale + 32 × i8).
const QK: usize = 32;
const QB: usize = 34;

/// Whether `bytes` is a checkpoint [`pack_int8`] wrote (either version).
pub fn is_packed(bytes: &[u8]) -> bool {
    SafeTensors::deserialize(bytes).is_ok_and(|st| {
        st.names().iter().any(|n| n.ends_with(SCALE_SUFFIX))
            || st
                .tensors()
                .iter()
                .any(|(_, v)| v.dtype() == Dtype::U8 && v.shape().len() == 2)
    })
}

/// Whether `name` (shape `shape`) is a projection the flow can run packed.
fn is_projection(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2
        && name.ends_with(".weight")
        && !name.contains("embeddings.")
        && shape[1].is_multiple_of(QK)
}

/// Pack an f32 safetensors checkpoint (see the module docs).
pub fn pack_int8(f32_safetensors: &[u8]) -> Result<Vec<u8>> {
    let st = SafeTensors::deserialize(f32_safetensors)?;
    let mut out: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, view) in st.tensors() {
        let shape = view.shape().to_vec();
        if view.dtype() != Dtype::F32 || shape.len() != 2 || !name.ends_with(".weight") {
            out.push((name, view.dtype(), shape, view.data().to_vec()));
            continue;
        }
        let w = f32s(view.data());
        let (rows, cols) = (shape[0], shape[1]);
        if is_projection(&name, &shape) {
            let blob = rlx_gguf::quantize::quantize_q8_0(&w)?;
            out.push((name, Dtype::U8, vec![rows, cols / QK * QB], blob));
            continue;
        }
        let (mut q, mut scales) = (Vec::with_capacity(rows * cols), Vec::with_capacity(rows));
        for row in w.chunks_exact(cols) {
            let amax = row.iter().fold(0f32, |m, x| m.max(x.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            q.extend(
                row.iter()
                    .map(|x| (x / s).round().clamp(-127.0, 127.0) as i8 as u8),
            );
            scales.push(s);
        }
        let sb = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
        out.push((format!("{name}{SCALE_SUFFIX}"), Dtype::F32, vec![rows], sb));
        out.push((name, Dtype::I8, shape, q));
    }
    let views = out
        .iter()
        .map(|(n, t, s, d)| Ok((n.clone(), TensorView::new(*t, s.clone(), d)?)))
        .collect::<Result<Vec<_>>>()?;
    Ok(safetensors::serialize(views, None)?)
}

/// A packed checkpoint held in memory as it was downloaded, decoded tensor by
/// tensor on demand: projections handed out as Q8_0 blobs, embedding rows
/// gathered one at a time, the rest widened to f32. Cloning shares the bytes.
#[derive(Clone)]
pub struct PackedWeights {
    bytes: Arc<[u8]>,
    /// name → (dtype, shape, absolute byte range)
    index: HashMap<String, (Dtype, Vec<usize>, std::ops::Range<usize>)>,
    /// int8 tables' scales, decoded once (they are small)
    scales: HashMap<String, Vec<f32>>,
}

impl PackedWeights {
    pub fn new(bytes: impl Into<Arc<[u8]>>) -> Result<Self> {
        let bytes: Arc<[u8]> = bytes.into();
        let (n, meta) = SafeTensors::read_metadata(&bytes)?;
        let base = 8 + n;
        let index: HashMap<_, _> = meta
            .tensors()
            .into_iter()
            .map(|(k, t)| {
                (
                    k,
                    (
                        t.dtype,
                        t.shape.clone(),
                        base + t.data_offsets.0..base + t.data_offsets.1,
                    ),
                )
            })
            .collect();
        let mut scales = HashMap::new();
        for (k, (_, _, r)) in &index {
            if let Some(table) = k.strip_suffix(SCALE_SUFFIX) {
                scales.insert(table.to_string(), f32s(&bytes[r.clone()]));
            }
        }
        Ok(Self {
            bytes,
            index,
            scales,
        })
    }

    fn data(&self, key: &str) -> Result<(Dtype, &[usize], &[u8])> {
        let (t, s, r) = self
            .index
            .get(key)
            .with_context(|| format!("no tensor {key:?}"))?;
        Ok((*t, s, &self.bytes[r.clone()]))
    }

    /// `[rows, cols]` of a 2-D tensor (a Q8_0 blob reports its f32 shape).
    pub fn dims(&self, key: &str) -> Option<(usize, usize)> {
        let (t, s, _) = self.index.get(key)?;
        match (*t, s.as_slice()) {
            (Dtype::U8, &[r, b]) => Some((r, b / QB * QK)),
            (_, &[r, c]) => Some((r, c)),
            _ => None,
        }
    }

    /// Row `i` of the 2-D table `key`, as f32, added into `out` (so a caller
    /// sums the token, position and type rows in one buffer).
    pub fn add_row(&self, key: &str, i: usize, out: &mut [f32]) -> Result<()> {
        let (t, s, d) = self.data(key)?;
        let cols = s[1];
        if i >= s[0] || out.len() != cols {
            bail!("{key}: row {i} of {s:?} into {}", out.len());
        }
        match t {
            Dtype::I8 => {
                let scale = self
                    .scales
                    .get(key)
                    .with_context(|| format!("{key}: no scales"))?[i];
                for (o, &b) in out.iter_mut().zip(&d[i * cols..(i + 1) * cols]) {
                    *o += b as i8 as f32 * scale;
                }
            }
            Dtype::F32 => {
                for (o, c) in out.iter_mut().zip(
                    d[i * cols * 4..(i + 1) * cols * 4]
                        .as_chunks::<4>()
                        .0
                        .iter(),
                ) {
                    *o += f32::from_le_bytes(*c);
                }
            }
            other => bail!("{key}: rows of {other:?} are not supported"),
        }
        Ok(())
    }

    /// Every tensor widened to f32 (the dense path; what v1 loading did).
    pub fn to_weight_map(&self) -> Result<WeightMap> {
        let mut tensors = HashMap::new();
        for k in self.index.keys().filter(|k| !k.ends_with(SCALE_SUFFIX)) {
            tensors.insert(k.clone(), self.widen(k)?);
        }
        Ok(WeightMap::from_tensors(tensors))
    }

    fn widen(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (t, s, d) = self.data(key)?;
        let shape = s.to_vec();
        let v: Vec<f32> = match t {
            Dtype::F32 => f32s(d),
            Dtype::I8 => {
                let scales = self
                    .scales
                    .get(key)
                    .with_context(|| format!("{key}: no scales"))?;
                let cols = d.len() / scales.len().max(1);
                d.iter()
                    .enumerate()
                    .map(|(i, &b)| b as i8 as f32 * scales[i / cols])
                    .collect()
            }
            Dtype::U8 if shape.len() == 2 => {
                let mut v = Vec::with_capacity(d.len() / QB * QK);
                for b in d.as_chunks::<QB>().0.iter() {
                    let scale = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
                    v.extend(b[2..].iter().map(|&q| q as i8 as f32 * scale));
                }
                return Ok((v, vec![shape[0], shape[1] / QB * QK]));
            }
            Dtype::F16 => d
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::BF16 => d
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::I64 => d
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b) as f32)
                .collect(),
            Dtype::I32 => d
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b) as f32)
                .collect(),
            other => bail!("{key}: unsupported dtype {other:?}"),
        };
        Ok((v, shape))
    }
}

impl WeightSource for PackedWeights {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (v, s) = self.widen(key)?;
        if !transpose {
            return Ok((v, s));
        }
        let [r, c] = s[..] else {
            bail!("transpose requires a rank-2 weight: {key}")
        };
        Ok((transpose_blocked(&v, r, c), vec![c, r]))
    }
    fn has(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }
    fn has_packed(&self, key: &str) -> bool {
        matches!(self.index.get(key), Some((Dtype::U8, s, _)) if s.len() == 2)
    }
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        if !self.has_packed(key) {
            return Ok(None);
        }
        let (rows, cols) = self.dims(key).unwrap();
        Ok(Some(GgufPackedLinear {
            w_q: self.data(key)?.2.to_vec(),
            scheme: QuantScheme::GgufQ8_0,
            in_dim: cols,
            out_dim: rows,
            bias: Vec::new(),
        }))
    }
    fn keys(&self) -> Vec<String> {
        self.index
            .keys()
            .filter(|k| !k.ends_with(SCALE_SUFFIX))
            .cloned()
            .collect()
    }
}

/// A packed checkpoint seen as dense: every projection widened to f32 (once,
/// at build), embedding tables still gathered by the caller. For a backend
/// whose int8 matmul is slower than its f32 one.
pub struct DenseView<'a>(pub &'a mut PackedWeights);

impl WeightSource for DenseView<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        self.0.take(key, transpose)
    }
    fn has(&self, key: &str) -> bool {
        self.0.has(key)
    }
    fn keys(&self) -> Vec<String> {
        self.0.keys()
    }
}

/// `[r, c]` → `[c, r]`, in cache-sized tiles.
fn transpose_blocked(v: &[f32], r: usize, c: usize) -> Vec<f32> {
    const T: usize = 32;
    let mut out = vec![0f32; r * c];
    for i0 in (0..r).step_by(T) {
        for j0 in (0..c).step_by(T) {
            for i in i0..(i0 + T).min(r) {
                for j in j0..(j0 + T).min(c) {
                    out[j * r + i] = v[i * c + j];
                }
            }
        }
    }
    out
}

/// Load a packed checkpoint as f32 weights (every tensor widened).
pub fn unpack(bytes: &[u8]) -> Result<WeightMap> {
    PackedWeights::new(bytes.to_vec())?.to_weight_map()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let bytes: Vec<Vec<u8>> = tensors
            .iter()
            .map(|(_, _, d)| d.iter().flat_map(|x| x.to_le_bytes()).collect())
            .collect();
        safetensors::serialize(
            tensors.iter().zip(&bytes).map(|((n, s, _), b)| {
                (
                    n.to_string(),
                    TensorView::new(Dtype::F32, s.clone(), b).unwrap(),
                )
            }),
            None,
        )
        .unwrap()
    }

    #[test]
    fn projections_pack_to_q8_0_and_tables_to_int8_rows() {
        let proj: Vec<f32> = (0..4 * 64)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) / 13.0)
            .collect();
        let table: Vec<f32> = (0..3 * 8)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) / 10.0)
            .collect();
        let src = checkpoint(&[
            (
                "encoder.layer.0.output.dense.weight",
                vec![4, 64],
                proj.clone(),
            ),
            (
                "embeddings.word_embeddings.weight",
                vec![3, 8],
                table.clone(),
            ),
            (
                "encoder.layer.0.output.dense.bias",
                vec![4],
                vec![1.5, -2.0, 0.0, 3.0],
            ),
        ]);
        let packed = pack_int8(&src).unwrap();
        assert!(is_packed(&packed) && !is_packed(&src));
        let mut w = PackedWeights::new(packed).unwrap();
        assert!(w.has_packed("encoder.layer.0.output.dense.weight"));
        assert!(!w.has_packed("embeddings.word_embeddings.weight"));
        // the blob widens back within half a step of each block's scale
        let (got, shape) = w
            .take("encoder.layer.0.output.dense.weight", false)
            .unwrap();
        assert_eq!(shape, vec![4, 64]);
        for (a, b) in proj.chunks(32).zip(got.chunks(32)) {
            let step = a.iter().fold(0f32, |m, x| m.max(x.abs())) / 127.0;
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() <= step * 0.51 + 1e-3, "{x} vs {y}");
            }
        }
        let p = w
            .take_packed("encoder.layer.0.output.dense.weight")
            .unwrap()
            .unwrap();
        assert_eq!((p.out_dim, p.in_dim, p.w_q.len()), (4, 64, 4 * 2 * 34));
        // a table row, gathered and summed into a buffer
        let mut row = vec![1.0f32; 8];
        w.add_row("embeddings.word_embeddings.weight", 2, &mut row)
            .unwrap();
        let step = table[16..24].iter().fold(0f32, |m, x| m.max(x.abs())) / 127.0;
        for (x, y) in table[16..24].iter().zip(&row) {
            assert!((x + 1.0 - y).abs() <= step / 2.0 + 1e-6);
        }
        assert_eq!(
            w.take("encoder.layer.0.output.dense.bias", false)
                .unwrap()
                .0,
            vec![1.5, -2.0, 0.0, 3.0]
        );
        // and the transposed dense path agrees with the plain one
        let (t, ts) = w.take("encoder.layer.0.output.dense.weight", true).unwrap();
        assert_eq!(ts, vec![64, 4]);
        assert_eq!(t[5 * 4 + 3], got[3 * 64 + 5]);
    }
}
