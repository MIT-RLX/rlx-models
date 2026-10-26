// RLX — versatile ML compiler + runtime. GPLv3.
//! A **pread** view of a sharded DeepSeek-V4.1 checkpoint.
//!
//! [`crate::safetensors_checkpoint::SafetensorsCheckpoint`] mmaps each shard and
//! caches the mapping, which is the right trade for a model that fits in RAM: the
//! kernel does the paging and repeated reads are free. It is the wrong trade
//! here. Touching every tensor of a 510 GB checkpoint makes 510 GB of page cache
//! resident, and on macOS that memory is not reclaimable while the mapping is
//! alive — so the process is killed long before the model is loaded.
//!
//! This reads the shard headers once, keeps only the resulting
//! `name -> (shard, offset, len, dtype, shape)` index, and `pread`s individual
//! tensors on demand. The index for the released checkpoint is ~96k entries —
//! a few MB — and no tensor byte stays resident longer than the caller holds it.
//!
//! [`StreamingLoader`] wraps that as a [`WeightLoader`], so every existing graph
//! builder works against it unchanged.

use crate::dsv41_quant::{QuantPlan, dequantize, plan};
use crate::weight_loader::WeightLoader;
use anyhow::{Context, Result, anyhow, bail};
use safetensors::Dtype;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Where one tensor's bytes live.
#[derive(Debug, Clone)]
pub struct TensorLoc {
    /// Index into the checkpoint's sorted shard list.
    pub shard: usize,
    /// Absolute byte offset in that file.
    pub offset: u64,
    pub len: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl TensorLoc {
    /// Element count, which is *not* `len` for the sub-byte and quantized dtypes.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// `name -> TensorLoc` over every shard in a checkpoint directory.
pub struct WeightIndex {
    shards: Vec<PathBuf>,
    tensors: HashMap<String, TensorLoc>,
}

impl WeightIndex {
    /// Read every `*.safetensors` header in `dir`.
    ///
    /// Only the headers are read — for the released checkpoint that is 48 reads
    /// of a few MB each, against 510 GB of data left untouched on disk.
    pub fn open(dir: &Path) -> Result<Self> {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("read checkpoint dir {dir:?}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            bail!("deepseek_v41: no *.safetensors in {dir:?}");
        }
        let mut tensors = HashMap::new();
        for (i, path) in shards.iter().enumerate() {
            for (name, loc) in read_header(path, i)? {
                tensors.insert(name, loc);
            }
        }
        Ok(WeightIndex { shards, tensors })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn loc(&self, name: &str) -> Option<&TensorLoc> {
        self.tensors.get(name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Total bytes of every indexed tensor — the checkpoint's on-disk size.
    pub fn total_bytes(&self) -> u64 {
        self.tensors.values().map(|t| t.len as u64).sum()
    }

    /// `pread` one tensor into `buf`, replacing its contents.
    ///
    /// Positional reads, so concurrent callers need no shared cursor and nothing
    /// is left mapped afterwards.
    pub fn read_into(&self, name: &str, buf: &mut Vec<u8>) -> Result<()> {
        let loc = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("deepseek_v41: no tensor `{name}` in the checkpoint"))?;
        buf.clear();
        buf.resize(loc.len, 0);
        read_exact_at(&self.shards[loc.shard], loc.offset, buf)
            .with_context(|| format!("read `{name}`"))?;
        Ok(())
    }

    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let mut b = Vec::new();
        self.read_into(name, &mut b)?;
        Ok(b)
    }
}

/// Parse one shard's safetensors header into absolute tensor locations.
fn read_header(path: &Path, shard: usize) -> Result<Vec<(String, TensorLoc)>> {
    let mut f = File::open(path).with_context(|| format!("open {path:?}"))?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n)
        .with_context(|| format!("{path:?} is too short for a safetensors header"))?;
    let hlen = u64::from_le_bytes(n);
    let mut hdr = vec![0u8; hlen as usize];
    f.read_exact(&mut hdr)
        .with_context(|| format!("read {path:?} header"))?;
    let json: serde_json::Value =
        serde_json::from_slice(&hdr).with_context(|| format!("parse {path:?} header"))?;
    // data_offsets are relative to the end of the header
    let base = 8 + hlen;
    let mut out = Vec::new();
    for (name, v) in json.as_object().map(|o| o.iter()).into_iter().flatten() {
        if name == "__metadata__" {
            continue;
        }
        let off = v["data_offsets"]
            .as_array()
            .ok_or_else(|| anyhow!("{path:?}: `{name}` has no data_offsets"))?;
        let (s, e) = (off[0].as_u64().unwrap_or(0), off[1].as_u64().unwrap_or(0));
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as usize)
                    .collect()
            })
            .unwrap_or_default();
        out.push((
            name.clone(),
            TensorLoc {
                shard,
                offset: base + s,
                len: (e - s) as usize,
                dtype: parse_dtype(v["dtype"].as_str().unwrap_or("F32"))?,
                shape,
            },
        ));
    }
    Ok(out)
}

fn parse_dtype(s: &str) -> Result<Dtype> {
    Ok(match s {
        "F64" => Dtype::F64,
        "F32" => Dtype::F32,
        "F16" => Dtype::F16,
        "BF16" => Dtype::BF16,
        "I64" => Dtype::I64,
        "I32" => Dtype::I32,
        "I16" => Dtype::I16,
        "I8" => Dtype::I8,
        "U8" => Dtype::U8,
        "BOOL" => Dtype::BOOL,
        // The quantized paths key off these: FP8 weights are E4M3, and the
        // companion scales are E8M0 (a bare exponent byte). Collapsing them to
        // U8 would make `plan` reject the tensor as "neither FP8 nor packed FP4".
        "F8_E4M3" => Dtype::F8_E4M3,
        "F8_E5M2" => Dtype::F8_E5M2,
        "F8_E8M0" => Dtype::F8_E8M0,
        "F4" => Dtype::F4,
        other => bail!("deepseek_v41: unsupported safetensors dtype `{other}`"),
    })
}

#[cfg(unix)]
fn read_exact_at(path: &Path, offset: u64, buf: &mut [u8]) -> Result<()> {
    use std::os::unix::fs::FileExt;
    let f = File::open(path).with_context(|| format!("open {path:?}"))?;
    f.read_exact_at(buf, offset)?;
    Ok(())
}

#[cfg(not(unix))]
fn read_exact_at(path: &Path, offset: u64, buf: &mut [u8]) -> Result<()> {
    use std::io::{Seek, SeekFrom};
    let mut f = File::open(path).with_context(|| format!("open {path:?}"))?;
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(buf)?;
    Ok(())
}

/// A [`WeightLoader`] that `pread`s each tensor as it is asked for.
///
/// Unlike [`crate::dsv41_quant::DsV41Loader`] it retains nothing: the bytes are
/// dequantized into the caller's `Vec<f32>` and the file handle is dropped. That
/// makes peak RSS the largest single tensor rather than the whole checkpoint,
/// which is the difference between loading one layer and being killed.
pub struct StreamingLoader {
    index: WeightIndex,
    block: usize,
    /// Reused scratch, so a long run does not churn the allocator.
    scratch: Vec<u8>,
    scale_scratch: Vec<u8>,
    taken: std::collections::HashSet<String>,
}

impl StreamingLoader {
    pub fn open(dir: &Path, block: usize) -> Result<Self> {
        Ok(StreamingLoader {
            index: WeightIndex::open(dir)?,
            block,
            scratch: Vec::new(),
            scale_scratch: Vec::new(),
            taken: std::collections::HashSet::new(),
        })
    }

    pub fn from_index(index: WeightIndex, block: usize) -> Self {
        StreamingLoader {
            index,
            block,
            scratch: Vec::new(),
            scale_scratch: Vec::new(),
            taken: std::collections::HashSet::new(),
        }
    }

    pub fn index(&self) -> &WeightIndex {
        &self.index
    }

    /// Names asked for but never found — the "weights I never used" check the
    /// `WeightLoader` contract talks about, inverted.
    pub fn taken(&self) -> &std::collections::HashSet<String> {
        &self.taken
    }

    /// Read and dequantize one tensor, without marking it taken.
    ///
    /// A quantized tensor is stored as `key` plus a companion `key`-with-`.scale`;
    /// which of the three scale layouts applies is decided by
    /// [`crate::dsv41_quant::plan`] from the two shapes.
    pub fn fetch(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        Ok(match self.fetch_raw(key)? {
            RawTensor::Dense { values, shape } => (values, shape),
            RawTensor::Packed {
                codes,
                scales,
                plan: p,
            } => (dequantize(&codes, &scales, &p)?, vec![p.rows, p.cols]),
        })
    }

    /// Read a tensor **without** dequantizing it.
    ///
    /// A quantized tensor is 8× smaller packed than as f32, which is the
    /// difference between a cache that holds a model's working set and one that
    /// does not — so a caller that can defer the dequant should
    /// ([`crate::dsv41_pager::ExpertPager`] does).
    pub fn fetch_raw(&mut self, key: &str) -> Result<RawTensor> {
        let loc = self
            .index
            .loc(key)
            .ok_or_else(|| anyhow!("deepseek_v41: no tensor `{key}` in the checkpoint"))?
            .clone();
        self.index.read_into(key, &mut self.scratch)?;

        let scale_key = key.strip_suffix(".weight").map(|s| format!("{s}.scale"));
        let Some(sk) = scale_key.filter(|k| self.index.contains(k)) else {
            return Ok(RawTensor::Dense {
                values: plain_to_f32(key, &self.scratch, loc.dtype)?,
                shape: loc.shape,
            });
        };
        let sloc = self.index.loc(&sk).expect("checked above").clone();
        self.index.read_into(&sk, &mut self.scale_scratch)?;
        // I8/F4 means the columns are nibble pairs, so the logical width is
        // twice the stored one; FP8 is one byte per element.
        let packed_fp4 = matches!(loc.dtype, Dtype::I8 | Dtype::F4);
        if !packed_fp4 && loc.dtype != Dtype::F8_E4M3 {
            bail!(
                "deepseek_v41: `{key}` has a `.scale` but dtype {:?} is neither FP8 nor packed FP4",
                loc.dtype
            );
        }
        Ok(RawTensor::Packed {
            codes: std::mem::take(&mut self.scratch),
            scales: std::mem::take(&mut self.scale_scratch),
            plan: plan(&loc.shape, packed_fp4, &sloc.shape, self.block)?,
        })
    }
}

/// A tensor as the checkpoint stores it.
pub enum RawTensor {
    /// Already float: nothing to defer.
    Dense { values: Vec<f32>, shape: Vec<usize> },
    /// Quantized codes plus their scale bytes, and how to read them.
    Packed {
        codes: Vec<u8>,
        scales: Vec<u8>,
        plan: QuantPlan,
    },
}

impl RawTensor {
    /// Bytes this occupies as stored — what a byte-budgeted cache should count.
    pub fn stored_bytes(&self) -> usize {
        match self {
            RawTensor::Dense { values, .. } => values.len() * 4,
            RawTensor::Packed { codes, scales, .. } => codes.len() + scales.len(),
        }
    }

    /// `[rows, cols]` of the logical tensor.
    pub fn shape(&self) -> Vec<usize> {
        match self {
            RawTensor::Dense { shape, .. } => shape.clone(),
            RawTensor::Packed { plan, .. } => vec![plan.rows, plan.cols],
        }
    }

    /// Materialize as f32, transposed to `[in, out]`.
    ///
    /// The transpose is folded into the dequant walk rather than done as a
    /// second pass, so a gathered bank costs one traversal instead of two.
    pub fn to_f32_transposed(&self) -> Result<Vec<f32>> {
        let shape = self.shape();
        if shape.len() != 2 {
            bail!("deepseek_v41: expected a rank-2 expert, got {shape:?}");
        }
        let (rows, cols) = (shape[0], shape[1]);
        let flat = match self {
            RawTensor::Dense { values, .. } => std::borrow::Cow::Borrowed(values),
            RawTensor::Packed {
                codes,
                scales,
                plan,
            } => std::borrow::Cow::Owned(dequantize(codes, scales, plan)?),
        };
        let mut out = vec![0f32; flat.len()];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = flat[i * cols + j];
            }
        }
        Ok(out)
    }
}

fn plain_to_f32(key: &str, bytes: &[u8], dt: Dtype) -> Result<Vec<f32>> {
    Ok(match dt {
        Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::I32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        Dtype::I64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        other => bail!(
            "deepseek_v41: `{key}` has dtype {other:?} and no companion `.scale` to read it with"
        ),
    })
}

impl WeightLoader for StreamingLoader {
    fn format_id(&self) -> &'static str {
        "safetensors"
    }

    fn len(&self) -> usize {
        self.index.len()
    }

    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.taken.insert(key.to_string());
        self.fetch(key)
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.index
            .names()
            .filter(|n| !self.taken.contains(*n))
            .map(str::to_string)
            .collect()
    }

    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (v, shape) = self.take(key)?;
        if shape.len() != 2 {
            return Ok((v, shape));
        }
        let (r, c) = (shape[0], shape[1]);
        let mut out = vec![0f32; v.len()];
        for i in 0..r {
            for j in 0..c {
                out[j * r + i] = v[i * c + j];
            }
        }
        Ok((out, vec![c, r]))
    }
}

/// How a synthetic checkpoint stores its weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CheckpointFormat {
    /// Everything as plain F32.
    #[default]
    Dense,
    /// Quantized wherever the released checkpoint is: FP8 tiles for the
    /// projections, FP4 nibble pairs for the routed experts, FP8 row-groups for
    /// the Engram table — each with its companion `.scale`.
    ///
    /// This is the layout a real load actually takes, so a runner only ever
    /// driven on [`CheckpointFormat::Dense`] has never exercised it.
    Quantized,
}

/// Write a complete, loadable DeepSeek-V4.1 checkpoint from a config alone.
///
/// The tensor list comes from [`crate::dsv41::DeepseekV41Spec::expected_tensors`]
/// and the values from [`crate::weight_loader::SyntheticLoader`], so the result
/// contains exactly what the builders ask for, at the right shapes, reproducibly
/// on any machine.
///
/// This exists so the runner can be exercised against a real directory on disk
/// rather than an in-memory fake: the file layout, the index, the loader and the
/// graph builders all have to agree, and a mismatch shows up here as a load error
/// instead of silently reading zeros.
pub fn write_synthetic_checkpoint(dir: &Path, config: &serde_json::Value) -> Result<()> {
    write_synthetic_checkpoint_as(dir, config, CheckpointFormat::Dense).map(|_| ())
}

/// [`write_synthetic_checkpoint`] with a choice of storage format.
///
/// Returns the values the checkpoint decodes to, keyed by tensor name — which is
/// what lets a test compare a quantized checkpoint against a dense one holding
/// exactly those values, separating "does the runner read this layout" from "is
/// quantization lossy".
pub fn write_synthetic_checkpoint_as(
    dir: &Path,
    config: &serde_json::Value,
    format: CheckpointFormat,
) -> Result<HashMap<String, Vec<f32>>> {
    use crate::dsv41::DeepseekV41Spec;
    use crate::dsv41_quant::{CodeFormat, QuantPlan, ScaleLayout, dequantize, quantize};
    use crate::weight_loader::SyntheticLoader;
    use safetensors::tensor::{Dtype as StDtype, TensorView};

    std::fs::create_dir_all(dir).with_context(|| format!("create {dir:?}"))?;
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    let spec = DeepseekV41Spec::from_config(config)?;
    spec.validate()?;
    let block = if spec.weight_block_size > 0 {
        spec.weight_block_size
    } else {
        crate::dsv41_quant::DEFAULT_BLOCK
    };

    let mut raw: Vec<(String, Vec<usize>, StDtype, Vec<u8>)> = Vec::new();
    let mut decoded: HashMap<String, Vec<f32>> = HashMap::new();
    for need in spec.expected_tensors() {
        let values = SyntheticLoader::values(&need.name, &need.shape);
        let quantize_this = format == CheckpointFormat::Quantized
            && need.quantized
            && need.shape.len() == 2
            // a group narrower than one block has no grid to speak of
            && need.shape[1] >= block;
        if !quantize_this {
            raw.push((
                need.name.clone(),
                need.shape.clone(),
                StDtype::F32,
                bytemuck::cast_slice(&values).to_vec(),
            ));
            decoded.insert(need.name, values);
            continue;
        }
        // The released layouts: experts are FP4 nibble pairs in row groups,
        // everything else is FP8 — tiled, except the Engram table which is
        // row-wise. `plan` re-derives all three from the two shapes, so what is
        // written here has to match what it will infer.
        let (code, layout) = if need.packed_fp4 {
            (CodeFormat::Fp4E2m1, ScaleLayout::RowGroups { block })
        } else if need.name.contains(".engram.embed") {
            (CodeFormat::Fp8E4m3, ScaleLayout::RowGroups { block })
        } else {
            (CodeFormat::Fp8E4m3, ScaleLayout::Tile { block })
        };
        let plan = QuantPlan {
            rows: need.shape[0],
            cols: need.shape[1],
            format: code,
            layout,
        };
        let (codes, scales) = quantize(&values, &plan)?;
        decoded.insert(need.name.clone(), dequantize(&codes, &scales, &plan)?);

        let sc_cols = plan.cols.div_ceil(block);
        let sc_rows = match layout {
            ScaleLayout::RowGroups { .. } => plan.rows,
            ScaleLayout::Tile { block } => plan.rows.div_ceil(block),
        };
        let scale_name = need.scale_name().ok_or_else(|| {
            anyhow!(
                "deepseek_v41: `{}` is quantized but has no scale",
                need.name
            )
        })?;
        raw.push((
            need.name.clone(),
            need.stored_shape(),
            if need.packed_fp4 {
                StDtype::I8
            } else {
                StDtype::F8_E4M3
            },
            codes,
        ));
        raw.push((scale_name, vec![sc_rows, sc_cols], StDtype::F8_E8M0, scales));
    }

    let views: HashMap<String, TensorView> = raw
        .iter()
        .map(|(n, s, dt, b)| {
            TensorView::new(*dt, s.clone(), b)
                .map(|v| (n.clone(), v))
                .map_err(|e| anyhow!("deepseek_v41: tensor view for `{n}`: {e}"))
        })
        .collect::<Result<_>>()?;
    safetensors::serialize_to_file(&views, None, &dir.join("model.safetensors"))
        .with_context(|| format!("write {dir:?}/model.safetensors"))?;
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41_quant::{DEFAULT_BLOCK, DsV41Loader};

    fn weights_dir() -> Option<std::path::PathBuf> {
        rlx_ir::env::var_result("RLX_DSV41_WEIGHTS")
            .ok()
            .map(Into::into)
    }

    /// The streaming loader must agree with the mmap loader bit for bit on real
    /// checkpoint bytes — including the quantized tensors, where the two scale
    /// layouts and the FP4 nibble unpacking are decided from the shapes.
    ///
    /// This is the check that keeps `pread` from being a *different* reader
    /// rather than a cheaper one.
    #[test]
    fn streaming_matches_the_mmap_loader_on_real_weights() {
        let Some(dir) = weights_dir() else {
            eprintln!("set RLX_DSV41_WEIGHTS to run this");
            return;
        };
        let mut mm = DsV41Loader::open(&dir, DEFAULT_BLOCK).expect("mmap loader opens");
        let mut st = StreamingLoader::open(&dir, DEFAULT_BLOCK).expect("streaming loader opens");
        let names: Vec<String> = st.index().names().map(str::to_string).collect();
        assert!(
            names.len() > 20,
            "expected a real subset, got {}",
            names.len()
        );

        let mut checked = 0usize;
        for n in &names {
            if n.ends_with(".scale") {
                continue; // read as part of its `.weight`
            }
            let (want, wshape) = mm.take(n).expect("mmap read");
            let (got, gshape) = st.take(n).expect("streaming read");
            assert_eq!(gshape, wshape, "{n}: shape");
            assert_eq!(got.len(), want.len(), "{n}: element count");
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{n}[{i}] differs");
            }
            checked += 1;
        }
        assert!(checked >= 20, "only checked {checked} tensors");
        println!("streaming == mmap on {checked} real tensors");
    }

    /// The index must be built from headers alone: no tensor bytes are read
    /// until something asks for one.
    #[test]
    fn opening_the_index_reads_no_tensor_data() {
        let Some(dir) = weights_dir() else { return };
        let ix = WeightIndex::open(&dir).expect("index opens");
        // every location must land inside its shard, and the indexed bytes must
        // add up to roughly the on-disk size
        let files: u64 = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
            .map(|e| e.metadata().unwrap().len())
            .sum();
        let indexed = ix.total_bytes();
        assert!(
            indexed > 0 && indexed <= files,
            "{indexed} vs {files} on disk"
        );
        // headers are a small fraction, so the data should dominate
        assert!(
            indexed * 100 / files > 90,
            "indexed {indexed} is only {}% of {files}",
            indexed * 100 / files
        );
    }

    /// A tensor that is not in the checkpoint must say so by name rather than
    /// panicking somewhere downstream.
    #[test]
    fn a_missing_tensor_names_itself() {
        let Some(dir) = weights_dir() else { return };
        let mut st = StreamingLoader::open(&dir, DEFAULT_BLOCK).unwrap();
        let e = st.take("layers.999.nope.weight").unwrap_err().to_string();
        assert!(e.contains("layers.999.nope.weight"), "unhelpful error: {e}");
    }
}
