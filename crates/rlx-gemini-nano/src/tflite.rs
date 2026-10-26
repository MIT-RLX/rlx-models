//! Minimal, bounds-checked reader for the TFLite flatbuffer container (`TFL3`).
//!
//! Gemini Nano ships as a TFLite file (`weights.bin` under Chrome's
//! `OptGuideOnDeviceModel/<version>/`). Only the parts of the schema needed to
//! enumerate tensors, locate their bytes and read quantization parameters are
//! decoded here; operator options are left opaque.
//!
//! Schema reference: `tensorflow/compiler/mlir/lite/schema/schema.fbs`. Field
//! slots below are the declaration order in that file.
//!
//! Large models (> 2 GiB) store tensor bytes outside the flatbuffer: the
//! `Buffer` table then carries an absolute file `offset` and `size` instead of
//! inline `data`. Both forms are handled by [`TfliteModel::buffer_bytes`].

use std::fmt;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};

/// File identifier at bytes 4..8 of every TFLite flatbuffer.
pub const TFLITE_IDENTIFIER: &[u8; 4] = b"TFL3";

/// `TensorType` from the schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorType {
    F32,
    F16,
    I32,
    U8,
    I64,
    String,
    Bool,
    I16,
    Complex64,
    I8,
    F64,
    Complex128,
    U64,
    Resource,
    Variant,
    U32,
    U16,
    I4,
    Bf16,
    I2,
    U4,
    F8E4M3,
    F8E5M2,
    Unknown(i8),
}

impl TensorType {
    pub fn from_raw(v: i8) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::I32,
            3 => Self::U8,
            4 => Self::I64,
            5 => Self::String,
            6 => Self::Bool,
            7 => Self::I16,
            8 => Self::Complex64,
            9 => Self::I8,
            10 => Self::F64,
            11 => Self::Complex128,
            12 => Self::U64,
            13 => Self::Resource,
            14 => Self::Variant,
            15 => Self::U32,
            16 => Self::U16,
            17 => Self::I4,
            18 => Self::Bf16,
            19 => Self::I2,
            20 => Self::U4,
            21 => Self::F8E4M3,
            22 => Self::F8E5M2,
            other => Self::Unknown(other),
        }
    }

    /// Storage width in bits for one element, if fixed.
    pub fn bits(self) -> Option<usize> {
        Some(match self {
            Self::F32 | Self::I32 | Self::U32 => 32,
            Self::F16 | Self::I16 | Self::U16 | Self::Bf16 => 16,
            Self::U8 | Self::I8 | Self::Bool | Self::F8E4M3 | Self::F8E5M2 => 8,
            Self::I64 | Self::U64 | Self::F64 | Self::Complex64 => 64,
            Self::Complex128 => 128,
            Self::I4 | Self::U4 => 4,
            Self::I2 => 2,
            Self::String | Self::Resource | Self::Variant | Self::Unknown(_) => return None,
        })
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(v) => write!(f, "type{v}"),
            other => write!(f, "{}", format!("{other:?}").to_lowercase()),
        }
    }
}

/// Per-tensor affine quantization (`QuantizationParameters`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Quantization {
    pub scale: Vec<f32>,
    pub zero_point: Vec<i64>,
    pub quantized_dimension: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<i32>,
    pub dtype: TensorType,
    /// Index into [`TfliteModel::buffers`]; 0 is the schema's empty sentinel.
    pub buffer: u32,
    pub quantization: Option<Quantization>,
    pub is_variable: bool,
    /// `Tensor.external_buffer`: bytes in a side file (schema's
    /// `Model.external_buffers`), not in this file. 0 = unused.
    pub external_buffer: u32,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().map(|&d| d.max(0) as usize).product()
    }

    /// Bytes the tensor occupies when packed densely (sub-byte types round up).
    pub fn packed_bytes(&self) -> Option<usize> {
        self.dtype.bits().map(|b| (self.numel() * b).div_ceil(8))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OperatorInfo {
    pub opcode_index: u32,
    pub inputs: Vec<i32>,
    pub outputs: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SubgraphInfo {
    pub name: String,
    pub tensors: Vec<TensorInfo>,
    pub inputs: Vec<i32>,
    pub outputs: Vec<i32>,
    pub operators: Vec<OperatorInfo>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OperatorCode {
    pub builtin: i32,
    pub custom: Option<String>,
    pub version: i32,
}

impl OperatorCode {
    /// Human-readable op name: the custom code, or the builtin's schema name.
    pub fn name(&self) -> String {
        match &self.custom {
            Some(c) => c.clone(),
            None => builtin_name(self.builtin)
                .map(str::to_string)
                .unwrap_or_else(|| format!("BUILTIN_{}", self.builtin)),
        }
    }
}

/// Where a buffer's bytes live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferLoc {
    Empty,
    /// Byte range inside the flatbuffer (`Buffer.data`).
    Inline {
        start: usize,
        len: usize,
    },
    /// Absolute file range (`Buffer.offset` / `Buffer.size`, used past 2 GiB).
    External {
        offset: u64,
        size: u64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetadataEntry {
    pub name: String,
    pub buffer: u32,
}

/// Bytes backing a model: an mmap of the file or an owned copy.
enum Backing {
    Mmap(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl Backing {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m,
            Self::Owned(v) => v,
        }
    }
}

/// A parsed TFLite model. Tensor bytes stay in the backing store until asked for.
pub struct TfliteModel {
    backing: Backing,
    pub version: u32,
    pub description: String,
    pub operator_codes: Vec<OperatorCode>,
    pub subgraphs: Vec<SubgraphInfo>,
    pub buffers: Vec<BufferLoc>,
    pub metadata: Vec<MetadataEntry>,
}

impl TfliteModel {
    /// Memory-map and parse `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: read-only map; the file is not expected to change underneath us.
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("mmap {}", path.display()))?;
        Self::parse(Backing::Mmap(mmap)).with_context(|| format!("parse {}", path.display()))
    }

    /// Parse an in-memory TFLite file.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::parse(Backing::Owned(bytes))
    }

    fn parse(backing: Backing) -> Result<Self> {
        let buf = backing.bytes();
        ensure!(
            buf.len() >= 8,
            "file too small for a flatbuffer ({} B)",
            buf.len()
        );
        ensure!(
            &buf[4..8] == TFLITE_IDENTIFIER,
            "not a TFLite file: identifier {:?}, expected \"TFL3\"",
            String::from_utf8_lossy(&buf[4..8])
        );
        let fb = Fb { buf };
        let model = fb.root()?;

        let version = model.u32(0, 0)?;
        let description = model.string(3)?.unwrap_or_default();

        let mut operator_codes = Vec::new();
        for oc in model.tables(1)? {
            let deprecated = oc.i8(0, 0)? as i32;
            let builtin = oc.i32(3, 0)?;
            operator_codes.push(OperatorCode {
                // Codes >= 127 live only in `builtin_code`; older files fill
                // only the deprecated byte. The larger of the two is correct.
                builtin: builtin.max(deprecated),
                custom: oc.string(1)?,
                version: oc.i32(2, 1)?,
            });
        }

        let mut subgraphs = Vec::new();
        for sg in model.tables(2)? {
            let mut tensors = Vec::new();
            for t in sg.tables(0)? {
                let quantization = match t.table(4)? {
                    None => None,
                    Some(q) => {
                        let scale = q.f32s(2)?;
                        let zero_point = q.i64s(3)?;
                        if scale.is_empty() && zero_point.is_empty() {
                            None
                        } else {
                            Some(Quantization {
                                scale,
                                zero_point,
                                quantized_dimension: q.i32(6, 0)?,
                            })
                        }
                    }
                };
                tensors.push(TensorInfo {
                    shape: t.i32s(0)?,
                    dtype: TensorType::from_raw(t.i8(1, 0)?),
                    buffer: t.u32(2, 0)?,
                    name: t.string(3)?.unwrap_or_default(),
                    quantization,
                    is_variable: t.u8(5, 0)? != 0,
                    external_buffer: t.u32(10, 0)?,
                });
            }
            let mut operators = Vec::new();
            for op in sg.tables(3)? {
                operators.push(OperatorInfo {
                    opcode_index: op.u32(0, 0)?,
                    inputs: op.i32s(1)?,
                    outputs: op.i32s(2)?,
                });
            }
            subgraphs.push(SubgraphInfo {
                tensors,
                inputs: sg.i32s(1)?,
                outputs: sg.i32s(2)?,
                operators,
                name: sg.string(4)?.unwrap_or_default(),
            });
        }

        let mut buffers = Vec::new();
        for b in model.tables(4)? {
            let offset = b.u64(1, 0)?;
            let size = b.u64(2, 0)?;
            // The converter writes offset = 1 as a placeholder before it fixes
            // up the real positions; anything > 1 is an absolute file offset.
            let loc = if offset > 1 {
                BufferLoc::External { offset, size }
            } else {
                match b.vector(0)? {
                    Some((start, len)) if len > 0 => BufferLoc::Inline { start, len },
                    _ => BufferLoc::Empty,
                }
            };
            buffers.push(loc);
        }

        let mut metadata = Vec::new();
        for m in model.tables(6)? {
            metadata.push(MetadataEntry {
                name: m.string(0)?.unwrap_or_default(),
                buffer: m.u32(1, 0)?,
            });
        }

        Ok(Self {
            backing,
            version,
            description,
            operator_codes,
            subgraphs,
            buffers,
            metadata,
        })
    }

    /// Raw bytes of buffer `index` (empty for the sentinel / unset buffers).
    pub fn buffer_bytes(&self, index: u32) -> Result<&[u8]> {
        let buf = self.backing.bytes();
        let loc = self
            .buffers
            .get(index as usize)
            .ok_or_else(|| anyhow!("buffer {index} out of range ({})", self.buffers.len()))?;
        let (start, len) = match *loc {
            BufferLoc::Empty => return Ok(&[]),
            BufferLoc::Inline { start, len } => (start, len),
            BufferLoc::External { offset, size } => (
                usize::try_from(offset).context("buffer offset")?,
                usize::try_from(size).context("buffer size")?,
            ),
        };
        let end = start
            .checked_add(len)
            .filter(|&e| e <= buf.len())
            .ok_or_else(|| {
                anyhow!(
                    "buffer {index} [{start}, +{len}) runs past end of file ({} B)",
                    buf.len()
                )
            })?;
        Ok(&buf[start..end])
    }

    /// Bytes of a tensor in the given subgraph.
    pub fn tensor_bytes(&self, tensor: &TensorInfo) -> Result<&[u8]> {
        ensure!(
            tensor.external_buffer == 0,
            "tensor {:?} lives in external buffer {} (side file), which is not supported",
            tensor.name,
            tensor.external_buffer
        );
        self.buffer_bytes(tensor.buffer)
    }

    /// Bytes of a named metadata entry, if present.
    pub fn metadata_bytes(&self, name: &str) -> Result<Option<&[u8]>> {
        match self.metadata.iter().find(|m| m.name == name) {
            Some(m) => self.buffer_bytes(m.buffer).map(Some),
            None => Ok(None),
        }
    }

    /// Find a tensor by exact name across all subgraphs.
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.subgraphs
            .iter()
            .flat_map(|s| s.tensors.iter())
            .find(|t| t.name == name)
    }

    /// Total file size of the backing store.
    pub fn file_len(&self) -> usize {
        self.backing.bytes().len()
    }
}

// ---------------------------------------------------------------------------
// Flatbuffer primitives. Every offset is checked; malformed input is an error,
// never a panic.

#[derive(Clone, Copy)]
struct Fb<'a> {
    buf: &'a [u8],
}

impl<'a> Fb<'a> {
    fn read<const N: usize>(&self, pos: usize) -> Result<[u8; N]> {
        let end = pos
            .checked_add(N)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| anyhow!("flatbuffer read of {N} B at {pos} out of bounds"))?;
        Ok(self.buf[pos..end].try_into().expect("length checked"))
    }

    fn u32_at(&self, pos: usize) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read(pos)?))
    }

    fn root(&self) -> Result<Table<'a>> {
        let pos = self.u32_at(0)? as usize;
        Table::at(*self, pos)
    }

    /// Follow a `uoffset_t` stored at `pos`.
    fn follow(&self, pos: usize) -> Result<usize> {
        let rel = self.u32_at(pos)? as usize;
        pos.checked_add(rel)
            .filter(|&p| p < self.buf.len())
            .ok_or_else(|| anyhow!("flatbuffer offset at {pos} points out of bounds"))
    }
}

#[derive(Clone, Copy)]
struct Table<'a> {
    fb: Fb<'a>,
    pos: usize,
    vtable: usize,
    vt_len: usize,
}

impl<'a> Table<'a> {
    fn at(fb: Fb<'a>, pos: usize) -> Result<Self> {
        let soff = i32::from_le_bytes(fb.read(pos)?) as i64;
        let vtable = usize::try_from(pos as i64 - soff)
            .map_err(|_| anyhow!("vtable for table at {pos} out of bounds"))?;
        let vt_len = u16::from_le_bytes(fb.read(vtable)?) as usize;
        ensure!(vt_len >= 4, "vtable at {vtable} too short ({vt_len})");
        Ok(Self {
            fb,
            pos,
            vtable,
            vt_len,
        })
    }

    /// Absolute position of field `slot`, or `None` when absent (default).
    fn field(&self, slot: usize) -> Result<Option<usize>> {
        let entry = 4 + 2 * slot;
        if entry + 2 > self.vt_len {
            return Ok(None);
        }
        let off = u16::from_le_bytes(self.fb.read(self.vtable + entry)?) as usize;
        Ok((off != 0).then_some(self.pos + off))
    }

    fn scalar<const N: usize>(&self, slot: usize) -> Result<Option<[u8; N]>> {
        match self.field(slot)? {
            Some(p) => self.fb.read(p).map(Some),
            None => Ok(None),
        }
    }

    fn u8(&self, slot: usize, d: u8) -> Result<u8> {
        Ok(self.scalar::<1>(slot)?.map_or(d, |b| b[0]))
    }

    fn i8(&self, slot: usize, d: i8) -> Result<i8> {
        Ok(self.scalar::<1>(slot)?.map_or(d, |b| b[0] as i8))
    }

    fn i32(&self, slot: usize, d: i32) -> Result<i32> {
        Ok(self.scalar(slot)?.map_or(d, i32::from_le_bytes))
    }

    fn u32(&self, slot: usize, d: u32) -> Result<u32> {
        Ok(self.scalar(slot)?.map_or(d, u32::from_le_bytes))
    }

    fn u64(&self, slot: usize, d: u64) -> Result<u64> {
        Ok(self.scalar(slot)?.map_or(d, u64::from_le_bytes))
    }

    /// Position of a vector field's first element and its element count.
    fn vector(&self, slot: usize) -> Result<Option<(usize, usize)>> {
        let Some(p) = self.field(slot)? else {
            return Ok(None);
        };
        let v = self.fb.follow(p)?;
        let len = self.fb.u32_at(v)? as usize;
        Ok(Some((v + 4, len)))
    }

    fn scalars<const N: usize, T>(&self, slot: usize, conv: fn([u8; N]) -> T) -> Result<Vec<T>> {
        let Some((start, len)) = self.vector(slot)? else {
            return Ok(Vec::new());
        };
        let bytes = len
            .checked_mul(N)
            .ok_or_else(|| anyhow!("vector length overflow"))?;
        ensure!(
            start + bytes <= self.fb.buf.len(),
            "vector at {start} ({len} x {N} B) out of bounds"
        );
        (0..len)
            .map(|i| self.fb.read(start + i * N).map(conv))
            .collect()
    }

    fn i32s(&self, slot: usize) -> Result<Vec<i32>> {
        self.scalars(slot, i32::from_le_bytes)
    }

    fn f32s(&self, slot: usize) -> Result<Vec<f32>> {
        self.scalars(slot, f32::from_le_bytes)
    }

    fn i64s(&self, slot: usize) -> Result<Vec<i64>> {
        self.scalars(slot, i64::from_le_bytes)
    }

    fn string(&self, slot: usize) -> Result<Option<String>> {
        let Some((start, len)) = self.vector(slot)? else {
            return Ok(None);
        };
        let end = start
            .checked_add(len)
            .filter(|&e| e <= self.fb.buf.len())
            .ok_or_else(|| anyhow!("string at {start} out of bounds"))?;
        Ok(Some(
            String::from_utf8_lossy(&self.fb.buf[start..end]).into_owned(),
        ))
    }

    fn table(&self, slot: usize) -> Result<Option<Table<'a>>> {
        match self.field(slot)? {
            Some(p) => Table::at(self.fb, self.fb.follow(p)?).map(Some),
            None => Ok(None),
        }
    }

    fn tables(&self, slot: usize) -> Result<Vec<Table<'a>>> {
        let Some((start, len)) = self.vector(slot)? else {
            return Ok(Vec::new());
        };
        if len > self.fb.buf.len() / 4 {
            bail!("table vector length {len} exceeds file size");
        }
        (0..len)
            .map(|i| Table::at(self.fb, self.fb.follow(start + 4 * i)?))
            .collect()
    }
}

/// Schema names of the builtin operators (`enum BuiltinOperator`, 0..=209).
pub fn builtin_name(code: i32) -> Option<&'static str> {
    const NAMES: &[&str] = &[
        "ADD",
        "AVERAGE_POOL_2D",
        "CONCATENATION",
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "DEPTH_TO_SPACE",
        "DEQUANTIZE",
        "EMBEDDING_LOOKUP",
        "FLOOR",
        "FULLY_CONNECTED",
        "HASHTABLE_LOOKUP",
        "L2_NORMALIZATION",
        "L2_POOL_2D",
        "LOCAL_RESPONSE_NORMALIZATION",
        "LOGISTIC",
        "LSH_PROJECTION",
        "LSTM",
        "MAX_POOL_2D",
        "MUL",
        "RELU",
        "RELU_N1_TO_1",
        "RELU6",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "RNN",
        "SOFTMAX",
        "SPACE_TO_DEPTH",
        "SVDF",
        "TANH",
        "CONCAT_EMBEDDINGS",
        "SKIP_GRAM",
        "CALL",
        "CUSTOM",
        "EMBEDDING_LOOKUP_SPARSE",
        "PAD",
        "UNIDIRECTIONAL_SEQUENCE_RNN",
        "GATHER",
        "BATCH_TO_SPACE_ND",
        "SPACE_TO_BATCH_ND",
        "TRANSPOSE",
        "MEAN",
        "SUB",
        "DIV",
        "SQUEEZE",
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        "STRIDED_SLICE",
        "BIDIRECTIONAL_SEQUENCE_RNN",
        "EXP",
        "TOPK_V2",
        "SPLIT",
        "LOG_SOFTMAX",
        "DELEGATE",
        "BIDIRECTIONAL_SEQUENCE_LSTM",
        "CAST",
        "PRELU",
        "MAXIMUM",
        "ARG_MAX",
        "MINIMUM",
        "LESS",
        "NEG",
        "PADV2",
        "GREATER",
        "GREATER_EQUAL",
        "LESS_EQUAL",
        "SELECT",
        "SLICE",
        "SIN",
        "TRANSPOSE_CONV",
        "SPARSE_TO_DENSE",
        "TILE",
        "EXPAND_DIMS",
        "EQUAL",
        "NOT_EQUAL",
        "LOG",
        "SUM",
        "SQRT",
        "RSQRT",
        "SHAPE",
        "POW",
        "ARG_MIN",
        "FAKE_QUANT",
        "REDUCE_PROD",
        "REDUCE_MAX",
        "PACK",
        "LOGICAL_OR",
        "ONE_HOT",
        "LOGICAL_AND",
        "LOGICAL_NOT",
        "UNPACK",
        "REDUCE_MIN",
        "FLOOR_DIV",
        "REDUCE_ANY",
        "SQUARE",
        "ZEROS_LIKE",
        "FILL",
        "FLOOR_MOD",
        "RANGE",
        "RESIZE_NEAREST_NEIGHBOR",
        "LEAKY_RELU",
        "SQUARED_DIFFERENCE",
        "MIRROR_PAD",
        "ABS",
        "SPLIT_V",
        "UNIQUE",
        "CEIL",
        "REVERSE_V2",
        "ADD_N",
        "GATHER_ND",
        "COS",
        "WHERE",
        "RANK",
        "ELU",
        "REVERSE_SEQUENCE",
        "MATRIX_DIAG",
        "QUANTIZE",
        "MATRIX_SET_DIAG",
        "ROUND",
        "HARD_SWISH",
        "IF",
        "WHILE",
        "NON_MAX_SUPPRESSION_V4",
        "NON_MAX_SUPPRESSION_V5",
        "SCATTER_ND",
        "SELECT_V2",
        "DENSIFY",
        "SEGMENT_SUM",
        "BATCH_MATMUL",
        "PLACEHOLDER_FOR_GREATER_OP_CODES",
        "CUMSUM",
        "CALL_ONCE",
        "BROADCAST_TO",
        "RFFT2D",
        "CONV_3D",
        "IMAG",
        "REAL",
        "COMPLEX_ABS",
        "HASHTABLE",
        "HASHTABLE_FIND",
        "HASHTABLE_IMPORT",
        "HASHTABLE_SIZE",
        "REDUCE_ALL",
        "CONV_3D_TRANSPOSE",
        "VAR_HANDLE",
        "READ_VARIABLE",
        "ASSIGN_VARIABLE",
        "BROADCAST_ARGS",
        "RANDOM_STANDARD_NORMAL",
        "BUCKETIZE",
        "RANDOM_UNIFORM",
        "MULTINOMIAL",
        "GELU",
        "DYNAMIC_UPDATE_SLICE",
        "RELU_0_TO_1",
        "UNSORTED_SEGMENT_PROD",
        "UNSORTED_SEGMENT_MAX",
        "UNSORTED_SEGMENT_SUM",
        "ATAN2",
        "UNSORTED_SEGMENT_MIN",
        "SIGN",
        "BITCAST",
        "BITWISE_XOR",
        "RIGHT_SHIFT",
        "STABLEHLO_LOGISTIC",
        "STABLEHLO_ADD",
        "STABLEHLO_DIVIDE",
        "STABLEHLO_MULTIPLY",
        "STABLEHLO_MAXIMUM",
        "STABLEHLO_RESHAPE",
        "STABLEHLO_CLAMP",
        "STABLEHLO_CONCATENATE",
        "STABLEHLO_BROADCAST_IN_DIM",
        "STABLEHLO_CONVOLUTION",
        "STABLEHLO_SLICE",
        "STABLEHLO_CUSTOM_CALL",
        "STABLEHLO_REDUCE",
        "STABLEHLO_ABS",
        "STABLEHLO_AND",
        "STABLEHLO_COSINE",
        "STABLEHLO_EXPONENTIAL",
        "STABLEHLO_FLOOR",
        "STABLEHLO_LOG",
        "STABLEHLO_MINIMUM",
        "STABLEHLO_NEGATE",
        "STABLEHLO_OR",
        "STABLEHLO_POWER",
        "STABLEHLO_REMAINDER",
        "STABLEHLO_RSQRT",
        "STABLEHLO_SELECT",
        "STABLEHLO_SUBTRACT",
        "STABLEHLO_TANH",
        "STABLEHLO_SCATTER",
        "STABLEHLO_COMPARE",
        "STABLEHLO_CONVERT",
        "STABLEHLO_DYNAMIC_SLICE",
        "STABLEHLO_DYNAMIC_UPDATE_SLICE",
        "STABLEHLO_PAD",
        "STABLEHLO_IOTA",
        "STABLEHLO_DOT_GENERAL",
        "STABLEHLO_REDUCE_WINDOW",
        "STABLEHLO_SORT",
        "STABLEHLO_WHILE",
        "STABLEHLO_GATHER",
        "STABLEHLO_TRANSPOSE",
        "DILATE",
        "STABLEHLO_RNG_BIT_GENERATOR",
        "REDUCE_WINDOW",
        "STABLEHLO_COMPOSITE",
        "STABLEHLO_SHIFT_LEFT",
        "STABLEHLO_CBRT",
        "STABLEHLO_CASE",
    ];
    usize::try_from(code)
        .ok()
        .and_then(|i| NAMES.get(i))
        .copied()
}
