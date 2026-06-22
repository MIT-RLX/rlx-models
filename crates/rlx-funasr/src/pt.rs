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

//! Native reader for PyTorch `model.pt` (the modern zip format): a minimal
//! pickle virtual machine recovers the `state_dict` tensor metadata
//! (`_rebuild_tensor_v2`), and the raw storages are read from the `data/`
//! entries and converted to `f32`. No Python, no torch dependency.

use std::collections::HashMap;
use std::io::Read;

use anyhow::{Context, Result, bail, ensure};
use half::{bf16, f16};
use zip::ZipArchive;

/// A loaded `state_dict`: tensor name → (row-major `f32` data, shape).
pub struct StateDict {
    /// Tensor name → (row-major `f32` data, shape).
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl StateDict {
    /// Load a `.pt`/`.pth`/`.bin` checkpoint, auto-detecting the modern (zip)
    /// or legacy (pre-1.6, non-zip) torch format.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        match ZipArchive::new(file) {
            Ok(zip) => Self::load_zip(zip, path),
            Err(_) => Self::load_legacy(path),
        }
    }

    fn load_zip(mut zip: ZipArchive<std::fs::File>, _path: &std::path::Path) -> Result<Self> {
        // Locate <prefix>data.pkl
        let mut pkl_name = None;
        for i in 0..zip.len() {
            let name = zip.by_index(i)?.name().to_string();
            if name.ends_with("data.pkl") {
                pkl_name = Some(name);
                break;
            }
        }
        let pkl_name = pkl_name.context("no data.pkl in checkpoint")?;
        let prefix = pkl_name.strip_suffix("data.pkl").unwrap_or("").to_string();

        let pkl_bytes = {
            let mut e = zip.by_name(&pkl_name)?;
            let mut b = Vec::new();
            e.read_to_end(&mut b)?;
            b
        };

        let root = Unpickler::new(&pkl_bytes).run()?;
        let dict = find_state_dict(&root).context("no tensor dictionary found in pickle")?;

        let mut tensors = HashMap::new();
        for (k, v) in dict {
            if let Value::Tensor(meta) = v {
                let storage_path = format!("{prefix}data/{}", meta.storage.key);
                let raw = {
                    let mut e = zip
                        .by_name(&storage_path)
                        .with_context(|| format!("missing storage {storage_path}"))?;
                    let mut b = Vec::new();
                    e.read_to_end(&mut b)?;
                    b
                };
                let data = materialize(&raw, &meta)?;
                tensors.insert(k, (data, meta.size.iter().map(|&x| x as usize).collect()));
            }
        }
        ensure!(!tensors.is_empty(), "checkpoint contained no tensors");
        Ok(Self { tensors })
    }

    /// Legacy (pre-1.6) format: five sequential pickles (magic, protocol,
    /// sys_info, the state dict with persistent storage refs, the storage-key
    /// list), then each storage as `[int64 numel][numel·elsize raw bytes]`,
    /// written in the order of the key list.
    fn load_legacy(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let mut up = Unpickler::new(&bytes);
        let _magic = up.load_one().context("legacy magic")?;
        let _proto = up.load_one().context("legacy protocol")?;
        let _sys = up.load_one().context("legacy sys_info")?;
        let main = up.load_one().context("legacy state dict")?;
        let keys = up.load_one().context("legacy storage keys")?;
        let storage_keys: Vec<String> = match keys {
            Value::List(v) | Value::Tuple(v) => v
                .into_iter()
                .filter_map(|x| if let Value::Str(s) = x { Some(s) } else { None })
                .collect(),
            _ => bail!("legacy: storage key list is not a list"),
        };

        let dict = find_state_dict(&main).context("no tensor dictionary in legacy pickle")?;
        // map storage key → element dtype (from the tensors that reference it)
        let mut dtype_of: HashMap<String, TorchDtype> = HashMap::new();
        for (_, v) in &dict {
            if let Value::Tensor(meta) = v {
                dtype_of.insert(meta.storage.key.clone(), meta.storage.dtype);
            }
        }

        // sequentially read the storage blocks from the current position
        let mut pos = up.pos;
        let mut storages: HashMap<String, &[u8]> = HashMap::new();
        for key in &storage_keys {
            let dt = *dtype_of
                .get(key)
                .with_context(|| format!("legacy: storage {key} has no referencing tensor"))?;
            ensure!(
                pos + 8 <= bytes.len(),
                "legacy: truncated before storage {key}"
            );
            let numel = i64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            let nbytes = numel * dt.elsize();
            ensure!(
                pos + nbytes <= bytes.len(),
                "legacy: truncated storage {key}"
            );
            storages.insert(key.clone(), &bytes[pos..pos + nbytes]);
            pos += nbytes;
        }

        let mut tensors = HashMap::new();
        for (name, v) in dict {
            if let Value::Tensor(meta) = v {
                let raw = storages
                    .get(&meta.storage.key)
                    .with_context(|| format!("legacy: missing storage {}", meta.storage.key))?;
                let data = materialize(raw, &meta)?;
                tensors.insert(
                    name,
                    (data, meta.size.iter().map(|&x| x as usize).collect()),
                );
            }
        }
        ensure!(
            !tensors.is_empty(),
            "legacy checkpoint contained no tensors"
        );
        Ok(Self { tensors })
    }
}

/// Walk the pickled object to find the dict that maps names → tensors.
fn find_state_dict(v: &Value) -> Option<Vec<(String, Value)>> {
    if let Value::Dict(items) = v {
        // direct: at least one tensor value
        let has_tensor = items.iter().any(|(_, val)| matches!(val, Value::Tensor(_)));
        if has_tensor {
            let mut out = Vec::new();
            for (k, val) in items {
                if let Value::Str(s) = k {
                    out.push((s.clone(), val.clone()));
                }
            }
            if !out.is_empty() {
                return Some(out);
            }
        }
        // nested: look under common wrapper keys
        for (k, val) in items {
            if let Value::Str(s) = k {
                if matches!(
                    s.as_str(),
                    "state_dict" | "model" | "model_state_dict" | "module"
                ) {
                    if let Some(d) = find_state_dict(val) {
                        return Some(d);
                    }
                }
            }
        }
        // fall back: any nested dict that resolves
        for (_, val) in items {
            if let Some(d) = find_state_dict(val) {
                return Some(d);
            }
        }
    }
    None
}

#[derive(Clone, Debug)]
struct StorageRef {
    dtype: TorchDtype,
    key: String,
}

#[derive(Clone, Debug)]
struct TensorMeta {
    storage: StorageRef,
    offset: i64,
    size: Vec<i64>,
    stride: Vec<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TorchDtype {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    U8,
    I8,
    Bool,
}

impl TorchDtype {
    fn from_storage_name(name: &str) -> Option<Self> {
        Some(match name {
            "DoubleStorage" => Self::F64,
            "FloatStorage" => Self::F32,
            "HalfStorage" => Self::F16,
            "BFloat16Storage" => Self::BF16,
            "LongStorage" => Self::I64,
            "IntStorage" => Self::I32,
            "ShortStorage" => Self::I16,
            "ByteStorage" => Self::U8,
            "CharStorage" => Self::I8,
            "BoolStorage" => Self::Bool,
            _ => return None,
        })
    }
    fn elsize(self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::U8 | Self::I8 | Self::Bool => 1,
        }
    }
}

/// Convert a raw storage buffer + tensor view to contiguous row-major `f32`.
fn materialize(raw: &[u8], meta: &TensorMeta) -> Result<Vec<f32>> {
    let dt = meta.storage.dtype;
    let es = dt.elsize();
    let numel: usize = meta.size.iter().map(|&x| x as usize).product();
    let read1 = |idx: usize| -> f32 {
        let off = idx * es;
        let b = &raw[off..off + es];
        match dt {
            TorchDtype::F64 => f64::from_le_bytes(b.try_into().unwrap()) as f32,
            TorchDtype::F32 => f32::from_le_bytes(b.try_into().unwrap()),
            TorchDtype::F16 => f16::from_le_bytes([b[0], b[1]]).to_f32(),
            TorchDtype::BF16 => bf16::from_le_bytes([b[0], b[1]]).to_f32(),
            TorchDtype::I64 => i64::from_le_bytes(b.try_into().unwrap()) as f32,
            TorchDtype::I32 => i32::from_le_bytes(b.try_into().unwrap()) as f32,
            TorchDtype::I16 => i16::from_le_bytes([b[0], b[1]]) as f32,
            TorchDtype::U8 => b[0] as f32,
            TorchDtype::I8 => b[0] as i8 as f32,
            TorchDtype::Bool => (b[0] != 0) as i32 as f32,
        }
    };

    // contiguous fast path
    let contig = contiguous_strides(&meta.size);
    let mut out = vec![0.0f32; numel];
    if meta.stride == contig {
        let base = meta.offset as usize;
        ensure!(
            (base + numel) * es <= raw.len(),
            "storage too small for tensor view"
        );
        for (i, o) in out.iter_mut().enumerate() {
            *o = read1(base + i);
        }
    } else {
        // generic strided gather
        let ndim = meta.size.len();
        let mut idx = vec![0i64; ndim];
        for o in out.iter_mut() {
            let mut flat = meta.offset;
            for d in 0..ndim {
                flat += idx[d] * meta.stride[d];
            }
            ensure!(
                (flat as usize) * es + es <= raw.len(),
                "strided index out of range"
            );
            *o = read1(flat as usize);
            // increment multi-index (row-major)
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < meta.size[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }
    Ok(out)
}

fn contiguous_strides(size: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; size.len()];
    for i in (0..size.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * size[i + 1];
    }
    s
}

// ── minimal pickle VM ─────────────────────────────────────────────────

#[derive(Clone, Debug)]
#[allow(dead_code)] // several scalar variants are parsed but not inspected
enum Value {
    None_,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String, String),
    Storage(StorageRef),
    Tensor(TensorMeta),
    Mark,
}

struct Unpickler<'a> {
    buf: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    memo: HashMap<u64, Value>,
}

impl<'a> Unpickler<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            stack: Vec::new(),
            memo: HashMap::new(),
        }
    }

    fn u8(&mut self) -> Result<u8> {
        let b = *self.buf.get(self.pos).context("pickle: unexpected EOF")?;
        self.pos += 1;
        Ok(b)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.pos + n <= self.buf.len(), "pickle: truncated");
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn line(&mut self) -> Result<String> {
        let start = self.pos;
        while self.pos < self.buf.len() && self.buf[self.pos] != b'\n' {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.buf[start..self.pos])?.to_string();
        self.pos += 1; // skip newline
        Ok(s)
    }
    fn pop(&mut self) -> Result<Value> {
        self.stack.pop().context("pickle: stack underflow")
    }
    fn pop_to_mark(&mut self) -> Result<Vec<Value>> {
        let mut items = Vec::new();
        loop {
            match self.stack.pop() {
                Some(Value::Mark) => break,
                Some(v) => items.push(v),
                None => bail!("pickle: mark not found"),
            }
        }
        items.reverse();
        Ok(items)
    }

    fn run(mut self) -> Result<Value> {
        self.load_one()
    }

    /// Interpret one pickle (until `STOP`), leaving `self.pos` just past it.
    /// Stack and memo are reset so the instance can be reused for the next
    /// pickle in a legacy multi-pickle stream.
    fn load_one(&mut self) -> Result<Value> {
        self.stack.clear();
        self.memo.clear();
        loop {
            let op = self.u8()?;
            match op {
                0x80 => {
                    self.u8()?; // PROTO version
                }
                0x95 => {
                    self.take(8)?; // FRAME length
                }
                b'.' => return self.pop(), // STOP
                b'(' => self.stack.push(Value::Mark),
                b'0' => {
                    self.pop()?;
                } // POP
                b'1' => {
                    self.pop_to_mark()?;
                } // POP_MARK
                b'N' => self.stack.push(Value::None_),
                0x88 => self.stack.push(Value::Bool(true)),
                0x89 => self.stack.push(Value::Bool(false)),
                b'J' => {
                    let v = i32::from_le_bytes(self.take(4)?.try_into().unwrap());
                    self.stack.push(Value::Int(v as i64));
                }
                b'K' => {
                    let v = self.u8()?;
                    self.stack.push(Value::Int(v as i64));
                }
                b'M' => {
                    let v = u16::from_le_bytes(self.take(2)?.try_into().unwrap());
                    self.stack.push(Value::Int(v as i64));
                }
                b'I' => {
                    let s = self.line()?;
                    self.stack.push(Value::Int(s.trim().parse().unwrap_or(0)));
                }
                b'L' => {
                    let s = self.line()?;
                    let s = s.trim().trim_end_matches('L');
                    self.stack.push(Value::Int(s.parse().unwrap_or(0)));
                }
                0x8a => {
                    // LONG1
                    let n = self.u8()? as usize;
                    let bytes = self.take(n)?;
                    self.stack.push(Value::Int(le_signed(bytes)));
                }
                0x8b => {
                    // LONG4
                    let n = u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize;
                    let bytes = self.take(n)?;
                    self.stack.push(Value::Int(le_signed(bytes)));
                }
                b'G' => {
                    let v = f64::from_be_bytes(self.take(8)?.try_into().unwrap());
                    self.stack.push(Value::Float(v));
                }
                b'F' => {
                    let s = self.line()?;
                    self.stack
                        .push(Value::Float(s.trim().parse().unwrap_or(0.0)));
                }
                // strings
                b'X' => {
                    let n = u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(s));
                }
                0x8c => {
                    let n = self.u8()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(s));
                }
                0x8d => {
                    let n = u64::from_le_bytes(self.take(8)?.try_into().unwrap()) as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(s));
                }
                b'U' => {
                    let n = self.u8()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(s));
                }
                b'T' => {
                    let n = u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(s));
                }
                b'S' => {
                    let s = self.line()?;
                    let s = s.trim().trim_matches(|c| c == '\'' || c == '"').to_string();
                    self.stack.push(Value::Str(s));
                }
                b'V' => {
                    let s = self.line()?;
                    self.stack.push(Value::Str(s));
                }
                // bytes
                b'C' => {
                    // SHORT_BINBYTES
                    let n = self.u8()? as usize;
                    let b = self.take(n)?.to_vec();
                    self.stack.push(Value::Bytes(b));
                }
                b'B' => {
                    // BINBYTES
                    let n = u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize;
                    let b = self.take(n)?.to_vec();
                    self.stack.push(Value::Bytes(b));
                }
                // collections
                b']' => self.stack.push(Value::List(Vec::new())),
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b'}' => self.stack.push(Value::Dict(Vec::new())),
                b't' => {
                    let items = self.pop_to_mark()?;
                    self.stack.push(Value::Tuple(items));
                }
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a]));
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b]));
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b, c]));
                }
                b'l' => {
                    let items = self.pop_to_mark()?;
                    self.stack.push(Value::List(items));
                }
                b'a' => {
                    let v = self.pop()?;
                    if let Some(Value::List(l)) = self.stack.last_mut() {
                        l.push(v);
                    }
                }
                b'e' => {
                    let items = self.pop_to_mark()?;
                    if let Some(Value::List(l)) = self.stack.last_mut() {
                        l.extend(items);
                    }
                }
                b'd' => {
                    let items = self.pop_to_mark()?;
                    let mut d = Vec::new();
                    for pair in items.chunks(2) {
                        if pair.len() == 2 {
                            d.push((pair[0].clone(), pair[1].clone()));
                        }
                    }
                    self.stack.push(Value::Dict(d));
                }
                b's' => {
                    let val = self.pop()?;
                    let key = self.pop()?;
                    if let Some(Value::Dict(d)) = self.stack.last_mut() {
                        d.push((key, val));
                    }
                }
                b'u' => {
                    let items = self.pop_to_mark()?;
                    if let Some(Value::Dict(d)) = self.stack.last_mut() {
                        for pair in items.chunks(2) {
                            if pair.len() == 2 {
                                d.push((pair[0].clone(), pair[1].clone()));
                            }
                        }
                    }
                }
                0x8f => self.stack.push(Value::List(Vec::new())), // EMPTY_SET
                0x90 => {
                    // ADDITEMS
                    let items = self.pop_to_mark()?;
                    if let Some(Value::List(l)) = self.stack.last_mut() {
                        l.extend(items);
                    }
                }
                0x91 => {
                    // FROZENSET
                    let items = self.pop_to_mark()?;
                    self.stack.push(Value::List(items));
                }
                // globals
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(Value::Global(
                        module.trim().to_string(),
                        name.trim().to_string(),
                    ));
                }
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let (m, n) = (str_of(&module), str_of(&name));
                    self.stack.push(Value::Global(m, n));
                }
                // reduce / build / newobj
                b'R' => {
                    let args = self.pop()?;
                    let func = self.pop()?;
                    self.stack.push(self.reduce(func, args)?);
                }
                0x81 => {
                    // NEWOBJ
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    self.stack.push(self.reduce(cls, args)?);
                }
                0x92 => {
                    // NEWOBJ_EX
                    let _kwargs = self.pop()?;
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    self.stack.push(self.reduce(cls, args)?);
                }
                b'b' => {
                    // BUILD: merge state dict into the object on the stack
                    let state = self.pop()?;
                    let obj = self.pop()?;
                    self.stack.push(build(obj, state));
                }
                b'i' | b'o' => {
                    // INST / OBJ — treat as opaque
                    let _ = self.pop_to_mark();
                    self.stack.push(Value::None_);
                }
                // persistent ids
                b'Q' => {
                    let pid = self.pop()?;
                    self.stack.push(persistent_load(pid)?);
                }
                b'P' => {
                    let _ = self.line()?;
                    self.stack.push(Value::None_);
                }
                // memo
                0x94 => {
                    let v = self.stack.last().cloned().unwrap_or(Value::None_);
                    let id = self.memo.len() as u64;
                    self.memo.insert(id, v);
                }
                b'q' => {
                    let id = self.u8()? as u64;
                    let v = self.stack.last().cloned().unwrap_or(Value::None_);
                    self.memo.insert(id, v);
                }
                b'r' => {
                    let id = u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as u64;
                    let v = self.stack.last().cloned().unwrap_or(Value::None_);
                    self.memo.insert(id, v);
                }
                b'p' => {
                    let s = self.line()?;
                    let id: u64 = s.trim().parse().unwrap_or(0);
                    let v = self.stack.last().cloned().unwrap_or(Value::None_);
                    self.memo.insert(id, v);
                }
                b'h' => {
                    let id = self.u8()? as u64;
                    let v = self
                        .memo
                        .get(&id)
                        .cloned()
                        .context("pickle: bad memo get")?;
                    self.stack.push(v);
                }
                b'j' => {
                    let id = u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as u64;
                    let v = self
                        .memo
                        .get(&id)
                        .cloned()
                        .context("pickle: bad memo get")?;
                    self.stack.push(v);
                }
                b'g' => {
                    let s = self.line()?;
                    let id: u64 = s.trim().parse().unwrap_or(0);
                    let v = self
                        .memo
                        .get(&id)
                        .cloned()
                        .context("pickle: bad memo get")?;
                    self.stack.push(v);
                }
                b'2' => {
                    let v = self.stack.last().cloned().context("pickle: dup empty")?;
                    self.stack.push(v);
                }
                other => bail!(
                    "pickle: unsupported opcode 0x{other:02x} ('{}')",
                    other as char
                ),
            }
        }
    }

    fn reduce(&self, func: Value, args: Value) -> Result<Value> {
        let argv = match args {
            Value::Tuple(v) => v,
            other => vec![other],
        };
        if let Value::Global(_module, name) = &func {
            match name.as_str() {
                "_rebuild_tensor_v2" | "_rebuild_tensor" => {
                    return rebuild_tensor(&argv);
                }
                "_rebuild_parameter" => {
                    // (tensor, requires_grad, backward_hooks)
                    return Ok(argv.into_iter().next().unwrap_or(Value::None_));
                }
                "OrderedDict" => return Ok(Value::Dict(Vec::new())),
                _ => {}
            }
        }
        // unknown callable: pass through first arg or None
        Ok(argv.into_iter().next().unwrap_or(Value::None_))
    }
}

fn build(obj: Value, state: Value) -> Value {
    match (obj, state) {
        (Value::Dict(mut d), Value::Dict(s)) => {
            d.extend(s);
            Value::Dict(d)
        }
        (obj, _) => obj,
    }
}

fn str_of(v: &Value) -> String {
    match v {
        Value::Str(s) => s.clone(),
        Value::Global(m, n) => format!("{m}.{n}"),
        _ => String::new(),
    }
}

fn le_signed(bytes: &[u8]) -> i64 {
    // Only the low 8 bytes fit an i64; longer pickled longs (e.g. the 10-byte
    // legacy magic number) are truncated — their value is never used.
    let mut v: i64 = 0;
    for (i, &b) in bytes.iter().enumerate().take(8) {
        v |= (b as i64) << (8 * i);
    }
    if bytes.len() <= 8 && !bytes.is_empty() && bytes[bytes.len() - 1] & 0x80 != 0 {
        let bits = 8 * bytes.len();
        if bits < 64 {
            v -= 1i64 << bits;
        }
    }
    v
}

/// `persistent_load(("storage", FloatStorage, key, location, numel))`.
fn persistent_load(pid: Value) -> Result<Value> {
    let items = match pid {
        Value::Tuple(v) | Value::List(v) => v,
        other => vec![other],
    };
    // find the storage_type Global and the key string
    let mut dtype = None;
    let mut key = None;
    for it in &items {
        match it {
            Value::Global(_m, n) => {
                if let Some(dt) = TorchDtype::from_storage_name(n) {
                    dtype = Some(dt);
                }
            }
            // first string after the "storage" tag is the storage key
            Value::Str(s) if key.is_none() && s != "storage" => key = Some(s.clone()),
            _ => {}
        }
    }
    let dtype = dtype.context("persistent_load: unknown storage dtype")?;
    let key = key.context("persistent_load: missing storage key")?;
    Ok(Value::Storage(StorageRef { dtype, key }))
}

/// `_rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, ...)`.
fn rebuild_tensor(args: &[Value]) -> Result<Value> {
    let storage = match args.first() {
        Some(Value::Storage(s)) => s.clone(),
        _ => bail!("_rebuild_tensor: first arg is not a storage"),
    };
    let offset = as_int(args.get(1)).unwrap_or(0);
    let size = as_int_vec(args.get(2));
    let stride = {
        let s = as_int_vec(args.get(3));
        if s.is_empty() && !size.is_empty() {
            contiguous_strides(&size)
        } else {
            s
        }
    };
    Ok(Value::Tensor(TensorMeta {
        storage,
        offset,
        size,
        stride,
    }))
}

fn as_int(v: Option<&Value>) -> Option<i64> {
    match v {
        Some(Value::Int(i)) => Some(*i),
        Some(Value::Bool(b)) => Some(*b as i64),
        _ => None,
    }
}

fn as_int_vec(v: Option<&Value>) -> Vec<i64> {
    match v {
        Some(Value::Tuple(items)) | Some(Value::List(items)) => {
            items.iter().filter_map(|x| as_int(Some(x))).collect()
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn le_signed_roundtrip() {
        assert_eq!(le_signed(&[0x05]), 5);
        assert_eq!(le_signed(&[0xfb]), -5);
        assert_eq!(le_signed(&[0x00, 0x01]), 256);
    }

    #[test]
    fn contiguous_strides_rowmajor() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn unpickle_simple_dict() {
        // pickle.dumps({"a": 1, "b": 2}, protocol=2)
        let data: &[u8] = &[
            0x80, 0x02, 0x7d, 0x71, 0x00, 0x28, 0x58, 0x01, 0x00, 0x00, 0x00, b'a', 0x71, 0x01,
            0x4b, 0x01, 0x58, 0x01, 0x00, 0x00, 0x00, b'b', 0x71, 0x02, 0x4b, 0x02, 0x75, 0x2e,
        ];
        let v = Unpickler::new(data).run().unwrap();
        match v {
            Value::Dict(d) => {
                assert_eq!(d.len(), 2);
                assert!(matches!(&d[0].0, Value::Str(s) if s == "a"));
                assert!(matches!(&d[0].1, Value::Int(1)));
            }
            _ => panic!("expected dict"),
        }
    }
}
