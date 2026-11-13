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

//! Graph building over a weight loader, and compiling + attaching the result.
//!
//! Every parameter is named after its tensor. Quantized GGUF projections stay
//! packed (`Op::DequantMatMul`, bytes uploaded straight from the file at
//! attach); everything else is widened to f32 (or kept at half width, see
//! [`WeightStorage`]). Any [`WeightLoader`] works: the model reads GGUF, and
//! `rlx-liquid-codec` builds the same stacks over HF safetensors.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rlx_core::weight_loader::{GgufLoader, WeightLoader};
use rlx_flow::CompileProfile;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

pub use rlx_core::weight_storage::WeightStorage;

/// How dense weights are kept: `RLX_WEIGHT_STORAGE` when set, else f32.
///
/// `auto` keeps the F16 LM / mmproj at 2 B/elem (lossless, half the memory)
/// and is verified on CUDA and Vulkan, but not the default: on x86 the CPU
/// f16-weight matmul (`rlx_cpu::blas::sgemm_half_rhs_widened`) segfaults
/// inside OpenBLAS here, and on ROCm hipRTC fails to compile the backend's
/// `cast_f32_to_half` kernel (integrated RDNA 3).
pub fn default_storage() -> WeightStorage {
    WeightStorage::from_env().unwrap_or(WeightStorage::F32)
}

/// A graph under construction plus the weights it reads.
pub struct Gb<'a> {
    pub g: Graph,
    pub params: HashMap<String, Vec<f32>>,
    /// Packed GGUF params to upload at attach: param name, tensor, and the
    /// tensor's byte range for a row slice (`None` = the whole tensor).
    pub packed: Vec<(String, String, Option<std::ops::Range<usize>>)>,
    /// Dense half-precision weights kept at 2 B/elem: name -> (bytes, dtype).
    pub typed: HashMap<String, (Vec<u8>, DType)>,
    pub w: &'a mut dyn WeightLoader,
    storage: WeightStorage,
    /// Keep GGUF quants packed (`DequantMatMul`); off widens them to dense.
    packed_ok: bool,
    shared: HashMap<String, NodeId>,
}

impl<'a> Gb<'a> {
    pub fn new(name: &str, w: &'a mut dyn WeightLoader, storage: WeightStorage) -> Self {
        Self {
            g: Graph::new(name),
            params: HashMap::new(),
            packed: Vec::new(),
            typed: HashMap::new(),
            w,
            storage,
            packed_ok: true,
            shared: HashMap::new(),
        }
    }

    /// Widen packed quants to dense weights (for backends without packed
    /// parameters).
    pub fn dense_only(mut self, dense: bool) -> Self {
        self.packed_ok = !dense;
        self
    }

    /// A host-made f32 parameter.
    pub fn constant(&mut self, name: &str, data: Vec<f32>, shape: &[usize]) -> NodeId {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>(), "{name}");
        let id = self.g.param(name, Shape::new(shape, DType::F32));
        self.params.insert(name.to_string(), data);
        id
    }

    pub fn input(&mut self, name: &str, shape: &[usize]) -> NodeId {
        self.g.input(name, Shape::new(shape, DType::F32))
    }

    /// A tensor widened to f32, row-major in safetensors order (outermost
    /// dimension first).
    pub fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.w
            .take(key)
            .with_context(|| format!("lfm3-audio: tensor {key}"))
    }

    /// A tensor as an f32 parameter of `shape` (same element count).
    pub fn tensor(&mut self, key: &str, shape: &[usize]) -> Result<NodeId> {
        let (data, have) = self.take(key)?;
        if data.len() != shape.iter().product::<usize>() {
            return Err(anyhow!("{key}: {have:?} does not fit {shape:?}"));
        }
        Ok(self.constant(key, data, shape))
    }

    /// A host-made f32 parameter registered once per name and reused.
    pub fn shared(
        &mut self,
        name: &str,
        shape: &[usize],
        make: impl FnOnce() -> Vec<f32>,
    ) -> NodeId {
        if let Some(&id) = self.shared.get(name) {
            return id;
        }
        let id = self.constant(name, make(), shape);
        self.shared.insert(name.to_string(), id);
        id
    }

    fn zero_beta(&mut self, n: usize) -> NodeId {
        self.shared(&format!("lfm3a.zero_beta.{n}"), &[n], || vec![0.0; n])
    }

    /// RMSNorm over the last axis with the gain stored at `key`.
    pub fn rms(&mut self, x: NodeId, key: &str, eps: f32) -> Result<NodeId> {
        let (gain, _) = self.take(key)?;
        let n = gain.len();
        let gamma = self.constant(key, gain, &[n]);
        let beta = self.zero_beta(n);
        Ok(self.g.rms_norm(x, gamma, beta, eps))
    }

    /// RMSNorm over the last axis with a unit gain.
    pub fn rms_unit(&mut self, x: NodeId, n: usize, eps: f32) -> NodeId {
        let gamma = self.shared(&format!("lfm3a.unit_gain.{n}"), &[n], || vec![1.0; n]);
        let beta = self.zero_beta(n);
        self.g.rms_norm(x, gamma, beta, eps)
    }

    /// `x @ Wᵀ` for the linear stored at `key` (`[out, in]` in safetensors
    /// order): packed GGUF quants become `DequantMatMul`, everything else a
    /// dense f32 `MatMul`. `x` is `[rows, in]`.
    pub fn linear(&mut self, x: NodeId, key: &str, rows: usize) -> Result<NodeId> {
        if let Some((scheme, shape)) = self.w.packed_meta(key).filter(|_| self.packed_ok) {
            let out = shape[0];
            let nbytes = match self.w.tensor_bytes_borrowed(key) {
                Some(b) => b.len(),
                None => {
                    let mut buf = Vec::new();
                    read_tensor_bytes(self.w, key, &mut buf)?;
                    buf.len()
                }
            };
            let w = self.g.param(key, Shape::new(&[nbytes], DType::U8));
            self.packed.push((key.to_string(), key.to_string(), None));
            return Ok(self.g.add_node(
                Op::DequantMatMul { scheme },
                vec![x, w],
                Shape::new(&[rows, out], DType::F32),
            ));
        }
        let (data, shape) = self
            .w
            .take_transposed(key)
            .with_context(|| format!("lfm3-audio: linear {key}"))?;
        // An F16 / BF16 checkpoint stays at its own width under `Auto` —
        // lossless, and half the device memory of widening (the F16 LM is
        // ~7 GB as f32).
        let dt = self.storage.resolve(self.w.stored_dtype(key));
        if matches!(dt, DType::F16 | DType::BF16) {
            let w = self.g.param(key, Shape::new(&shape, dt));
            let bytes = rlx_core::weight_storage::encode_f32(&data, dt);
            self.typed.insert(key.to_string(), (bytes, dt));
            return Ok(self.g.mm(x, w));
        }
        let w = self.constant(key, data, &shape);
        Ok(self.g.mm(x, w))
    }

    /// Whether `key` will load as a packed quant in this graph.
    pub fn is_packed(&self, key: &str) -> bool {
        self.packed_ok && self.w.packed_meta(key).is_some()
    }

    /// Bytes per output row of the packed tensor `key` (rows are contiguous).
    fn packed_row_bytes(&self, key: &str) -> Result<(rlx_ir::quant::QuantScheme, usize, usize)> {
        let (scheme, shape) = self
            .w
            .packed_meta(key)
            .ok_or_else(|| anyhow!("{key}: not a packed quant"))?;
        let k = shape[shape.len() - 1];
        let dtype = self
            .w
            .gguf_file()
            .and_then(|f| f.get(key))
            .map(|t| t.dtype)
            .ok_or_else(|| anyhow!("{key}: not in the GGUF"))?;
        let row_bytes = rlx_gguf::bytes_for_public(dtype, k)
            .ok_or_else(|| anyhow!("{key}: {dtype:?} rows of {k} are not whole blocks"))?;
        Ok((scheme, shape.iter().rev().skip(1).product(), row_bytes))
    }

    /// `x @ W[r0..r0+n]ᵀ` over output rows `r0..r0 + n` of the packed tensor
    /// `key`, as its own param over that byte range. `x` is `[rows, in]`.
    pub fn linear_rows(
        &mut self,
        x: NodeId,
        key: &str,
        rows: usize,
        r0: usize,
        n: usize,
    ) -> Result<NodeId> {
        let (scheme, _, row_bytes) = self.packed_row_bytes(key)?;
        let name = format!("{key}#rows{r0}");
        let w = self.g.param(&name, Shape::new(&[n * row_bytes], DType::U8));
        self.packed.push((
            name,
            key.to_string(),
            Some(r0 * row_bytes..(r0 + n) * row_bytes),
        ));
        Ok(self.g.add_node(
            Op::DequantMatMul { scheme },
            vec![x, w],
            Shape::new(&[rows, n], DType::F32),
        ))
    }

    /// [`linear`](Self::linear) over output rows in slices of at most
    /// `max_out` (outputs concatenated). For a packed weight each slice is its
    /// own param over a byte range of the tensor — quant rows are contiguous —
    /// which keeps a backend's per-weight dequant scratch to one slice: Metal
    /// sizes it to the largest packed weight, and the 128000-row tied head
    /// would make that ~1 GB of f32 per graph.
    pub fn linear_sliced(
        &mut self,
        x: NodeId,
        key: &str,
        rows: usize,
        max_out: usize,
    ) -> Result<NodeId> {
        if !self.is_packed(key) {
            return self.linear(x, key, rows);
        }
        let (_, out, _) = self.packed_row_bytes(key)?;
        if out <= max_out {
            return self.linear(x, key, rows);
        }
        let mut parts = Vec::new();
        let mut r0 = 0;
        while r0 < out {
            let n = max_out.min(out - r0);
            parts.push(self.linear_rows(x, key, rows, r0, n)?);
            r0 += n;
        }
        Ok(self.g.concat_(parts, 1))
    }

    /// [`linear`](Self::linear) plus the bias stored at `bias_key`.
    pub fn linear_bias(
        &mut self,
        x: NodeId,
        key: &str,
        bias_key: &str,
        rows: usize,
    ) -> Result<NodeId> {
        let y = self.linear(x, key, rows)?;
        let (b, _) = self.take(bias_key)?;
        let n = b.len();
        let b = self.constant(bias_key, b, &[1, n]);
        Ok(self.g.add(y, b))
    }
}

/// Whether `device` takes packed GGUF parameters. WebGL uploads f32 textures
/// only, so its graphs get dense weights.
pub fn dense_only(device: Device) -> bool {
    matches!(device, Device::OpenGl)
}

/// Compile options shared by every graph here: the packed-GGUF profile (it
/// keeps K-quant `DequantMatMul` nodes away from fusion passes that assume
/// dense f32 weights).
fn compile_opts(device: Device) -> rlx_runtime::CompileOptions {
    rlx_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &CompileProfile::qwen3_prefill(),
        device,
    )
}

/// The device a packed-GGUF graph actually runs on (wgpu/Vulkan may be routed
/// to the host by the shared `RLX_PACKED_GGUF_*_HOST` settings).
pub fn exec_device(device: Device) -> Device {
    rlx_core::flow_bridge::packed_gguf_execution_device(device)
}

/// Compile `gb`'s graph for `device` and upload its weights.
///
/// `scope` names the weight file: compiles under one scope share each weight's
/// storage where the backend can (Metal on unified memory; CUDA with
/// `RLX_CUDA_SHARED_PARAMS=1`), so the LM's decode and prefill graphs hold one
/// copy of the model, not two. Host-side f32 params are freed one by one as
/// they are uploaded.
pub fn compile(gb: Gb<'_>, device: Device, scope: &Path) -> Result<CompiledGraph> {
    let Gb {
        g,
        params,
        packed,
        typed,
        w,
        ..
    } = gb;
    let device = exec_device(device);
    let opts = compile_opts(device);
    let label = format!("lfm3-audio:{}", scope.display());
    if crate::diag::timing() {
        let (mut act, mut par) = (0usize, 0usize);
        for n in g.nodes() {
            let b = n.shape.size_bytes().unwrap_or(0);
            if matches!(n.op, Op::Param { .. }) {
                par += b;
            } else {
                act += b;
            }
        }
        crate::diag::trace(&format!(
            "built {}: {} nodes, params {} MiB, activations {} MiB before reuse",
            g.name,
            g.nodes().len(),
            par >> 20,
            act >> 20
        ));
    }
    rlx_runtime::with_param_sharing_scope(&label, || {
        let mut compiled = rlx_core::flow_bridge::packed_gguf_compile_guard(device, || {
            Session::new(device).compile_with(g, &opts)
        });
        crate::diag::trace("compiled");
        for (name, data) in params {
            compiled.set_param(&name, &data);
        }
        crate::diag::trace("f32 params uploaded");
        for (name, (bytes, dtype)) in typed {
            compiled.set_param_typed(&name, &bytes, dtype);
        }
        // Whole tensors are `pread` into one reused scratch. Slices are copied
        // straight from the file mapping: clean file-backed pages do not count
        // toward the process footprint (iOS's limit), where reading the whole
        // tensor into the scratch first would (the tied head is ~215 MB).
        let mut scratch = Vec::new();
        let mut held: Option<&str> = None;
        for (name, key, range) in &packed {
            if let Some(r) = range
                && let Some(mapped) = w.tensor_bytes_borrowed(key)
            {
                compiled.set_param_typed(name, &mapped[r.clone()], DType::U8);
                continue;
            }
            if held != Some(key.as_str()) {
                read_tensor_bytes(w, key, &mut scratch)?;
                held = Some(key.as_str());
            }
            let bytes = match range {
                Some(r) => &scratch[r.clone()],
                None => &scratch[..],
            };
            compiled.set_param_typed(name, bytes, DType::U8);
        }
        crate::diag::trace("packed params uploaded");
        Ok(compiled)
    })
}

/// The stored bytes of tensor `key` into `buf`: `pread` when the loader
/// streams, else a copy from its mapping.
pub fn read_tensor_bytes(w: &dyn WeightLoader, key: &str, buf: &mut Vec<u8>) -> Result<()> {
    if !w.read_tensor_bytes_into(key, buf)? {
        *buf = w
            .tensor_bytes_borrowed(key)
            .ok_or_else(|| anyhow!("lfm3-audio: {key}: bytes unavailable"))?
            .to_vec();
    }
    Ok(())
}

/// Open a GGUF file as a loader.
pub fn open(path: &Path) -> Result<GgufLoader> {
    GgufLoader::from_file(
        path.to_str()
            .ok_or_else(|| anyhow!("non-UTF-8 path {}", path.display()))?,
    )
    .with_context(|| format!("open {}", path.display()))
}
