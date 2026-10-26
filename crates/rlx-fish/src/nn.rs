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

//! Graph building blocks shared by every fish network.
//!
//! All four fish transformers — the S2 slow backbone and fast decoder, the two
//! codec RVQ transformers and the codec encoder transformer, and the Drama 3
//! codebook tower — are the same pre-norm block with the same tensor names
//! (`attention.wqkv`, `feed_forward.w1/w2/w3`, …). They differ only in a few
//! switches ([`BlockSpec`]), so one builder serves all of them.
//!
//! Parameters are *declared* while the graph is built and *filled* after it is
//! compiled ([`ParamSpec`] / [`upload`]), one tensor at a time straight from
//! the checkpoint. The S2 backbone is 14.5 GB in f32; holding a host copy next
//! to the graph's own would double the peak.

use anyhow::{Context, Result, bail, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, NodeId, Op, RopeStyle, Shape};
use rlx_runtime::CompiledGraph;

const F32: DType = DType::F32;

/// Read access to checkpoint tensors as row-major f32.
pub trait TensorSource: Sync {
    /// The tensor and its shape.
    fn tensor(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>)>;

    /// The stored bytes, dtype and shape when the tensor is half precision
    /// (bf16/f16) — lets a half-precision graph param skip the f32 detour.
    fn half_bytes(&self, _key: &str) -> Option<(Vec<u8>, DType, Vec<usize>)> {
        None
    }
}

impl TensorSource for rlx_core::safetensors_checkpoint::SafetensorsCheckpoint {
    fn tensor(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.load_tensor_f32(key)
    }

    fn half_bytes(&self, key: &str) -> Option<(Vec<u8>, DType, Vec<usize>)> {
        let (bytes, dtype, shape) = self.tensor_raw(key).ok()?;
        let dtype = match dtype {
            safetensors::Dtype::BF16 => DType::BF16,
            safetensors::Dtype::F16 => DType::F16,
            _ => return None,
        };
        Some((bytes, dtype, shape))
    }
}

impl TensorSource for rlx_torch_ckpt::PtModel {
    fn tensor(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let t = rlx_torch_ckpt::PtModel::tensor(self, key)?;
        Ok((t.data, t.shape))
    }
}

/// How a declared parameter is filled.
pub enum Init {
    /// Host data computed at build time.
    Data(Vec<f32>),
    /// A checkpoint tensor, transformed on the way in.
    Tensor { key: String, op: TensorOp },
}

/// What happens to a checkpoint tensor before upload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorOp {
    /// As stored.
    Raw,
    /// `[out, in]` → `[in, out]`: an `nn.Linear` weight for `x @ W`.
    Transpose,
    /// `w + 1`: a zero-centered RMSNorm gain (`out · (1 + w)`).
    PlusOne,
}

/// One parameter of a built graph.
pub struct ParamSpec {
    pub name: String,
    pub shape: Vec<usize>,
    /// Storage dtype in the graph (matmul weights may be bf16/f16).
    pub dtype: DType,
    pub init: Init,
}

/// A graph under construction plus the parameters it declares.
pub struct Builder {
    pub g: Graph,
    pub params: Vec<ParamSpec>,
    /// Storage dtype of the checkpoint matrices [`Self::linear`] declares.
    /// Backends widen bf16/f16 weights and multiply in f32, so for a bf16
    /// checkpoint `BF16` gives the same products at half the bytes.
    pub matrix_dtype: DType,
    /// The backend's attention kernel takes K/V with fewer heads than Q
    /// (GQA) — no head expansion needed. See [`native_gqa`].
    pub native_gqa: bool,
}

/// Backends whose attention infers GQA from K's head count (CPU, Metal, wgpu).
/// Elsewhere [`block`] expands K/V (or, for one query row, regroups Q).
pub fn native_gqa(device: rlx_runtime::Device) -> bool {
    use rlx_runtime::Device;
    matches!(device, Device::Cpu | Device::Metal | Device::Gpu)
        && !rlx_ir::env::flag("RLX_FISH_GQA_EXPAND")
}

impl Builder {
    pub fn new(name: &str) -> Self {
        Self {
            g: Graph::new(name),
            params: Vec::new(),
            matrix_dtype: F32,
            native_gqa: false,
        }
    }

    /// [`Self::new`] storing matmul weights as `dtype`.
    pub fn with_matrix_dtype(name: &str, dtype: DType) -> Self {
        Self {
            matrix_dtype: dtype,
            ..Self::new(name)
        }
    }

    pub fn input(&mut self, name: &str, dims: &[usize]) -> NodeId {
        self.g.input(name, Shape::new(dims, F32))
    }

    fn declare(&mut self, name: String, shape: &[usize], dtype: DType, init: Init) -> NodeId {
        if let Some(id) = self.g.param_id(&name) {
            return id;
        }
        let id = self.g.param(name.clone(), Shape::new(shape, dtype));
        self.params.push(ParamSpec {
            name,
            shape: shape.to_vec(),
            dtype,
            init,
        });
        id
    }

    /// A parameter filled from host data.
    pub fn data(&mut self, name: &str, data: Vec<f32>, shape: &[usize]) -> NodeId {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>(), "{name}");
        self.declare(name.to_string(), shape, F32, Init::Data(data))
    }

    /// A checkpoint tensor (`shape` is the *uploaded* shape, after `op`).
    pub fn weight(&mut self, key: &str, shape: &[usize], op: TensorOp) -> NodeId {
        self.weight_as(key, shape, op, F32)
    }

    fn weight_as(&mut self, key: &str, shape: &[usize], op: TensorOp, dtype: DType) -> NodeId {
        let name = match op {
            TensorOp::Raw => key.to_string(),
            TensorOp::Transpose => format!("{key}#t"),
            TensorOp::PlusOne => format!("{key}#p1"),
        };
        self.declare(
            name,
            shape,
            dtype,
            Init::Tensor {
                key: key.to_string(),
                op,
            },
        )
    }

    /// `x @ Wᵀ` for an `nn.Linear` weight `[out, in]` stored under `key`.
    pub fn linear(&mut self, x: NodeId, key: &str, d_in: usize, d_out: usize) -> NodeId {
        let w = self.weight_as(key, &[d_in, d_out], TensorOp::Transpose, self.matrix_dtype);
        self.g.mm(x, w)
    }

    /// `linear` plus a `[d_out]` bias (broadcast over the leading axes).
    pub fn linear_bias(&mut self, x: NodeId, key: &str, d_in: usize, d_out: usize) -> NodeId {
        let y = self.linear(x, &format!("{key}.weight"), d_in, d_out);
        let b = self.weight(&format!("{key}.bias"), &[d_out], TensorOp::Raw);
        self.g.add(y, b)
    }

    /// A shared all-zero `[n]` vector (the `beta` of an RMSNorm).
    pub fn zeros(&mut self, n: usize) -> NodeId {
        self.data(&format!("fish.zero.{n}"), vec![0.0; n], &[n])
    }

    /// RMSNorm over the last axis with the gain stored under `key`.
    pub fn rms_norm(
        &mut self,
        x: NodeId,
        key: &str,
        n: usize,
        eps: f32,
        zero_centered: bool,
    ) -> NodeId {
        let op = if zero_centered {
            TensorOp::PlusOne
        } else {
            TensorOp::Raw
        };
        let w = self.weight(key, &[n], op);
        let z = self.zeros(n);
        self.g.rms_norm(x, w, z, eps)
    }

    /// RMSNorm applied to each `head_dim` slice of `x: [1, s, heads·head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn per_head_rms(
        &mut self,
        x: NodeId,
        key: &str,
        seq: usize,
        heads: usize,
        head_dim: usize,
        eps: f32,
        zero_centered: bool,
    ) -> NodeId {
        let r = self
            .g
            .reshape_(x, vec![(seq * heads) as i64, head_dim as i64]);
        let n = self.rms_norm(r, key, head_dim, eps, zero_centered);
        self.g
            .reshape_(n, vec![1, seq as i64, (heads * head_dim) as i64])
    }

    /// Broadcast `x` to `dims` (NumPy rules: size-1 axes repeat).
    pub fn expand(&mut self, x: NodeId, dims: &[usize]) -> NodeId {
        self.g.add_node(
            Op::Expand {
                target_shape: dims.iter().map(|&d| d as i64).collect(),
            },
            vec![x],
            Shape::new(dims, F32),
        )
    }

    /// Broadcast `kv` heads of `x: [1, s, kv·d]` to `kv·group` heads
    /// (`repeat_interleave` order: head `h` reads kv head `h / group`).
    pub fn repeat_kv(
        &mut self,
        x: NodeId,
        seq: usize,
        kv: usize,
        d: usize,
        group: usize,
    ) -> NodeId {
        if group == 1 {
            return x;
        }
        let (s, kv_i, d_i) = (seq as i64, kv as i64, d as i64);
        let r = self.g.reshape_(x, vec![1, s, kv_i, 1, d_i]);
        let e = self.g.add_node(
            Op::Expand {
                target_shape: vec![1, s, kv_i, group as i64, d_i],
            },
            vec![r],
            Shape::new(&[1, seq, kv, group, d], F32),
        );
        self.g.reshape_(e, vec![1, s, (kv * group) as i64 * d_i])
    }
}

/// Attention geometry of one block.
#[derive(Clone, Debug)]
pub struct AttnSpec {
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    /// Per-head RMSNorm on q and k (`attention.q_norm` / `k_norm`).
    pub qk_norm: bool,
    /// Leading channels of each head that RoPE rotates (`head_dim` = all).
    pub rope_dim: usize,
    pub rope_style: RopeStyle,
    /// `wqkv` rows are `[Q | Gate | K | V]` and the output is
    /// `wo(attn · sigmoid(gate))` (Drama 3 tower).
    pub output_gate: bool,
}

/// One pre-norm transformer block.
#[derive(Clone, Debug)]
pub struct BlockSpec {
    pub dim: usize,
    pub ffn: usize,
    pub eps: f32,
    pub attn: AttnSpec,
    /// `attention_layer_scale.gamma` / `ffn_layer_scale.gamma` (codec).
    pub layer_scale: bool,
    /// Norm gains scale by `1 + w` (Drama 3).
    pub zero_centered: bool,
}

/// What the block's queries attend to.
pub enum Keys {
    /// The block's own rows, under a structural mask.
    Own(MaskKind),
    /// `past ‖ own` rows: `past_k` / `past_v` are `[1, past, kv·d]` inputs and
    /// `mask` is `[1, past + seq]` (1 = attend) — a decode step over a bucket.
    Cached {
        past_k: NodeId,
        past_v: NodeId,
        past: usize,
        mask: NodeId,
    },
}

/// Output of [`block`]: the new residual stream and this block's K/V rows
/// (after RoPE, before GQA expansion) for a KV cache.
pub struct BlockOut {
    pub x: NodeId,
    pub k: NodeId,
    pub v: NodeId,
}

/// Emit one fish transformer block on `x: [1, seq, dim]`.
///
/// `prefix` ends in `.` (`layers.3.`); `cos` / `sin` are `[seq, rope_dim/2]`.
#[allow(clippy::too_many_arguments)]
pub fn block(
    b: &mut Builder,
    spec: &BlockSpec,
    prefix: &str,
    x: NodeId,
    seq: usize,
    cos: NodeId,
    sin: NodeId,
    keys: Keys,
) -> BlockOut {
    let a = &spec.attn;
    let (dim, eps, zc) = (spec.dim, spec.eps, spec.zero_centered);
    let q_w = a.n_head * a.head_dim;
    let kv_w = a.n_kv * a.head_dim;

    let xn = b.rms_norm(x, &format!("{prefix}attention_norm.weight"), dim, eps, zc);
    let qkv_w = if a.output_gate {
        2 * q_w + 2 * kv_w
    } else {
        q_w + 2 * kv_w
    };
    let qkv = b.linear(xn, &format!("{prefix}attention.wqkv.weight"), dim, qkv_w);
    let mut off = 0;
    let mut take = |b: &mut Builder, w: usize| {
        let n = b.g.narrow_(qkv, 2, off, w);
        off += w;
        n
    };
    let mut q = take(b, q_w);
    let gate = a.output_gate.then(|| take(b, q_w));
    let mut k = take(b, kv_w);
    let v = take(b, kv_w);

    if a.qk_norm {
        let (qn, kn) = (
            format!("{prefix}attention.q_norm.weight"),
            format!("{prefix}attention.k_norm.weight"),
        );
        q = b.per_head_rms(q, &qn, seq, a.n_head, a.head_dim, eps, zc);
        k = b.per_head_rms(k, &kn, seq, a.n_kv, a.head_dim, eps, zc);
    }
    if a.rope_dim == a.head_dim {
        q = b.g.rope_styled(q, cos, sin, a.head_dim, a.rope_style);
        k = b.g.rope_styled(k, cos, sin, a.head_dim, a.rope_style);
    } else {
        q =
            b.g.rope_n_styled(q, cos, sin, a.head_dim, a.rope_dim, a.rope_style);
        k =
            b.g.rope_n_styled(k, cos, sin, a.head_dim, a.rope_dim, a.rope_style);
    }

    let group = a.n_head / a.n_kv;
    let (nh, nkv, d) = (a.n_head as i64, a.n_kv as i64, a.head_dim as i64);
    // `[1, rows, heads, D]` views for the attention op.
    let heads = |b: &mut Builder, x: NodeId, rows: usize, h: i64| {
        b.g.reshape_(x, vec![1, rows as i64, h, d])
    };
    let q4 = heads(b, q, seq, nh);
    let shape4 = Shape::new(&[1, seq, a.n_head, a.head_dim], F32);
    let (k_all, v_all, total, mask) = match keys {
        Keys::Own(mask) => (k, v, seq, Err(mask)),
        Keys::Cached {
            past_k,
            past_v,
            past,
            mask,
        } => (
            b.g.concat_(vec![past_k, k], 1),
            b.g.concat_(vec![past_v, v], 1),
            past + seq,
            Ok(mask),
        ),
    };
    let attend =
        |b: &mut Builder, q: NodeId, k: NodeId, v: NodeId, heads: usize, shape: Shape| match mask {
            Err(kind) => b.g.attention_kind(q, k, v, heads, a.head_dim, kind, shape),
            Ok(m) => b.g.attention(q, k, v, m, heads, a.head_dim, shape),
        };
    let attn = if b.native_gqa || group == 1 {
        // The kernel reads each KV head for its `group` query heads (GQA
        // inferred from K's head count) — no expanded copy of the keys.
        let (k4, v4) = (heads(b, k_all, total, nkv), heads(b, v_all, total, nkv));
        attend(b, q4, k4, v4, a.n_head, shape4)
    } else if seq == 1 {
        // A backend without GQA, one query row: regroup its heads as `group`
        // query rows × `n_kv` heads (head h = kv·group + g), so each KV head
        // still serves its own query heads straight from the cache.
        let qg = b.g.reshape_(q, vec![1, nkv, group as i64, d]);
        let qg = b.g.transpose_(qg, vec![0, 2, 1, 3]);
        let (k4, v4) = (heads(b, k_all, total, nkv), heads(b, v_all, total, nkv));
        let o = attend(
            b,
            qg,
            k4,
            v4,
            a.n_kv,
            Shape::new(&[1, group, a.n_kv, a.head_dim], F32),
        );
        b.g.transpose_(o, vec![0, 2, 1, 3])
    } else {
        let kr = b.repeat_kv(k_all, total, a.n_kv, a.head_dim, group);
        let vr = b.repeat_kv(v_all, total, a.n_kv, a.head_dim, group);
        let (kr, vr) = (heads(b, kr, total, nh), heads(b, vr, total, nh));
        attend(b, q4, kr, vr, a.n_head, shape4)
    };
    let attn = b.g.reshape_(attn, vec![1, seq as i64, q_w as i64]);
    finish_block(b, spec, prefix, x, attn, gate, k, v)
}

/// Output gate, `wo`, residual and the SwiGLU half of [`block`].
#[allow(clippy::too_many_arguments)]
fn finish_block(
    b: &mut Builder,
    spec: &BlockSpec,
    prefix: &str,
    x: NodeId,
    attn: NodeId,
    gate: Option<NodeId>,
    k: NodeId,
    v: NodeId,
) -> BlockOut {
    let a = &spec.attn;
    let dim = spec.dim;
    let eps = spec.eps;
    let zc = spec.zero_centered;
    let q_w = a.n_head * a.head_dim;
    let attn = match gate {
        Some(gt) => {
            let s = b.g.sigmoid(gt);
            b.g.mul(attn, s)
        }
        None => attn,
    };
    let mut o = b.linear(attn, &format!("{prefix}attention.wo.weight"), q_w, dim);
    if spec.layer_scale {
        let gamma = b.weight(
            &format!("{prefix}attention_layer_scale.gamma"),
            &[dim],
            TensorOp::Raw,
        );
        o = b.g.mul(o, gamma);
    }
    let h = b.g.add(x, o);

    let hn = b.rms_norm(h, &format!("{prefix}ffn_norm.weight"), dim, eps, zc);
    let g1 = b.linear(
        hn,
        &format!("{prefix}feed_forward.w1.weight"),
        dim,
        spec.ffn,
    );
    let u = b.linear(
        hn,
        &format!("{prefix}feed_forward.w3.weight"),
        dim,
        spec.ffn,
    );
    let act = {
        let s = b.g.silu(g1);
        b.g.mul(s, u)
    };
    let mut d = b.linear(
        act,
        &format!("{prefix}feed_forward.w2.weight"),
        spec.ffn,
        dim,
    );
    if spec.layer_scale {
        let gamma = b.weight(
            &format!("{prefix}ffn_layer_scale.gamma"),
            &[dim],
            TensorOp::Raw,
        );
        d = b.g.mul(d, gamma);
    }
    BlockOut {
        x: b.g.add(h, d),
        k,
        v,
    }
}

/// RoPE tables `[positions, rot_dim/2]` the way fish computes them: inverse
/// frequencies and angles in f32 (`1 / base^(2i/rot_dim)`, `t · f`), and —
/// for the S2 backbone and the codec, whose `freqs_cis` buffers are bfloat16 —
/// cos/sin rounded to bf16. Matching that rounding is what keeps the decoders
/// on the reference's token path; the rotation itself runs in f32.
pub fn rope_tables(
    positions: &[usize],
    rot_dim: usize,
    base: f64,
    bf16: bool,
) -> (Vec<f32>, Vec<f32>) {
    let half = rot_dim / 2;
    let base = base as f32;
    let freqs: Vec<f32> = (0..half)
        .map(|i| 1.0f32 / base.powf((2 * i) as f32 / rot_dim as f32))
        .collect();
    let round = |v: f32| {
        if bf16 {
            half::bf16::from_f32(v).to_f32()
        } else {
            v
        }
    };
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);
    for &p in positions {
        for &f in &freqs {
            let a = p as f32 * f;
            cos.push(round(a.cos()));
            sin.push(round(a.sin()));
        }
    }
    (cos, sin)
}

/// Fill every declared parameter of `compiled` from `specs`, reading
/// checkpoint tensors from `src` one at a time.
pub fn upload(
    compiled: &mut CompiledGraph,
    specs: Vec<ParamSpec>,
    src: &dyn TensorSource,
) -> Result<()> {
    use rayon::prelude::*;
    // Prepare a few tensors at a time in parallel (read, transpose, convert),
    // then hand them over in order. Bounded so the host never holds more
    // than a handful of matrices next to the graph's copy.
    const BATCH: usize = 8;
    let mut specs = specs.into_iter().peekable();
    while specs.peek().is_some() {
        // A weight another graph of this model already placed (Metal's shared
        // weight store maps it into this graph too) needs no second upload.
        let batch: Vec<ParamSpec> = specs
            .by_ref()
            .filter(|p| !compiled.param_already_set(&p.name))
            .take(BATCH)
            .collect();
        if batch.is_empty() {
            continue;
        }
        let ready: Vec<Result<(String, Payload)>> = batch
            .into_par_iter()
            .map(|spec| prepare(spec, src))
            .collect();
        for r in ready {
            match r? {
                (name, Payload::F32(d)) => compiled.set_param(&name, &d),
                (name, Payload::Typed(b, dt)) => compiled.set_param_typed(&name, &b, dt),
            }
        }
    }
    compiled.finalize_params();
    Ok(())
}

enum Payload {
    F32(Vec<f32>),
    Typed(Vec<u8>, DType),
}

fn prepare(spec: ParamSpec, src: &dyn TensorSource) -> Result<(String, Payload)> {
    let want: usize = spec.shape.iter().product();
    // A half-precision matrix stored in the checkpoint at the same width:
    // transpose the 16-bit words directly.
    if let Init::Tensor { key, op } = &spec.init
        && spec.dtype != F32
        && matches!(op, TensorOp::Raw | TensorOp::Transpose)
        && let Some((bytes, dt, shape)) = src.half_bytes(key)
        && dt == spec.dtype
    {
        ensure!(
            shape.iter().product::<usize>() == want,
            "{key}: {shape:?} does not fill {:?}",
            spec.shape
        );
        let bytes = if *op == TensorOp::Transpose {
            ensure!(
                shape.len() == 2,
                "{key}: transpose needs a matrix, got {shape:?}"
            );
            let words: Vec<u16> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            transpose(&words, shape[0], shape[1])
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect()
        } else {
            bytes
        };
        return Ok((spec.name, Payload::Typed(bytes, dt)));
    }
    let data = match spec.init {
        Init::Data(d) => d,
        Init::Tensor { key, op } => {
            let (data, shape) = src.tensor(&key).with_context(|| format!("load {key}"))?;
            let n: usize = shape.iter().product();
            ensure!(n == want, "{key}: {shape:?} does not fill {:?}", spec.shape);
            match op {
                TensorOp::Raw => data,
                TensorOp::PlusOne => data.into_iter().map(|v| v + 1.0).collect(),
                TensorOp::Transpose => {
                    if shape.len() != 2 {
                        bail!("{key}: transpose needs a matrix, got {shape:?}");
                    }
                    transpose(&data, shape[0], shape[1])
                }
            }
        }
    };
    ensure!(
        data.len() == want,
        "{}: {} values for {:?}",
        spec.name,
        data.len(),
        spec.shape
    );
    Ok(if spec.dtype == F32 {
        (spec.name, Payload::F32(data))
    } else {
        let bytes = rlx_flow::storage::encode_f32(&data, spec.dtype);
        (spec.name, Payload::Typed(bytes, spec.dtype))
    })
}

/// Row-major `[rows, cols]` → `[cols, rows]`, tiled so a 100 M-element weight
/// does not thrash the cache.
pub fn transpose<T: Copy + Default>(src: &[T], rows: usize, cols: usize) -> Vec<T> {
    const TILE: usize = 64;
    let mut out = vec![T::default(); rows * cols];
    for r0 in (0..rows).step_by(TILE) {
        for c0 in (0..cols).step_by(TILE) {
            for r in r0..(r0 + TILE).min(rows) {
                for c in c0..(c0 + TILE).min(cols) {
                    out[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_roundtrip() {
        let (r, c) = (70, 131);
        let a: Vec<f32> = (0..r * c).map(|i| i as f32).collect();
        let t = transpose(&a, r, c);
        assert_eq!(t[5 * r + 3], a[3 * c + 5]);
        assert_eq!(transpose(&t, c, r), a);
    }

    #[test]
    fn rope_tables_match_fish_bf16() {
        // position 0 is the identity; bf16 rounding is visible at large t.
        let (c, s) = rope_tables(&[0, 1000], 128, 1_000_000.0, true);
        assert_eq!(&c[..64], &[1.0; 64]);
        assert!(s[..64].iter().all(|&v| v == 0.0));
        let v = c[64];
        assert_eq!(v, half::bf16::from_f32(v).to_f32());
    }
}
