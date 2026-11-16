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

//! rlx-ir graphs for Pocket TTS, compiled per device so they run on every RLX
//! backend.
//!
//! - [`flow_lm`] — one autoregressive step of the FlowLM backbone plus the
//!   sampler and EOS heads. The KV cache lives on the host and is fed back as
//!   bucketed inputs.
//! - [`mimi`] — the Mimi decoder as a *streaming block*: a fixed number of
//!   latent frames in, their audio out, with every causal conv's left context,
//!   every transposed conv's overlap tail and the transformer's sliding-window
//!   KV carried as explicit state inputs/outputs. One compiled graph covers any
//!   utterance length.
//!   The Mimi encoder (voice cloning) is built the same way.
//!
//! Backend-portability choices shared with `rlx-mimi`: 1-D convs are NCHW convs
//! with the length in H (`[1, C, T, 1]`, kernel `[k, 1]`), padding is explicit,
//! transposed convs are zero insertion + a regular conv, and ELU is composed
//! from `relu`/`exp`.

pub mod flow_lm;
pub mod mimi;

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{CompiledGraph, Device};

use crate::checkpoint::Checkpoint;

const F32: DType = DType::F32;

/// A param's values, held until the graph is compiled. The checkpoint is
/// bf16, so nearly every weight is exactly a bf16 value: held as one, it
/// takes half the memory (the builder's weights and the compiled graph's
/// would otherwise both be f32 at the peak — past a browser's wasm memory).
pub(crate) enum ParamData {
    F32(Vec<f32>),
    /// Values that are exactly bf16 (lossless).
    Bf16(Vec<half::bf16>),
}

impl ParamData {
    /// As bf16 when every value is one exactly, else as given.
    fn new(data: Vec<f32>) -> Self {
        // One pass: narrowed while checked, given up at the first value that
        // is not exactly bf16.
        let mut out = Vec::with_capacity(data.len());
        for v in &data {
            let h = half::bf16::from_f32(*v);
            if h.to_f32().to_bits() != v.to_bits() {
                return ParamData::F32(data);
            }
            out.push(h);
        }
        ParamData::Bf16(out)
    }

    pub fn into_f32(self) -> Vec<f32> {
        match self {
            ParamData::F32(v) => v,
            ParamData::Bf16(v) => v.into_iter().map(half::bf16::to_f32).collect(),
        }
    }
}

/// Graph under construction plus the params it references.
pub(crate) struct Builder {
    pub g: Graph,
    pub params: HashMap<String, ParamData>,
    next: usize,
}

impl Builder {
    pub fn new(name: &str) -> Self {
        Self {
            g: Graph::new(name),
            params: HashMap::new(),
            next: 0,
        }
    }

    /// A param with a generated unique name.
    pub fn param(&mut self, data: Vec<f32>, shape: &[usize]) -> NodeId {
        let name = format!("p{}", self.next);
        self.next += 1;
        self.named_param(&name, data, shape)
    }

    /// A param already held as it should be (see [`ParamData`]).
    fn param_data(&mut self, data: ParamData, shape: &[usize]) -> NodeId {
        let name = format!("p{}", self.next);
        self.next += 1;
        let id = self.g.param(name.clone(), Shape::new(shape, F32));
        self.params.insert(name, data);
        id
    }

    pub fn named_param(&mut self, name: &str, data: Vec<f32>, shape: &[usize]) -> NodeId {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "param {name} shape {shape:?}"
        );
        let id = self.g.param(name.to_string(), Shape::new(shape, F32));
        self.params.insert(name.to_string(), ParamData::new(data));
        id
    }

    /// The params as f32 (for an export).
    #[cfg_attr(not(feature = "export"), allow(dead_code))]
    pub fn params_f32(self) -> (Graph, HashMap<String, Vec<f32>>) {
        (
            self.g,
            self.params
                .into_iter()
                .map(|(k, v)| (k, v.into_f32()))
                .collect(),
        )
    }

    pub fn input(&mut self, name: &str, shape: &[usize]) -> NodeId {
        self.g.input(name.to_string(), Shape::new(shape, F32))
    }

    pub fn dims(&self, x: NodeId) -> Vec<usize> {
        self.g
            .shape(x)
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect()
    }

    /// `x @ Wᵀ (+ b)` for a torch `Linear` weight `[out, in]`; `x` is `[.., in]`.
    pub fn linear(&mut self, x: NodeId, w: &Tensor, b: Option<&Tensor>) -> Result<NodeId> {
        let (out_d, in_d) = w.dims2()?;
        // Transposed straight to bf16 when it is exactly bf16 (the
        // checkpoint's weights are): no f32 copy in between.
        let data = match transpose2_bf16(&w.data, out_d, in_d) {
            Some(t) => ParamData::Bf16(t),
            None => ParamData::F32(transpose2(&w.data, out_d, in_d)),
        };
        let wt = self.param_data(data, &[in_d, out_d]);
        let y = self.g.mm(x, wt);
        Ok(match b {
            Some(b) => {
                if b.data.len() != out_d {
                    bail!("bias has {} values, weight has {out_d} rows", b.data.len());
                }
                let bp = self.param(b.data.clone(), &[out_d]);
                self.g.add(y, bp)
            }
            None => y,
        })
    }

    /// LayerNorm over the last axis; `affine = None` is `elementwise_affine=False`.
    pub fn layer_norm(
        &mut self,
        x: NodeId,
        affine: Option<(&Tensor, &Tensor)>,
        eps: f32,
    ) -> NodeId {
        let d = *self.dims(x).last().expect("rank >= 1");
        let (w, b) = match affine {
            Some((w, b)) => (w.data.clone(), b.data.clone()),
            None => (vec![1.0; d], vec![0.0; d]),
        };
        let wp = self.param(w, &[d]);
        let bp = self.param(b, &[d]);
        self.g.ln(x, wp, bp, eps)
    }

    /// Broadcastable scalar.
    pub fn scalar(&mut self, v: f32) -> NodeId {
        self.param(vec![v], &[1])
    }

    pub fn expand(&mut self, x: NodeId, target: &[usize]) -> NodeId {
        let tgt: Vec<i64> = target.iter().map(|&d| d as i64).collect();
        self.g.add_node(
            Op::Expand { target_shape: tgt },
            vec![x],
            Shape::new(target, F32),
        )
    }

    /// Copy a node so it can be a graph output on its own (some backends alias
    /// an output with a view of another tensor otherwise).
    pub fn own(&mut self, x: NodeId) -> NodeId {
        let one = self.scalar(1.0);
        self.g.mul(x, one)
    }

    /// ELU (α = 1) as `relu(x) + exp(min(x, 0)) - 1`.
    pub fn elu(&mut self, x: NodeId) -> NodeId {
        let shape = self.g.shape(x).clone();
        let r = self.g.relu(x);
        let negx = self
            .g
            .add_node(Op::Activation(Activation::Neg), vec![x], shape.clone());
        let relu_neg = self.g.relu(negx);
        let minx = self
            .g
            .add_node(Op::Activation(Activation::Neg), vec![relu_neg], shape);
        let e = self.g.exp(minx);
        let re = self.g.add(r, e);
        let one = self.scalar(1.0);
        self.g.sub(re, one)
    }

    /// Interleaved (GPT-J) RoPE on `[.., H, D]` with `cos`/`sin` broadcastable
    /// to `[.., H, D/2]` — the pairing pocket-tts uses (`view(.., D/2, 2)`).
    pub fn rope_interleaved(&mut self, x: NodeId, cos: NodeId, sin: NodeId) -> NodeId {
        let dims = self.dims(x);
        let rank = dims.len();
        let d = dims[rank - 1];
        let half = d / 2;
        let lead: Vec<i64> = dims[..rank - 1].iter().map(|&v| v as i64).collect();
        let mut pairs = lead.clone();
        pairs.extend([half as i64, 2]);
        let x5 = self.g.reshape_(x, pairs);
        let mut halves = lead.clone();
        halves.push(half as i64);
        let xr = self.g.narrow_(x5, rank, 0, 1);
        let xi = self.g.narrow_(x5, rank, 1, 1);
        let xr = self.g.reshape_(xr, halves.clone());
        let xi = self.g.reshape_(xi, halves.clone());
        let a = self.g.mul(xr, cos);
        let b = self.g.mul(xi, sin);
        let out_r = self.g.sub(a, b);
        let c = self.g.mul(xr, sin);
        let e = self.g.mul(xi, cos);
        let out_i = self.g.add(c, e);
        let mut one = halves;
        one.push(1);
        let out_r = self.g.reshape_(out_r, one.clone());
        let out_i = self.g.reshape_(out_i, one);
        let stacked = self.g.concat_(vec![out_r, out_i], rank);
        let mut full = lead;
        full.push(d as i64);
        self.g.reshape_(stacked, full)
    }

    /// Plain conv on `[1, C, T, 1]`, no padding. `w` is `[c_out, c_in/groups, k]`.
    pub fn conv1d(
        &mut self,
        x: NodeId,
        w: &[f32],
        w_shape: [usize; 3],
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> NodeId {
        let [c_out, c_in_g, k] = w_shape;
        let wp = self.param(w.to_vec(), &[c_out, c_in_g, k, 1]);
        self.g
            .conv2d(x, wp, [k, 1], [stride, 1], [0, 0], [dilation, 1], groups)
    }

    /// Per-channel bias on `[1, C, T, 1]`.
    pub fn channel_bias(&mut self, x: NodeId, b: &[f32]) -> NodeId {
        let bp = self.param(b.to_vec(), &[1, b.len(), 1, 1]);
        self.g.add(x, bp)
    }

    /// Zeros `[1, C, T, 1]` (a param, so every backend gets a real buffer).
    pub fn zeros_ct(&mut self, c: usize, t: usize) -> NodeId {
        self.param(vec![0.0; c * t], &[1, c, t, 1])
    }

    /// Transposed conv without the right trim and without bias:
    /// `[1, C_in, T, 1]` → `[1, C_out, (T-1)·S + K, 1]`. `w` is torch's
    /// `[C_in, C_out/groups, K]`.
    ///
    /// Polyphase: output `m·S + p` is `Σ_q x[m - q] · w[p + q·S]`, a plain
    /// conv over `x` with phase `p`'s `⌈K/S⌉` taps. The `S` phases are one conv
    /// with `S`× the output channels (channel `oc·S + p`), interleaved after.
    /// Zero insertion spent `S`× the work on inserted zeros (in Mimi's
    /// decoder, ~70% of its multiply-adds).
    pub fn conv_transpose1d_full(
        &mut self,
        x: NodeId,
        w: &Tensor,
        stride: usize,
        groups: usize,
    ) -> Result<NodeId> {
        if stride == 1 || rlx_ir::env::flag("RLX_POCKET_TTS_ZERO_INSERT") {
            return self.conv_transpose1d_zero_insert(x, w, stride, groups);
        }
        let (in_ch, c_out_g, k) = w.dims3()?;
        let t = self.dims(x)[2];
        let c_out = c_out_g * groups;
        let c_in_g = in_ch / groups;
        let kp = k.div_ceil(stride);
        // [C_out·S, C_in/groups, ⌈K/S⌉], each phase's taps reversed (a conv
        // correlates): tap q' of phase p is w[p + (kp-1-q')·S], zero past K.
        let mut phased = vec![0f32; c_out * stride * c_in_g * kp];
        for g in 0..groups {
            for ocp in 0..c_out_g {
                let oc = g * c_out_g + ocp;
                for p in 0..stride {
                    let row = oc * stride + p;
                    for icp in 0..c_in_g {
                        let ic = g * c_in_g + icp;
                        for qq in 0..kp {
                            let j = p + (kp - 1 - qq) * stride;
                            if j < k {
                                phased[(row * c_in_g + icp) * kp + qq] =
                                    w.data[(ic * c_out_g + ocp) * k + j];
                            }
                        }
                    }
                }
            }
        }
        // x[m - q] for m in 0..T + kp - 1: kp - 1 zeros each side.
        let x = if kp > 1 {
            let zl = self.zeros_ct(in_ch, kp - 1);
            let zr = self.zeros_ct(in_ch, kp - 1);
            self.g.concat_(vec![zl, x, zr], 2)
        } else {
            x
        };
        let l = t + kp - 1;
        let y = self.conv1d(x, &phased, [c_out * stride, c_in_g, kp], 1, 1, groups);
        // [1, C_out·S, L, 1] → [1, C_out, L, S] → [1, C_out, L·S, 1].
        let y = self
            .g
            .reshape_(y, vec![1, c_out as i64, stride as i64, l as i64]);
        let y = self.g.transpose_(y, vec![0, 1, 3, 2]);
        let y = self
            .g
            .reshape_(y, vec![1, c_out as i64, (l * stride) as i64, 1]);
        Ok(self.g.narrow_(y, 2, 0, (t - 1) * stride + k))
    }

    /// [`Self::conv_transpose1d_full`] as zero insertion plus a regular conv
    /// with the flipped kernel: `S`× the multiply-adds, all but one in `S`
    /// of them by an inserted zero (RLX_POCKET_TTS_ZERO_INSERT=1, to compare).
    fn conv_transpose1d_zero_insert(
        &mut self,
        x: NodeId,
        w: &Tensor,
        stride: usize,
        groups: usize,
    ) -> Result<NodeId> {
        let (in_ch, c_out_g, k) = w.dims3()?;
        let t = self.dims(x)[2];
        let c_out = c_out_g * groups;
        let c_in_g = in_ch / groups;
        let mut flipped = vec![0f32; c_out * c_in_g * k];
        for g in 0..groups {
            for ocp in 0..c_out_g {
                let oc = g * c_out_g + ocp;
                for icp in 0..c_in_g {
                    let ic = g * c_in_g + icp;
                    for j in 0..k {
                        flipped[(oc * c_in_g + icp) * k + j] =
                            w.data[(ic * c_out_g + ocp) * k + (k - 1 - j)];
                    }
                }
            }
        }
        // Zero insertion: [1, C, T, 1] → [1, C, (T-1)·S + 1, 1].
        let u = if stride > 1 {
            let z = self.param(
                vec![0.0; in_ch * t * (stride - 1)],
                &[1, in_ch, t, stride - 1],
            );
            let cat = self.g.concat_(vec![x, z], 3);
            let r = self
                .g
                .reshape_(cat, vec![1, in_ch as i64, (t * stride) as i64, 1]);
            self.g.narrow_(r, 2, 0, (t - 1) * stride + 1)
        } else {
            x
        };
        let lu = (t - 1) * stride + 1;
        let zl = self.zeros_ct(in_ch, k - 1);
        let zr = self.zeros_ct(in_ch, k - 1);
        let padded = self.g.concat_(vec![zl, u, zr], 2);
        debug_assert_eq!(self.dims(padded)[2], lu + 2 * (k - 1));
        Ok(self.conv1d(padded, &flipped, [c_out, c_in_g, k], 1, 1, groups))
    }
}

/// A checkpoint tensor in f32.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn dims2(&self) -> Result<(usize, usize)> {
        match self.shape.as_slice() {
            [a, b] => Ok((*a, *b)),
            s => Err(anyhow!("expected a 2-D tensor, got {s:?}")),
        }
    }
    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        match self.shape.as_slice() {
            [a, b, c] => Ok((*a, *b, *c)),
            s => Err(anyhow!("expected a 3-D tensor, got {s:?}")),
        }
    }
}

pub(crate) fn transpose2(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    // In 32 × 32 tiles: both sides stay in cache (a weight is up to
    // 1024 × 4096; element by element, every write missed).
    const T: usize = 32;
    let mut out = vec![0f32; rows * cols];
    for r0 in (0..rows).step_by(T) {
        for c0 in (0..cols).step_by(T) {
            for r in r0..(r0 + T).min(rows) {
                let row = &data[r * cols..r * cols + cols];
                for c in c0..(c0 + T).min(cols) {
                    out[c * rows + r] = row[c];
                }
            }
        }
    }
    out
}

/// [`transpose2`] into bf16, when every value is exactly one; else None.
fn transpose2_bf16(data: &[f32], rows: usize, cols: usize) -> Option<Vec<half::bf16>> {
    const T: usize = 32;
    let mut out = vec![half::bf16::ZERO; rows * cols];
    for r0 in (0..rows).step_by(T) {
        for c0 in (0..cols).step_by(T) {
            for r in r0..(r0 + T).min(rows) {
                let row = &data[r * cols..r * cols + cols];
                for c in c0..(c0 + T).min(cols) {
                    let v = row[c];
                    let h = half::bf16::from_f32(v);
                    if h.to_f32().to_bits() != v.to_bits() {
                        return None;
                    }
                    out[c * rows + r] = h;
                }
            }
        }
    }
    Some(out)
}

/// Fetch a tensor by name, with the name in the error.
pub(crate) fn get(ck: &Checkpoint, name: &str) -> Result<Tensor> {
    ck.tensor(name)
        .with_context(|| format!("checkpoint tensor {name}"))
}

/// Compile `b.g` for `device` and upload its params.
pub(crate) fn compile(mut b: Builder, device: Device) -> Result<CompiledGraph> {
    if !rlx_runtime::is_available(device) {
        bail!("device {device:?} is not available in this build");
    }
    // In a browser (wasm32, where memory is short) on the CPU, matmul weights
    // stay bf16 in the compiled graph — half the memory; its matmuls widen
    // them as they read them. Only those whose values are exactly bf16 (the
    // checkpoint's): no weight changes. Natively the f32 BLAS path is
    // faster (~17%) and memory is not short; other backends promote to f32
    // anyway. WebGL computes in f32 but is handed the bf16 values as they
    // are: it keeps them so, widening each only to upload it once.
    let half_weights = match device {
        Device::Cpu => cfg!(target_arch = "wasm32"),
        // wgpu keeps a BF16 matmul weight packed (2 B/elem) and unpacks it
        // in the shader (`matmul_bf16w`); RLX_POCKET_TTS_F32_GPU turns it off.
        Device::Gpu | Device::WebGpu => !rlx_ir::env::flag("RLX_POCKET_TTS_F32_GPU"),
        _ => false,
    };
    if half_weights {
        rlx_ir::retarget::matmul_weights_as(&mut b.g, DType::BF16);
        for node in b.g.nodes_mut() {
            if let Op::Param { name } = &node.op
                && node.shape.dtype() == DType::BF16
                && !matches!(b.params.get(name), Some(ParamData::Bf16(_)))
            {
                node.shape = Shape::from_dims(node.shape.dims(), F32);
            }
        }
    }
    let declared: HashMap<String, DType> =
        b.g.nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Param { name } => Some((name.clone(), n.shape.dtype())),
                _ => None,
            })
            .collect();
    let mut compiled =
        rlx_core::flow_util::compile_graph_default_with_params(device, b.g, HashMap::new())
            .map_err(|e| anyhow!("compile: {e}"))?;
    // One param at a time (each freed as it goes in), in the precision its
    // param declares (bf16 ones to WebGL as they are, above).
    for (name, data) in b.params {
        match (data, declared.get(&name)) {
            (ParamData::Bf16(v), d) if d == Some(&DType::BF16) || device == Device::OpenGl => {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                drop(v);
                compiled.set_param_typed(&name, &bytes, DType::BF16);
            }
            (data, _) => compiled.set_param(&name, &data.into_f32()),
        }
    }
    compiled.finalize_params();
    Ok(compiled)
}

/// Interleaved-RoPE angles at `positions` for head width `head_dim`, matching
/// `pocket_tts.modules.rope.apply_rope` in f32: returns `(cos, sin)`, each
/// `[positions.len() * head_dim/2]`.
pub fn rope_tables(
    positions: impl Iterator<Item = usize>,
    head_dim: usize,
    max_period: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    // torch: `ds * (-math.log(max_period) * 2 / D)` — the factor is a Python
    // double rounded to f32 when it multiplies the f32 `ds`.
    let factor = (-(max_period as f64).ln() * 2.0 / head_dim as f64) as f32;
    let freqs: Vec<f32> = (0..half).map(|i| (i as f32 * factor).exp()).collect();
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for p in positions {
        let t = p as f32;
        for &f in &freqs {
            let a = f * t;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod conv_transpose_tests {
    use super::*;

    /// torch's `conv_transpose1d` (no bias, no trim), directly.
    fn reference(
        x: &[f32],
        c_in: usize,
        t: usize,
        w: &Tensor,
        s: usize,
        groups: usize,
    ) -> Vec<f32> {
        let (_, c_out_g, k) = w.dims3().unwrap();
        let c_out = c_out_g * groups;
        let c_in_g = c_in / groups;
        let len = (t - 1) * s + k;
        let mut y = vec![0f32; c_out * len];
        for ic in 0..c_in {
            let g = ic / c_in_g;
            for ocp in 0..c_out_g {
                let oc = g * c_out_g + ocp;
                for m in 0..t {
                    for j in 0..k {
                        y[oc * len + m * s + j] +=
                            x[ic * t + m] * w.data[(ic * c_out_g + ocp) * k + j];
                    }
                }
            }
        }
        y
    }

    #[test]
    fn polyphase_is_a_transposed_conv() {
        let mut seed = 7u32;
        let mut noise = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
                })
                .collect()
        };
        // (c_in, c_out/groups, k, stride, groups, t): Mimi's shapes (k = 2S,
        // depthwise ×16), and a k that S does not divide.
        for (c_in, c_out_g, k, s, groups, t) in [
            (8, 4, 12, 6, 1, 5),
            (6, 3, 10, 5, 1, 4),
            (4, 1, 32, 16, 4, 3),
            (5, 2, 7, 3, 1, 6),
            (3, 2, 2, 4, 1, 3),
        ] {
            let w = Tensor {
                data: noise(c_in * c_out_g * k),
                shape: vec![c_in, c_out_g, k],
            };
            let x = noise(c_in * t);
            let want = reference(&x, c_in, t, &w, s, groups);
            let mut b = Builder::new("convtr");
            let xi = b.input("x", &[1, c_in, t, 1]);
            let y = b.conv_transpose1d_full(xi, &w, s, groups).unwrap();
            b.g.set_outputs(vec![y]);
            let mut c = compile(b, Device::Cpu).unwrap();
            let got = c.run(&[("x", &x)]).remove(0);
            assert_eq!(got.len(), want.len(), "({c_in},{c_out_g},{k},{s},{groups})");
            for (i, (a, e)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - e).abs() < 1e-5,
                    "({c_in},{c_out_g},{k},{s},{groups}) [{i}]: {a} vs {e}"
                );
            }
        }
    }
}

#[cfg(test)]
mod transpose_tests {
    #[test]
    fn tiled_transpose_is_a_transpose() {
        for (rows, cols) in [(1, 1), (3, 5), (33, 70), (64, 32), (100, 3)] {
            let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
            let t = super::transpose2(&data, rows, cols);
            for r in 0..rows {
                for c in 0..cols {
                    assert_eq!(
                        t[c * rows + r],
                        data[r * cols + c],
                        "{rows}×{cols} at {r},{c}"
                    );
                }
            }
        }
    }
}
