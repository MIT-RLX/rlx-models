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

/// Graph under construction plus the params it references.
pub(crate) struct Builder {
    pub g: Graph,
    pub params: HashMap<String, Vec<f32>>,
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

    pub fn named_param(&mut self, name: &str, data: Vec<f32>, shape: &[usize]) -> NodeId {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "param {name} shape {shape:?}"
        );
        let id = self.g.param(name.to_string(), Shape::new(shape, F32));
        self.params.insert(name.to_string(), data);
        id
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
        let wt = self.param(transpose2(&w.data, out_d, in_d), &[in_d, out_d]);
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
    /// `[C_in, C_out/groups, K]`. Decomposed into zero insertion plus a regular
    /// conv with the flipped kernel.
    pub fn conv_transpose1d_full(
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
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Fetch a tensor by name, with the name in the error.
pub(crate) fn get(ck: &Checkpoint, name: &str) -> Result<Tensor> {
    ck.tensor(name)
        .with_context(|| format!("checkpoint tensor {name}"))
}

/// Compile `b.g` for `device` and upload its params.
pub(crate) fn compile(b: Builder, device: Device) -> Result<CompiledGraph> {
    if !rlx_runtime::is_available(device) {
        bail!("device {device:?} is not available in this build");
    }
    let mut compiled =
        rlx_core::flow_util::compile_graph_default_with_params(device, b.g, b.params)
            .map_err(|e| anyhow!("compile: {e}"))?;
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
