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

//! Graph-building helpers shared by the TADA sub-models.
//!
//! Every builder in this crate accumulates its constants as named params
//! alongside the HIR, so a graph can be compiled once and have its weights
//! bound afterwards. This mirrors `rlx-dac`'s pattern rather than inventing a
//! second one.

use crate::weights::TensorStore;
use anyhow::Result;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;

pub const F32: DType = DType::F32;

/// Named tensors produced alongside a built graph.
pub type NamedTensors = Vec<(String, Vec<f32>)>;

/// Row-major `[rows, cols]` → `[cols, rows]`.
///
/// Delegates to `rlx-core`'s cache-blocked transpose. Checkpoint weights arrive
/// `[out, in]` and every matmul here wants `[in, out]`, so this runs over every
/// parameter of every graph — the naive loop cost 1.4 s just for the diffusion
/// head.
pub fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    rlx_core::weight_map::transpose_2d(data, rows, cols)
}

/// HIR builder that auto-names every constant it interns.
///
/// Constants can be interned under a caller-supplied key, in which case a
/// repeat request returns the existing node instead of a second copy. That
/// matters for any graph that reuses a weight — the diffusion head unrolls ten
/// ODE steps over the same ~350 M parameters, so without deduplication the
/// param table would be ten times the model.
pub struct Ctx<'a, 'b> {
    pub g: &'a mut HirMut<'b>,
    params: NamedTensors,
    interned: HashMap<String, HirNodeId>,
    next: usize,
}

impl<'a, 'b> Ctx<'a, 'b> {
    pub fn new(g: &'a mut HirMut<'b>) -> Self {
        Self {
            g,
            params: Vec::new(),
            interned: HashMap::new(),
            next: 0,
        }
    }

    pub fn into_params(self) -> NamedTensors {
        self.params
    }

    /// Intern `data` as a param of `shape` and return its node.
    pub fn param(&mut self, data: Vec<f32>, shape: &[usize]) -> HirNodeId {
        let n: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n,
            "param w{} shape {shape:?} wants {n} values, got {}",
            self.next,
            data.len()
        );
        let name = format!("w{}", self.next);
        self.next += 1;
        let id = self.g.param(name.clone(), Shape::new(shape, F32));
        self.params.push((name, data));
        id
    }

    /// Intern under `key`, reusing the node (and its storage) on repeat calls.
    ///
    /// `build` is only invoked on a miss, so a caller can key on a weight name
    /// and skip the host-side transpose entirely when the weight is reused.
    pub fn param_keyed(
        &mut self,
        key: &str,
        shape: &[usize],
        build: impl FnOnce() -> Vec<f32>,
    ) -> HirNodeId {
        if let Some(&id) = self.interned.get(key) {
            return id;
        }
        let id = self.param(build(), shape);
        self.interned.insert(key.to_string(), id);
        id
    }

    /// Fallible [`Self::param_keyed`] — `build` only runs on a miss, so a
    /// caller can read straight from a mapped checkpoint and skip the read
    /// entirely when the weight is already interned.
    pub fn param_keyed_try(
        &mut self,
        key: &str,
        shape: &[usize],
        build: impl FnOnce() -> Result<Vec<f32>>,
    ) -> Result<HirNodeId> {
        if let Some(&id) = self.interned.get(key) {
            return Ok(id);
        }
        let id = self.param(build()?, shape);
        self.interned.insert(key.to_string(), id);
        Ok(id)
    }

    /// Bias-free keyed linear that loads its `[out, in]` weight lazily.
    pub fn linear_keyed_try(
        &mut self,
        key: &str,
        x: HirNodeId,
        out_dim: usize,
        in_dim: usize,
        load: impl FnOnce() -> Result<Vec<f32>>,
    ) -> Result<HirNodeId> {
        let wp = self.param_keyed_try(key, &[in_dim, out_dim], || {
            Ok(transpose(&load()?, out_dim, in_dim))
        })?;
        Ok(self.apply_linear(x, wp, None, out_dim, in_dim))
    }

    /// A bias-carrying `Linear` whose weight and bias are pulled from a
    /// checkpoint on a cache miss and transposed into `[in, out]`.
    ///
    /// The codec's attention stack and the aligner both want exactly this and
    /// had grown their own copies; the key shape is always `{key}.weight` /
    /// `{key}.bias`. Interning keeps a weight from being materialized twice
    /// when one graph uses it more than once.
    pub fn store_linear(
        &mut self,
        store: &TensorStore,
        x: HirNodeId,
        weight_key: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<HirNodeId> {
        // Keyed on the checkpoint name, which is unique across every stack in
        // the graph. A shorthand key (`l0.qkv`) is not: the aligner's encoder
        // and the codec's both number their layers from zero, so the second
        // stack built into a given `Ctx` would silently reuse the first's
        // weights — visible only as subtly wrong audio.
        let wp = self.param_keyed_try(&format!("{weight_key}.w"), &[in_dim, out_dim], || {
            Ok(transpose(
                &store.get(&format!("{weight_key}.weight"))?,
                out_dim,
                in_dim,
            ))
        })?;
        let bp = self.param_keyed_try(&format!("{weight_key}.b"), &[1, out_dim], || {
            store.get(&format!("{weight_key}.bias"))
        })?;
        Ok(self.apply_linear(x, wp, Some(bp), out_dim, in_dim))
    }

    /// Intern a LayerNorm's `(gamma, beta)` pair from a checkpoint, lazily.
    pub fn store_norm(
        &mut self,
        store: &TensorStore,
        key: &str,
        dim: usize,
    ) -> Result<(HirNodeId, HirNodeId)> {
        let g = self.param_keyed_try(&format!("{key}.w"), &[dim], || {
            store.get(&format!("{key}.weight"))
        })?;
        let b = self.param_keyed_try(&format!("{key}.b"), &[dim], || {
            store.get(&format!("{key}.bias"))
        })?;
        Ok((g, b))
    }

    /// `y = x·Wᵀ (+ b)` for `x` of rank ≥ 2 whose last axis is `in_dim`.
    ///
    /// `w` arrives in checkpoint orientation `[out, in]` and is transposed
    /// host-side, because rlx's `mm` wants `[in, out]`.
    pub fn linear(
        &mut self,
        x: HirNodeId,
        w_row_major: &[f32],
        out_dim: usize,
        in_dim: usize,
        bias: Option<&[f32]>,
    ) -> HirNodeId {
        let wp = self.param(transpose(w_row_major, out_dim, in_dim), &[in_dim, out_dim]);
        let bp = bias.map(|b| self.param(b.to_vec(), &[1, out_dim]));
        self.apply_linear(x, wp, bp, out_dim, in_dim)
    }

    fn apply_linear(
        &mut self,
        x: HirNodeId,
        wp: HirNodeId,
        bias: Option<HirNodeId>,
        out_dim: usize,
        in_dim: usize,
    ) -> HirNodeId {
        let shape = self.g.shape(x).clone();
        let rank = shape.rank();
        let rows: usize = (0..rank - 1)
            .map(|i| shape.dim(i).unwrap_static())
            .product();
        let x2 = self.g.reshape_(x, vec![rows as i64, in_dim as i64]);
        let mut y = self.g.mm(x2, wp);
        if let Some(bp) = bias {
            let be = self.g.expand_(bp, vec![rows as i64, out_dim as i64]);
            y = self.g.add(y, be);
        }
        let mut out: Vec<i64> = (0..rank - 1)
            .map(|i| shape.dim(i).unwrap_static() as i64)
            .collect();
        out.push(out_dim as i64);
        self.g.reshape_(y, out)
    }

    /// GPT-J / interleaved RoPE on `[batch, seq, heads, head_dim]`.
    ///
    /// Pairs adjacent channels `(2i, 2i+1)` and rotates each by the angle at
    /// table column `i` — the convention TADA's codec attention uses (its
    /// `_apply_rope` reshapes to `(…, head_dim/2, 2)` and rotates the trailing
    /// pair). Built from elementwise ops instead of `Op::Rope` so the operand
    /// layout is unambiguous on every backend.
    ///
    /// `cos`/`sin` are `[seq, head_dim/2]` params.
    pub fn rope_interleaved(
        &mut self,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> HirNodeId {
        let half = head_dim / 2;
        let (b, s, h, hf) = (batch as i64, seq as i64, heads as i64, half as i64);
        let x5 = self.g.reshape_(x, vec![b, s, h, hf, 2]);
        let x1 = self.g.narrow_(x5, 4, 0, 1);
        let x2 = self.g.narrow_(x5, 4, 1, 1);
        // [seq, half] → [1, seq, 1, half, 1] so it broadcasts over batch/heads.
        let cos5 = self.g.reshape_(cos, vec![1, s, 1, hf, 1]);
        let sin5 = self.g.reshape_(sin, vec![1, s, 1, hf, 1]);
        let x1c = self.g.mul(x1, cos5);
        let x2s = self.g.mul(x2, sin5);
        let x2c = self.g.mul(x2, cos5);
        let x1s = self.g.mul(x1, sin5);
        let r1 = self.g.sub(x1c, x2s);
        let r2 = self.g.add(x2c, x1s);
        let cat = self.g.concat_(vec![r1, r2], 4);
        self.g.reshape_(cat, vec![b, s, h, head_dim as i64])
    }
}

/// Compile `graph` on `device` and bind `params`.
pub fn compile(device: Device, graph: Graph, params: NamedTensors) -> CompiledGraph {
    let session = Session::new(device);
    let mut compiled = session.compile(graph);
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    compiled.finalize_params();
    compiled
}

/// Lower a finished [`HirModule`] to a [`Graph`], naming the stage in errors.
pub fn lower(hir: HirModule, what: &str) -> Result<Graph> {
    Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{what}: HIR lowering failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::hir::HirModule;

    /// Run a one-node graph on CPU and read back the single output.
    fn run(build: impl FnOnce(&mut Ctx) -> HirNodeId, inputs: &[(&str, &[f32])]) -> Vec<f32> {
        let mut hir = HirModule::new("t");
        let mut g = HirMut::new(&mut hir);
        let mut ctx = Ctx::new(&mut g);
        let out = build(&mut ctx);
        let params = ctx.into_params();
        hir.set_outputs(vec![out]);
        let graph = lower(hir, "test").unwrap();
        let mut c = compile(Device::Cpu, graph, params);
        c.run(inputs).into_iter().next().unwrap()
    }

    #[test]
    fn linear_applies_weight_and_bias() {
        // x = [1, 2]; W = [[1, 0], [0, 1], [1, 1]]; b = [10, 20, 30]
        let out = run(
            |ctx| {
                let x = ctx.g.input("x", Shape::new(&[1, 2], F32));
                ctx.linear(
                    x,
                    &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    3,
                    2,
                    Some(&[10.0, 20.0, 30.0]),
                )
            },
            &[("x", &[1.0, 2.0])],
        );
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn interleaved_rope_rotates_adjacent_pairs() {
        // One head, head_dim 2, one position, angle π/2 → (1, 0) becomes (0, 1).
        let out = run(
            |ctx| {
                let x = ctx.g.input("x", Shape::new(&[1, 1, 1, 2], F32));
                let cos = ctx.param(vec![0.0], &[1, 1]);
                let sin = ctx.param(vec![1.0], &[1, 1]);
                ctx.rope_interleaved(x, cos, sin, 1, 1, 1, 2)
            },
            &[("x", &[1.0, 0.0])],
        );
        assert!((out[0] - 0.0).abs() < 1e-6, "{out:?}");
        assert!((out[1] - 1.0).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn interleaved_rope_at_angle_zero_is_the_identity() {
        let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let out = run(
            |ctx| {
                let xi = ctx.g.input("x", Shape::new(&[1, 1, 2, 4], F32));
                let cos = ctx.param(vec![1.0, 1.0], &[1, 2]);
                let sin = ctx.param(vec![0.0, 0.0], &[1, 2]);
                ctx.rope_interleaved(xi, cos, sin, 1, 1, 2, 4)
            },
            &[("x", &x)],
        );
        for (a, b) in out.iter().zip(&x) {
            assert!((a - b).abs() < 1e-6, "{out:?}");
        }
    }
}
