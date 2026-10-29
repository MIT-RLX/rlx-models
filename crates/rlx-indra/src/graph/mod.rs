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

//! Native RLX graphs for the Indra networks, written against the module
//! structure of the upstream release (weights from [`crate::convert`]).
//!
//! Every graph is static-shape: [`Compiled`] caches one compilation per input
//! shape, and the callers bucket batch and width so a page needs few shapes.

pub mod classifier;
pub mod det;
pub mod order;
pub mod rec;

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::flow_bridge::compile_options_for_profile;
use rlx_core::flow_util::{attach_built_params, graph_from_hir};
use rlx_core::weight_map::WeightMap;
use rlx_flow::CompileProfile;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::{Activation, Op, ReduceOp};
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// `BatchNormalization` / `LayerNormalization` epsilon of every Indra network.
pub const EPS: f32 = 1e-5;

/// Float weights of one network, by module path.
pub struct Weights {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl Weights {
    pub fn load(path: &Path) -> Result<Self> {
        let mut wm = WeightMap::from_file(path.to_str().context("non-UTF-8 path")?)
            .with_context(|| format!("loading {}", path.display()))?;
        Ok(Self {
            tensors: wm.take_all(),
        })
    }

    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    pub fn get(&self, key: &str) -> Result<(&[f32], &[usize])> {
        self.tensors
            .get(key)
            .map(|(d, s)| (d.as_slice(), s.as_slice()))
            .ok_or_else(|| anyhow!("missing weight {key}"))
    }

    pub fn shape(&self, key: &str) -> Result<&[usize]> {
        Ok(self.get(key)?.1)
    }
}

/// HIR under construction plus the parameter data it references.
pub struct Builder<'w> {
    pub hir: HirModule,
    pub params: HashMap<String, Vec<f32>>,
    pub w: &'w Weights,
}

impl<'w> Builder<'w> {
    pub fn new(name: &str, w: &'w Weights) -> Self {
        Self {
            hir: HirModule::new(name),
            params: HashMap::new(),
            w,
        }
    }

    pub fn m(&mut self) -> HirMut<'_> {
        HirMut::new(&mut self.hir)
    }

    pub fn dims(&mut self, x: HirNodeId) -> Vec<usize> {
        let s = self.m().shape(x).clone();
        (0..s.rank()).map(|i| s.dim(i).unwrap_static()).collect()
    }

    pub fn input(&mut self, name: &str, dims: &[usize]) -> HirNodeId {
        self.m().input(name, Shape::new(dims, DType::F32))
    }

    /// A parameter with explicit data (derived from weights, or a constant).
    pub fn constant(&mut self, key: &str, dims: &[usize], data: Vec<f32>) -> HirNodeId {
        debug_assert_eq!(dims.iter().product::<usize>(), data.len());
        let id = self.m().param(key, Shape::new(dims, DType::F32));
        self.params.insert(key.to_string(), data);
        id
    }

    /// A stored weight, reshaped to `dims` (same element count).
    pub fn weight_as(&mut self, key: &str, dims: &[usize]) -> Result<HirNodeId> {
        let (data, shape) = self.w.get(key)?;
        if shape.iter().product::<usize>() != dims.iter().product::<usize>() {
            bail!("weight {key}: {shape:?} cannot be viewed as {dims:?}");
        }
        let data = data.to_vec();
        Ok(self.constant(key, dims, data))
    }

    pub fn weight(&mut self, key: &str) -> Result<HirNodeId> {
        let dims = self.w.shape(key)?.to_vec();
        self.weight_as(key, &dims)
    }

    pub fn act(&mut self, x: HirNodeId, a: Activation) -> HirNodeId {
        let s = self.m().shape(x).clone();
        self.m().activation(a, x, s)
    }

    /// Broadcast `y` up to `x`'s shape (size-1 axes), explicitly: one `Expand`
    /// kernel on every backend.
    pub fn expand_to(&mut self, y: HirNodeId, x: HirNodeId) -> HirNodeId {
        let (dy, dx) = (self.dims(y), self.dims(x));
        if dy == dx {
            return y;
        }
        let t = dx.iter().map(|&d| d as i64).collect();
        self.m().expand_(y, t)
    }

    pub fn mul_b(&mut self, x: HirNodeId, y: HirNodeId) -> HirNodeId {
        let y = self.expand_to(y, x);
        self.m().mul(x, y)
    }

    pub fn add_b(&mut self, x: HirNodeId, y: HirNodeId) -> HirNodeId {
        let y = self.expand_to(y, x);
        self.m().add(x, y)
    }

    pub fn scale(&mut self, x: HirNodeId, s: f32, key: &str) -> HirNodeId {
        let rank = self.dims(x).len();
        let c = self.constant(key, &vec![1; rank], vec![s]);
        self.mul_b(x, c)
    }

    /// NCHW `Conv` from `{prefix}.weight` (`[O, I/g, kh, kw]`, or `[O, I/g, k]`
    /// for a 1-D conv run as `kh = 1`) and `{prefix}.bias` when stored.
    pub fn conv(
        &mut self,
        x: HirNodeId,
        prefix: &str,
        stride: [usize; 2],
        pad: [usize; 2],
    ) -> Result<HirNodeId> {
        let wkey = format!("{prefix}.weight");
        let ws = self.w.shape(&wkey)?.to_vec();
        let (o, ig, kh, kw) = match ws.as_slice() {
            [o, i, kh, kw] => (*o, *i, *kh, *kw),
            [o, i, k] => (*o, *i, 1, *k),
            s => bail!("{wkey}: unexpected conv weight shape {s:?}"),
        };
        let [b, c, h, wd] = self.dims(x)[..] else {
            bail!("{prefix}: conv input must be NCHW");
        };
        if c % ig != 0 {
            bail!("{prefix}: {c} input channels, weight takes {ig} per group");
        }
        let groups = c / ig;
        let oh = (h + 2 * pad[0] - kh) / stride[0] + 1;
        let ow = (wd + 2 * pad[1] - kw) / stride[1] + 1;
        let wid = self.weight_as(&wkey, &[o, ig, kh, kw])?;
        let out = Shape::new(&[b, o, oh, ow], DType::F32);
        let y = self.m().conv2d(x, wid, [kh, kw], stride, pad, groups, out);
        let bkey = format!("{prefix}.bias");
        if !self.w.has(&bkey) {
            return Ok(y);
        }
        let bias = self.weight_as(&bkey, &[1, o, 1, 1])?;
        Ok(self.add_b(y, bias))
    }

    /// `conv` with "same" padding (`k / 2`) and an optional activation.
    pub fn conv_act(
        &mut self,
        x: HirNodeId,
        prefix: &str,
        stride: usize,
        act: Option<Activation>,
    ) -> Result<HirNodeId> {
        let ws = self.w.shape(&format!("{prefix}.weight"))?.to_vec();
        let (kh, kw) = match ws.as_slice() {
            [_, _, kh, kw] => (*kh, *kw),
            [_, _, k] => (1, *k),
            s => bail!("{prefix}: unexpected conv weight shape {s:?}"),
        };
        let y = self.conv(x, prefix, [stride, stride], [kh / 2, kw / 2])?;
        Ok(match act {
            Some(a) => self.act(y, a),
            None => y,
        })
    }

    /// Inference `BatchNormalization` over axis 1, folded to one scale + shift.
    pub fn batch_norm(&mut self, x: HirNodeId, prefix: &str) -> Result<HirNodeId> {
        let g = |s: &str| {
            self.w
                .get(&format!("{prefix}.{s}"))
                .map(|(d, _)| d.to_vec())
        };
        let (gamma, beta, mean, var) = (
            g("weight")?,
            g("bias")?,
            g("running_mean")?,
            g("running_var")?,
        );
        let scale: Vec<f32> = gamma
            .iter()
            .zip(&var)
            .map(|(g, v)| g / (v + EPS).sqrt())
            .collect();
        let shift: Vec<f32> = beta
            .iter()
            .zip(&mean)
            .zip(&scale)
            .map(|((b, m), s)| b - m * s)
            .collect();
        let mut d = vec![1; self.dims(x).len()];
        d[1] = scale.len();
        let s = self.constant(&format!("{prefix}.scale"), &d, scale);
        let t = self.constant(&format!("{prefix}.shift"), &d, shift);
        let y = self.mul_b(x, s);
        Ok(self.add_b(y, t))
    }

    /// `x @ W + b` over the last axis (`{prefix}.weight` stored `[in, out]`).
    pub fn linear(&mut self, x: HirNodeId, prefix: &str) -> Result<HirNodeId> {
        let w = self.weight(&format!("{prefix}.weight"))?;
        let y = self.mm_last(x, w);
        let b = self.weight(&format!("{prefix}.bias"))?;
        Ok(self.add_b(y, b))
    }

    /// `Gemm` with `transB = 1` (`{prefix}.weight` stored `[out, in]`), `x: [B, in]`.
    pub fn gemm_t(&mut self, x: HirNodeId, prefix: &str) -> Result<HirNodeId> {
        let ws = self.w.shape(&format!("{prefix}.weight"))?.to_vec();
        let wt: Vec<f32> = {
            let (d, _) = self.w.get(&format!("{prefix}.weight"))?;
            let (o, i) = (ws[0], ws[1]);
            let mut t = vec![0.0; o * i];
            for r in 0..o {
                for c in 0..i {
                    t[c * o + r] = d[r * i + c];
                }
            }
            t
        };
        let w = self.constant(&format!("{prefix}.weight_t"), &[ws[1], ws[0]], wt);
        let y = self.m().mm(x, w);
        let b = self.weight(&format!("{prefix}.bias"))?;
        Ok(self.add_b(y, b))
    }

    /// Matmul of `[.., K] @ [K, N]` with leading axes folded into rows.
    pub fn mm_last(&mut self, x: HirNodeId, w: HirNodeId) -> HirNodeId {
        let d = self.dims(x);
        let n = self.dims(w)[1];
        let k = *d.last().unwrap();
        let rows: usize = d[..d.len() - 1].iter().product();
        let x2 = self.m().reshape_(x, vec![rows as i64, k as i64]);
        let y = self.m().mm(x2, w);
        let mut out: Vec<i64> = d[..d.len() - 1].iter().map(|&v| v as i64).collect();
        out.push(n as i64);
        self.m().reshape_(y, out)
    }

    pub fn layer_norm(&mut self, x: HirNodeId, prefix: &str) -> Result<HirNodeId> {
        let g = self.weight(&format!("{prefix}.weight"))?;
        let b = self.weight(&format!("{prefix}.bias"))?;
        Ok(self.m().ln(x, g, b, EPS))
    }

    pub fn reduce(
        &mut self,
        x: HirNodeId,
        op: ReduceOp,
        axes: Vec<usize>,
        keep: bool,
    ) -> HirNodeId {
        let s = rlx_ir::shape::reduce_shape(self.m().shape(x), &axes, keep).expect("reduce shape");
        self.m().add_node(
            Op::Reduce {
                op,
                axes,
                keep_dim: keep,
            },
            vec![x],
            s,
        )
    }

    pub fn pool(&mut self, x: HirNodeId, kind: ReduceOp, k: usize) -> HirNodeId {
        let [b, c, h, w] = self.dims(x)[..] else {
            panic!("pool input must be NCHW");
        };
        let out = Shape::new(&[b, c, h / k, w / k], DType::F32);
        self.m().add_node(
            Op::Pool {
                kind,
                kernel_size: vec![k, k],
                stride: vec![k, k],
                padding: vec![0, 0],
            },
            vec![x],
            out,
        )
    }

    pub fn finish(
        mut self,
        outputs: Vec<HirNodeId>,
    ) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> {
        self.m().set_outputs(outputs);
        graph_from_hir(self.hir, self.params)
    }
}

type BuildFn =
    dyn Fn(&Weights, &[usize]) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)> + Send + Sync;

/// One network: its weights and a compilation per input shape key.
pub struct Compiled {
    weights: Weights,
    device: Device,
    build: Box<BuildFn>,
    cache: Mutex<HashMap<Vec<usize>, CompiledGraph>>,
}

impl Compiled {
    pub fn new(
        weights: Weights,
        device: Device,
        build: impl Fn(&Weights, &[usize]) -> Result<(rlx_ir::Graph, HashMap<String, Vec<f32>>)>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            weights,
            device,
            build: Box::new(build),
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// Run the graph compiled for `key` (built on first use).
    pub fn run(&self, key: &[usize], inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| anyhow!("compile cache poisoned"))?;
        if !cache.contains_key(key) {
            let (graph, params) = (self.build)(&self.weights, key)?;
            let opts = compile_options_for_profile(&CompileProfile::encoder(), self.device);
            let mut compiled = Session::new(self.device).compile_with(graph, &opts);
            attach_built_params(&mut compiled, params, &[]);
            cache.insert(key.to_vec(), compiled);
        }
        let compiled = cache.get_mut(key).expect("compiled above");
        compiled.try_run(inputs).map_err(|e| anyhow!("{e}"))
    }
}
