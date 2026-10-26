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

//! The compiled Timestral network: patch embedding → time/variate transformer →
//! quantile-head projection.
//!
//! Two upstream steps are folded into the weights so the graph stays plain matmuls:
//! the `1/sqrt(head_dim)` attention scale goes into the query columns, and RoPE's
//! `rotate_half(x W)` becomes `x (W P)` — two extra projections per time attention —
//! so the rotation is `q * cos + q_rot * sin` with host-built `[R*S, D]` tables.
//!
//! Graph inputs (`S = l + n_dec`):
//!
//! | name | shape | |
//! |---|---|---|
//! | `patches` | `[R*l, 2*patch_size]` | normalized values ‖ missing flags |
//! | `dec` | `[R*n_dec, D]` | forecast-query tokens |
//! | `rope_cos`, `rope_sin` | `[R*S, D]` | time positions, repeated per head |
//! | `time_bias` | `[R, S, S]` | additive time mask |
//! | `var_bias` | `[S, R, R]` | additive variate mask (when the schedule has one) |
//!
//! Output: raw head values `[R*n_dec, patch_size * Q]` (before the monotone map),
//! then, with [`GraphDims::hidden`], the transformer output `[R*S, D]`.

use std::collections::HashMap;

use ndarray::Array2;
use rlx_ir::hir::{HirGraphExt, HirModule, HirMut, HirNodeId};
use rlx_ir::op::Activation;
use rlx_ir::{DType, Shape};

use crate::config::{LayerKind, TimestralConfig};
pub use crate::model::{GraphDims, has_variate_attention, rope_half_tables, rope_tables};
use crate::weights::{ResidualBlockWeight, TimestralWeights};

const F32: DType = DType::F32;

/// Graph-ready parameters, keyed by the names [`build_graph`] declares.
///
/// Built once per model: matrices are stored `[in, out]`, the query scale and RoPE's
/// rotation are already folded in.
pub fn graph_params(cfg: &TimestralConfig, w: &TimestralWeights) -> HashMap<String, Vec<f32>> {
    let mut p = HashMap::new();
    let d = cfg.d_model;

    res_block_params(&mut p, "patch", &w.patch_proj);
    res_block_params(&mut p, "head", &w.head);
    p.insert("out_norm".into(), w.out_norm.to_vec());
    p.insert("zero_bias".into(), vec![0.0; d]);

    for (i, layer) in w.layers.iter().enumerate() {
        for (j, (norm, attn)) in layer.attn_norms.iter().zip(&layer.attns).enumerate() {
            let t = format!("l{i}.a{j}");
            p.insert(format!("{t}.norm"), norm.to_vec());
            // One wide projection per attention: q, k, v, then the gate. The
            // fused attention op applies the 1/sqrt(hd) scale and `Op::Rope`
            // the rotation, so neither is folded in here (the training layout,
            // `graph_params_train`, still folds both).
            let wq = attn.qkv.slice(ndarray::s![..d, ..]).to_owned();
            let wk = attn.qkv.slice(ndarray::s![d..2 * d, ..]).to_owned();
            let wv = attn.qkv.slice(ndarray::s![2 * d.., ..]).to_owned();
            let mut parts = vec![wq, wk, wv];
            if let Some(g) = &attn.gate {
                parts.push(g.clone());
            }
            let views: Vec<_> = parts.iter().map(|m| m.view()).collect();
            p.insert(
                format!("{t}.proj"),
                transposed(&ndarray::concatenate(ndarray::Axis(0), &views).unwrap()),
            );
            p.insert(format!("{t}.o"), transposed(&attn.out));
        }
        let t = format!("l{i}.mlp");
        p.insert(format!("{t}.norm"), layer.mlp_norm.to_vec());
        // `[gate | up]` as one `[D, 2H]` weight (upstream's `w12`): two
        // matmuls on the same input made the CPU compiler concatenate the two
        // weights on every call; one wide matmul feeds `FusedSwiGLU` directly.
        let gate_up =
            ndarray::concatenate(ndarray::Axis(0), &[layer.w_gate.view(), layer.w_up.view()])
                .unwrap();
        p.insert(format!("{t}.gate_up"), transposed(&gate_up));
        p.insert(format!("{t}.down"), transposed(&layer.w3));
    }
    p
}

/// Parameters for a [`GraphDims::train`] graph: as [`graph_params`], but each
/// attention keeps its query (pre-scaled), key, value and gate weights, and
/// each MLP its gate and up weights, as separate `[in, out]` params instead of
/// one pre-built projection, plus the constant `rope_perm` that rotates them
/// in-graph. [`TRAIN_CONSTANTS`] are not
/// trained.
pub fn graph_params_train(
    cfg: &TimestralConfig,
    w: &TimestralWeights,
) -> HashMap<String, Vec<f32>> {
    let mut p = graph_params(cfg, w);
    let d = cfg.d_model;
    let q_scale = 1.0 / (cfg.head_dim() as f32).sqrt();
    for (i, layer) in w.layers.iter().enumerate() {
        for (j, attn) in layer.attns.iter().enumerate() {
            let t = format!("l{i}.a{j}");
            p.remove(&format!("{t}.proj"));
            let q = attn.qkv.slice(ndarray::s![..d, ..]).mapv(|v| v * q_scale);
            p.insert(format!("{t}.q"), transposed(&q));
            p.insert(
                format!("{t}.k"),
                transposed(&attn.qkv.slice(ndarray::s![d..2 * d, ..]).to_owned()),
            );
            p.insert(
                format!("{t}.v"),
                transposed(&attn.qkv.slice(ndarray::s![2 * d.., ..]).to_owned()),
            );
            if let Some(g) = &attn.gate {
                p.insert(format!("{t}.gate"), transposed(g));
            }
        }
    }
    for (i, layer) in w.layers.iter().enumerate() {
        let t = format!("l{i}.mlp");
        p.remove(&format!("{t}.gate_up"));
        p.insert(format!("{t}.gate"), transposed(&layer.w_gate));
        p.insert(format!("{t}.up"), transposed(&layer.w_up));
    }
    p.insert("rope_perm".into(), rope_perm(cfg));
    p
}

/// Params of a training graph that hold constants, never trained.
pub const TRAIN_CONSTANTS: [&str; 2] = ["rope_perm", "zero_bias"];

/// `[D, D]` signed permutation `P` with `W P` = RoPE's `rotate_half` applied to
/// the output columns of an `[in, out]` weight `W`, head by head.
fn rope_perm(cfg: &TimestralConfig) -> Vec<f32> {
    let (d, hd) = (cfg.d_model, cfg.head_dim());
    let half = hd / 2;
    let mut m = vec![0f32; d * d];
    for c in 0..d {
        let k = c % hd;
        let (j, sign) = if k < half {
            (c + half, -1.0)
        } else {
            (c - half, 1.0)
        };
        m[j * d + c] = sign;
    }
    m
}

fn res_block_params(p: &mut HashMap<String, Vec<f32>>, tag: &str, w: &ResidualBlockWeight) {
    p.insert(format!("{tag}.hidden"), transposed(&w.hidden));
    p.insert(format!("{tag}.output"), transposed(&w.output));
    p.insert(format!("{tag}.residual"), transposed(&w.residual));
    let biases = [
        ("hidden_b", &w.hidden_b),
        ("output_b", &w.output_b),
        ("residual_b", &w.residual_b),
    ];
    for (name, b) in biases {
        if let Some(b) = b {
            p.insert(format!("{tag}.{name}"), b.to_vec());
        }
    }
}

/// `[out, in]` → row-major `[in, out]`.
fn transposed(w: &Array2<f32>) -> Vec<f32> {
    w.t().iter().copied().collect()
}

/// Rows of `W P`: output row `(h, k)` is `-W(h, k + hd/2)` for `k < hd/2`, else `W(h, k - hd/2)`.
#[cfg(test)]
fn rotate_half_rows(w: &Array2<f32>, nh: usize, hd: usize) -> Array2<f32> {
    let half = hd / 2;
    let mut out = Array2::<f32>::zeros(w.raw_dim());
    for h in 0..nh {
        for k in 0..hd {
            let row = h * hd + k;
            if k < half {
                out.row_mut(row).assign(&w.row(row + half).mapv(|v| -v));
            } else {
                out.row_mut(row).assign(&w.row(row - half));
            }
        }
    }
    out
}

/// Declare the graph for `dims`. Parameter values come from [`graph_params`].
pub fn build_graph(cfg: &TimestralConfig, dims: GraphDims) -> HirModule {
    let mut hir = HirModule::new("timestral");
    let mut b = Builder {
        cfg,
        dims,
        hir: &mut hir,
        params: HashMap::new(),
    };
    b.build();
    hir
}

struct Builder<'a> {
    cfg: &'a TimestralConfig,
    dims: GraphDims,
    hir: &'a mut HirModule,
    /// One node per param name: binding is by name and reaches a single node,
    /// so a second `Param` of the same name would keep its initial zeros.
    params: HashMap<String, HirNodeId>,
}

impl Builder<'_> {
    fn g(&mut self) -> HirMut<'_> {
        HirMut::new(self.hir)
    }

    fn param(&mut self, name: &str, shape: &[usize]) -> HirNodeId {
        if let Some(&id) = self.params.get(name) {
            return id;
        }
        let id = self.hir.param(name, Shape::new(shape, F32));
        self.params.insert(name.to_string(), id);
        id
    }

    fn linear(&mut self, x: HirNodeId, name: &str, inp: usize, out: usize) -> HirNodeId {
        let w = self.param(name, &[inp, out]);
        self.g().mm(x, w)
    }

    fn rms(&mut self, x: HirNodeId, name: &str) -> HirNodeId {
        let d = self.cfg.d_model;
        let gamma = self.param(name, &[d]);
        let beta = self.param("zero_bias", &[d]);
        let eps = self.cfg.norm_eps;
        self.g().rms_norm(x, gamma, beta, eps)
    }

    fn build(&mut self) {
        let cfg = self.cfg;
        let GraphDims { r, l, n_dec, .. } = self.dims;
        let s = self.dims.s();
        let d = cfg.d_model;
        let ps = cfg.patch_size;

        let patches = self.hir.input("patches", Shape::new(&[r * l, 2 * ps], F32));
        let dec = self.hir.input("dec", Shape::new(&[r * n_dec, d], F32));
        // Inference feeds `Op::Rope` one row of `head_dim / 2` per token; the
        // training graph rotates by hand with full-width rows.
        let rope_w = if self.dims.train {
            d
        } else {
            cfg.head_dim() / 2
        };
        let cos = self
            .hir
            .input("rope_cos", Shape::new(&[r * s, rope_w], F32));
        let sin = self
            .hir
            .input("rope_sin", Shape::new(&[r * s, rope_w], F32));
        let time_bias = self.hir.input("time_bias", Shape::new(&[r, s, s], F32));
        let var_bias = has_variate_attention(cfg)
            .then(|| self.hir.input("var_bias", Shape::new(&[s, r, r], F32)));

        let enc = self.res_block("patch", patches, 2 * ps, d, true, r * l);
        let enc = self.g().reshape_(enc, vec![r as i64, l as i64, d as i64]);
        let dec = self
            .g()
            .reshape_(dec, vec![r as i64, n_dec as i64, d as i64]);
        let x = self.g().concat_(vec![enc, dec], 1);
        let mut x = self.g().reshape_(x, vec![(r * s) as i64, d as i64]);

        let schedule: Vec<LayerKind> = cfg.block_schedule.clone();
        for (i, kind) in schedule.iter().enumerate() {
            for (j, &is_time) in kind.axes().iter().enumerate() {
                let t = format!("l{i}.a{j}");
                let xn = self.rms(x, &format!("{t}.norm"));
                let a = if is_time {
                    self.time_attention(&t, xn, cos, sin, time_bias)
                } else {
                    self.variate_attention(&t, xn, var_bias.expect("variate bias input"))
                };
                x = self.g().add(x, a);
            }
            let t = format!("l{i}.mlp");
            let xn = self.rms(x, &format!("{t}.norm"));
            let m = self.mlp(&t, xn, cfg.mlp_hidden_dim);
            x = self.g().add(x, m);
        }
        let hidden = self.rms(x, "out_norm");

        let h3 = self
            .g()
            .reshape_(hidden, vec![r as i64, s as i64, d as i64]);
        let dec_h = self.g().narrow_(h3, 1, l, n_dec);
        let dec_h = self.g().reshape_(dec_h, vec![(r * n_dec) as i64, d as i64]);
        let head = self.res_block("head", dec_h, d, ps * cfg.num_quantiles(), false, r * n_dec);
        self.hir.outputs = if self.dims.hidden {
            vec![head, hidden]
        } else {
            vec![head]
        };
    }

    /// `output(silu(hidden(x) + b)) + b + residual(x) + b`.
    fn res_block(
        &mut self,
        tag: &str,
        x: HirNodeId,
        inp: usize,
        out: usize,
        biased: bool,
        rows: usize,
    ) -> HirNodeId {
        let hid = 4 * self.cfg.d_model;
        let mut h = self.linear(x, &format!("{tag}.hidden"), inp, hid);
        if biased {
            h = self.bias(h, &format!("{tag}.hidden_b"), hid);
        }
        h = self
            .g()
            .activation(Activation::Silu, h, Shape::new(&[rows, hid], F32));
        let mut o = self.linear(h, &format!("{tag}.output"), hid, out);
        if biased {
            o = self.bias(o, &format!("{tag}.output_b"), out);
        }
        let mut res = self.linear(x, &format!("{tag}.residual"), inp, out);
        if biased {
            res = self.bias(res, &format!("{tag}.residual_b"), out);
        }
        self.g().add(o, res)
    }

    fn bias(&mut self, x: HirNodeId, name: &str, n: usize) -> HirNodeId {
        let b = self.param(name, &[n]);
        self.g().add(x, b)
    }

    /// `down(gate(x) * silu(up(x)))`.
    fn mlp(&mut self, t: &str, x: HirNodeId, hidden: usize) -> HirNodeId {
        let d = self.cfg.d_model;
        let rows = self.dims.r * self.dims.s();
        let (gate, up) = if self.dims.train {
            let gate = self.linear(x, &format!("{t}.gate"), d, hidden);
            (gate, self.linear(x, &format!("{t}.up"), d, hidden))
        } else {
            let both = self.linear(x, &format!("{t}.gate_up"), d, 2 * hidden);
            let gate = self.g().narrow_(both, 1, 0, hidden);
            (gate, self.g().narrow_(both, 1, hidden, hidden))
        };
        let up = self
            .g()
            .activation(Activation::Silu, up, Shape::new(&[rows, hidden], F32));
        let h = self.g().mul(gate, up);
        self.linear(h, &format!("{t}.down"), hidden, d)
    }

    /// Multi-head attention over `[batch, seq]` rows of `x` (`[batch*seq, D]`),
    /// with an additive `[batch, seq, seq]` bias; returns `[batch*seq, D]` before `wO`.
    fn sdpa(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        bias: HirNodeId,
        batch: usize,
        seq: usize,
    ) -> HirNodeId {
        let nh = self.cfg.num_heads as i64;
        let hd = self.cfg.head_dim() as i64;
        let (b, sq) = (batch as i64, seq as i64);
        let q = self.g().reshape_(q, vec![b, sq, nh, hd]);
        let q = self.g().transpose_(q, vec![0, 2, 1, 3]);
        let k = self.g().reshape_(k, vec![b, sq, nh, hd]);
        let k_t = self.g().transpose_(k, vec![0, 2, 3, 1]);
        let v = self.g().reshape_(v, vec![b, sq, nh, hd]);
        let v = self.g().transpose_(v, vec![0, 2, 1, 3]);
        let scores = self.g().mm(q, k_t);
        let bias = self.g().reshape_(bias, vec![b, 1, sq, sq]);
        let scores = self.g().add(scores, bias);
        let probs = self.g().sm(scores, -1);
        let ctx = self.g().mm(probs, v);
        let ctx = self.g().transpose_(ctx, vec![0, 2, 1, 3]);
        self.g().reshape_(ctx, vec![b * sq, nh * hd])
    }

    /// `Op::Attention` over `[batch*seq, D]` rows with an additive `[batch,
    /// seq, seq]` bias shared by the heads: no head transposes and no
    /// materialized score tensor, unlike [`Self::sdpa`] (kept for training).
    fn fused_attention(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        bias: HirNodeId,
        batch: usize,
        seq: usize,
    ) -> HirNodeId {
        let (nh, hd, d) = (self.cfg.num_heads, self.cfg.head_dim(), self.cfg.d_model);
        let (b, sq) = (batch as i64, seq as i64);
        let q = self.g().reshape_(q, vec![b, sq, d as i64]);
        let k = self.g().reshape_(k, vec![b, sq, d as i64]);
        let v = self.g().reshape_(v, vec![b, sq, d as i64]);
        let bias = self.g().reshape_(bias, vec![b, 1, sq, sq]);
        let bias = self.g().expand_(bias, vec![b, nh as i64, sq, sq]);
        let ctx = self
            .g()
            .attention_bias(q, k, v, bias, nh, hd, Shape::new(&[batch, seq, d], F32));
        self.g().reshape_(ctx, vec![b * sq, d as i64])
    }

    /// The fused input projection `[rows, k*D]` (q, k, v, then q_rot, k_rot for time
    /// attention, then the gate) and a slicer for its `D`-wide parts.
    fn project(&mut self, t: &str, x: HirNodeId, is_time: bool) -> (HirNodeId, usize) {
        let d = self.cfg.d_model;
        let gated = self.cfg.gated(is_time);
        if !self.dims.train {
            // q, k, v (+ gate): RoPE and the score scale are applied by ops.
            let parts = 3 + gated as usize;
            return (self.linear(x, &format!("{t}.proj"), d, parts * d), parts);
        }
        let parts = 3 + if is_time { 2 } else { 0 } + gated as usize;
        // Same column layout as `graph_params` builds on the host.
        let q = self.param(&format!("{t}.q"), &[d, d]);
        let k = self.param(&format!("{t}.k"), &[d, d]);
        let v = self.param(&format!("{t}.v"), &[d, d]);
        let mut cols = vec![q, k, v];
        if is_time {
            let perm = self.param("rope_perm", &[d, d]);
            cols.push(self.g().mm(q, perm));
            cols.push(self.g().mm(k, perm));
        }
        if gated {
            cols.push(self.param(&format!("{t}.gate"), &[d, d]));
        }
        let w = self.g().concat_(cols, 1);
        (self.g().mm(x, w), parts)
    }

    fn part(&mut self, proj: HirNodeId, i: usize) -> HirNodeId {
        let d = self.cfg.d_model;
        self.g().narrow_(proj, 1, i * d, d)
    }

    /// `wO(ctx * sigmoid(gate))`, the gate being the last projection part when present.
    fn gate_and_project(
        &mut self,
        t: &str,
        is_time: bool,
        ctx: HirNodeId,
        proj: (HirNodeId, usize),
        rows: usize,
    ) -> HirNodeId {
        let d = self.cfg.d_model;
        let ctx = if self.cfg.gated(is_time) {
            let g = self.part(proj.0, proj.1 - 1);
            let g = self
                .g()
                .activation(Activation::Sigmoid, g, Shape::new(&[rows, d], F32));
            self.g().mul(ctx, g)
        } else {
            ctx
        };
        self.linear(ctx, &format!("{t}.o"), d, d)
    }

    fn time_attention(
        &mut self,
        t: &str,
        xn: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        bias: HirNodeId,
    ) -> HirNodeId {
        let (r, s) = (self.dims.r, self.dims.s());
        let proj = self.project(t, xn, true);
        if !self.dims.train {
            let (d, hd) = (self.cfg.d_model, self.cfg.head_dim());
            let rows = Shape::new(&[r * s, d], F32);
            let q = self.part(proj.0, 0);
            let q = self.hir.rope(q, cos, sin, hd, hd, rows.clone());
            let k = self.part(proj.0, 1);
            let k = self.hir.rope(k, cos, sin, hd, hd, rows);
            let v = self.part(proj.0, 2);
            let ctx = self.fused_attention(q, k, v, bias, r, s);
            return self.gate_and_project(t, true, ctx, proj, r * s);
        }
        let rope = |b: &mut Self, i: usize| {
            let x = b.part(proj.0, i);
            let x_rot = b.part(proj.0, i + 3);
            let a = b.g().mul(x, cos);
            let c = b.g().mul(x_rot, sin);
            b.g().add(a, c)
        };
        let q = rope(self, 0);
        let k = rope(self, 1);
        let v = self.part(proj.0, 2);
        let ctx = self.sdpa(q, k, v, bias, r, s);
        self.gate_and_project(t, true, ctx, proj, r * s)
    }

    /// Attention across the rows of each patch slot: `[R, S]` → `[S, R]` and back.
    fn variate_attention(&mut self, t: &str, xn: HirNodeId, bias: HirNodeId) -> HirNodeId {
        let d = self.cfg.d_model;
        let (r, s) = (self.dims.r, self.dims.s());
        let xt = self.g().reshape_(xn, vec![r as i64, s as i64, d as i64]);
        let xt = self.g().transpose_(xt, vec![1, 0, 2]);
        let xt = self.g().reshape_(xt, vec![(s * r) as i64, d as i64]);
        let proj = self.project(t, xt, false);
        let q = self.part(proj.0, 0);
        let k = self.part(proj.0, 1);
        let v = self.part(proj.0, 2);
        let ctx = if self.dims.train {
            self.sdpa(q, k, v, bias, s, r)
        } else {
            self.fused_attention(q, k, v, bias, s, r)
        };
        let out = self.gate_and_project(t, false, ctx, proj, s * r);
        let out = self.g().reshape_(out, vec![s as i64, r as i64, d as i64]);
        let out = self.g().transpose_(out, vec![1, 0, 2]);
        self.g().reshape_(out, vec![(r * s) as i64, d as i64])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_lowers_with_variate_attention() {
        let cfg = TimestralConfig::synth_tiny();
        let w = TimestralWeights::synth(&cfg, 3);
        let dims = GraphDims {
            r: 2,
            l: 4,
            n_dec: 2,
            hidden: true,
            train: false,
        };
        let hir = build_graph(&cfg, dims);
        hir.lower_to_mir().expect("lower");
        let params = graph_params(&cfg, &w);
        // q, k, v, gate (RoPE and the score scale are ops, not folded in).
        let d = cfg.d_model;
        assert_eq!(params["l1.a0.proj"].len(), d * 4 * d);
        assert_eq!(params["l1.a1.proj"].len(), d * 4 * d);
    }

    /// The in-graph `q P` of a training graph equals `rotate_half_rows` of the
    /// (scaled) query weight: the training graph's hand-built RoPE.
    #[test]
    fn rope_perm_matches_host_rotation() {
        let cfg = TimestralConfig::synth_tiny();
        let w = TimestralWeights::synth(&cfg, 5);
        let d = cfg.d_model;
        let train = graph_params_train(&cfg, &w);
        let (q, perm) = (&train["l0.a0.q"], &train["rope_perm"]);
        let scale = 1.0 / (cfg.head_dim() as f32).sqrt();
        let wq = w.layers[0].attns[0]
            .qkv
            .slice(ndarray::s![..d, ..])
            .mapv(|v| v * scale);
        let rot = transposed(&rotate_half_rows(&wq, cfg.num_heads, cfg.head_dim()));
        for i in 0..d {
            for c in 0..d {
                let qp: f32 = (0..d).map(|j| q[i * d + j] * perm[j * d + c]).sum();
                assert_eq!(qp, rot[i * d + c], "row {i} col {c}");
            }
        }
    }

    #[test]
    fn rope_identity_past_rope_dim() {
        let cfg = TimestralConfig::synth_tiny();
        let (cos, sin) = rope_tables(&cfg, &[5]);
        let hd = cfg.head_dim();
        // hd = 16, rope_dim = 8: dims 4..8 and 12..16 of each head do not rotate.
        for k in [4, 7, 12, 15] {
            assert_eq!((cos[k], sin[k]), (1.0, 0.0));
            assert_eq!((cos[hd + k], sin[hd + k]), (1.0, 0.0));
        }
        assert_eq!(cos[0], cos[8]);
        assert_ne!(sin[0], 0.0);
    }
}
