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

//! The stochastic duration predictor ([`crate::sdp::Sdp`]) as an rlx
//! graph: `dp_in` `[1, 192, t]`, the flow noise `[2, t]` and the length
//! scale `[1]` in, each phoneme's frames out (`durations` `[t]`, f32).
//!
//! The rational-quadratic spline's inverse is computed for every element
//! — its bin a count of the boundaries at or below it, each bin's values
//! picked by a one-hot — and the elements outside `[−5, 5]` passed through
//! (`Op::Where`): the boolean-mask indexing the ONNX export has, without a
//! data-dependent shape.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_ir::op::{Activation, CmpOp};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};

const NUM_BINS: usize = 10;
const TAIL: f64 = 5.0;
const FILTER: usize = 192;
const EPS_LN: f32 = 1e-5;
const MIN_BW: f64 = 1e-3;
const MIN_BH: f64 = 1e-3;
const MIN_DER: f64 = 1e-3;
const DILATIONS: [usize; 3] = [1, 3, 9];

struct B<'a> {
    g: Graph,
    /// Each weight's dims, as the manifest gives them.
    dims: &'a HashMap<String, Vec<usize>>,
    t: usize,
}

impl B<'_> {
    fn c(&mut self, v: f64) -> NodeId {
        self.g.constant(v, DType::F32)
    }

    fn p(&mut self, name: &str, shape: &[usize]) -> NodeId {
        self.g.param(name, Shape::new(shape, DType::F32))
    }

    /// `x` `[1, ci, 1, t]` through the conv `name` (`same` padding, its
    /// bias added) → `[1, co, 1, t]`.
    fn conv(&mut self, x: NodeId, name: &str, dilation: usize, groups: usize) -> NodeId {
        let d = self.dims[&format!("{name}.weight")].clone();
        let (co, ci, k) = (d[0], d[1], d[2]);
        let w = self.p(&format!("{name}.weight"), &[co, ci, 1, k]);
        let pad = (k - 1) * dilation / 2;
        let y = self
            .g
            .conv2d(x, w, [1, k], [1, 1], [0, pad], [1, dilation], groups);
        let b = self.p(&format!("{name}.bias"), &[co]);
        let b = self.g.reshape_(b, vec![1, co as i64, 1, 1]);
        self.g.add(y, b)
    }

    /// VITS' LayerNorm over the channels of `x` `[1, c, 1, t]`.
    fn norm(&mut self, x: NodeId, name: &str) -> NodeId {
        let t = self.t as i64;
        let x = self.g.reshape_(x, vec![FILTER as i64, t]);
        let x = self.g.transpose_(x, vec![1, 0]);
        let gamma = self.p(&format!("{name}.gamma"), &[FILTER]);
        let beta = self.p(&format!("{name}.beta"), &[FILTER]);
        let y = self.g.ln(x, gamma, beta, EPS_LN);
        let y = self.g.transpose_(y, vec![1, 0]);
        self.g.reshape_(y, vec![1, FILTER as i64, 1, t])
    }

    /// DDSConv: three dilated depthwise → norm → GELU → 1×1 → norm → GELU,
    /// each residual; `cond` added first.
    fn ddsconv(&mut self, x: NodeId, prefix: &str, cond: Option<NodeId>) -> NodeId {
        let mut x = match cond {
            Some(c) => self.g.add(x, c),
            None => x,
        };
        for (i, dil) in DILATIONS.iter().enumerate() {
            let y = self.conv(x, &format!("{prefix}.convs_sep.{i}"), *dil, FILTER);
            let y = self.norm(y, &format!("{prefix}.norms_1.{i}"));
            let y = self.g.gelu(y);
            let y = self.conv(y, &format!("{prefix}.convs_1x1.{i}"), 1, 1);
            let y = self.norm(y, &format!("{prefix}.norms_2.{i}"));
            let y = self.g.gelu(y);
            x = self.g.add(x, y);
        }
        x
    }

    /// `a op b`, broadcast.
    fn cmp(&mut self, op: CmpOp, a: NodeId, b: NodeId) -> NodeId {
        let dims = |n: NodeId| -> Vec<usize> {
            self.g
                .node(n)
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect()
        };
        let (da, db) = (dims(a), dims(b));
        let rank = da.len().max(db.len());
        let at = |d: &[usize], i: usize| {
            if i + d.len() >= rank {
                d[i + d.len() - rank]
            } else {
                1
            }
        };
        let dims: Vec<usize> = (0..rank).map(|i| at(&da, i).max(at(&db, i))).collect();
        self.g
            .add_node(Op::Compare(op), vec![a, b], Shape::new(&dims, DType::Bool))
    }

    fn act(&mut self, a: Activation, x: NodeId) -> NodeId {
        let shape = self.g.node(x).shape.clone();
        self.g.add_node(Op::Activation(a), vec![x], shape)
    }

    /// `[t, n]` of `edge`, `inner` (`[t, n − 2]`) and `edge` again.
    fn bracket(&mut self, inner: NodeId, first: f64, last: f64) -> NodeId {
        let column = |b: &mut Self, v: f64| {
            let c = b.g.full(&[1, 1], v as f32, DType::F32);
            b.g.add_node(
                Op::Expand {
                    target_shape: vec![b.t as i64, 1],
                },
                vec![c],
                Shape::new(&[b.t, 1], DType::F32),
            )
        };
        let a = column(self, first);
        let b = column(self, last);
        self.g.concat_(vec![a, inner, b], 1)
    }

    /// Softmaxed `u` `[t, NUM_BINS]` as the bins' edges in `[−TAIL, TAIL]`
    /// (`[t, NUM_BINS + 1]`) and their widths.
    fn edges(&mut self, u: NodeId, min: f64) -> (NodeId, NodeId) {
        let s = self.g.sm(u, 1);
        let k = self.c(1.0 - min * NUM_BINS as f64);
        let s = self.g.mul(s, k);
        let m = self.c(min);
        let w = self.g.add(s, m);
        let cum = self.g.add_node(
            Op::Cumsum {
                axis: 1,
                exclusive: false,
            },
            vec![w],
            Shape::new(&[self.t, NUM_BINS], DType::F32),
        );
        let two = self.c(2.0 * TAIL);
        let cum = self.g.mul(cum, two);
        let tail = self.c(TAIL);
        let cum = self.g.sub(cum, tail);
        let inner = self.g.narrow_(cum, 1, 0, NUM_BINS - 1);
        let edges = self.bracket(inner, -TAIL, TAIL);
        let hi = self.g.narrow_(edges, 1, 1, NUM_BINS);
        let lo = self.g.narrow_(edges, 1, 0, NUM_BINS);
        let widths = self.g.sub(hi, lo);
        (edges, widths)
    }

    /// Row `bin` of each of `x`'s rows, by the one-hot `hot` `[t, n]`.
    fn pick(&mut self, x: NodeId, hot: NodeId) -> NodeId {
        let y = self.g.mul(x, hot);
        self.g.sum(y, vec![1], false)
    }

    /// The inverse rational-quadratic spline of `x` `[t]` by `proj`
    /// `[1, 29, 1, t]`; outside `[−TAIL, TAIL]` the identity.
    fn spline_inverse(&mut self, x: NodeId, proj: NodeId) -> NodeId {
        let t = self.t;
        let p = self
            .g
            .reshape_(proj, vec![(3 * NUM_BINS - 1) as i64, t as i64]);
        let p = self.g.transpose_(p, vec![1, 0]);
        let scale = self.c(1.0 / (FILTER as f64).sqrt());
        let uw = self.g.narrow_(p, 1, 0, NUM_BINS);
        let uw = self.g.mul(uw, scale);
        let uh = self.g.narrow_(p, 1, NUM_BINS, NUM_BINS);
        let uh = self.g.mul(uh, scale);
        let ud = self.g.narrow_(p, 1, 2 * NUM_BINS, NUM_BINS - 1);
        let (cw, widths) = self.edges(uw, MIN_BW);
        let (ch, heights) = self.edges(uh, MIN_BH);
        // The derivatives at the edges: softplus (`x` past 20) plus the
        // minimum, the outer two fixed.
        let edge = ((1.0 - MIN_DER).exp() - 1.0).ln();
        let raw = self.bracket(ud, edge, edge);
        let e = self.g.exp(raw);
        let one = self.c(1.0);
        let e1 = self.g.add(e, one);
        let sp = self.g.log(e1);
        let twenty = self.c(20.0);
        let big = self.cmp(CmpOp::Gt, raw, twenty);
        let sp = self.g.add_node(
            Op::Where,
            vec![big, raw, sp],
            Shape::new(&[t, NUM_BINS + 1], DType::F32),
        );
        let md = self.c(MIN_DER);
        let der = self.g.add(sp, md);

        // The bin: the edges at or below `x` (the last nudged up), less one.
        let nudge: Vec<f32> = (0..=NUM_BINS)
            .map(|k| if k == NUM_BINS { 1e-6 } else { 0.0 })
            .collect();
        let nudge = self.g.add_node(
            Op::Constant {
                data: nudge.iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
            vec![],
            Shape::new(&[1, NUM_BINS + 1], DType::F32),
        );
        let bounds = self.g.add(ch, nudge);
        let xc = self.g.reshape_(x, vec![t as i64, 1]);
        let ge = self.cmp(CmpOp::Ge, xc, bounds);
        let ge = self.g.cast(ge, DType::F32);
        let count = self.g.sum(ge, vec![1], true);
        let count = self.g.sub(count, one);
        let bin = self.g.clamp_(count, 0.0, (NUM_BINS - 1) as f32);
        let iota: Vec<f32> = (0..=NUM_BINS).map(|k| k as f32).collect();
        let iota = self.g.add_node(
            Op::Constant {
                data: iota.iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
            vec![],
            Shape::new(&[1, NUM_BINS + 1], DType::F32),
        );
        let hot = self.cmp(CmpOp::Eq, iota, bin);
        let hot = self.g.cast(hot, DType::F32);
        let hot_n = self.g.narrow_(hot, 1, 0, NUM_BINS);
        let bin1 = self.g.add(bin, one);
        let hot1 = self.cmp(CmpOp::Eq, iota, bin1);
        let hot1 = self.g.cast(hot1, DType::F32);

        let in_cw = self.pick(cw, hot);
        let in_w = self.pick(widths, hot_n);
        let in_ch = self.pick(ch, hot);
        let in_h = self.pick(heights, hot_n);
        let in_d = self.pick(der, hot);
        let in_d1 = self.pick(der, hot1);
        let delta = self.g.div(in_h, in_w);

        let dy = self.g.sub(x, in_ch);
        let two = self.c(2.0);
        let d2 = self.g.mul(two, delta);
        let s = self.g.add(in_d, in_d1);
        let common = self.g.sub(s, d2);
        let a1 = self.g.mul(dy, common);
        let dd = self.g.sub(delta, in_d);
        let a2 = self.g.mul(in_h, dd);
        let a = self.g.add(a1, a2);
        let b1 = self.g.mul(in_h, in_d);
        let b2 = self.g.mul(dy, common);
        let b = self.g.sub(b1, b2);
        let nd = self.g.neg(delta);
        let c = self.g.mul(nd, dy);
        let bb = self.g.mul(b, b);
        let four = self.c(4.0);
        let ac = self.g.mul(a, c);
        let ac4 = self.g.mul(four, ac);
        let disc = self.g.sub(bb, ac4);
        let sq = self.g.sqrt(disc);
        let nb = self.g.neg(b);
        let den = self.g.sub(nb, sq);
        let c2 = self.g.mul(two, c);
        let root = self.g.div(c2, den);
        let rw = self.g.mul(root, in_w);
        let y = self.g.add(rw, in_cw);

        let lo = self.c(-TAIL);
        let hi = self.c(TAIL);
        let above = self.cmp(CmpOp::Ge, x, lo);
        let below = self.cmp(CmpOp::Le, x, hi);
        let above = self.g.cast(above, DType::F32);
        let below = self.g.cast(below, DType::F32);
        let inside = self.g.mul(above, below);
        let inside = self.g.cast(inside, DType::Bool);
        self.g
            .add_node(Op::Where, vec![inside, y, x], Shape::new(&[t], DType::F32))
    }

    /// A ConvFlow in reverse: `z` `[2, t]`'s second row through the spline
    /// its first row and `cond` give.
    fn convflow_reverse(&mut self, z: NodeId, idx: usize, cond: NodeId) -> NodeId {
        let t = self.t as i64;
        let pfx = format!("dp.flows.{idx}");
        let x0 = self.g.narrow_(z, 0, 0, 1);
        let x1 = self.g.narrow_(z, 0, 1, 1);
        let x1 = self.g.reshape_(x1, vec![t]);
        let h = self.g.reshape_(x0, vec![1, 1, 1, t]);
        let h = self.conv(h, &format!("{pfx}.pre"), 1, 1);
        let h = self.ddsconv(h, &format!("{pfx}.convs"), Some(cond));
        let proj = self.conv(h, &format!("{pfx}.proj"), 1, 1);
        let y = self.spline_inverse(x1, proj);
        let y = self.g.reshape_(y, vec![1, t]);
        self.g.concat_(vec![x0, y], 0)
    }

    fn flip(&mut self, z: NodeId) -> NodeId {
        let a = self.g.narrow_(z, 0, 0, 1);
        let b = self.g.narrow_(z, 0, 1, 1);
        self.g.concat_(vec![b, a], 0)
    }
}

/// The duration predictor at `t` phonemes; `dims` its weights' dims.
pub fn build(dims: &HashMap<String, Vec<usize>>, t: usize) -> Graph {
    let mut b = B {
        g: Graph::new("durations"),
        dims,
        t,
    };
    let dp_in = b.g.input("dp_in", Shape::new(&[1, FILTER, t], DType::F32));
    let noise = b.g.input("noise", Shape::new(&[2, t], DType::F32));
    let length_scale = b.g.input("length_scale", Shape::new(&[1], DType::F32));
    let x = b.g.reshape_(dp_in, vec![1, FILTER as i64, 1, t as i64]);
    let x = b.conv(x, "dp.pre", 1, 1);
    let x = b.ddsconv(x, "dp.convs", None);
    let cond = b.conv(x, "dp.proj", 1, 1);

    let mut z = b.flip(noise);
    for (i, idx) in [7, 5, 3].into_iter().enumerate() {
        if i > 0 {
            z = b.flip(z);
        }
        z = b.convflow_reverse(z, idx, cond);
    }
    z = b.flip(z);
    // ElementwiseAffine in reverse (no `logs`: the scale is 1).
    let m = b.p("dp.flows.0.m", &[2, 1]);
    let z = b.g.sub(z, m);

    let logw = b.g.narrow_(z, 0, 0, 1);
    let logw = b.g.reshape_(logw, vec![t as i64]);
    let w = b.g.exp(logw);
    let w = b.g.mul(w, length_scale);
    let w = b.act(Activation::Ceil, w);
    let zero = b.c(0.0);
    let w = b.g.add_node(
        Op::Binary(rlx_ir::op::BinaryOp::Max),
        vec![w, zero],
        Shape::new(&[t], DType::F32),
    );
    b.g.node_mut(w).name = Some("durations".into());
    b.g.set_outputs(vec![w]);
    b.g
}

/// The duration predictor's weights in a split bundle (`dp_weights.f32`,
/// `dp_manifest.json`): each one's dims and values, by name.
#[allow(clippy::type_complexity)]
pub fn weights(
    split_dir: &Path,
) -> Result<(HashMap<String, Vec<usize>>, HashMap<String, Vec<f32>>)> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(split_dir.join("dp_manifest.json"))?)?;
    let blob = std::fs::read(split_dir.join("dp_weights.f32"))?;
    let flat: Vec<f32> = blob
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let (mut dims, mut values) = (HashMap::new(), HashMap::new());
    for (name, m) in manifest["weights"]
        .as_object()
        .context("dp_manifest.json has no weights")?
    {
        let at = |k: &str| m[k].as_u64().map(|v| v as usize);
        let (off, n) = (
            at("offset").context("offset")?,
            at("numel").context("numel")?,
        );
        let d: Vec<usize> = m["shape"]
            .as_array()
            .context("shape")?
            .iter()
            .filter_map(|v| v.as_u64().map(|v| v as usize))
            .collect();
        anyhow::ensure!(off + n <= flat.len(), "dp weight {name} out of range");
        dims.insert(name.clone(), d);
        values.insert(name.clone(), flat[off..off + n].to_vec());
    }
    anyhow::ensure!(
        !dims.contains_key("dp.flows.0.logs"),
        "the duration predictor's ElementwiseAffine has a scale: not this graph"
    );
    Ok((dims, values))
}
