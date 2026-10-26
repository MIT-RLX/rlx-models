//! Building a raw `rlx-ir` [`Graph`] from a [`WeightMap`] — the plumbing every
//! hand-written model port repeats.
//!
//! * shape-checked extraction: [`take_vec`], [`take_tensor`], [`take_linear`],
//!   [`take_conv`], [`move_key`];
//! * declaring a checkpoint tensor as a graph param while recording its data in a
//!   [`Params`] list: [`declare`], [`vec_param`], [`lin_param`], [`conv_param`],
//!   [`bias_param`], [`mat_param`], [`take`], [`fold_bn`];
//! * [`ParamBuilder`] / [`LayerNormBuilder`]: the same for a builder struct that names
//!   params `p0, p1, …` in declaration order.
//!
//! These came verbatim from the exg model ports, where each crate carried its own
//! copy; keep them bit-for-bit stable (op order fixes param names and node ids).
//! `elu` and the softmaxes are written as compositions on purpose — they round
//! differently from `Activation::Elu` / `Op::Softmax`, and the ports' parity tests
//! pin the composite. [`fold_bn`] uses `BN_EPS = 1e-5`.

use std::collections::HashMap;

use anyhow::{Result, ensure};
use rlx_ir::op::{Activation, Op, ReduceOp};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};

use crate::weight_map::{WeightMap, transpose_2d as transpose2d};

/// Take a 1-D tensor of exactly `n` elements.
pub fn take_vec(wm: &mut WeightMap, key: &str, n: usize) -> Result<Vec<f32>> {
    let (b, bs) = wm.take(key)?;
    ensure!(bs == [n], "{key} expected [{n}], got {bs:?}");
    Ok(b)
}

/// Take a tensor whose shape is exactly `dims`.
pub fn take_tensor(wm: &mut WeightMap, key: &str, dims: &[usize]) -> Result<Vec<f32>> {
    let (w, ws) = wm.take(key)?;
    ensure!(ws == dims, "{key} expected {dims:?}, got {ws:?}");
    Ok(w)
}

/// Take an `nn.Linear` `[out, in]` weight + `[out]` bias.
pub fn take_linear(
    wm: &mut WeightMap,
    key: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (w, ws) = wm.take(&format!("{key}.weight"))?;
    ensure!(
        ws == [out_dim, in_dim],
        "{key}.weight expected [{out_dim},{in_dim}], got {ws:?}"
    );
    let b = take_vec(wm, &format!("{key}.bias"), out_dim)?;
    Ok((w, b))
}

/// Take a 2-D conv weight `[c_out, c_in/groups, kh, kw]` (no bias).
pub fn take_conv(
    wm: &mut WeightMap,
    key: &str,
    c_out: usize,
    c_in_g: usize,
    kh: usize,
    kw: usize,
) -> Result<Vec<f32>> {
    let (w, ws) = wm.take(&format!("{key}.weight"))?;
    ensure!(
        ws == [c_out, c_in_g, kh, kw],
        "{key}.weight expected [{c_out},{c_in_g},{kh},{kw}], got {ws:?}"
    );
    Ok(w)
}

/// Move `from` out of `src` into `dst` under the name `to`.
pub fn move_key(
    src: &mut WeightMap,
    dst: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    from: &str,
    to: &str,
) -> Result<()> {
    let (data, shape) = src.take(from)?;
    dst.insert(to.to_string(), (data, shape));
    Ok(())
}

const F32: DType = DType::F32;
const BN_EPS: f32 = 1e-5;

/// Named `(param_name, row-major data)` pairs to bind before running.
pub type Params = Vec<(String, Vec<f32>)>;

/// Declare `data` as graph param `name` with shape `dims`.
pub fn declare(
    g: &mut Graph,
    params: &mut Params,
    name: &str,
    data: Vec<f32>,
    dims: &[usize],
) -> NodeId {
    let id = g.param(name, Shape::new(dims, F32));
    params.push((name.to_string(), data));
    id
}

/// Declare a 1-D parameter taken verbatim from `wm`.
pub fn vec_param(
    g: &mut Graph,
    params: &mut Params,
    wm: &mut WeightMap,
    key: &str,
    expect: usize,
) -> Result<NodeId> {
    let (data, shape) = wm.take(key)?;
    ensure!(
        data.len() == expect,
        "{key}: expected {expect} elements, got shape {shape:?}"
    );
    let id = g.param(key, Shape::new(&[expect], F32));
    params.push((key.to_string(), data));
    Ok(id)
}

/// Declare a 2-D `[rows, cols]` tensor taken verbatim from `wm`.
pub fn mat_param(
    g: &mut Graph,
    params: &mut Params,
    wm: &mut WeightMap,
    key: &str,
    rows: usize,
    cols: usize,
) -> Result<NodeId> {
    let (data, shape) = wm.take(key)?;
    ensure!(
        shape == vec![rows, cols],
        "{key}: expected [{rows},{cols}], got {shape:?}"
    );
    let id = g.param(key, Shape::new(&[rows, cols], F32));
    params.push((key.to_string(), data));
    Ok(id)
}

/// Declare a torch Linear weight `[out, in]` stored transposed to `[in, out]`; returns `(node, out, in)`.
pub fn lin_param(
    g: &mut Graph,
    params: &mut Params,
    wm: &mut WeightMap,
    key: &str,
) -> Result<(NodeId, usize, usize)> {
    let (data, shape) = wm.take(key)?;
    ensure!(
        shape.len() == 2,
        "{key}: expected 2-D weight, got {shape:?}"
    );
    let (out_d, in_d) = (shape[0], shape[1]);
    let t = transpose2d(&data, out_d, in_d);
    let id = g.param(key, Shape::new(&[in_d, out_d], F32));
    params.push((key.to_string(), t));
    Ok((id, out_d, in_d))
}

/// Declare a conv weight `[c_out, c_in, kh, kw]` taken verbatim from `wm`.
pub fn conv_param(
    g: &mut Graph,
    params: &mut Params,
    wm: &mut WeightMap,
    key: &str,
    c_out: usize,
    c_in: usize,
    kh: usize,
    kw: usize,
) -> Result<NodeId> {
    let (data, shape) = wm.take(key)?;
    ensure!(
        shape.len() == 4
            && shape[0] == c_out
            && shape[1] == c_in
            && shape[2] == kh
            && shape[3] == kw,
        "{key}: expected [{c_out},{c_in},{kh},{kw}], got {shape:?}"
    );
    let id = g.param(key, Shape::new(&[c_out, c_in, kh, kw], F32));
    params.push((key.to_string(), data));
    Ok(id)
}

/// Declare a per-channel bias as `[1, c_out, 1, 1]`.
pub fn bias_param(
    g: &mut Graph,
    params: &mut Params,
    wm: &mut WeightMap,
    key: &str,
    c_out: usize,
) -> Result<NodeId> {
    let (data, _) = wm.take(key)?;
    ensure!(
        data.len() == c_out,
        "{key}: expected {c_out} elements, got {}",
        data.len()
    );
    let id = g.param(key, Shape::new(&[1, c_out, 1, 1], F32));
    params.push((key.to_string(), data));
    Ok(id)
}

/// Declare param `name` with `shape`, taking its data from `wm` unchecked.
pub fn take(
    g: &mut Graph,
    params: &mut Params,
    wm: &mut WeightMap,
    name: &str,
    shape: Shape,
) -> Result<NodeId> {
    let node = g.param(name.to_string(), shape);
    let (data, _s) = wm.take(name)?;
    params.push((name.to_string(), data));
    Ok(node)
}

/// Eval-mode BatchNorm folded to a per-channel affine `x·scale + shift` on `[B, ch, H, W]`.
pub fn fold_bn(
    g: &mut Graph,
    params: &mut Params,
    wm: &mut WeightMap,
    key: &str,
    ch: usize,
    x: NodeId,
) -> Result<NodeId> {
    let (gamma, _) = wm.take(&format!("{key}.weight"))?;
    let (beta, _) = wm.take(&format!("{key}.bias"))?;
    let (mean, _) = wm.take(&format!("{key}.running_mean"))?;
    let (var, _) = wm.take(&format!("{key}.running_var"))?;
    ensure!(
        gamma.len() == ch && beta.len() == ch && mean.len() == ch && var.len() == ch,
        "{key}: batchnorm params must all be length {ch}"
    );
    let mut scale = vec![0f32; ch];
    let mut shift = vec![0f32; ch];
    for c in 0..ch {
        let s = gamma[c] / (var[c] + BN_EPS).sqrt();
        scale[c] = s;
        shift[c] = beta[c] - mean[c] * s;
    }
    let s = g.param(format!("{key}.__scale"), Shape::new(&[1, ch, 1, 1], F32));
    params.push((format!("{key}.__scale"), scale));
    let sh = g.param(format!("{key}.__shift"), Shape::new(&[1, ch, 1, 1], F32));
    params.push((format!("{key}.__shift"), shift));
    let y = g.mul(x, s);
    Ok(g.add(y, sh))
}

/// ELU(α=1) as `relu(x) + exp(min(x, 0)) − 1`.
pub fn elu(g: &mut Graph, x: NodeId) -> NodeId {
    let r = g.relu(x);
    let nx = g.neg(x);
    let rnx = g.relu(nx);
    let minx = g.neg(rnx);
    let e = g.exp(minx);
    let ep = g.add(r, e); // relu(x) + exp(min(x,0))
    let one = g.constant(1.0, F32);
    g.sub(ep, one)
}

/// Softmax over the last axis of `[n, s, K]`.
/// Fold a BatchNorm's `weight`, `bias`, `running_mean`, `running_var` into a
/// per-channel `(scale, shift)` with the caller's epsilon:
/// `scale = γ/√(σ²+eps)`, `shift = β − μ·scale`.
///
/// Nine model crates carried this verbatim with their own `BN_EPS` (1e-5 or
/// 1e-3 — which is why `eps` is an argument rather than [`fold_bn`]'s constant).
pub fn fold_bn_scale_shift(
    wm: &mut WeightMap,
    key: &str,
    ch: usize,
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let gamma = take_vec(wm, &format!("{key}.weight"), ch)?;
    let beta = take_vec(wm, &format!("{key}.bias"), ch)?;
    let mean = take_vec(wm, &format!("{key}.running_mean"), ch)?;
    let var = take_vec(wm, &format!("{key}.running_var"), ch)?;
    let mut scale = vec![0f32; ch];
    let mut shift = vec![0f32; ch];
    for c in 0..ch {
        let s = gamma[c] / (var[c] + eps).sqrt();
        scale[c] = s;
        shift[c] = beta[c] - mean[c] * s;
    }
    Ok((scale, shift))
}

/// Numerically-robust softmax over any `axis` of a tensor of shape `full`.
pub fn softmax_axis(g: &mut Graph, x: NodeId, full: &[usize], axis: usize) -> NodeId {
    let mut kd = full.to_vec();
    kd[axis] = 1;
    let mx = g.add_node(
        Op::Reduce {
            op: ReduceOp::Max,
            axes: vec![axis],
            keep_dim: true,
        },
        vec![x],
        Shape::new(&kd, F32),
    );
    let e = g.sub(x, mx);
    let e = g.exp(e);
    let s = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![axis],
            keep_dim: true,
        },
        vec![e],
        Shape::new(&kd, F32),
    );
    g.div(e, s)
}

/// A single-axis reduction of a tensor of shape `dims`, keeping or dropping the axis.
pub fn reduce_axis(
    g: &mut Graph,
    op: ReduceOp,
    x: NodeId,
    dims: &[usize],
    axis: usize,
    keep: bool,
) -> NodeId {
    let mut out: Vec<usize> = dims.to_vec();
    if keep {
        out[axis] = 1;
    } else {
        out.remove(axis);
    }
    g.add_node(
        Op::Reduce {
            op,
            axes: vec![axis],
            keep_dim: keep,
        },
        vec![x],
        Shape::new(&out, F32),
    )
}

/// A torch `[out_d, in_d]` linear weight as a `[in_d, out_d]` param named `key`,
/// refusing any other stored shape (unlike [`lin_param`], which reads the shape).
pub fn lin_param_checked(
    g: &mut Graph,
    params: &mut Params,
    wm: &mut WeightMap,
    key: &str,
    out_d: usize,
    in_d: usize,
) -> Result<NodeId> {
    let (data, shape) = wm.take(key)?;
    ensure!(
        shape.len() == 2 && shape[0] == out_d && shape[1] == in_d,
        "{key}: expected torch [{out_d},{in_d}], got {shape:?}"
    );
    let t = transpose2d(&data, out_d, in_d);
    let id = g.param(key, Shape::new(&[in_d, out_d], F32));
    params.push((key.to_string(), t));
    Ok(id)
}

pub fn softmax_last(g: &mut Graph, scores: NodeId, n: usize, s: usize) -> NodeId {
    let shape = Shape::new(&[n, s, 1], DType::F32);
    let mx = g.add_node(
        Op::Reduce {
            op: ReduceOp::Max,
            axes: vec![2],
            keep_dim: true,
        },
        vec![scores],
        shape.clone(),
    );
    let e = g.sub(scores, mx);
    let e = g.exp(e);
    let sm = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![2],
            keep_dim: true,
        },
        vec![e],
        shape,
    );
    g.div(e, sm)
}

/// A graph builder whose params are named `p0, p1, …` in declaration order.
///
/// The model crates each had an identical `struct Builder { g, params, n, .. }` with
/// these methods; implement [`ParamBuilder::parts`] to get them. Every method keeps
/// the op order of those copies, so param names and node ids are unchanged.
pub trait ParamBuilder {
    /// The graph, the recorded `(name, data)` params, and the next param index.
    fn parts(&mut self) -> (&mut Graph, &mut Params, &mut usize);

    /// Declare `data` as the next param `p{n}` with shape `dims`.
    fn param(&mut self, data: Vec<f32>, dims: &[usize]) -> NodeId {
        debug_assert_eq!(data.len(), dims.iter().product::<usize>());
        let (g, params, n) = self.parts();
        let name = format!("p{}", *n);
        *n += 1;
        let id = g.param(name.clone(), Shape::new(dims, F32));
        params.push((name, data));
        id
    }

    /// `y = x @ Wᵀ + b`, `W` in torch `[out,in]` layout. `x` may be rank-2 or -3.
    fn linear(
        &mut self,
        wm: &mut WeightMap,
        key: &str,
        x: NodeId,
        out: usize,
        inp: usize,
    ) -> Result<NodeId> {
        let (w, b) = take_linear(wm, key, out, inp)?;
        let wt = self.param(transpose2d(&w, out, inp), &[inp, out]);
        let y = self.parts().0.mm(x, wt);
        let bn = self.param(b, &[out]);
        Ok(self.parts().0.add(y, bn))
    }

    /// Manual softmax over axis 3 of `[d0,d1,d2,K]` (last axis = keys).
    fn softmax_last4(&mut self, x: NodeId, d0: usize, d1: usize, d2: usize) -> NodeId {
        let g = self.parts().0;
        let sh = Shape::new(&[d0, d1, d2, 1], F32);
        let m = g.reduce(x, ReduceOp::Max, vec![3], true, sh.clone());
        let e = g.sub(x, m);
        let e = g.exp(e);
        let s = g.reduce(e, ReduceOp::Sum, vec![3], true, sh);
        g.div(e, s)
    }

    /// Numerically-robust softmax over the last axis of `[b, s, s]`.
    fn softmax_last(&mut self, x: NodeId, b: usize, s: usize) -> NodeId {
        softmax_last(self.parts().0, x, b, s)
    }

    /// An `f32` scalar constant.
    fn cst(&mut self, v: f64) -> NodeId {
        self.parts().0.constant(v, F32)
    }

    /// Elementwise sigmoid.
    fn sigmoid(&mut self, x: NodeId) -> NodeId {
        let g = self.parts().0;
        let s = g.shape(x).clone();
        g.activation(Activation::Sigmoid, x, s)
    }

    /// A channel-axis affine `[c]` broadcast as `[1,c,1,1]` over an NCHW tensor.
    fn affine(&mut self, x: NodeId, scale: Vec<f32>, shift: Vec<f32>, c: usize) -> NodeId {
        let sn = self.param(scale, &[1, c, 1, 1]);
        let shn = self.param(shift, &[1, c, 1, 1]);
        let g = self.parts().0;
        let y = g.mul(x, sn);
        g.add(y, shn)
    }

    /// Folded-BatchNorm affine over `[N, c, l]`: `x·scale + shift`.
    fn bn(&mut self, x: NodeId, scale: Vec<f32>, shift: Vec<f32>, c: usize) -> NodeId {
        let s = self.param(scale, &[1, c, 1]);
        let sh = self.param(shift, &[1, c, 1]);
        let g = self.parts().0;
        let y = g.mul(x, s);
        g.add(y, sh)
    }
}

/// [`ParamBuilder`] plus LayerNorm; `LN_EPS` is the crate's own epsilon.
pub trait LayerNormBuilder: ParamBuilder {
    /// LayerNorm epsilon (each model crate sets its own).
    const LN_EPS: f32;

    /// LayerNorm over the last axis with weights consumed from `wm`.
    fn ln(&mut self, wm: &mut WeightMap, key: &str, x: NodeId, d: usize) -> Result<NodeId> {
        let gw = take_vec(wm, &format!("{key}.weight"), d)?;
        let gb = take_vec(wm, &format!("{key}.bias"), d)?;
        let gwn = self.param(gw, &[d]);
        let gbn = self.param(gb, &[d]);
        Ok(self.parts().0.ln(x, gwn, gbn, Self::LN_EPS))
    }
}
