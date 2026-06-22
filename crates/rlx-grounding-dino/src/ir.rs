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

//! Reusable HIR graph-building helpers for the IR (on-device) paths.
//!
//! Weights enter the checkpoint in PyTorch `[out, in]` layout; for the IR graph
//! `mm(x[rows, in], w[in, out])` we transpose them to `[in, out]` at param-load
//! time. These helpers run identically on every backend (CPU/Metal/MLX/CUDA/
//! WGPU/Vulkan) because they compose only ops that exist on all of them.

use rlx_ir::{DType, HirGraphExt, HirMut, HirNodeId, Shape};
use std::collections::HashMap;

pub type Params = HashMap<String, Vec<f32>>;

/// Transpose a row-major `[out, in]` weight to `[in, out]`.
pub fn transpose2d(w: &[f32], out: usize, in_dim: usize) -> Vec<f32> {
    let mut t = vec![0f32; in_dim * out];
    for o in 0..out {
        for i in 0..in_dim {
            t[i * out + o] = w[o * in_dim + i];
        }
    }
    t
}

/// Register a `[in, out]` param (transposed from PyTorch `[out, in]`) and return
/// its node.
pub fn weight_param(
    g: &mut HirMut<'_>,
    params: &mut Params,
    name: &str,
    w_pt: &[f32],
    out_dim: usize,
    in_dim: usize,
    scale: f32,
) -> HirNodeId {
    let mut wt = transpose2d(w_pt, out_dim, in_dim);
    if scale != 1.0 {
        for v in wt.iter_mut() {
            *v *= scale;
        }
    }
    params.insert(name.to_string(), wt);
    g.param(name, Shape::new(&[in_dim, out_dim], DType::F32))
}

/// Register a `[n]` vector param and return its node.
pub fn vec_param(
    g: &mut HirMut<'_>,
    params: &mut Params,
    name: &str,
    data: &[f32],
    scale: f32,
) -> HirNodeId {
    let v: Vec<f32> = if scale != 1.0 {
        data.iter().map(|x| x * scale).collect()
    } else {
        data.to_vec()
    };
    let n = v.len();
    params.insert(name.to_string(), v);
    g.param(name, Shape::new(&[n], DType::F32))
}

/// `y = x @ w^T + b` (optionally scaling the weight + bias by `scale`).
#[allow(clippy::too_many_arguments)]
pub fn linear(
    g: &mut HirMut<'_>,
    params: &mut Params,
    name: &str,
    x: HirNodeId,
    in_dim: usize,
    out_dim: usize,
    w_pt: &[f32],
    b: &[f32],
    scale: f32,
) -> HirNodeId {
    let w = weight_param(
        g,
        params,
        &format!("{name}.w"),
        w_pt,
        out_dim,
        in_dim,
        scale,
    );
    let mm = g.mm(x, w);
    if b.is_empty() {
        mm
    } else {
        let bp = vec_param(g, params, &format!("{name}.b"), b, scale);
        g.add(mm, bp)
    }
}

/// LayerNorm with registered gamma/beta params.
pub fn layer_norm(
    g: &mut HirMut<'_>,
    params: &mut Params,
    name: &str,
    x: HirNodeId,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> HirNodeId {
    let gp = vec_param(g, params, &format!("{name}.g"), gamma, 1.0);
    let bp = vec_param(g, params, &format!("{name}.b"), beta, 1.0);
    g.ln(x, gp, bp, eps)
}

/// Multi-head attention built from primitive ops with an additive bias INPUT.
///
/// `q_in`/`k_in`/`v_in` are nodes of shape `[l, d]`. `bias_node` is an input of
/// shape `[1, lq, lk]` (broadcast over heads), added to the scaled scores before
/// softmax. The `1/sqrt(head_dim)` scale is folded into the query weight. All
/// projections use PyTorch `[out, in]` weights. Returns `[lq, d]`.
#[allow(clippy::too_many_arguments)]
pub fn mha(
    g: &mut HirMut<'_>,
    params: &mut Params,
    name: &str,
    q_in: HirNodeId,
    k_in: HirNodeId,
    v_in: HirNodeId,
    lq: usize,
    lk: usize,
    d: usize,
    n_heads: usize,
    qw: &[f32],
    qb: &[f32],
    kw: &[f32],
    kb: &[f32],
    vw: &[f32],
    vb: &[f32],
    ow: &[f32],
    ob: &[f32],
    bias_node: HirNodeId,
) -> HirNodeId {
    let hd = d / n_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    // Fold the attention scale into the query projection.
    let q = linear(g, params, &format!("{name}.q"), q_in, d, d, qw, qb, scale);
    let k = linear(g, params, &format!("{name}.k"), k_in, d, d, kw, kb, 1.0);
    let v = linear(g, params, &format!("{name}.v"), v_in, d, d, vw, vb, 1.0);

    // [l, d] → [heads, l, hd]
    let q = g.reshape_(q, vec![lq as i64, n_heads as i64, hd as i64]);
    let q = g.transpose_(q, vec![1, 0, 2]); // [heads, lq, hd]
    let k = g.reshape_(k, vec![lk as i64, n_heads as i64, hd as i64]);
    let kt = g.transpose_(k, vec![1, 2, 0]); // [heads, hd, lk]
    let v = g.reshape_(v, vec![lk as i64, n_heads as i64, hd as i64]);
    let v = g.transpose_(v, vec![1, 0, 2]); // [heads, lk, hd]

    let scores = g.mm(q, kt); // [heads, lq, lk]
    let scores = g.add(scores, bias_node); // broadcast [1, lq, lk]
    let probs = g.sm(scores, -1); // softmax over lk
    let ctx = g.mm(probs, v); // [heads, lq, hd]
    let ctx = g.transpose_(ctx, vec![1, 0, 2]); // [lq, heads, hd]
    let ctx = g.reshape_(ctx, vec![lq as i64, d as i64]);
    linear(g, params, &format!("{name}.o"), ctx, d, d, ow, ob, 1.0)
}

/// Compile a built HIR graph + params on `device` and run with the given inputs.
/// Returns the flat output vectors.
pub fn compile_and_run(
    hir: rlx_ir::HirModule,
    params: Params,
    device: rlx_runtime::Device,
    inputs: &[(&str, &[f32])],
) -> anyhow::Result<Vec<Vec<f32>>> {
    use rlx_flow::CompileProfile;
    use rlx_runtime::Session;
    let (graph, params) = rlx_core::flow_util::graph_from_hir(hir, params)?;
    let opts =
        rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
    let mut compiled = Session::new(device).compile_with(graph, &opts);
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    Ok(compiled.run(inputs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::HirModule;
    use rlx_runtime::Device;

    fn det(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 13 + seed * 7) % 17) as f32 - 8.0) * 0.03)
            .collect()
    }

    #[test]
    fn ir_mha_matches_native() {
        // The masked-attention kernel the enhancer/decoder reuse: IR vs host.
        let (l, d, nh) = (3usize, 4usize, 2usize);
        let x = det(l * d, 1);
        let (qw, kw, vw, ow) = (det(d * d, 2), det(d * d, 3), det(d * d, 4), det(d * d, 5));
        let zb = vec![0f32; d];
        let bias = vec![0f32; l * l]; // no mask

        // Native reference.
        let native = crate::nn::mha(
            &x,
            &x,
            &x,
            l,
            l,
            d,
            nh,
            &qw,
            &zb,
            &kw,
            &zb,
            &vw,
            &zb,
            &ow,
            &zb,
            crate::nn::AttnBias::Shared(&bias),
        );

        // IR graph.
        let mut hir = HirModule::new("mha");
        let mut params = Params::new();
        let mut g = HirMut::new(&mut hir);
        let xn = g.input("x", Shape::new(&[l, d], DType::F32));
        let bn = g.input("bias", Shape::new(&[1, l, l], DType::F32));
        let out = super::mha(
            &mut g,
            &mut params,
            "a",
            xn,
            xn,
            xn,
            l,
            l,
            d,
            nh,
            &qw,
            &zb,
            &kw,
            &zb,
            &vw,
            &zb,
            &ow,
            &zb,
            bn,
        );
        g.set_outputs(vec![out]);
        let outs =
            compile_and_run(hir, params, Device::Cpu, &[("x", &x), ("bias", &bias)]).unwrap();

        let mut max_err = 0f32;
        for (a, b) in native.iter().zip(outs[0].iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-4, "ir vs native mha max_err={max_err}");
    }

    #[test]
    fn ir_relu_ffn_matches_host() {
        // linear → relu → linear (the FFN the enhancer/decoder reuse).
        let (rows, d, inter) = (3usize, 4usize, 8usize);
        let x = det(rows * d, 9);
        let w1 = det(inter * d, 10);
        let b1 = det(inter, 11);
        let w2 = det(d * inter, 12);
        let b2 = det(d, 13);

        // Host reference.
        let mut h = crate::nn::linear(&x, rows, d, &w1, inter, &b1);
        crate::nn::relu(&mut h);
        let host = crate::nn::linear(&h, rows, inter, &w2, d, &b2);

        // IR graph.
        let mut hir = HirModule::new("ffn");
        let mut params = Params::new();
        let mut g = HirMut::new(&mut hir);
        let xn = g.input("x", Shape::new(&[rows, d], DType::F32));
        let f1 = super::linear(&mut g, &mut params, "fc1", xn, d, inter, &w1, &b1, 1.0);
        let act = g.relu(f1);
        let f2 = super::linear(&mut g, &mut params, "fc2", act, inter, d, &w2, &b2, 1.0);
        g.set_outputs(vec![f2]);
        let outs = compile_and_run(hir, params, Device::Cpu, &[("x", &x)]).unwrap();

        let mut max_err = 0f32;
        for (a, b) in host.iter().zip(outs[0].iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-5, "ir vs host ffn max_err={max_err}");
    }

    #[test]
    fn ir_linear_matches_host() {
        // y = x @ w^T + b on CPU IR vs hand math.
        let (rows, in_dim, out_dim) = (2usize, 3usize, 2usize);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0]; // [out,in]
        let b = vec![0.5, -0.5];

        let mut hir = HirModule::new("lin");
        let mut params = Params::new();
        let mut g = HirMut::new(&mut hir);
        let xn = g.input("x", Shape::new(&[rows, in_dim], DType::F32));
        let y = linear(&mut g, &mut params, "l", xn, in_dim, out_dim, &w, &b, 1.0);
        g.set_outputs(vec![y]);

        let out = compile_and_run(hir, params, Device::Cpu, &[("x", &x)]).unwrap();
        // row0: [1, 5]+b=[1.5,4.5]; row1: [4, 11]+b=[4.5,10.5]
        let expected = [1.5, 4.5, 4.5, 10.5];
        for (a, e) in out[0].iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "{a} vs {e}");
        }
    }
}
