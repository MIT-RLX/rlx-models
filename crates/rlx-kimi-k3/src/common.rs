// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Shared graph-builder helpers for Kimi-K3 — the custom **situ** activation and
//! scalar constants.

use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{Activation, Op};
use rlx_ir::{DType, HirGraphExt, Shape};
use std::collections::HashMap;

/// Register a param `data` of `shape` under `name` and return its node.
pub fn reg(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: Vec<f32>,
    shape: &[usize],
) -> HirNodeId {
    debug_assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "{name} shape mismatch"
    );
    params.insert(name.to_string(), data);
    g.param(name, Shape::new(shape, DType::F32))
}

/// `x[.,in] @ w[in,out]`, registering `w` (row-major `[in, out]`) under
/// `{prefix}.{name}`. HF `nn.Linear` weights are `[out, in]` and must be
/// transposed by the loader before being passed here.
#[allow(clippy::too_many_arguments)]
pub fn linear(
    g: &mut HirMut,
    params: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    name: &str,
    x: HirNodeId,
    w: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> HirNodeId {
    let wid = reg(
        g,
        params,
        &format!("{prefix}.{name}"),
        w.to_vec(),
        &[in_dim, out_dim],
    );
    g.mm(x, wid)
}

/// A broadcastable f32 scalar constant node (shape `[1]`).
pub fn scalar_const(g: &mut HirMut, value: f32) -> HirNodeId {
    g.add_node(
        Op::Constant {
            data: value.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    )
}

/// An elementwise activation op preserving the input `shape`.
pub fn act(g: &mut HirMut, kind: Activation, x: HirNodeId, shape: Shape) -> HirNodeId {
    g.add_node(Op::Activation(kind), vec![x], shape)
}

/// Sigmoid via the activation op (there is no `HirGraphExt::sigmoid`), preserving
/// the input shape.
pub fn sigmoid(g: &mut HirMut, x: HirNodeId, shape: Shape) -> HirNodeId {
    act(g, Activation::Sigmoid, x, shape)
}

/// The Kimi **situ** GLU activation applied to a concatenated `gate_up` tensor
/// `[rows, 2*d]` (last-axis split into `gate = [..:d]`, `up = [..d:]`):
///
/// ```text
///   situ_a = beta * tanh(gate / beta) * sigmoid(gate)
///   up'    = linear_beta * tanh(up / linear_beta)   (only if linear_beta set)
///   out    = situ_a * up'                            -> [rows, d]
/// ```
///
/// `tanh(g)·sigmoid(g)` — distinct from silu (`g·sigmoid(g)`).
pub fn situ(
    g: &mut HirMut,
    gate_up: HirNodeId,
    rows: usize,
    d: usize,
    beta: f32,
    linear_beta: Option<f32>,
) -> HirNodeId {
    let f = DType::F32;
    let half = Shape::new(&[rows, d], f);
    let gate = g.narrow_(gate_up, 1, 0, d);
    let up = g.narrow_(gate_up, 1, d, d);

    // situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    let beta_c = scalar_const(g, beta);
    let gate_scaled = g.div(gate, beta_c);
    let gate_tanh = g.tanh(gate_scaled);
    let beta_c2 = scalar_const(g, beta);
    let situ_a = g.mul(beta_c2, gate_tanh);
    let gate_sig = sigmoid(g, gate, half.clone());
    let situ_a = g.mul(situ_a, gate_sig);

    // up' = linear_beta * tanh(up / linear_beta), else up.
    let up = match linear_beta {
        Some(lb) => {
            let lb_c = scalar_const(g, lb);
            let up_scaled = g.div(up, lb_c);
            let up_tanh = g.tanh(up_scaled);
            let lb_c2 = scalar_const(g, lb);
            g.mul(lb_c2, up_tanh)
        }
        None => up,
    };

    g.mul(situ_a, up)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_core::flow_util::{built_from_hir, compile_built};
    use rlx_ir::hir::HirModule;
    use rlx_runtime::Device;
    use std::collections::HashMap;

    // Reference situ on the host.
    fn situ_ref(gate: f32, up: f32, beta: f32, linear_beta: Option<f32>) -> f32 {
        let situ_a = beta * (gate / beta).tanh() * (1.0 / (1.0 + (-gate).exp()));
        let up = match linear_beta {
            Some(lb) => lb * (up / lb).tanh(),
            None => up,
        };
        situ_a * up
    }

    #[test]
    fn situ_matches_reference() {
        let (rows, d) = (2usize, 3usize);
        let beta = 4.0f32;
        let linear_beta = Some(25.0f32);

        let mut hir = HirModule::new("situ_test");
        let mut g = HirMut::new(&mut hir);
        let x = g.input("x", Shape::new(&[rows, 2 * d], DType::F32));
        let out = situ(&mut g, x, rows, d, beta, linear_beta);
        g.set_outputs(vec![out]);
        let built = built_from_hir(hir, HashMap::new()).expect("build situ graph");
        let mut compiled = compile_built(built, Device::Cpu).expect("compile situ");

        // gate = first d, up = last d, per row.
        let gate = [0.5f32, -1.2, 2.0, -0.3, 0.8, 1.5];
        let up = [1.0f32, 0.4, -0.7, 2.2, -1.1, 0.6];
        let mut xin = Vec::new();
        for r in 0..rows {
            xin.extend_from_slice(&gate[r * d..r * d + d]);
            xin.extend_from_slice(&up[r * d..r * d + d]);
        }
        let y = compiled
            .run(&[("x", xin.as_slice())])
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(y.len(), rows * d);
        for i in 0..rows * d {
            let want = situ_ref(gate[i], up[i], beta, linear_beta);
            assert!((y[i] - want).abs() < 1e-5, "situ[{i}] = {} != {want}", y[i]);
        }
    }
}
