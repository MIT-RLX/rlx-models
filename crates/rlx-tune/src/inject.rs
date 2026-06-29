// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Inject LoRA adapters into an existing model forward graph.
//!
//! `inject_lora` rewrites a `Graph`, replacing every targeted `MatMul(x, W)`
//! (where `W` is a `Param` whose name matches the spec) with a LoRA-augmented
//! linear `x·W + (x·A)·B`. `A`/`B` are fresh trainable params; the base `W`
//! stays frozen (excluded from the optimizer's `wrt`). The graph is rebuilt
//! node-by-node so topological order is preserved and every other op is copied
//! verbatim. The delta uses plain matmuls (autodiff-supported on every
//! backend); initialize `B` to zero so injection doesn't change the forward
//! pass until training begins.

use crate::adapter::{DoraSpec, LoraSpec};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, NodeId, Shape};

/// A newly-injected adapter parameter (`A` or `B`): its name, graph node, and
/// shape — enough to build a [`crate::ParamSlot`] and an initial value.
#[derive(Debug, Clone)]
pub struct AdapterParam {
    pub name: String,
    pub node: NodeId,
    pub shape: Vec<usize>,
}

/// How the LoRA forward is emitted into the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuseMode {
    /// One first-class `LoraMatMul` op — `x·W + scale·(x·A)·B`. Lowers to a
    /// fused kernel on CPU/Metal/MLX (and decomposes for autodiff via the
    /// pre-AD unfuse pass, so it trains too). Fewer nodes, faster forward.
    Fused,
    /// Explicit `base + scale·((x·A)·B)` matmuls. Portable everywhere with no
    /// reliance on the fused op/kernel.
    Unfused,
}

/// Emit `y = x·W + scale·((x·A)·B)` in either fusion mode. `r` is the rank;
/// `out_shape` is the matmul output shape; `x` has its last dim contracted.
fn lora_forward(
    g: &mut Graph,
    x: NodeId,
    w: NodeId,
    a: NodeId,
    b: NodeId,
    r: usize,
    out_shape: Shape,
    scale: f32,
    mode: FuseMode,
) -> NodeId {
    match mode {
        FuseMode::Fused => g.lora_matmul(x, w, a, b, scale, out_shape),
        FuseMode::Unfused => {
            let f = DType::F32;
            let base = g.matmul(x, w, out_shape.clone());
            let mut xa_dims: Vec<usize> = g
                .shape(x)
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect();
            *xa_dims.last_mut().expect("matmul lhs rank >= 1") = r;
            let xa = g.matmul(x, a, Shape::new(&xa_dims, f));
            let delta = g.matmul(xa, b, out_shape);
            let delta = if (scale - 1.0).abs() > f32::EPSILON {
                let s = g.constant(scale as f64, f);
                g.mul(delta, s) // rank-0 scalar broadcasts
            } else {
                delta
            };
            g.add(base, delta)
        }
    }
}

/// Rewrite `forward` with LoRA on every matching `MatMul`, in the chosen
/// [`FuseMode`]. Returns the new graph and the injected `A`/`B` params.
pub fn inject_lora(forward: &Graph, spec: &LoraSpec, mode: FuseMode) -> (Graph, Vec<AdapterParam>) {
    let r = spec.rank;
    let f = DType::F32;
    let mut new = Graph::new(forward.name.clone());
    let mut map: Vec<NodeId> = Vec::with_capacity(forward.nodes().len());
    let mut params: Vec<AdapterParam> = Vec::new();

    for node in forward.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| map[i.0 as usize]).collect();

        // Target = a MatMul whose weight operand (input 1) is a matching Param.
        let target = if matches!(node.op, Op::MatMul) && node.inputs.len() == 2 {
            match &forward.node(node.inputs[1]).op {
                Op::Param { name } if r > 0 && spec.targets(name) => Some(name.clone()),
                _ => None,
            }
        } else {
            None
        };

        let new_id = match target {
            Some(wname) => {
                let x = inputs[0];
                let w = inputs[1];
                let wshape = forward.node(node.inputs[1]).shape.clone(); // [k, n]
                let k = wshape.dim(0).unwrap_static();
                let n = wshape.dim(wshape.rank() - 1).unwrap_static();

                let a_name = format!("lora.{wname}.a");
                let b_name = format!("lora.{wname}.b");
                let a = new.param(a_name.clone(), Shape::new(&[k, r], f));
                let b = new.param(b_name.clone(), Shape::new(&[r, n], f));
                params.push(AdapterParam {
                    name: a_name,
                    node: a,
                    shape: vec![k, r],
                });
                params.push(AdapterParam {
                    name: b_name,
                    node: b,
                    shape: vec![r, n],
                });
                lora_forward(
                    &mut new,
                    x,
                    w,
                    a,
                    b,
                    r,
                    node.shape.clone(),
                    spec.scale(),
                    mode,
                )
            }
            None => new.append_node(
                node.op.clone(),
                inputs,
                node.shape.clone(),
                node.name.clone(),
            ),
        };
        map.push(new_id);
    }

    let outs = forward.outputs.iter().map(|o| map[o.0 as usize]).collect();
    new.set_outputs(outs);
    (new, params)
}

/// Rewrite `forward` with **DoRA** (weight-decomposed LoRA) on every matching
/// `MatMul(x, W)`: the effective weight becomes
/// `m ⊙ (W + A·B) / ‖W + A·B‖_c`, expressed as the LoRA forward scaled
/// per-output-column. New trainable params per target: `A`, `B`, and a
/// magnitude vector `m [n]`. Initialize `A` random, `B = 0`, and
/// `m = column_norms(W)` (see [`crate::adapter::column_norms`]) so injection
/// leaves the forward unchanged until training. The `alpha/rank` scale is
/// folded into `B`'s init. Returns the new graph + injected params in the order
/// `[A, B, m]` per target. `mode` controls how the LoRA direction is emitted;
/// the column norm + magnitude rescale are identical either way.
pub fn inject_dora(forward: &Graph, spec: &DoraSpec, mode: FuseMode) -> (Graph, Vec<AdapterParam>) {
    let r = spec.lora.rank;
    let f = DType::F32;
    let mut new = Graph::new(forward.name.clone());
    let mut map: Vec<NodeId> = Vec::with_capacity(forward.nodes().len());
    let mut params: Vec<AdapterParam> = Vec::new();

    for node in forward.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| map[i.0 as usize]).collect();

        let target = if matches!(node.op, Op::MatMul) && node.inputs.len() == 2 {
            match &forward.node(node.inputs[1]).op {
                Op::Param { name } if r > 0 && spec.lora.targets(name) => Some(name.clone()),
                _ => None,
            }
        } else {
            None
        };

        let new_id = match target {
            Some(wname) => {
                let x = inputs[0];
                let w = inputs[1];
                let wshape = forward.node(node.inputs[1]).shape.clone(); // [k, n]
                let k = wshape.dim(0).unwrap_static();
                let n = wshape.dim(wshape.rank() - 1).unwrap_static();

                let a = new.param(format!("dora.{wname}.a"), Shape::new(&[k, r], f));
                let b = new.param(format!("dora.{wname}.b"), Shape::new(&[r, n], f));
                let mag = new.param(format!("dora.{wname}.m"), Shape::new(&[n], f));
                params.push(AdapterParam {
                    name: format!("dora.{wname}.a"),
                    node: a,
                    shape: vec![k, r],
                });
                params.push(AdapterParam {
                    name: format!("dora.{wname}.b"),
                    node: b,
                    shape: vec![r, n],
                });
                params.push(AdapterParam {
                    name: format!("dora.{wname}.m"),
                    node: mag,
                    shape: vec![n],
                });

                let scale = spec.lora.scale();
                // Combined weight Wc = W + scale·(A·B), its column norm, and
                // the per-column rescale m / ‖Wc‖_c.
                let ab = new.matmul(a, b, Shape::new(&[k, n], f));
                let ab = if (scale - 1.0).abs() > f32::EPSILON {
                    let s = new.constant(scale as f64, f);
                    new.mul(ab, s)
                } else {
                    ab
                };
                let wc = new.add(w, ab);
                let sq = new.mul(wc, wc);
                let ssum = new.sum(sq, vec![0], false); // [n]
                let norm = new.sqrt(ssum); // [n]
                let rescale = new.div(mag, norm); // [n]

                // LoRA direction (fused/unfused), then per-column magnitude.
                let lora_out =
                    lora_forward(&mut new, x, w, a, b, r, node.shape.clone(), scale, mode);
                new.mul(lora_out, rescale) // broadcast [n] over the batch
            }
            None => new.append_node(
                node.op.clone(),
                inputs,
                node.shape.clone(),
                node.name.clone(),
            ),
        };
        map.push(new_id);
    }

    let outs = forward.outputs.iter().map(|o| map[o.0 as usize]).collect();
    new.set_outputs(outs);
    (new, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::{Adam, ParamSlot, train};
    use rlx_runtime::{CompileOptions, Device, Session};
    use std::collections::HashMap;

    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / u32::MAX as f32 - 0.5) * 0.2
            })
            .collect()
    }

    fn host_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for t in 0..k {
                    acc += a[i * k + t] * b[t * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    #[test]
    fn inject_finds_targets_and_preserves_forward_when_b_zero() {
        // y = x·W2. Inject LoRA into "W2"; with B=0 the rewritten graph must
        // produce the identical forward output (delta = 0).
        let (m, k, n) = (4usize, 3usize, 2usize);
        let f = DType::F32;
        let mut g = Graph::new("base");
        let x = g.input("x", Shape::new(&[m, k], f));
        let w2 = g.param("W2", Shape::new(&[k, n], f));
        let y = g.matmul(x, w2, Shape::new(&[m, n], f));
        g.set_outputs(vec![y]);

        let spec = LoraSpec::new(3, 6.0, vec!["W2".into()]);
        let (rewritten, params) = inject_lora(&g, &spec, FuseMode::Unfused);
        assert_eq!(params.len(), 2, "one A + one B for the single target");
        assert_eq!(params[0].name, "lora.W2.a");
        assert_eq!(params[0].shape, vec![k, 3]);
        assert_eq!(params[1].shape, vec![3, n]);

        let xd = pseudo(m * k, 1);
        let w2d = pseudo(k * n, 2);
        let ad = pseudo(k * 3, 3);
        let bd = vec![0.0f32; 3 * n]; // B = 0 → delta = 0

        let run = |graph: Graph, set_lora: bool| {
            let mut c = Session::new(Device::Cpu).compile_with(graph, &CompileOptions::new());
            c.set_param("W2", &w2d);
            if set_lora {
                c.set_param("lora.W2.a", &ad);
                c.set_param("lora.W2.b", &bd);
            }
            c.run(&[("x", &xd)])[0].clone()
        };
        let base_out = run(g, false);
        let lora_out = run(rewritten, true);
        for (a, b) in base_out.iter().zip(&lora_out) {
            assert!(
                (a - b).abs() < 1e-6,
                "B=0 must leave the forward unchanged: {a} vs {b}"
            );
        }
    }

    #[test]
    fn injected_lora_trains_a_residual() {
        // Base W2 frozen; target = x·(W2 + M_true). The injected LoRA must
        // learn the residual M_true (loss → ~0).
        let (m, k, n, r) = (4usize, 3usize, 2usize, 3usize);
        let f = DType::F32;
        let mut g = Graph::new("residual");
        let x = g.input("x", Shape::new(&[m, k], f));
        let w2 = g.param("W2", Shape::new(&[k, n], f));
        let y = g.matmul(x, w2, Shape::new(&[m, n], f));
        let t = g.input("t", Shape::new(&[m, n], f));
        let diff = g.sub(y, t);
        let sq = g.mul(diff, diff);
        let flat = g.reshape_(sq, vec![(m * n) as i64]);
        let loss = g.mean(flat, vec![0], false);
        g.set_outputs(vec![loss]);

        let spec = LoraSpec::new(r, r as f32, vec!["W2".into()]);
        let (rewritten, adapters) = inject_lora(&g, &spec, FuseMode::Unfused);

        let xd = pseudo(m * k, 1);
        let w2d = pseudo(k * n, 2);
        let m_true = pseudo(k * n, 5);
        let w_plus: Vec<f32> = w2d.iter().zip(&m_true).map(|(a, b)| a + b).collect();
        let td = host_matmul(&xd, &w_plus, m, k, n);

        let mut params = HashMap::new();
        params.insert("W2".to_string(), w2d); // frozen base
        params.insert(adapters[0].name.clone(), pseudo(k * r, 3)); // A random
        params.insert(adapters[1].name.clone(), vec![0.0; r * n]); // B = 0
        let wrt: Vec<ParamSlot> = adapters
            .iter()
            .map(|p| ParamSlot {
                name: p.name.clone(),
                node: p.node,
            })
            .collect();
        let inputs = vec![("x".to_string(), xd), ("t".to_string(), td)];

        let mut opt = Adam::new(0.05);
        let losses = train(rewritten, &wrt, &mut params, &inputs, &mut opt, 300, None).unwrap();
        let (first, last) = (losses[0], *losses.last().unwrap());
        assert!(
            first > 1e-4,
            "initial residual loss should be nonzero: {first}"
        );
        assert!(
            last < first * 0.05,
            "injected LoRA should fit the residual: {first} -> {last}"
        );
    }

    #[test]
    fn inject_dora_preserves_forward_with_b_zero_and_norm_magnitude() {
        // DoRA with B=0 and m = ‖W‖_c reduces to the base forward.
        let (m, k, n) = (4usize, 3usize, 2usize);
        let f = DType::F32;
        let mut g = Graph::new("base");
        let x = g.input("x", Shape::new(&[m, k], f));
        let w2 = g.param("W2", Shape::new(&[k, n], f));
        let y = g.matmul(x, w2, Shape::new(&[m, n], f));
        g.set_outputs(vec![y]);

        let spec = DoraSpec {
            lora: LoraSpec::new(3, 3.0, vec!["W2".into()]),
        };
        let (rewritten, params) = inject_dora(&g, &spec, FuseMode::Unfused);
        assert_eq!(params.len(), 3, "A, B, m for the single target");
        assert_eq!(params[2].name, "dora.W2.m");
        assert_eq!(params[2].shape, vec![n]);

        let xd = pseudo(m * k, 1);
        let w2d = pseudo(k * n, 2);
        let ad = pseudo(k * 3, 3);
        let bd = vec![0.0f32; 3 * n];
        let md = crate::adapter::column_norms(&w2d, k, n); // m = ‖W‖_c

        let base_out = {
            let mut c = Session::new(Device::Cpu).compile_with(g, &CompileOptions::new());
            c.set_param("W2", &w2d);
            c.run(&[("x", &xd)])[0].clone()
        };
        let dora_out = {
            let mut c = Session::new(Device::Cpu).compile_with(rewritten, &CompileOptions::new());
            c.set_param("W2", &w2d);
            c.set_param("dora.W2.a", &ad);
            c.set_param("dora.W2.b", &bd);
            c.set_param("dora.W2.m", &md);
            c.run(&[("x", &xd)])[0].clone()
        };
        for (a, b) in base_out.iter().zip(&dora_out) {
            assert!(
                (a - b).abs() < 1e-4,
                "DoRA(B=0, m=‖W‖_c) must equal base: {a} vs {b}"
            );
        }
    }

    #[test]
    fn injected_dora_trains_a_residual() {
        let (m, k, n, r) = (4usize, 3usize, 2usize, 3usize);
        let f = DType::F32;
        let mut g = Graph::new("dora_fit");
        let x = g.input("x", Shape::new(&[m, k], f));
        let w2 = g.param("W2", Shape::new(&[k, n], f));
        let y = g.matmul(x, w2, Shape::new(&[m, n], f));
        let t = g.input("t", Shape::new(&[m, n], f));
        let diff = g.sub(y, t);
        let sq = g.mul(diff, diff);
        let flat = g.reshape_(sq, vec![(m * n) as i64]);
        let loss = g.mean(flat, vec![0], false);
        g.set_outputs(vec![loss]);

        let spec = DoraSpec {
            lora: LoraSpec::new(r, r as f32, vec!["W2".into()]),
        };
        let (rewritten, adapters) = inject_dora(&g, &spec, FuseMode::Unfused);

        let xd = pseudo(m * k, 1);
        let w2d = pseudo(k * n, 2);
        let m_true = pseudo(k * n, 5);
        let w_plus: Vec<f32> = w2d.iter().zip(&m_true).map(|(a, b)| a + b).collect();
        let td = host_matmul(&xd, &w_plus, m, k, n);

        let mut params = HashMap::new();
        params.insert("W2".to_string(), w2d.clone());
        params.insert(adapters[0].name.clone(), pseudo(k * r, 3)); // A
        params.insert(adapters[1].name.clone(), vec![0.0; r * n]); // B = 0
        params.insert(
            adapters[2].name.clone(),
            crate::adapter::column_norms(&w2d, k, n),
        ); // m
        let wrt: Vec<ParamSlot> = adapters
            .iter()
            .map(|p| ParamSlot {
                name: p.name.clone(),
                node: p.node,
            })
            .collect();
        let inputs = vec![("x".to_string(), xd), ("t".to_string(), td)];

        let mut opt = Adam::new(0.05);
        let losses = train(rewritten, &wrt, &mut params, &inputs, &mut opt, 400, None).unwrap();
        let (first, last) = (losses[0], *losses.last().unwrap());
        assert!(
            first > 1e-4,
            "initial residual loss should be nonzero: {first}"
        );
        assert!(
            last < first * 0.2,
            "injected DoRA should fit the residual: {first} -> {last}"
        );
    }

    #[test]
    fn fused_lora_matches_unfused_forward() {
        // The fused LoraMatMul op and the explicit decomposition must produce
        // identical forward output for the same params + scale (B != 0).
        let (m, k, n, r) = (4usize, 3usize, 2usize, 2usize);
        let f = DType::F32;
        let build = || {
            let mut g = Graph::new("base");
            let x = g.input("x", Shape::new(&[m, k], f));
            let w2 = g.param("W2", Shape::new(&[k, n], f));
            let y = g.matmul(x, w2, Shape::new(&[m, n], f));
            g.set_outputs(vec![y]);
            g
        };
        let spec = LoraSpec::new(r, 2.0 * r as f32, vec!["W2".into()]); // scale = 2.0
        let (g_fused, _) = inject_lora(&build(), &spec, FuseMode::Fused);
        let (g_unfused, _) = inject_lora(&build(), &spec, FuseMode::Unfused);

        let xd = pseudo(m * k, 1);
        let w2d = pseudo(k * n, 2);
        let ad = pseudo(k * r, 3);
        let bd = pseudo(r * n, 4); // nonzero so scale·delta is exercised
        let run = |graph: Graph| {
            let mut c = Session::new(Device::Cpu).compile_with(graph, &CompileOptions::new());
            c.set_param("W2", &w2d);
            c.set_param("lora.W2.a", &ad);
            c.set_param("lora.W2.b", &bd);
            c.run(&[("x", &xd)])[0].clone()
        };
        let fused = run(g_fused);
        let unfused = run(g_unfused);
        for (a, b) in fused.iter().zip(&unfused) {
            assert!((a - b).abs() < 1e-5, "fused vs unfused forward: {a} vs {b}");
        }
    }

    #[test]
    fn fused_lora_trains() {
        // Fused injection still trains — autodiff unfuses LoraMatMul.
        let (m, k, n, r) = (4usize, 3usize, 2usize, 3usize);
        let f = DType::F32;
        let mut g = Graph::new("fused_fit");
        let x = g.input("x", Shape::new(&[m, k], f));
        let w2 = g.param("W2", Shape::new(&[k, n], f));
        let y = g.matmul(x, w2, Shape::new(&[m, n], f));
        let t = g.input("t", Shape::new(&[m, n], f));
        let diff = g.sub(y, t);
        let sq = g.mul(diff, diff);
        let flat = g.reshape_(sq, vec![(m * n) as i64]);
        let loss = g.mean(flat, vec![0], false);
        g.set_outputs(vec![loss]);

        let spec = LoraSpec::new(r, r as f32, vec!["W2".into()]);
        let (rewritten, adapters) = inject_lora(&g, &spec, FuseMode::Fused);

        let xd = pseudo(m * k, 1);
        let w2d = pseudo(k * n, 2);
        let m_true = pseudo(k * n, 5);
        let w_plus: Vec<f32> = w2d.iter().zip(&m_true).map(|(a, b)| a + b).collect();
        let td = host_matmul(&xd, &w_plus, m, k, n);

        let mut params = HashMap::new();
        params.insert("W2".to_string(), w2d);
        params.insert(adapters[0].name.clone(), pseudo(k * r, 3));
        params.insert(adapters[1].name.clone(), vec![0.0; r * n]);
        let wrt: Vec<ParamSlot> = adapters
            .iter()
            .map(|p| ParamSlot {
                name: p.name.clone(),
                node: p.node,
            })
            .collect();
        let inputs = vec![("x".to_string(), xd), ("t".to_string(), td)];
        let mut opt = Adam::new(0.05);
        let losses = train(rewritten, &wrt, &mut params, &inputs, &mut opt, 300, None).unwrap();
        let (first, last) = (losses[0], *losses.last().unwrap());
        assert!(
            last < first * 0.1,
            "fused LoRA should train: {first} -> {last}"
        );
    }
}
