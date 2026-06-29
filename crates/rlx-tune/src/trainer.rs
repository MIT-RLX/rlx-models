// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! The training loop: compile the backward of a forward graph once, then run
//! gradient steps with a host-side optimizer.
//!
//! Generic over the model graph: the caller builds a `Graph` whose single
//! output is the scalar loss, lists the trainable param nodes, and provides
//! initial values + per-step inputs. [`train`] differentiates via
//! `rlx_autodiff::grad_with_loss`, compiles on CPU, and steps. Backward
//! autodiff + compilation are backend-portable; this drives it on CPU.

use crate::distributed::GradComm;
use anyhow::{Result, bail};
use rlx_autodiff::grad_with_loss;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{CompileOptions, Device, Session};
use std::collections::HashMap;

/// A trainable parameter: its graph node and the param name used to seed /
/// update it in the compiled executable.
#[derive(Debug, Clone)]
pub struct ParamSlot {
    pub name: String,
    pub node: NodeId,
}

/// Host-side optimizer interface (operates on flat f32 slices, so it is
/// backend-free). Mirrors `rlx_optim::Optimizer`.
pub trait Optimizer {
    fn step(&mut self, name: &str, param: &mut [f32], grad: &[f32]);
}

/// Adam with per-parameter moment state.
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    t: i32,
    state: HashMap<String, (Vec<f32>, Vec<f32>)>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            state: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, name: &str, param: &mut [f32], grad: &[f32]) {
        self.t += 1;
        let (m, v) = self
            .state
            .entry(name.to_string())
            .or_insert_with(|| (vec![0.0; param.len()], vec![0.0; param.len()]));
        let bc1 = 1.0 - self.beta1.powi(self.t);
        let bc2 = 1.0 - self.beta2.powi(self.t);
        for i in 0..param.len() {
            let g = grad.get(i).copied().unwrap_or(0.0);
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;
            let mhat = m[i] / bc1;
            let vhat = v[i] / bc2;
            param[i] -= self.lr * mhat / (vhat.sqrt() + self.eps);
        }
    }
}

/// Emit a LoRA-augmented linear `y = x·W + (x·A)·B` with plain matmuls
/// (autodiff-safe on every backend). Shapes: `x [m,k]`, `w [k,n]`, `a [k,r]`,
/// `b [r,n]` → `y [m,n]`. The `alpha/rank` scale is folded into `B`'s values
/// by the caller (standard LoRA practice), so it isn't a graph op here.
pub fn lora_linear(
    g: &mut Graph,
    x: NodeId,
    w: NodeId,
    a: NodeId,
    b: NodeId,
    m: usize,
    n: usize,
    r: usize,
) -> NodeId {
    let f = DType::F32;
    let base = g.matmul(x, w, Shape::new(&[m, n], f));
    let xa = g.matmul(x, a, Shape::new(&[m, r], f));
    let delta = g.matmul(xa, b, Shape::new(&[m, n], f));
    g.add(base, delta)
}

/// Compile the backward of `forward` (whose single output must be the scalar
/// loss) w.r.t. `wrt`, then run `steps` optimization iterations, feeding
/// `inputs` each step. Updates `params` in place; returns the per-step loss.
///
/// When `comm` is `Some`, this is **data-parallel**: every rank runs the same
/// loop on its own `inputs` (data shard); trainable weights are broadcast from
/// rank 0 once, and each gradient is mean-all-reduced across ranks before the
/// optimizer step — so all ranks stay in lockstep with identical weights.
/// Collectives run over `wrt` in its (rank-stable) order, never `params`'
/// nondeterministic `HashMap` order.
pub fn train(
    forward: Graph,
    wrt: &[ParamSlot],
    params: &mut HashMap<String, Vec<f32>>,
    inputs: &[(String, Vec<f32>)],
    opt: &mut dyn Optimizer,
    steps: usize,
    comm: Option<&dyn GradComm>,
) -> Result<Vec<f32>> {
    let wrt_ids: Vec<NodeId> = wrt.iter().map(|s| s.node).collect();
    let backward = grad_with_loss(&forward, &wrt_ids);

    let mut compiled = Session::new(Device::Cpu).compile_with(backward, &CompileOptions::new());

    // Data-parallel: sync trainable weights from rank 0 so all ranks start
    // identical (frozen params are assumed loaded identically on every rank).
    if let Some(c) = comm {
        if c.world_size() > 1 {
            for slot in wrt {
                if let Some(p) = params.get_mut(&slot.name) {
                    c.broadcast(0, p);
                }
            }
        }
    }
    for (name, data) in params.iter() {
        compiled.set_param(name, data);
    }
    // `grad_with_loss` exposes a `d_output` cotangent input; seed it with 1.0
    // (∂loss/∂loss) so the backward pass produces real gradients.
    let seed = [1.0f32];
    let mut run_inputs: Vec<(&str, &[f32])> = inputs
        .iter()
        .map(|(n, d)| (n.as_str(), d.as_slice()))
        .collect();
    run_inputs.push(("d_output", &seed));

    let mut losses = Vec::with_capacity(steps);
    for _ in 0..steps {
        let outs = compiled.run(&run_inputs);
        if outs.is_empty() || outs[0].is_empty() {
            bail!("backward graph produced no loss output");
        }
        losses.push(outs[0][0]);
        // outs = [loss, grad(wrt[0]), grad(wrt[1]), ...]. In data-parallel
        // mode each gradient is averaged across ranks before the step.
        for (i, slot) in wrt.iter().enumerate() {
            let mut grad = outs.get(1 + i).cloned().unwrap_or_default();
            if let Some(c) = comm {
                c.all_reduce_mean(&mut grad);
            }
            let p = params
                .get_mut(&slot.name)
                .ok_or_else(|| anyhow::anyhow!("missing param value for {}", slot.name))?;
            opt.step(&slot.name, p, &grad);
            compiled.set_param(&slot.name, p);
        }
    }
    Ok(losses)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic small "random" values in [-0.1, 0.1].
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
    fn lora_overfits_a_linear_target() {
        // Fit y = (x·A)·B to a target T = x·M_true with a frozen zero base.
        // r = k so the low-rank product can represent M_true exactly.
        let (m, k, n, r) = (4usize, 3usize, 2usize, 3usize);
        let f = DType::F32;

        let mut g = Graph::new("lora_fit");
        let x = g.input("x", Shape::new(&[m, k], f));
        let w = g.param("w", Shape::new(&[k, n], f));
        let a = g.param("a", Shape::new(&[k, r], f));
        let b = g.param("b", Shape::new(&[r, n], f));
        let y = lora_linear(&mut g, x, w, a, b, m, n, r);
        let t = g.input("t", Shape::new(&[m, n], f));
        let diff = g.sub(y, t);
        let sq = g.mul(diff, diff);
        let flat = g.reshape_(sq, vec![(m * n) as i64]);
        let loss = g.mean(flat, vec![0], false);
        g.set_outputs(vec![loss]);

        let xd = pseudo(m * k, 1);
        let m_true = pseudo(k * n, 2);
        let td = host_matmul(&xd, &m_true, m, k, n);

        let mut params = HashMap::new();
        params.insert("w".to_string(), vec![0.0; k * n]); // frozen zero base
        params.insert("a".to_string(), pseudo(k * r, 3));
        params.insert("b".to_string(), pseudo(r * n, 4));

        let wrt = vec![
            ParamSlot {
                name: "a".into(),
                node: a,
            },
            ParamSlot {
                name: "b".into(),
                node: b,
            },
        ];
        let inputs = vec![("x".to_string(), xd), ("t".to_string(), td)];

        let mut opt = Adam::new(0.05);
        let losses = train(g, &wrt, &mut params, &inputs, &mut opt, 300, None).unwrap();

        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(first > 1e-4, "initial loss should be nonzero, got {first}");
        assert!(
            last < first * 0.02,
            "training did not reduce loss: {first} -> {last}"
        );
    }
}
