// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! GPU-resident (fused) Adam optimizer.
//!
//! The default [`Trainer`](crate::Trainer) runs the backward on the device but
//! the optimizer on the **host**: every step downloads the gradients, updates
//! params on the CPU, and re-uploads them. On a GPU that host round-trip
//! dominates for small/medium models (the update itself is trivial), so GPU
//! throughput stops scaling with batch size.
//!
//! This module **fuses the Adam update into the graph**: after
//! [`grad_with_loss`] the update `m' = β₁m + (1-β₁)g`, `v' = β₂v + (1-β₂)g²`,
//! `p' = p - lr·m̂/(√v̂+ε)` is appended as ordinary ops, so forward + backward +
//! optimizer step run as **one on-device computation**. The moment buffers and
//! params are kept **device-resident** across steps via
//! [`bind_gpu_handle`](rlx_runtime::CompiledGraph::bind_gpu_handle) +
//! [`set_gpu_handle_feed`](rlx_runtime::CompiledGraph::set_gpu_handle_feed) —
//! the graph's updated outputs are written back into the resident buffers with
//! no host transfer. Backends without handle support (and CPU) fall back to a
//! correct host-chained path, so the fused update is identical everywhere.

use std::collections::HashMap;

use anyhow::{Result, bail};
use rlx_autodiff::grad_with_loss;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};

use crate::trainer::{AdamConfig, ParamSlot};

/// Layout of one fused parameter: its moment-input names and where its updated
/// `param' / m' / v'` land in the fused graph's outputs.
#[derive(Clone, Debug)]
struct FusedParam {
    name: String,
    m_name: String,
    v_name: String,
    p_out: usize,
    m_out: usize,
    v_out: usize,
}

/// The fused forward+backward+Adam graph plus its output layout.
struct FusedStep {
    graph: Graph,
    params: Vec<FusedParam>,
}

/// Append the Adam update to the backward of `forward` w.r.t. `wrt`.
///
/// Outputs become `[loss, p'₀, m'₀, v'₀, p'₁, …]`. New per-param inputs
/// `"{name}__m"` / `"{name}__v"` carry the moments; scalar inputs `"__lr"`,
/// `"__bc1"`, `"__bc2"` carry the learning rate and Adam bias corrections.
fn fused_adam_graph(
    forward: &Graph,
    wrt: &[ParamSlot],
    cfg: &AdamConfig,
    weight_decay: f32,
) -> FusedStep {
    let wrt_ids: Vec<NodeId> = wrt.iter().map(|s| s.node).collect();
    let mut g = grad_with_loss(forward, &wrt_ids);
    let f = DType::F32;
    let loss = g.outputs[0];
    let grads: Vec<NodeId> = (0..wrt.len()).map(|i| g.outputs[1 + i]).collect();

    // Shared scalars + constants.
    let lr = g.input("__lr", Shape::scalar(f));
    let bc1 = g.input("__bc1", Shape::scalar(f));
    let bc2 = g.input("__bc2", Shape::scalar(f));
    let c_b1 = g.constant(cfg.beta1 as f64, f);
    let c_1mb1 = g.constant((1.0 - cfg.beta1) as f64, f);
    let c_b2 = g.constant(cfg.beta2 as f64, f);
    let c_1mb2 = g.constant((1.0 - cfg.beta2) as f64, f);
    let c_eps = g.constant(cfg.eps as f64, f);
    // AdamW decoupled weight decay: p -= lr·wd·p (as a shared `lr·wd` scalar).
    let lr_wd = (weight_decay != 0.0).then(|| {
        let c_wd = g.constant(weight_decay as f64, f);
        g.mul(lr, c_wd)
    });

    let mut outputs = vec![loss];
    let mut params = Vec::with_capacity(wrt.len());
    let mut param_ids = Vec::with_capacity(wrt.len());
    for (i, slot) in wrt.iter().enumerate() {
        let p = g
            .param_id(&slot.name)
            .expect("wrt param present in backward graph");
        param_ids.push(p);
        let shape = g.shape(p).clone();
        let grad = grads[i];
        let m_name = format!("{}__m", slot.name);
        let v_name = format!("{}__v", slot.name);
        let m = g.input(&m_name, shape.clone());
        let v = g.input(&v_name, shape.clone());

        // m' = β₁·m + (1-β₁)·g
        let mb1 = g.mul(m, c_b1);
        let g1 = g.mul(grad, c_1mb1);
        let m_new = g.add(mb1, g1);
        // v' = β₂·v + (1-β₂)·g²
        let g2 = g.mul(grad, grad);
        let vb2 = g.mul(v, c_b2);
        let g2b = g.mul(g2, c_1mb2);
        let v_new = g.add(vb2, g2b);
        // p' = p - lr·(m'/bc1) / (√(v'/bc2) + ε)
        let mhat = g.div(m_new, bc1);
        let vhat = g.div(v_new, bc2);
        let root = g.sqrt(vhat);
        let denom = g.add(root, c_eps);
        let ratio = g.div(mhat, denom);
        let update = g.mul(ratio, lr);
        let stepped = g.sub(p, update);
        let p_new = match lr_wd {
            Some(lrwd) => {
                let decay = g.mul(p, lrwd);
                g.sub(stepped, decay)
            }
            None => stepped,
        };

        let p_out = outputs.len();
        outputs.push(p_new);
        let m_out = outputs.len();
        outputs.push(m_new);
        let v_out = outputs.len();
        outputs.push(v_new);
        params.push(FusedParam {
            name: slot.name.clone(),
            m_name,
            v_name,
            p_out,
            m_out,
            v_out,
        });
    }
    g.set_outputs(outputs);
    // Autodiff is done, so re-tag the trainable params as **inputs**: this lets
    // them be bound to device-resident GPU handles (which only accept graph
    // inputs) and fed each step, keeping weights on-device. The node id (and so
    // every reference to it) is unchanged.
    for (slot, &p) in wrt.iter().zip(&param_ids) {
        g.node_mut(p).op = Op::Input {
            name: slot.name.clone(),
        };
    }
    FusedStep { graph: g, params }
}

/// A trainer whose Adam optimizer runs **inside** the compiled graph, keeping
/// params + moments device-resident (on GPU backends that support handles).
pub struct ResidentTrainer {
    compiled: CompiledGraph,
    params: Vec<FusedParam>,
    cfg: AdamConfig,
    t: i32,
    /// Host mirrors (source of truth on the fallback path; snapshot cache on
    /// the resident path — refreshed lazily by [`params`](Self::params)).
    param_vals: HashMap<String, Vec<f32>>,
    m_vals: HashMap<String, Vec<f32>>,
    v_vals: HashMap<String, Vec<f32>>,
    /// True when params + moments live in device handles fed by the graph.
    resident: bool,
}

impl ResidentTrainer {
    /// Compile the fused step for `device`, seeding params from `params0`
    /// (moments start at zero). `weight_decay > 0` gives AdamW (decoupled decay
    /// `p -= lr·wd·p`). Frozen (non-`wrt`) params in `params0` are uploaded
    /// once. Binds device-resident handles when the backend supports them, else
    /// uses the host-chained path (identical results).
    pub fn new(
        forward: &Graph,
        wrt: &[ParamSlot],
        params0: &HashMap<String, Vec<f32>>,
        cfg: &AdamConfig,
        weight_decay: f32,
        device: Device,
    ) -> Result<Self> {
        let step = fused_adam_graph(forward, wrt, cfg, weight_decay);
        let mut compiled = Session::new(device).compile_with(step.graph, &CompileOptions::new());

        // Upload frozen params (the wrt params are handled below).
        let wrt_names: Vec<&str> = step.params.iter().map(|p| p.name.as_str()).collect();
        for (name, data) in params0 {
            if !wrt_names.contains(&name.as_str()) {
                compiled.set_param(name, data);
            }
        }

        let mut param_vals = HashMap::new();
        let mut m_vals = HashMap::new();
        let mut v_vals = HashMap::new();
        for fp in &step.params {
            let p0 = params0
                .get(&fp.name)
                .ok_or_else(|| anyhow::anyhow!("missing param value for {}", fp.name))?
                .clone();
            let zeros = vec![0.0f32; p0.len()];
            param_vals.insert(fp.name.clone(), p0);
            m_vals.insert(fp.name.clone(), zeros.clone());
            v_vals.insert(fp.name.clone(), zeros);
        }

        // Try to make params + moments device-resident: bind each as a GPU
        // handle and feed its updated output back into it. All-or-nothing.
        let resident =
            try_bind_resident(&mut compiled, &step.params, &param_vals, &m_vals, &v_vals);

        Ok(Self {
            compiled,
            params: step.params,
            cfg: *cfg,
            t: 0,
            param_vals,
            m_vals,
            v_vals,
            resident,
        })
    }

    /// Whether params + moments are device-resident (no host round-trip/step).
    pub fn is_resident(&self) -> bool {
        self.resident
    }

    /// Set the learning rate for subsequent steps — the lr is a scalar graph
    /// input, so this drives an external LR schedule (warmup / decay) with no
    /// recompile and no cost on the resident path.
    pub fn set_lr(&mut self, lr: f32) {
        self.cfg.lr = lr;
    }

    /// One optimization step on `inputs` (the model's data inputs). Returns the
    /// scalar loss.
    pub fn step(&mut self, inputs: &[(&str, &[f32])]) -> f32 {
        self.t += 1;
        let seed = [1.0f32];
        let lr = [self.cfg.lr];
        let bc1 = [1.0 - self.cfg.beta1.powi(self.t)];
        let bc2 = [1.0 - self.cfg.beta2.powi(self.t)];

        let mut run_inputs: Vec<(&str, &[f32])> = inputs.to_vec();
        run_inputs.push(("d_output", &seed));
        run_inputs.push(("__lr", &lr));
        run_inputs.push(("__bc1", &bc1));
        run_inputs.push(("__bc2", &bc2));

        if self.resident {
            // Params + moments come from resident handles; the feeds write the
            // updated outputs back on-device (D2D). Read back **only the loss**
            // (index 0) — `p'/m'/v'` never leave the device.
            let outs = self.compiled.run_read_outputs(&run_inputs, Some(&[0]));
            outs.first()
                .and_then(|o| o.first().copied())
                .unwrap_or(f32::NAN)
        } else {
            // Host path: feed params + moments as inputs, read the updated
            // values back.
            for fp in &self.params {
                run_inputs.push((fp.name.as_str(), &self.param_vals[&fp.name]));
                run_inputs.push((fp.m_name.as_str(), &self.m_vals[&fp.name]));
                run_inputs.push((fp.v_name.as_str(), &self.v_vals[&fp.name]));
            }
            let outs = self.compiled.run(&run_inputs);
            for fp in &self.params {
                copy_out(&mut self.param_vals, &fp.name, outs.get(fp.p_out));
                copy_out(&mut self.m_vals, &fp.name, outs.get(fp.m_out));
                copy_out(&mut self.v_vals, &fp.name, outs.get(fp.v_out));
            }
            outs.first()
                .and_then(|o| o.first().copied())
                .unwrap_or(f32::NAN)
        }
    }

    /// Current trainable weights by name (reads back from device on the
    /// resident path).
    pub fn params(&mut self) -> HashMap<String, Vec<f32>> {
        if self.resident {
            for fp in &self.params {
                if let Some(v) = self.compiled.read_gpu_handle(&fp.name) {
                    self.param_vals.insert(fp.name.clone(), v);
                }
            }
        }
        self.param_vals.clone()
    }

    /// Run `steps` iterations feeding the fixed `inputs`; returns per-step loss.
    pub fn run_fixed(&mut self, inputs: &[(&str, &[f32])], steps: usize) -> Result<Vec<f32>> {
        if self.params.is_empty() {
            bail!("no trainable params");
        }
        Ok((0..steps).map(|_| self.step(inputs)).collect())
    }
}

fn copy_out(map: &mut HashMap<String, Vec<f32>>, name: &str, out: Option<&Vec<f32>>) {
    if let (Some(dst), Some(src)) = (map.get_mut(name), out) {
        let n = dst.len().min(src.len());
        dst[..n].copy_from_slice(&src[..n]);
    }
}

/// Bind params + moments as device-resident handles and feed the updated
/// outputs back into them. Returns true only if every bind + feed succeeded.
fn try_bind_resident(
    compiled: &mut CompiledGraph,
    params: &[FusedParam],
    param_vals: &HashMap<String, Vec<f32>>,
    m_vals: &HashMap<String, Vec<f32>>,
    v_vals: &HashMap<String, Vec<f32>>,
) -> bool {
    let mut ok = true;
    for fp in params {
        ok &= compiled.bind_gpu_handle(&fp.name, &param_vals[&fp.name]);
        ok &= compiled.bind_gpu_handle(&fp.m_name, &m_vals[&fp.name]);
        ok &= compiled.bind_gpu_handle(&fp.v_name, &v_vals[&fp.name]);
        ok &= compiled.set_gpu_handle_feed(&fp.name, fp.p_out);
        ok &= compiled.set_gpu_handle_feed(&fp.m_name, fp.m_out);
        ok &= compiled.set_gpu_handle_feed(&fp.v_name, fp.v_out);
    }
    ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::lora_linear;

    // Fit y = (x·A)·B to a target, frozen zero base. Returns graph + a/b nodes.
    fn build(m: usize, k: usize, n: usize, r: usize) -> (Graph, NodeId, NodeId) {
        let f = DType::F32;
        let mut g = Graph::new("resident_fit");
        let x = g.input("x", Shape::new(&[m, k], f));
        let w = g.param("w", Shape::new(&[k, n], f));
        let a = g.param("a", Shape::new(&[k, r], f));
        let b = g.param("b", Shape::new(&[r, n], f));
        let y = lora_linear(&mut g, x, w, a, b, m, n, r);
        let t = g.input("t", Shape::new(&[m, n], f));
        let d = g.sub(y, t);
        let sq = g.mul(d, d);
        let flat = g.reshape_(sq, vec![(m * n) as i64]);
        let loss = g.mean(flat, vec![0], false);
        g.set_outputs(vec![loss]);
        (g, a, b)
    }

    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / 16_777_216.0 - 0.5) * 0.2
            })
            .collect()
    }

    fn host_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut o = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                o[i * n + j] = (0..k).map(|t| a[i * k + t] * b[t * n + j]).sum();
            }
        }
        o
    }

    #[test]
    fn fused_adam_matches_host_adam() {
        // The fused in-graph Adam must equal a host loop running the *same*
        // (per-step-t) Adam math — checked on the loss trajectory, which is
        // well-defined (unlike a/b, whose product A·B is what's identified).
        let (m, k, n, r) = (4usize, 3, 2, 3);
        let (g, a, b) = build(m, k, n, r);
        let xd = pseudo(m * k, 1);
        let td = pseudo(m * n, 9);
        let a0 = pseudo(k * r, 3);
        let b0 = pseudo(r * n, 4);
        let (steps, lr) = (30usize, 0.05f32);
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);

        // Host reference: backward graph + inline per-step Adam.
        let backward = grad_with_loss(&g, &[a, b]);
        let mut host = Session::new(Device::Cpu).compile_with(backward, &CompileOptions::new());
        host.set_param("w", &vec![0.0; k * n]);
        let (mut pa, mut pb) = (a0.clone(), b0.clone());
        let (mut ma, mut va) = (vec![0.0; pa.len()], vec![0.0; pa.len()]);
        let (mut mb, mut vb) = (vec![0.0; pb.len()], vec![0.0; pb.len()]);
        let seed = [1.0f32];
        let mut host_losses = Vec::new();
        for step in 1..=steps as i32 {
            host.set_param("a", &pa);
            host.set_param("b", &pb);
            let outs = host.run(&[("x", &xd), ("t", &td), ("d_output", &seed)]);
            host_losses.push(outs[0][0]);
            let (bc1, bc2) = (1.0 - b1.powi(step), 1.0 - b2.powi(step));
            let upd = |p: &mut [f32], mm: &mut [f32], vv: &mut [f32], gr: &[f32]| {
                for i in 0..p.len() {
                    mm[i] = b1 * mm[i] + (1.0 - b1) * gr[i];
                    vv[i] = b2 * vv[i] + (1.0 - b2) * gr[i] * gr[i];
                    p[i] -= lr * (mm[i] / bc1) / ((vv[i] / bc2).sqrt() + eps);
                }
            };
            upd(&mut pa, &mut ma, &mut va, &outs[1]);
            upd(&mut pb, &mut mb, &mut vb, &outs[2]);
        }

        // Fused resident trainer (host-chained on CPU).
        let mut p0 = HashMap::new();
        p0.insert("w".to_string(), vec![0.0; k * n]);
        p0.insert("a".to_string(), a0);
        p0.insert("b".to_string(), b0);
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
        let mut rt =
            ResidentTrainer::new(&g, &wrt, &p0, &AdamConfig::new(lr), 0.0, Device::Cpu).unwrap();
        let fused_losses = rt.run_fixed(&[("x", &xd), ("t", &td)], steps).unwrap();

        for (f, h) in fused_losses.iter().zip(&host_losses) {
            assert!((f - h).abs() < 1e-6, "fused loss {f} vs host {h}");
        }
    }

    #[test]
    fn fused_adamw_matches_host() {
        // Decoupled weight decay (AdamW): fused must match a host AdamW loop.
        let (m, k, n, r) = (4usize, 3, 2, 3);
        let (g, a, b) = build(m, k, n, r);
        let xd = pseudo(m * k, 1);
        let td = pseudo(m * n, 9);
        let a0 = pseudo(k * r, 3);
        let b0 = pseudo(r * n, 4);
        let (steps, lr, wd) = (30usize, 0.05f32, 0.02f32);
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);

        let backward = grad_with_loss(&g, &[a, b]);
        let mut host = Session::new(Device::Cpu).compile_with(backward, &CompileOptions::new());
        host.set_param("w", &vec![0.0; k * n]);
        let (mut pa, mut pb) = (a0.clone(), b0.clone());
        let (mut ma, mut va) = (vec![0.0; pa.len()], vec![0.0; pa.len()]);
        let (mut mb, mut vb) = (vec![0.0; pb.len()], vec![0.0; pb.len()]);
        let seed = [1.0f32];
        let mut host_losses = Vec::new();
        for step in 1..=steps as i32 {
            host.set_param("a", &pa);
            host.set_param("b", &pb);
            let outs = host.run(&[("x", &xd), ("t", &td), ("d_output", &seed)]);
            host_losses.push(outs[0][0]);
            let (bc1, bc2) = (1.0 - b1.powi(step), 1.0 - b2.powi(step));
            let upd = |p: &mut [f32], mm: &mut [f32], vv: &mut [f32], gr: &[f32]| {
                for i in 0..p.len() {
                    mm[i] = b1 * mm[i] + (1.0 - b1) * gr[i];
                    vv[i] = b2 * vv[i] + (1.0 - b2) * gr[i] * gr[i];
                    let adam = lr * (mm[i] / bc1) / ((vv[i] / bc2).sqrt() + eps);
                    p[i] -= adam + lr * wd * p[i]; // decoupled decay on original p
                }
            };
            upd(&mut pa, &mut ma, &mut va, &outs[1]);
            upd(&mut pb, &mut mb, &mut vb, &outs[2]);
        }

        let mut p0 = HashMap::new();
        p0.insert("w".to_string(), vec![0.0; k * n]);
        p0.insert("a".to_string(), a0);
        p0.insert("b".to_string(), b0);
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
        let mut rt =
            ResidentTrainer::new(&g, &wrt, &p0, &AdamConfig::new(lr), wd, Device::Cpu).unwrap();
        let fused = rt.run_fixed(&[("x", &xd), ("t", &td)], steps).unwrap();
        for (f, h) in fused.iter().zip(&host_losses) {
            assert!((f - h).abs() < 1e-6, "adamw fused {f} vs host {h}");
        }
    }

    // On CUDA the params + moments should live in device handles (no host
    // round-trip) and still converge. Only built with `--features cuda`.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_resident_trains() {
        let (m, k, n, r) = (4usize, 3, 2, 3);
        let (g, a, b) = build(m, k, n, r);
        let xd = pseudo(m * k, 1);
        let m_true: Vec<f32> = pseudo(k * n, 2).iter().map(|v| v * 10.0).collect();
        let td = host_matmul(&xd, &m_true, m, k, n);
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
        let mut p0 = HashMap::new();
        p0.insert("w".to_string(), vec![0.0; k * n]);
        p0.insert("a".to_string(), pseudo(k * r, 3));
        p0.insert("b".to_string(), pseudo(r * n, 4));
        let mut rt =
            ResidentTrainer::new(&g, &wrt, &p0, &AdamConfig::new(0.05), 0.0, Device::Cuda).unwrap();
        assert!(
            rt.is_resident(),
            "expected device-resident optimizer state on CUDA"
        );
        let losses = rt.run_fixed(&[("x", &xd), ("t", &td)], 300).unwrap();
        assert!(
            *losses.last().unwrap() < 1e-3,
            "cuda resident did not converge: {:?}",
            losses.last()
        );
    }

    #[test]
    fn fused_reduces_loss_to_zero() {
        // Exactly-fittable target (td = x·M) so the loss can reach ~0.
        let (m, k, n, r) = (4usize, 3, 2, 3);
        let (g, a, b) = build(m, k, n, r);
        let xd = pseudo(m * k, 1);
        let m_true: Vec<f32> = pseudo(k * n, 2).iter().map(|v| v * 10.0).collect();
        let td = host_matmul(&xd, &m_true, m, k, n);
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
        let mut p0 = HashMap::new();
        p0.insert("w".to_string(), vec![0.0; k * n]);
        p0.insert("a".to_string(), pseudo(k * r, 3));
        p0.insert("b".to_string(), pseudo(r * n, 4));
        let mut rt =
            ResidentTrainer::new(&g, &wrt, &p0, &AdamConfig::new(0.05), 0.0, Device::Cpu).unwrap();
        let losses = rt.run_fixed(&[("x", &xd), ("t", &td)], 300).unwrap();
        assert!(losses[0] > 1e-4, "initial loss {}", losses[0]);
        assert!(
            *losses.last().unwrap() < 1e-4,
            "did not converge: {:?}",
            losses.last()
        );
    }
}
