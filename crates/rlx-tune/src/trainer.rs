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

use crate::distributed::{GradComm, ReduceDtype};
use anyhow::{Context, Result, bail};
use rlx_autodiff::grad_with_loss;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};
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
/// When `comm` is `Some` (and world size > 1), this is **data-parallel**:
/// every rank runs the same loop on its own `inputs` (data shard); trainable
/// weights are broadcast from rank 0 once, and every step all gradients are
/// packed into one contiguous **bucket** and mean-all-reduced across ranks in
/// a single collective before the optimizer step — turning K per-parameter
/// reduces into one, so all ranks stay in lockstep with identical weights.
/// The bucket is laid out over `wrt` in its (rank-stable) order, never
/// `params`' nondeterministic `HashMap` order, and reused across steps so the
/// loop allocates nothing. For zero-config bring-up build `comm` from the
/// environment with [`crate::from_env`].
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

    // A single collective per step over ALL gradients (one "bucket") is far
    // cheaper than one collective per parameter: for K adapter tensors it
    // turns K latency-bound all-reduces into one bandwidth-optimal one. So
    // treat `comm` as active only when it will actually reduce (world > 1);
    // `Option<&dyn GradComm>` is `Copy`, so this stays usable below.
    let comm = comm.filter(|c| c.world_size() > 1);

    // Contiguous [offset, len) layout of the trainable params in `wrt` order
    // (rank-stable, unlike the `params` HashMap). grad(wrt[i]) has the same
    // length as the param, so one layout serves both the gradient reduce and
    // the weight broadcast. Doubles as an up-front check that every `wrt`
    // param has a value.
    let mut layout = Vec::with_capacity(wrt.len());
    let mut fused_len = 0usize;
    for slot in wrt {
        let len = params
            .get(&slot.name)
            .ok_or_else(|| anyhow::anyhow!("missing param value for {}", slot.name))?
            .len();
        layout.push((fused_len, len));
        fused_len += len;
    }

    // Data-parallel: sync trainable weights from rank 0 so every rank starts
    // identical (frozen params are assumed loaded identically on every rank),
    // fused into one broadcast of the whole bucket.
    if let Some(c) = comm {
        let mut bcast = vec![0.0f32; fused_len];
        for (slot, &(off, len)) in wrt.iter().zip(&layout) {
            bcast[off..off + len].copy_from_slice(&params[&slot.name]);
        }
        c.broadcast(0, &mut bcast);
        for (slot, &(off, len)) in wrt.iter().zip(&layout) {
            if let Some(p) = params.get_mut(&slot.name) {
                p.copy_from_slice(&bcast[off..off + len]);
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

    // Reused across steps so the fused reduce allocates nothing in the loop
    // (empty when not distributed).
    let mut bucket = vec![0.0f32; if comm.is_some() { fused_len } else { 0 }];

    let mut losses = Vec::with_capacity(steps);
    for _ in 0..steps {
        let outs = compiled.run(&run_inputs);
        if outs.is_empty() || outs[0].is_empty() {
            bail!("backward graph produced no loss output");
        }
        losses.push(outs[0][0]);
        // outs = [loss, grad(wrt[0]), grad(wrt[1]), ...].
        if let Some(c) = comm {
            // Pack every gradient into one contiguous buffer, average it
            // across ranks in a single collective, then scatter back.
            for (i, &(off, len)) in layout.iter().enumerate() {
                let g = outs.get(1 + i).map(Vec::as_slice).unwrap_or(&[]);
                let n = len.min(g.len());
                bucket[off..off + n].copy_from_slice(&g[..n]);
                bucket[off + n..off + len].fill(0.0);
            }
            c.all_reduce_mean(&mut bucket);
            for (i, slot) in wrt.iter().enumerate() {
                let (off, len) = layout[i];
                let p = params
                    .get_mut(&slot.name)
                    .expect("param existence checked above");
                opt.step(&slot.name, p, &bucket[off..off + len]);
                compiled.set_param(&slot.name, p);
            }
        } else {
            for (i, slot) in wrt.iter().enumerate() {
                let grad = outs.get(1 + i).map(Vec::as_slice).unwrap_or(&[]);
                let p = params
                    .get_mut(&slot.name)
                    .expect("param existence checked above");
                opt.step(&slot.name, p, grad);
                compiled.set_param(&slot.name, p);
            }
        }
    }
    Ok(losses)
}

// ===========================================================================
// train_dp — feature-rich data-parallel training
// ===========================================================================

/// Adam hyperparameters for [`train_dp`]'s fused optimizer.
#[derive(Clone, Copy, Debug)]
pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Decoupled weight decay (AdamW). `0.0` = plain Adam. Applied as
    /// `param -= lr·wd·param` each step, independent of the moment estimates.
    pub weight_decay: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

impl AdamConfig {
    /// Adam with learning rate `lr` and default `beta1=0.9`, `beta2=0.999`,
    /// `eps=1e-8`.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            ..Self::default()
        }
    }
    /// Set the momentum coefficients (β₁, β₂).
    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
    /// Set the numerical-stability epsilon.
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
    /// Set decoupled weight decay (turns Adam into AdamW).
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

/// Adam over a **flat** parameter slice — position-indexed moments (no per-name
/// map), so a rank can own any contiguous slice of the fused bucket, even one
/// that straddles parameter boundaries (the enabler for ZeRO-1 optimizer-state
/// sharding). Its per-element update equals the per-parameter [`Adam`], so
/// sharded / overlapped / plain training all match single-process bit-closely.
struct FlatAdam {
    cfg: AdamConfig,
    t: i32,
    m: Vec<f32>,
    v: Vec<f32>,
    bc1: f32, // bias corrections for the in-progress step
    bc2: f32,
}

impl FlatAdam {
    fn new(cfg: AdamConfig, n: usize) -> Self {
        Self {
            cfg,
            t: 0,
            m: vec![0.0; n],
            v: vec![0.0; n],
            bc1: 1.0,
            bc2: 1.0,
        }
    }

    /// Set the learning rate for subsequent steps (drives the LR schedule).
    fn set_lr(&mut self, lr: f32) {
        self.cfg.lr = lr;
    }

    /// Advance the timestep once per optimization step (before applying any
    /// range), fixing this step's bias corrections.
    fn begin_step(&mut self) {
        self.t += 1;
        self.bc1 = 1.0 - self.cfg.beta1.powi(self.t);
        self.bc2 = 1.0 - self.cfg.beta2.powi(self.t);
    }

    /// Apply the update to `param` (a slice of the bucket whose element `i` is
    /// flat index `base + i` in the moment state) from `grad`, using this
    /// step's bias corrections. Called once (full range) or per chunk.
    fn apply_range(&mut self, base: usize, param: &mut [f32], grad: &[f32]) {
        let AdamConfig {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        } = self.cfg;
        for i in 0..param.len() {
            let g = grad.get(i).copied().unwrap_or(0.0);
            let idx = base + i;
            self.m[idx] = beta1 * self.m[idx] + (1.0 - beta1) * g;
            self.v[idx] = beta2 * self.v[idx] + (1.0 - beta2) * g * g;
            let mhat = self.m[idx] / self.bc1;
            let vhat = self.v[idx] / self.bc2;
            // Decoupled weight decay (AdamW): shrink the weight independent of
            // the adaptive step. No-op when weight_decay == 0 (plain Adam).
            param[i] -= lr * (mhat / (vhat.sqrt() + eps) + weight_decay * param[i]);
        }
    }
}

/// The Muon + AdamW optimizer group for the single-process step path: 2-D
/// weight matrices → **canonical** [`rlx_optim::Muon`] (Nesterov-orthogonalized
/// momentum), everything 1-D → [`rlx_optim::AdamW`]. Per-tensor (keyed by name),
/// so it needs whole matrices — hence single-process / unsharded only. Uses the
/// same reference implementations the rest of rlx does, no bespoke update rule.
struct MuonGroup {
    muon: rlx_optim::Muon,
    adamw: rlx_optim::AdamW,
    shapes: Vec<Vec<usize>>,
}

impl MuonGroup {
    fn new(cfg: &AdamConfig, shapes: Vec<Vec<usize>>) -> Self {
        Self {
            // 5 Newton–Schulz iterations (the paper default). ns=3 is ~1.5×
            // cheaper but measurably hurts convergence on the codec's
            // ill-conditioned weight matrices (loss 0.64→0.74), so keep 5.
            muon: rlx_optim::Muon::new(cfg.lr).with_weight_decay(cfg.weight_decay),
            adamw: rlx_optim::AdamW::new(cfg.lr)
                .with_betas(cfg.beta1, cfg.beta2)
                .with_weight_decay(cfg.weight_decay)
                .with_eps(cfg.eps),
            shapes,
        }
    }

    /// Apply per parameter over the full bucket: 2-D → Muon, else → AdamW.
    fn apply(
        &mut self,
        lr: f32,
        wrt: &[ParamSlot],
        layout: &[(usize, usize)],
        pbucket: &mut [f32],
        gbucket: &[f32],
    ) {
        use rlx_optim::Optimizer;
        self.muon.set_lr(lr);
        self.adamw.set_lr(lr);
        for (i, (slot, &(off, len))) in wrt.iter().zip(layout).enumerate() {
            let shape = &self.shapes[i];
            let p = &mut pbucket[off..off + len];
            let g = &gbucket[off..off + len];
            if shape.len() == 2 && shape[0] >= 2 && shape[1] >= 2 {
                self.muon.step(&slot.name, shape, p, g);
            } else {
                self.adamw.step(&slot.name, shape, p, g);
            }
        }
    }
}

/// Learning-rate decay after warmup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LrSchedule {
    /// Hold `adam.lr` (after any warmup).
    #[default]
    Constant,
    /// Linear decay to `min_lr_ratio * adam.lr` over the post-warmup steps.
    Linear,
    /// Cosine decay to `min_lr_ratio * adam.lr` over the post-warmup steps.
    Cosine,
}

/// Configuration for [`train_dp`], the feature-rich data-parallel trainer.
#[derive(Clone, Copy, Debug)]
pub struct DpConfig {
    /// Compute device the backward graph compiles to. Defaults to
    /// [`Device::Cpu`]; `Device::Metal` / `Device::Cuda` run the forward +
    /// backward on the GPU (needs the matching `metal` / `cuda` build feature).
    pub device: Device,
    /// Adam hyperparameters (the fused optimizer).
    pub adam: AdamConfig,
    /// ZeRO-1: shard optimizer state across ranks — each rank stores Adam
    /// moments for only `1/world` of the parameters (the dominant training
    /// memory), reconstructing full weights each step via all-gather.
    /// Numerically identical to unsharded; saves ~`2·params·(world-1)/world`
    /// floats of optimizer state per rank. No effect when `world == 1`.
    pub shard_optimizer: bool,
    /// Overlap the gradient all-reduce with the optimizer step by pipelining
    /// the bucket in `chunks` pieces on a background thread — numerically exact
    /// (same reduces, same steps), hiding comm behind host update compute.
    /// Ignored when `world == 1`, `chunks <= 1`, or `shard_optimizer` is set.
    pub overlap: bool,
    /// Pipeline granularity for `overlap` (also the shard chunk count for
    /// overlapped sharding). `0` picks a default (`4`).
    pub chunks: usize,
    /// Gradient accumulation: run this many micro-batches (from the data
    /// provider) and average their gradients before **one** reduce + optimizer
    /// step. Cuts communication `grad_accum×` and grows the effective batch.
    /// `0` or `1` = no accumulation. Only [`train_dp_with`] varies the data
    /// across micro-batches; [`train_dp`] repeats its fixed batch.
    pub grad_accum: usize,
    /// Wire precision for the gradient all-reduce ([`ReduceDtype`]).
    pub reduce_dtype: ReduceDtype,
    /// Clip the **global** gradient L2 norm to this value before the optimizer
    /// step. Distributed-correct: the norm is over the reduced gradient
    /// (identical on every rank), so all ranks scale identically and stay in
    /// sync. `None` disables it. Enabling it disables `overlap` (the global
    /// norm needs the whole reduced bucket before any step).
    pub max_grad_norm: Option<f32>,
    /// Linear learning-rate warmup over the first `warmup_steps` steps
    /// (`0` = no warmup).
    pub warmup_steps: usize,
    /// Learning-rate decay after warmup ([`LrSchedule`]).
    pub lr_schedule: LrSchedule,
    /// Decay floor as a fraction of `adam.lr` (e.g. `0.1` decays to 10% of the
    /// base lr). Only used when `lr_schedule != Constant`.
    pub min_lr_ratio: f32,
    /// Invoke the step callback every `log_every` steps (and always on the last
    /// step). `0` disables it.
    pub log_every: usize,
    /// Use **Muon** (Newton–Schulz orthogonalized momentum) for 2-D weight
    /// matrices, with AdamW for everything else (1-D norms/biases/scales). Only
    /// active on the single-process / unsharded step path; sharded/overlapped
    /// DP falls back to AdamW (Muon needs whole matrices, not bucket chunks).
    pub muon: bool,
}

impl Default for DpConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            adam: AdamConfig::default(),
            shard_optimizer: false,
            overlap: false,
            chunks: 0,
            grad_accum: 1,
            reduce_dtype: ReduceDtype::F32,
            max_grad_norm: None,
            warmup_steps: 0,
            lr_schedule: LrSchedule::Constant,
            min_lr_ratio: 0.0,
            log_every: 0,
            muon: false,
        }
    }
}

/// Fluent builder. Fields stay public (struct-literal `DpConfig { .. }` still
/// works); these just read better for the common path:
///
/// ```
/// use rlx_tune::DpConfig;
/// let cfg = DpConfig::new(2e-4)   // learning rate
///     .shard()                    // ZeRO-1 optimizer-state sharding
///     .overlap()                  // hide comm behind the optimizer step
///     .bf16()                     // bf16 gradients on the wire
///     .clip(1.0)                  // global-norm gradient clipping
///     .warmup(100).cosine(0.1)    // 100-step warmup, cosine decay to 10%
///     .grad_accum(4)              // 4 micro-batches per step
///     .log_every(50);
/// ```
impl DpConfig {
    /// A config with learning rate `lr`; everything else default.
    pub fn new(lr: f32) -> Self {
        Self {
            adam: AdamConfig::new(lr),
            ..Self::default()
        }
    }
    /// Compile / run on `device` (e.g. `Device::Metal`, `Device::Cuda`).
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
    /// Run on the Apple Metal GPU (needs the `metal` build feature).
    pub fn metal(self) -> Self {
        self.device(Device::Metal)
    }
    /// Run on an NVIDIA CUDA GPU (needs the `cuda` build feature).
    pub fn cuda(self) -> Self {
        self.device(Device::Cuda)
    }
    /// Replace the whole [`AdamConfig`] (betas / eps).
    pub fn adam(mut self, adam: AdamConfig) -> Self {
        self.adam = adam;
        self
    }
    /// Set the learning rate.
    pub fn lr(mut self, lr: f32) -> Self {
        self.adam.lr = lr;
        self
    }
    /// Enable ZeRO-1 optimizer-state sharding (`shard_optimizer`).
    pub fn shard(mut self) -> Self {
        self.shard_optimizer = true;
        self
    }
    /// Overlap gradient communication with the optimizer step (`overlap`).
    pub fn overlap(mut self) -> Self {
        self.overlap = true;
        self
    }
    /// Set the overlap / sharded-overlap pipeline chunk count.
    pub fn chunks(mut self, chunks: usize) -> Self {
        self.chunks = chunks;
        self
    }
    /// Accumulate `g` micro-batches per optimizer step (`grad_accum`).
    pub fn grad_accum(mut self, g: usize) -> Self {
        self.grad_accum = g;
        self
    }
    /// Reduce gradients in bfloat16 on the wire (`reduce_dtype = Bf16`).
    pub fn bf16(mut self) -> Self {
        self.reduce_dtype = ReduceDtype::Bf16;
        self
    }
    /// Set decoupled weight decay (AdamW) on the fused optimizer.
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.adam.weight_decay = wd;
        self
    }
    /// Use Muon (orthogonalized momentum) for 2-D weights + AdamW for the rest.
    pub fn muon(mut self) -> Self {
        self.muon = true;
        self
    }
    /// Set the gradient-reduce wire precision.
    pub fn reduce_dtype(mut self, dtype: ReduceDtype) -> Self {
        self.reduce_dtype = dtype;
        self
    }
    /// Clip the global gradient L2 norm to `max_norm` (`max_grad_norm`).
    pub fn clip(mut self, max_norm: f32) -> Self {
        self.max_grad_norm = Some(max_norm);
        self
    }
    /// Linear LR warmup over the first `steps` steps.
    pub fn warmup(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }
    /// Cosine LR decay to `min_ratio × lr` after warmup.
    pub fn cosine(mut self, min_ratio: f32) -> Self {
        self.lr_schedule = LrSchedule::Cosine;
        self.min_lr_ratio = min_ratio;
        self
    }
    /// Linear LR decay to `min_ratio × lr` after warmup.
    pub fn linear_decay(mut self, min_ratio: f32) -> Self {
        self.lr_schedule = LrSchedule::Linear;
        self.min_lr_ratio = min_ratio;
        self
    }
    /// Invoke the step callback every `n` steps (and always the last).
    pub fn log_every(mut self, n: usize) -> Self {
        self.log_every = n;
        self
    }

    /// A one-line summary of the enabled knobs — handy to log at startup.
    /// Reflects the *requested* config (runtime may skip `overlap` when
    /// `max_grad_norm` is set; see [`train_dp`]).
    pub fn describe(&self) -> String {
        let mut s = format!("lr={:.2e}", self.adam.lr);
        if self.device != Device::Cpu {
            s.push_str(&format!(" {:?}", self.device));
        }
        if self.shard_optimizer {
            s.push_str(" shard");
        }
        if self.overlap {
            s.push_str(&format!(
                " overlap(x{})",
                if self.chunks == 0 { 4 } else { self.chunks }
            ));
        }
        if self.grad_accum > 1 {
            s.push_str(&format!(" accum={}", self.grad_accum));
        }
        if self.reduce_dtype == ReduceDtype::Bf16 {
            s.push_str(" bf16");
        }
        if let Some(c) = self.max_grad_norm {
            s.push_str(&format!(" clip={c}"));
        }
        if self.warmup_steps > 0 {
            s.push_str(&format!(" warmup={}", self.warmup_steps));
        }
        match self.lr_schedule {
            LrSchedule::Constant => {}
            LrSchedule::Linear => s.push_str(&format!(" linear→{:.0}%", self.min_lr_ratio * 100.0)),
            LrSchedule::Cosine => s.push_str(&format!(" cosine→{:.0}%", self.min_lr_ratio * 100.0)),
        }
        s
    }
}

/// Per-step timing + throughput, handed to [`train_dp`]'s callback.
#[derive(Clone, Copy, Debug)]
pub struct StepMetrics {
    pub step: usize,
    pub loss: f32,
    /// Effective learning rate this step (after warmup / schedule).
    pub lr: f32,
    /// Wall-clock of the backward graph run (ms).
    pub compute_ms: f64,
    /// Gradient-collective wall-clock this step (ms). With `overlap`, this is
    /// the time **not** hidden behind the optimizer step.
    pub comm_ms: f64,
    /// Total wall-clock for the step (ms).
    pub step_ms: f64,
    /// Gradient elements reduced (the fused bucket size).
    pub reduced_elems: usize,
    /// World size (1 = single process).
    pub world_size: u32,
}

impl StepMetrics {
    /// Fraction of the step spent computing rather than communicating —
    /// `compute / (compute + comm)`. Near 1.0 means comm is well hidden.
    pub fn compute_fraction(&self) -> f64 {
        let denom = self.compute_ms + self.comm_ms;
        if denom > 0.0 {
            self.compute_ms / denom
        } else {
            1.0
        }
    }
}

impl std::fmt::Display for StepMetrics {
    /// The standard one-line progress row — use `println!("{m}")` in `on_step`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "step {:>5} | loss {:.6} | lr {:.4} | compute {:.2}ms comm {:.2}ms ({:.0}% compute) | world {}",
            self.step,
            self.loss,
            self.lr,
            self.compute_ms,
            self.comm_ms,
            self.compute_fraction() * 100.0,
            self.world_size,
        )
    }
}

/// Feature-rich **data-parallel** training on a **fixed** batch — everything
/// [`train`] does, plus optimizer-state **sharding** (ZeRO-1), comm/compute
/// **overlap**, **mixed-precision** reduction, gradient **clipping**, an **LR
/// schedule**, and per-step **timing**, all driven by [`DpConfig`]. Build `comm`
/// from the environment with [`crate::from_env`] (or hand-wire one).
///
/// The fused Adam's per-element update matches [`train`]'s [`Adam`], so results
/// are identical to single-process training on the union of the shards.
/// `comm == None` (or `world_size == 1`) runs plain single-process training.
///
/// For **varying data per step** (real training) or **gradient accumulation**,
/// use [`train_dp_with`], which takes a data provider; this is the fixed-batch
/// convenience wrapper.
#[allow(clippy::too_many_arguments)]
pub fn train_dp(
    forward: Graph,
    wrt: &[ParamSlot],
    params: &mut HashMap<String, Vec<f32>>,
    inputs: &[(String, Vec<f32>)],
    steps: usize,
    comm: Option<&dyn GradComm>,
    cfg: &DpConfig,
    on_step: impl FnMut(&StepMetrics),
) -> Result<Vec<f32>> {
    train_dp_with(
        forward,
        wrt,
        params,
        steps,
        comm,
        cfg,
        |_step, _micro| inputs.to_vec(),
        on_step,
    )
}

/// [`train_dp`] with a **data provider** and **gradient accumulation**.
///
/// `next_batch(step, micro)` returns the model inputs (same names/shapes the
/// compiled graph expects) for optimization `step`, micro-batch `micro ∈
/// 0..grad_accum`. With `cfg.grad_accum > 1` the per-micro-batch gradients are
/// **averaged** before a single reduce + optimizer step — the same effective
/// batch as one `grad_accum×`-larger batch, with `grad_accum×` less
/// communication. All micro-batches must share the graph's input shapes.
#[allow(clippy::too_many_arguments)]
pub fn train_dp_with(
    forward: Graph,
    wrt: &[ParamSlot],
    params: &mut HashMap<String, Vec<f32>>,
    steps: usize,
    comm: Option<&dyn GradComm>,
    cfg: &DpConfig,
    next_batch: impl FnMut(usize, usize) -> Vec<(String, Vec<f32>)>,
    on_step: impl FnMut(&StepMetrics),
) -> Result<Vec<f32>> {
    let mut trainer = Trainer::new(forward, wrt, params, steps, comm, cfg)?;
    let losses = trainer.run(next_batch, on_step)?;
    trainer.write_params_into(params);
    Ok(losses)
}

/// A running data-parallel training session you drive yourself.
///
/// [`train_dp`] / [`train_dp_with`] are `Trainer::new(...).run(...)`; use the
/// `Trainer` directly when you want a **custom loop** (per-step eval, early
/// stop, custom logging) or **checkpointing**: call [`step`](Trainer::step) in
/// your own loop, [`checkpoint`](Trainer::checkpoint) / [`restore`](Trainer::restore)
/// to save/resume, and [`params`](Trainer::params) to read the current weights.
///
/// It owns the compiled backward, the fused parameter bucket, and the (possibly
/// sharded) optimizer state.
pub struct Trainer<'c> {
    comm: Option<&'c dyn GradComm>,
    cfg: DpConfig,
    total_steps: usize,
    compiled: CompiledGraph,
    wrt: Vec<ParamSlot>,
    layout: Vec<(usize, usize)>,
    real_len: usize,
    padded_len: usize,
    world: usize,
    rank: usize,
    sharded: bool,
    sc: usize,
    chunks: usize,
    overlap_dp: bool,
    shard: usize,
    pbucket: Vec<f32>,
    gbucket: Vec<f32>,
    opt: FlatAdam,
    /// Muon+AdamW group; `Some` only when `cfg.muon` and the step path is the
    /// single-process / unsharded one (Muon needs whole matrices). Takes over
    /// the optimizer apply from `opt` when present.
    muon: Option<MuonGroup>,
    step: usize,
}

impl<'c> Trainer<'c> {
    /// Build a trainer: compile the backward of `forward` w.r.t. `wrt`, seed the
    /// parameter bucket from `params`, and (when distributed) broadcast it so
    /// every rank starts identical. `total_steps` drives the LR schedule.
    pub fn new(
        forward: Graph,
        wrt: &[ParamSlot],
        params: &HashMap<String, Vec<f32>>,
        total_steps: usize,
        comm: Option<&'c dyn GradComm>,
        cfg: &DpConfig,
    ) -> Result<Self> {
        let wrt_ids: Vec<NodeId> = wrt.iter().map(|s| s.node).collect();
        let backward = grad_with_loss(&forward, &wrt_ids);
        let mut compiled = Session::new(cfg.device).compile_with(backward, &CompileOptions::new());

        // Engage the collective only when it will actually reduce.
        let comm = comm.filter(|c| c.world_size() > 1);
        let world = comm.map(|c| c.world_size()).unwrap_or(1) as usize;
        let rank = comm.map(|c| c.rank()).unwrap_or(0) as usize;

        // Fused [offset, len) layout of the trainable params in rank-stable
        // `wrt` order; also checks every `wrt` param has a value.
        let mut layout = Vec::with_capacity(wrt.len());
        let mut shapes: Vec<Vec<usize>> = Vec::with_capacity(wrt.len());
        let mut real_len = 0usize;
        for slot in wrt {
            let len = params
                .get(&slot.name)
                .ok_or_else(|| anyhow::anyhow!("missing param value for {}", slot.name))?
                .len();
            layout.push((real_len, len));
            // Static param shape from the forward graph (for Muon's matrix ops).
            shapes.push(
                forward
                    .shape(slot.node)
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect(),
            );
            real_len += len;
        }

        let sharded = cfg.shard_optimizer && world > 1;
        let chunks = if cfg.chunks == 0 { 4 } else { cfg.chunks };
        // Overlap needs the whole reduced bucket for a global-norm clip, so it
        // is mutually exclusive with clipping.
        let can_overlap =
            cfg.overlap && comm.is_some() && chunks > 1 && cfg.max_grad_norm.is_none();
        let overlap_dp = can_overlap && !sharded && real_len >= chunks;
        // Sharded overlap uses `sc` block-cyclic chunks so each chunk's
        // reduce-scatter can pipeline with the optimizer step.
        let sc = if sharded && can_overlap && real_len >= chunks * world {
            chunks
        } else {
            1
        };

        // Sharding needs equal shards → pad the bucket to a multiple of `world`
        // (or `world*sc` for overlapped sharding). Padding slots are dummy
        // params (grad 0), never written back.
        let pad_mult = if sharded { world * sc } else { 1 };
        let padded_len = real_len.div_ceil(pad_mult) * pad_mult;
        let shard = padded_len / world; // world >= 1 always

        // The flat parameter bucket is the source of truth across steps.
        let mut pbucket = vec![0.0f32; padded_len];
        for (slot, &(off, len)) in wrt.iter().zip(&layout) {
            pbucket[off..off + len].copy_from_slice(&params[&slot.name]);
        }
        // Start every rank identical (frozen params assumed identical already).
        if let Some(c) = comm {
            c.broadcast(0, &mut pbucket[..real_len]);
        }

        let opt = FlatAdam::new(cfg.adam, if sharded { shard } else { padded_len });
        // Muon orthogonalizes whole weight matrices, incompatible with the
        // bucket-chunk-straddling sharded/overlapped apply → only the plain
        // single-step path. Otherwise the flat AdamW path handles everything.
        let muon = if cfg.muon && !sharded && !overlap_dp {
            Some(MuonGroup::new(&cfg.adam, shapes))
        } else {
            None
        };

        // Frozen (non-`wrt`) params are uploaded once.
        for (name, data) in params.iter() {
            compiled.set_param(name, data);
        }

        Ok(Self {
            comm,
            cfg: *cfg,
            total_steps,
            compiled,
            wrt: wrt.to_vec(),
            layout,
            real_len,
            padded_len,
            world,
            rank,
            sharded,
            sc,
            chunks,
            overlap_dp,
            shard,
            pbucket,
            gbucket: vec![0.0f32; padded_len],
            opt,
            muon,
            step: 0,
        })
    }

    /// Run one optimization step: `grad_accum` micro-batches from `next_batch`,
    /// gradient reduce/shard/overlap per the config, LR schedule, clip, and the
    /// Adam update. Returns the step's [`StepMetrics`].
    pub fn step(
        &mut self,
        next_batch: &mut impl FnMut(usize, usize) -> Vec<(String, Vec<f32>)>,
    ) -> Result<StepMetrics> {
        use std::time::Instant;
        let step = self.step;
        let comm = self.comm;
        let t_step = Instant::now();
        let seed = [1.0f32];

        // Push current weights into the session (only `wrt` params change).
        for (slot, &(off, len)) in self.wrt.iter().zip(&self.layout) {
            self.compiled
                .set_param(&slot.name, &self.pbucket[off..off + len]);
        }

        // ---- accumulate gradients over micro-batches (compute) ----
        let ga = self.cfg.grad_accum.max(1);
        self.gbucket.iter_mut().for_each(|g| *g = 0.0);
        let mut loss_sum = 0.0f32;
        let mut compute_ms = 0.0f64;
        for micro in 0..ga {
            let batch = next_batch(step, micro);
            let mut run_inputs: Vec<(&str, &[f32])> = batch
                .iter()
                .map(|(n, d)| (n.as_str(), d.as_slice()))
                .collect();
            run_inputs.push(("d_output", &seed));
            let t_compute = Instant::now();
            let outs = self.compiled.run(&run_inputs);
            compute_ms += t_compute.elapsed().as_secs_f64() * 1e3;
            if outs.is_empty() || outs[0].is_empty() {
                bail!("backward graph produced no loss output");
            }
            loss_sum += outs[0][0];
            for (i, &(off, len)) in self.layout.iter().enumerate() {
                let g = outs.get(1 + i).map(Vec::as_slice).unwrap_or(&[]);
                let n = len.min(g.len());
                for (dst, &gv) in self.gbucket[off..off + n].iter_mut().zip(&g[..n]) {
                    *dst += gv;
                }
            }
        }
        if ga > 1 {
            let inv = 1.0 / ga as f32;
            self.gbucket[..self.real_len]
                .iter_mut()
                .for_each(|g| *g *= inv);
        }
        let loss = loss_sum / ga as f32;

        // ---- learning-rate schedule (warmup + decay) ----
        let lr = effective_lr(
            self.cfg.adam.lr,
            step,
            self.total_steps,
            self.cfg.warmup_steps,
            self.cfg.lr_schedule,
            self.cfg.min_lr_ratio,
        );
        self.opt.set_lr(lr);

        // ---- reduce + optimizer step ----
        self.opt.begin_step();
        let comm_ms = self.reduce_and_step(comm);

        self.step += 1;
        Ok(StepMetrics {
            step,
            loss,
            lr,
            compute_ms,
            comm_ms,
            step_ms: t_step.elapsed().as_secs_f64() * 1e3,
            reduced_elems: self.real_len,
            world_size: self.world as u32,
        })
    }

    /// Reduce the accumulated `gbucket` and apply the optimizer, dispatching on
    /// sharding / overlap / clipping. Returns `comm_ms`.
    fn reduce_and_step(&mut self, comm: Option<&dyn GradComm>) -> f64 {
        use std::time::Instant;
        let (world, rank, shard, real_len) = (self.world, self.rank, self.shard, self.real_len);
        if self.sharded {
            let c = comm.expect("sharded implies comm");
            if self.sc > 1 {
                // Overlapped ZeRO-1: pipeline each chunk's reduce-scatter (bg
                // thread) with the optimizer step, then batch the all-gathers.
                sharded_overlap_step(
                    c,
                    &self.gbucket,
                    &mut self.pbucket,
                    &mut self.opt,
                    world,
                    rank,
                    self.sc,
                )
            } else {
                // ZeRO-1: reduce-scatter → (clip) → step our shard → all-gather.
                let t_comm = Instant::now();
                let mut shard_grad = vec![0.0f32; shard];
                c.reduce_scatter_mean(&self.gbucket, &mut shard_grad);
                if let Some(max) = self.cfg.max_grad_norm {
                    // Global norm needs every shard's sum-of-squares: reduce the
                    // per-rank partial (mean→sum) so all ranks agree on the scale.
                    let mut ss = [shard_grad.iter().map(|g| g * g).sum::<f32>()];
                    if world > 1 {
                        c.all_reduce_mean(&mut ss);
                        ss[0] *= world as f32;
                    }
                    let s = clip_scale(ss[0], max);
                    if s != 1.0 {
                        shard_grad.iter_mut().for_each(|g| *g *= s);
                    }
                }
                let base = rank * shard;
                let mut shard_param = self.pbucket[base..base + shard].to_vec();
                self.opt.apply_range(0, &mut shard_param, &shard_grad);
                c.all_gather_into(&shard_param, &mut self.pbucket);
                t_comm.elapsed().as_secs_f64() * 1e3
            }
        } else if self.overlap_dp {
            overlapped_reduce_step(
                comm.expect("overlap implies comm"),
                &mut self.gbucket[..real_len],
                &mut self.pbucket[..real_len],
                &mut self.opt,
                self.chunks,
                self.cfg.reduce_dtype,
            )
        } else {
            let t_comm = Instant::now();
            if let Some(c) = comm {
                c.all_reduce_mean_typed(&mut self.gbucket[..real_len], self.cfg.reduce_dtype);
            }
            let comm_ms = t_comm.elapsed().as_secs_f64() * 1e3;
            // The reduced gradient is identical on every rank, so the global
            // norm (and clip scale) match without any extra communication.
            if let Some(max) = self.cfg.max_grad_norm {
                let ss = self.gbucket[..real_len].iter().map(|g| g * g).sum::<f32>();
                let s = clip_scale(ss, max);
                if s != 1.0 {
                    self.gbucket[..real_len].iter_mut().for_each(|g| *g *= s);
                }
            }
            if self.muon.is_some() {
                let lr = self.opt.cfg.lr;
                let Self {
                    muon,
                    wrt,
                    layout,
                    pbucket,
                    gbucket,
                    ..
                } = self;
                muon.as_mut().unwrap().apply(
                    lr,
                    wrt,
                    layout,
                    &mut pbucket[..real_len],
                    &gbucket[..real_len],
                );
            } else {
                self.opt
                    .apply_range(0, &mut self.pbucket[..real_len], &self.gbucket[..real_len]);
            }
            comm_ms
        }
    }

    /// Run steps until `total_steps` (respecting a prior [`restore`](Self::restore)),
    /// invoking `on_step` every `log_every` steps. Returns per-step loss.
    pub fn run(
        &mut self,
        mut next_batch: impl FnMut(usize, usize) -> Vec<(String, Vec<f32>)>,
        mut on_step: impl FnMut(&StepMetrics),
    ) -> Result<Vec<f32>> {
        let mut losses = Vec::new();
        while self.step < self.total_steps {
            let m = self.step(&mut next_batch)?;
            losses.push(m.loss);
            let last = self.step == self.total_steps;
            if self.cfg.log_every != 0 && (last || self.step.is_multiple_of(self.cfg.log_every)) {
                on_step(&m);
            }
        }
        Ok(losses)
    }

    /// [`run`](Self::run) with **background data prefetch**: a producer thread
    /// generates each step's `grad_accum` micro-batches (calling `next_batch`)
    /// while this thread computes the current step, so data loading /
    /// augmentation overlaps compute instead of stalling it — a throughput win
    /// whenever `next_batch` is non-trivial (real datasets, disk, augmentation).
    ///
    /// Numerically identical to [`run`](Self::run) (same batches, same order);
    /// `next_batch` just runs on another thread, so it must be `Send`. A bounded
    /// channel keeps the producer at most two steps ahead (double-buffering).
    pub fn run_prefetched(
        &mut self,
        next_batch: impl FnMut(usize, usize) -> Vec<(String, Vec<f32>)> + Send,
        mut on_step: impl FnMut(&StepMetrics),
    ) -> Result<Vec<f32>> {
        let ga = self.cfg.grad_accum.max(1);
        let (start, total) = (self.step, self.total_steps);
        let mut losses = Vec::new();
        // One step's worth of micro-batches.
        type Step = Vec<Vec<(String, Vec<f32>)>>;

        std::thread::scope(|s| -> Result<()> {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Step>(2);
            let mut produce = next_batch;
            s.spawn(move || {
                for step in start..total {
                    let batches: Step = (0..ga).map(|m| produce(step, m)).collect();
                    if tx.send(batches).is_err() {
                        break; // consumer done / dropped the receiver
                    }
                }
            });
            while self.step < total {
                let Ok(batches) = rx.recv() else { break };
                let mut it = batches.into_iter();
                let m = self.step(&mut |_step, _micro| {
                    it.next().expect("prefetch: micro-batch underflow")
                })?;
                losses.push(m.loss);
                let last = self.step == total;
                if self.cfg.log_every != 0 && (last || self.step.is_multiple_of(self.cfg.log_every))
                {
                    on_step(&m);
                }
            }
            Ok(())
        })?;
        Ok(losses)
    }

    /// The current trainable weights (by `wrt` name).
    pub fn params(&self) -> HashMap<String, Vec<f32>> {
        self.wrt
            .iter()
            .zip(&self.layout)
            .map(|(slot, &(off, len))| (slot.name.clone(), self.pbucket[off..off + len].to_vec()))
            .collect()
    }

    /// Copy the current trainable weights into `params` in place.
    pub fn write_params_into(&self, params: &mut HashMap<String, Vec<f32>>) {
        for (slot, &(off, len)) in self.wrt.iter().zip(&self.layout) {
            if let Some(p) = params.get_mut(&slot.name) {
                p.copy_from_slice(&self.pbucket[off..off + len]);
            }
        }
    }

    /// Overwrite the trainable weights in place, leaving Adam state and the step
    /// index untouched. This is the write-back counterpart to [`params`](Self::params):
    /// local-SGD uses it to fold periodically-averaged weights back in between
    /// steps (each rank keeps its own optimizer moments — standard local-SGD).
    /// The next [`step`](Self::step) pushes these into the session before forward.
    pub fn set_params(&mut self, params: &HashMap<String, Vec<f32>>) {
        for (slot, &(off, len)) in self.wrt.iter().zip(&self.layout) {
            if let Some(p) = params.get(&slot.name) {
                let n = len.min(p.len());
                self.pbucket[off..off + n].copy_from_slice(&p[..n]);
            }
        }
    }

    /// The next step index (0-based) — how many steps have completed.
    pub fn step_index(&self) -> usize {
        self.step
    }

    /// Steps not yet run toward `total_steps` (`total_steps - completed`).
    pub fn total_steps_remaining(&self) -> usize {
        self.total_steps.saturating_sub(self.step)
    }

    /// Capture a [`Checkpoint`]: current weights + full Adam state + step, in
    /// the fused bucket's `wrt` order. When sharded this **all-gathers** the
    /// optimizer moments, so it must be called by every rank in lockstep; the
    /// result is identical on all ranks (write it from one). World-size-agnostic
    /// — a run sharded over N ranks can [`restore`](Self::restore) on M.
    pub fn checkpoint(&self) -> Checkpoint {
        let (m_full, v_full) = if self.sharded {
            (
                self.gather_shard_to_full(&self.opt.m),
                self.gather_shard_to_full(&self.opt.v),
            )
        } else {
            (self.opt.m.clone(), self.opt.v.clone())
        };
        Checkpoint {
            step: self.step,
            t: self.opt.t,
            params: self
                .wrt
                .iter()
                .zip(&self.layout)
                .map(|(s, &(o, l))| (s.name.clone(), self.pbucket[o..o + l].to_vec()))
                .collect(),
            m: m_full[..self.real_len].to_vec(),
            v: v_full[..self.real_len].to_vec(),
        }
    }

    /// Restore weights, Adam state, and step index from a [`Checkpoint`]. All
    /// ranks must restore from the **same** checkpoint (typically loaded from a
    /// shared path). Re-shards the optimizer state into this run's layout, so a
    /// checkpoint saved under one world size resumes under another.
    pub fn restore(&mut self, ck: &Checkpoint) {
        self.step = ck.step;
        self.opt.t = ck.t;
        for (slot, &(off, len)) in self.wrt.iter().zip(&self.layout) {
            if let Some((_, vals)) = ck.params.iter().find(|(n, _)| n == &slot.name) {
                let n = len.min(vals.len());
                self.pbucket[off..off + n].copy_from_slice(&vals[..n]);
            }
        }
        let mut m_full = vec![0.0f32; self.padded_len];
        let mut v_full = vec![0.0f32; self.padded_len];
        let nm = self.real_len.min(ck.m.len());
        let nv = self.real_len.min(ck.v.len());
        m_full[..nm].copy_from_slice(&ck.m[..nm]);
        v_full[..nv].copy_from_slice(&ck.v[..nv]);
        if self.sharded {
            self.opt.m = self.extract_shard(&m_full);
            self.opt.v = self.extract_shard(&v_full);
        } else {
            self.opt.m = m_full;
            self.opt.v = v_full;
        }
    }

    /// Gather a per-rank shard (block-cyclic when `sc > 1`, else contiguous)
    /// into the full padded bucket. Collective; all ranks participate.
    fn gather_shard_to_full(&self, shard: &[f32]) -> Vec<f32> {
        let mut full = vec![0.0f32; self.padded_len];
        let c = match self.comm {
            Some(c) => c,
            None => {
                full[..shard.len()].copy_from_slice(shard);
                return full;
            }
        };
        if self.sc <= 1 {
            c.all_gather_into(shard, &mut full);
        } else {
            let chunk_e = self.padded_len / self.sc;
            let sub_e = chunk_e / self.world;
            for cc in 0..self.sc {
                let mut gathered = vec![0.0f32; chunk_e];
                c.all_gather_into(&shard[cc * sub_e..(cc + 1) * sub_e], &mut gathered);
                full[cc * chunk_e..(cc + 1) * chunk_e].copy_from_slice(&gathered);
            }
        }
        full
    }

    /// Extract this rank's shard (block-cyclic when `sc > 1`, else contiguous)
    /// from a full padded bucket. Local; the inverse of `gather_shard_to_full`.
    fn extract_shard(&self, full: &[f32]) -> Vec<f32> {
        if self.sc <= 1 {
            let base = self.rank * self.shard;
            full[base..base + self.shard].to_vec()
        } else {
            let chunk_e = self.padded_len / self.sc;
            let sub_e = chunk_e / self.world;
            let mut shard = vec![0.0f32; self.shard];
            for cc in 0..self.sc {
                let src = cc * chunk_e + self.rank * sub_e;
                shard[cc * sub_e..(cc + 1) * sub_e].copy_from_slice(&full[src..src + sub_e]);
            }
            shard
        }
    }
}

/// A training checkpoint: trainable weights + the full Adam state (moments +
/// timestep) + the step index, all in the fused bucket's `wrt` order.
/// **World-size-agnostic** — saved after gathering any sharded optimizer state,
/// so a run sharded across N ranks can resume on M. Save/load with
/// [`save`](Checkpoint::save) / [`load`](Checkpoint::load).
#[derive(Clone, Debug)]
pub struct Checkpoint {
    /// Steps completed (resume continues from here).
    pub step: usize,
    /// Adam timestep (for bias correction).
    pub t: i32,
    /// Trainable params by name.
    pub params: Vec<(String, Vec<f32>)>,
    /// Adam first moment, in `wrt`-bucket order.
    pub m: Vec<f32>,
    /// Adam second moment, in `wrt`-bucket order.
    pub v: Vec<f32>,
}

impl Checkpoint {
    /// Serialize to a compact self-describing binary file.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RLXCKPT1");
        buf.extend_from_slice(&(self.step as u64).to_le_bytes());
        buf.extend_from_slice(&self.t.to_le_bytes());
        buf.extend_from_slice(&(self.params.len() as u32).to_le_bytes());
        for (name, vals) in &self.params {
            buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            write_f32s(&mut buf, vals);
        }
        write_f32s(&mut buf, &self.m);
        write_f32s(&mut buf, &self.v);
        std::fs::write(path, buf).context("writing checkpoint")?;
        Ok(())
    }

    /// Load a checkpoint written by [`save`](Self::save).
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let buf = std::fs::read(path).context("reading checkpoint")?;
        let mut r = ByteReader::new(&buf);
        if r.take(8)? != b"RLXCKPT1" {
            bail!("not an RLX checkpoint (bad magic)");
        }
        let step = r.u64()? as usize;
        let t = r.i32()?;
        let np = r.u32()? as usize;
        let mut params = Vec::with_capacity(np);
        for _ in 0..np {
            let nl = r.u32()? as usize;
            let name = String::from_utf8(r.take(nl)?.to_vec()).context("checkpoint param name")?;
            params.push((name, r.f32s()?));
        }
        let m = r.f32s()?;
        let v = r.f32s()?;
        Ok(Checkpoint {
            step,
            t,
            params,
            m,
            v,
        })
    }
}

fn write_f32s(buf: &mut Vec<u8>, xs: &[f32]) {
    buf.extend_from_slice(&(xs.len() as u32).to_le_bytes());
    for &x in xs {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

/// Minimal little-endian byte reader for [`Checkpoint::load`].
struct ByteReader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.b.len() {
            bail!("checkpoint truncated");
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32s(&mut self) -> Result<Vec<f32>> {
        let n = self.u32()? as usize;
        let bytes = self.take(n * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

/// Overlap the chunked gradient all-reduce with the optimizer step: a
/// background thread reduces each chunk sequentially and hands it to this
/// thread, which applies Adam to the matching weight chunk while the next
/// chunk is still reducing. Numerically identical to a single fused reduce +
/// step (elementwise mean and elementwise Adam). Returns the collective time
/// **not** hidden behind the optimizer (ms) — the honest `comm_ms`.
fn overlapped_reduce_step(
    comm: &dyn GradComm,
    grad: &mut [f32],
    param: &mut [f32],
    opt: &mut FlatAdam,
    chunks: usize,
    dtype: ReduceDtype,
) -> f64 {
    use std::sync::mpsc;
    use std::time::Instant;

    let n = grad.len();
    let base = n / chunks;
    let rem = n % chunks;
    let bounds: Vec<(usize, usize)> = (0..chunks)
        .map(|i| {
            let s = i * base + i.min(rem);
            let e = (i + 1) * base + (i + 1).min(rem);
            (s, e)
        })
        .collect();

    let grad_ro: &[f32] = grad; // read-only view shared with the background thread
    let bounds_bg = bounds.clone();
    let mut wait_ms = 0.0f64;

    std::thread::scope(|s| {
        let (tx, rx) = mpsc::channel::<(usize, Vec<f32>)>();
        s.spawn(move || {
            for (i, &(a, b)) in bounds_bg.iter().enumerate() {
                let mut c = grad_ro[a..b].to_vec();
                comm.all_reduce_mean_typed(&mut c, dtype);
                if tx.send((i, c)).is_err() {
                    break;
                }
            }
        });
        for _ in 0..chunks {
            let t = Instant::now();
            let (i, reduced) = rx.recv().expect("overlap channel closed early");
            wait_ms += t.elapsed().as_secs_f64() * 1e3;
            let (a, _b) = bounds[i];
            opt.apply_range(a, &mut param[a..a + reduced.len()], &reduced);
        }
    });
    wait_ms
}

/// Overlapped ZeRO-1 step. The bucket is **block-cyclic** over `sc` chunks:
/// chunk `c` is `[c·chunkE, (c+1)·chunkE)` of the padded bucket, and within it
/// rank `r` owns the `r`-th `subE = chunkE/world` block — so each element still
/// has exactly one owner and the update matches single-process. A background
/// thread reduce-scatters each chunk's gradient while this thread runs the
/// optimizer on already-arrived blocks (hiding the reduce-scatter comm behind
/// the step); the all-gathers then reconstruct the full weights. `opt`'s shard
/// state is indexed `c·subE + j` for chunk `c`, offset `j`. Returns `comm_ms`.
fn sharded_overlap_step(
    comm: &dyn GradComm,
    grad: &[f32],
    pbucket: &mut [f32],
    opt: &mut FlatAdam,
    world: usize,
    rank: usize,
    sc: usize,
) -> f64 {
    use std::sync::mpsc;
    use std::time::Instant;

    let padded = grad.len();
    let chunk_e = padded / sc;
    let sub_e = chunk_e / world;

    // Extract this rank's owned block of each chunk into a contiguous shard.
    let mut shard_param = vec![0.0f32; sc * sub_e];
    for c in 0..sc {
        let src = c * chunk_e + rank * sub_e;
        shard_param[c * sub_e..(c + 1) * sub_e].copy_from_slice(&pbucket[src..src + sub_e]);
    }

    let grad_ro: &[f32] = grad;
    let mut comm_ms = 0.0f64;

    // Phase 1: reduce-scatter each chunk (bg comm) ∥ optimizer step (here).
    std::thread::scope(|s| {
        let (tx, rx) = mpsc::channel::<(usize, Vec<f32>)>();
        s.spawn(move || {
            for c in 0..sc {
                let mut sub = vec![0.0f32; sub_e];
                comm.reduce_scatter_mean(&grad_ro[c * chunk_e..(c + 1) * chunk_e], &mut sub);
                if tx.send((c, sub)).is_err() {
                    break;
                }
            }
        });
        for _ in 0..sc {
            let t = Instant::now();
            let (c, sub_grad) = rx.recv().expect("sharded overlap channel closed");
            comm_ms += t.elapsed().as_secs_f64() * 1e3;
            opt.apply_range(
                c * sub_e,
                &mut shard_param[c * sub_e..(c + 1) * sub_e],
                &sub_grad,
            );
        }
    });

    // Phase 2: all-gather each chunk's updated blocks → full weights. The
    // gather is rank-ordered, so chunk `c` reassembles contiguously.
    let t = Instant::now();
    for c in 0..sc {
        let mut gathered = vec![0.0f32; chunk_e];
        comm.all_gather_into(&shard_param[c * sub_e..(c + 1) * sub_e], &mut gathered);
        pbucket[c * chunk_e..(c + 1) * chunk_e].copy_from_slice(&gathered);
    }
    comm_ms += t.elapsed().as_secs_f64() * 1e3;
    comm_ms
}

/// Global-norm clip scale for an already-globally-reduced gradient with
/// sum-of-squares `sum_sq`: `1.0` unless `‖g‖ = √sum_sq` exceeds `max`, in which
/// case `max / ‖g‖` (so the clipped norm is exactly `max`).
fn clip_scale(sum_sq: f32, max: f32) -> f32 {
    let norm = sum_sq.max(0.0).sqrt();
    if norm > max && norm > 0.0 {
        max / norm
    } else {
        1.0
    }
}

/// Effective learning rate at `step` (0-based) of `total`: a linear ramp over
/// the first `warmup` steps, then `sched` decay from `base` down to
/// `min_ratio * base` across the remaining steps.
fn effective_lr(
    base: f32,
    step: usize,
    total: usize,
    warmup: usize,
    sched: LrSchedule,
    min_ratio: f32,
) -> f32 {
    if warmup > 0 && step < warmup {
        return base * (step as f32 + 1.0) / warmup as f32;
    }
    if sched == LrSchedule::Constant {
        return base;
    }
    let min_lr = base * min_ratio;
    let decay_total = total.saturating_sub(warmup).max(1);
    let t = ((step.saturating_sub(warmup)) as f32 / decay_total as f32).clamp(0.0, 1.0);
    let factor = match sched {
        LrSchedule::Linear => 1.0 - t,
        LrSchedule::Cosine => 0.5 * (1.0 + (std::f32::consts::PI * t).cos()),
        LrSchedule::Constant => 1.0,
    };
    min_lr + (base - min_lr) * factor
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

    #[test]
    fn effective_lr_warmup_and_decay() {
        let base = 0.1;
        // Linear warmup over 4 steps: lr = base*(step+1)/4.
        assert!(
            (effective_lr(base, 0, 100, 4, LrSchedule::Cosine, 0.0) - base * 0.25).abs() < 1e-6
        );
        assert!((effective_lr(base, 3, 100, 4, LrSchedule::Cosine, 0.0) - base).abs() < 1e-6);
        // Constant holds base after warmup.
        assert!((effective_lr(base, 50, 100, 4, LrSchedule::Constant, 0.0) - base).abs() < 1e-6);
        // Cosine: starts at base right after warmup, ends at the floor.
        let start = effective_lr(base, 4, 100, 4, LrSchedule::Cosine, 0.1);
        let end = effective_lr(base, 99, 100, 4, LrSchedule::Cosine, 0.1);
        assert!((start - base).abs() < 1e-3, "cosine start {start}");
        assert!((end - base * 0.1).abs() < 1e-3, "cosine end {end}");
        // Linear decay is monotonically non-increasing post-warmup.
        let a = effective_lr(base, 10, 100, 0, LrSchedule::Linear, 0.0);
        let b = effective_lr(base, 60, 100, 0, LrSchedule::Linear, 0.0);
        assert!(a > b, "linear decay should decrease: {a} !> {b}");
    }

    #[test]
    fn clip_scale_caps_the_norm() {
        // ‖g‖ = 5 clipped to 1 → scale 0.2; below the cap → no change.
        assert!((clip_scale(25.0, 1.0) - 0.2).abs() < 1e-6);
        assert_eq!(clip_scale(0.25, 1.0), 1.0); // ‖g‖=0.5 < 1
        assert_eq!(clip_scale(0.0, 1.0), 1.0);
    }

    #[test]
    fn dp_config_builder_sets_fields() {
        let cfg = DpConfig::new(2e-4)
            .shard()
            .overlap()
            .chunks(8)
            .bf16()
            .clip(1.0)
            .warmup(100)
            .cosine(0.1)
            .grad_accum(4)
            .log_every(50);
        assert_eq!(cfg.adam.lr, 2e-4);
        assert!(cfg.shard_optimizer && cfg.overlap);
        assert_eq!(cfg.chunks, 8);
        assert_eq!(cfg.reduce_dtype, ReduceDtype::Bf16);
        assert_eq!(cfg.max_grad_norm, Some(1.0));
        assert_eq!(cfg.warmup_steps, 100);
        assert_eq!(cfg.lr_schedule, LrSchedule::Cosine);
        assert!((cfg.min_lr_ratio - 0.1).abs() < 1e-6);
        assert_eq!(cfg.grad_accum, 4);
        assert_eq!(cfg.log_every, 50);
        let d = cfg.describe();
        assert!(
            d.contains("shard") && d.contains("bf16") && d.contains("cosine"),
            "{d}"
        );
        // AdamConfig builder too.
        let a = AdamConfig::new(1e-3).betas(0.8, 0.95).eps(1e-6);
        assert_eq!((a.beta1, a.beta2, a.eps), (0.8, 0.95, 1e-6));
    }

    #[test]
    fn step_metrics_display() {
        let m = StepMetrics {
            step: 3,
            loss: 0.5,
            lr: 0.01,
            compute_ms: 8.0,
            comm_ms: 2.0,
            step_ms: 10.0,
            reduced_elems: 100,
            world_size: 4,
        };
        let s = format!("{m}");
        assert!(s.contains("loss 0.500000") && s.contains("world 4"), "{s}");
        assert!(s.contains("80% compute"), "{s}"); // 8/(8+2)
    }
}
