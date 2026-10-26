// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Streamed training: one decoder layer resident at a time, exact backprop.
//!
//! The forward walks the layers, keeping only each layer's *input* (and which
//! experts each MoE layer picked); the backward walks them in reverse, rebuilding
//! each layer from its stashed input and taking its vector–Jacobian product
//! ([`crate::layer_exec::backward_layer`]). That is activation checkpointing at
//! layer granularity: peak memory is one layer's weights and activations plus
//! one `[batch, seq, hidden]` vector per layer, so a model far larger than RAM
//! trains, and the gradient is the same one the whole-graph trainer computes.
//!
//! [`Stage`] owns a contiguous layer range with its trainable values and
//! optimizer state — the unit a pipeline-parallel worker runs ([`crate::dist`]).
//! [`StreamTrainer`] is the single-process driver: embedding, every layer as one
//! stage, and the head, optionally data-parallel over a [`rlx_tune::GradComm`].

use crate::flow::{FlowConfig, LayerWeights};
use crate::layer_exec::{
    LayerRouting, LayerSpec, backward_layer, forward_layer_weights, head_loss_backward,
};
use crate::source::{HeadWeights, WeightSource};
use crate::train::{AdamW, Batch, TrainSpec, global_norm};
use anyhow::{Context, Result, ensure};
use rlx_runtime::Device;
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

pub type Params = HashMap<String, Vec<f32>>;

/// Hyperparameters of a streamed run.
#[derive(Debug, Clone)]
pub struct StreamOptions {
    pub lr: f32,
    pub weight_decay: f32,
    /// Clip the global gradient norm to this (over every stage and the head).
    pub clip: Option<f32>,
    pub device: Device,
    /// LoRA adapter init seed (must match across stages / ranks).
    pub seed: u64,
    /// Keep every layer's weights in RAM after first load instead of re-reading
    /// them for the backward and the next step. Only for ranges that fit.
    pub cache_layers: bool,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            weight_decay: 0.0,
            clip: Some(1.0),
            device: Device::Cpu,
            seed: 0,
            cache_layers: false,
        }
    }
}

/// What the forward kept for the backward.
struct Stash {
    /// Input `h` of each layer in the range.
    h: Vec<Vec<f32>>,
    /// Snapshot count entering each layer (its snapshots are `snaps[..m]`).
    m: Vec<usize>,
    routing: Vec<Option<LayerRouting>>,
    /// The snapshot list as it grew through the range (prefix-shared by every
    /// layer, so it is stored once).
    snaps: Vec<Vec<f32>>,
}

/// A contiguous range of layers with its trainable values and optimizer state.
pub struct Stage {
    pub src: Box<dyn WeightSource + Send>,
    pub cfg: FlowConfig,
    pub spec: TrainSpec,
    pub range: Range<usize>,
    pub opts: StreamOptions,
    /// Trained values by canonical name (weights touched so far, or adapters).
    pub store: Params,
    pub opt: AdamW,
    /// Gradients accumulated since the last [`Stage::apply`].
    pub grads: Params,
    /// Micro-batches accumulated into `grads`.
    pub accumulated: usize,
    /// Graph shapes of every param that has had a gradient (for export).
    pub shapes: HashMap<String, Vec<usize>>,
    /// Starting values of params first seen in a backward (full mode).
    values: Params,
    /// One stash per in-flight micro-batch (pipelining keeps several).
    stashes: HashMap<u64, Stash>,
    cache: HashMap<usize, LayerWeights>,
    /// Compiled graphs reused across steps when `opts.cache_layers` is set.
    graphs: std::cell::RefCell<crate::layer_exec::GraphCache>,
}

impl Stage {
    pub fn new(
        src: Box<dyn WeightSource + Send>,
        cfg: FlowConfig,
        spec: TrainSpec,
        range: Range<usize>,
        opts: StreamOptions,
    ) -> Self {
        let opt = AdamW::new(opts.lr, opts.weight_decay);
        Self {
            src,
            cfg,
            spec,
            range,
            opts,
            store: Params::new(),
            opt,
            grads: Params::new(),
            accumulated: 0,
            shapes: HashMap::new(),
            values: Params::new(),
            stashes: HashMap::new(),
            cache: HashMap::new(),
            graphs: Default::default(),
        }
    }

    fn layer(&mut self, i: usize) -> Result<LayerWeights> {
        if let Some(lw) = self.cache.remove(&i) {
            return Ok(lw);
        }
        self.src.layer(i, &self.cfg)
    }

    fn give_back(&mut self, i: usize, lw: LayerWeights) {
        if self.opts.cache_layers {
            self.cache.insert(i, lw);
        }
    }

    /// Run the range forward on `(h, snaps)`, stashing what the backward needs.
    pub fn forward(
        &mut self,
        h: Vec<f32>,
        snaps: Vec<Vec<f32>>,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        self.forward_mb(0, h, snaps)
    }

    /// [`Self::forward`] for micro-batch `mb`; several may be in flight, each
    /// backed by its own stash until its [`Self::backward_mb`].
    pub fn forward_mb(
        &mut self,
        mb: u64,
        h: Vec<f32>,
        snaps: Vec<Vec<f32>>,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let mut stash = Stash {
            h: Vec::with_capacity(self.range.len()),
            m: Vec::with_capacity(self.range.len()),
            routing: Vec::with_capacity(self.range.len()),
            snaps,
        };
        let mut h = h;
        let store = std::mem::take(&mut self.store);
        let res = (|| -> Result<()> {
            for i in self.range.clone() {
                let lw = self.layer(i)?;
                let m = stash.snaps.len();
                let spec = layer_spec(&self.cfg, &self.opts, &self.spec, &store, &self.graphs);
                let out =
                    forward_layer_weights(&mut *self.src, i, &lw, &h, &stash.snaps, &spec, false)?;
                self.give_back(i, lw);
                stash.m.push(m);
                stash.routing.push(out.routing);
                stash.h.push(std::mem::replace(&mut h, out.h_out));
                stash.snaps = out.snaps_out;
            }
            Ok(())
        })();
        self.store = store;
        res?;
        let snaps = stash.snaps.clone();
        self.stashes.insert(mb, stash);
        Ok((h, snaps))
    }

    /// Backward through the range from `ct_h` / `ct_snaps` (cotangents of the
    /// range's outputs). Accumulates param gradients into [`Self::grads`] and
    /// returns the cotangents of the range's inputs.
    pub fn backward(
        &mut self,
        ct_h: Vec<f32>,
        ct_snaps: Vec<Vec<f32>>,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        self.backward_mb(0, ct_h, ct_snaps)
    }

    /// [`Self::backward`] for micro-batch `mb`.
    pub fn backward_mb(
        &mut self,
        mb: u64,
        ct_h: Vec<f32>,
        ct_snaps: Vec<Vec<f32>>,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let stash = self
            .stashes
            .remove(&mb)
            .with_context(|| format!("backward of micro-batch {mb} without its forward"))?;
        ensure!(
            ct_snaps.len() == stash.snaps.len(),
            "{} snapshot cotangents for {} snapshots",
            ct_snaps.len(),
            stash.snaps.len()
        );
        let mut ct_h = ct_h;
        let mut ct_snaps = ct_snaps;
        let store = std::mem::take(&mut self.store);
        let spec_train = self.spec.clone();
        let res = (|| -> Result<()> {
            for (li, i) in self.range.clone().enumerate().rev() {
                let lw = self.layer(i)?;
                let m = stash.m[li];
                let m_out = m + usize::from(crate::layer_exec::is_boundary(i, &self.cfg));
                let spec = layer_spec(&self.cfg, &self.opts, &self.spec, &store, &self.graphs);
                let wrt = |n: &str| spec_train.is_trainable(n);
                let b = backward_layer(
                    &mut *self.src,
                    i,
                    &lw,
                    &stash.h[li],
                    &stash.snaps[..m],
                    stash.routing[li].as_ref(),
                    &ct_h,
                    &ct_snaps[..m_out],
                    &wrt,
                    &spec,
                )?;
                self.give_back(i, lw);
                ct_h = b.d_h;
                ct_snaps.truncate(m);
                for (dst, d) in ct_snaps.iter_mut().zip(b.d_snaps) {
                    *dst = d;
                }
                for (name, g) in b.grads {
                    accumulate(&mut self.grads, name, g);
                }
                for (name, v) in b.values {
                    self.values.entry(name).or_insert(v);
                }
                self.shapes.extend(b.shapes);
            }
            Ok(())
        })();
        self.store = store;
        res?;
        self.accumulated += 1;
        Ok((ct_h, ct_snaps))
    }

    /// Sum of squared gradient entries (for a global-norm clip across stages).
    pub fn grad_sq_norm(&self) -> f64 {
        let n = global_norm(self.grads.values()) as f64;
        n * n
    }

    /// Apply the accumulated gradients scaled by `scale` (e.g. `1/accum ·
    /// clip_factor`), then clear them. Params first touched here start from the
    /// value the backward saw.
    pub fn apply(&mut self, scale: f32) {
        apply_grads(
            &mut self.store,
            &mut self.values,
            &mut self.opt,
            std::mem::take(&mut self.grads),
            scale,
        );
        self.accumulated = 0;
    }

    /// Drop accumulated gradients without stepping.
    pub fn zero_grads(&mut self) {
        self.grads.clear();
        self.accumulated = 0;
    }
}

fn layer_spec<'a>(
    cfg: &'a FlowConfig,
    opts: &StreamOptions,
    spec: &'a TrainSpec,
    store: &'a Params,
    graphs: &'a std::cell::RefCell<crate::layer_exec::GraphCache>,
) -> LayerSpec<'a> {
    LayerSpec {
        cfg,
        device: opts.device,
        lora: spec.lora_spec(),
        overrides: store,
        seed: opts.seed,
        cache: opts.cache_layers.then_some(graphs),
    }
}

pub(crate) fn accumulate(grads: &mut Params, name: String, g: Vec<f32>) {
    match grads.get_mut(&name) {
        Some(acc) => crate::layer_exec::add_into(acc, &g),
        None => {
            grads.insert(name, g);
        }
    }
}

fn apply_grads(
    store: &mut Params,
    values: &mut Params,
    opt: &mut AdamW,
    grads: Params,
    scale: f32,
) {
    let mut names: Vec<String> = grads.keys().cloned().collect();
    names.sort();
    let mut grads = grads;
    for name in names {
        let mut g = grads.remove(&name).unwrap_or_default();
        if scale != 1.0 {
            g.iter_mut().for_each(|x| *x *= scale);
        }
        if !store.contains_key(&name) {
            let Some(v) = values.remove(&name) else {
                continue;
            };
            store.insert(name.clone(), v);
        }
        let p = store.get_mut(&name).expect("inserted above");
        opt.step(&name, p, &g);
    }
    opt.end_step();
}

/// Loss and gradient norm of one streamed step.
#[derive(Debug, Clone, Copy)]
pub struct StreamStep {
    pub step: usize,
    pub loss: f32,
    pub grad_norm: f32,
    pub ms: f64,
}

impl std::fmt::Display for StreamStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "step {:>5} | loss {:.6} | grad-norm {:.4} | {:.0} ms",
            self.step, self.loss, self.grad_norm, self.ms
        )
    }
}

/// Single-process streamed trainer: embedding → every layer (one [`Stage`]) →
/// head → loss, then the whole backward, then one optimizer step. Data-parallel
/// across processes when given a `comm`: each rank streams its own batch and the
/// gradients are mean-reduced before the step.
pub struct StreamTrainer {
    pub stage: Stage,
    head: Option<HeadWeights>,
    /// Head trained values / optimizer live with the trainer, not the stage.
    head_values: Params,
    comm: Option<Box<dyn rlx_tune::GradComm>>,
    pub step: usize,
}

impl StreamTrainer {
    pub fn new(
        src: Box<dyn WeightSource + Send>,
        cfg: FlowConfig,
        spec: TrainSpec,
        opts: StreamOptions,
        layers: Option<usize>,
    ) -> Self {
        let n = layers.unwrap_or(src.num_layers()).min(src.num_layers());
        Self {
            stage: Stage::new(src, cfg, spec, 0..n, opts),
            head: None,
            head_values: Params::new(),
            comm: None,
            step: 0,
        }
    }

    /// Reduce gradients across `comm`'s ranks before every step.
    pub fn with_comm(mut self, comm: Box<dyn rlx_tune::GradComm>) -> Self {
        self.comm = Some(comm);
        self
    }

    fn head(&mut self) -> Result<HeadWeights> {
        if let Some(h) = &self.head {
            return Ok(h.clone());
        }
        let h = self.stage.src.head(&self.stage.cfg)?;
        if self.stage.opts.cache_layers {
            self.head = Some(h.clone());
        }
        Ok(h)
    }

    /// Forward + backward on one batch; gradients accumulate on the stage.
    /// Returns the loss.
    pub fn accumulate(&mut self, batch: &Batch) -> Result<f32> {
        let cfg = self.stage.cfg.clone();
        ensure!(
            batch.tokens.len() == cfg.batch * cfg.seq,
            "batch has {} tokens, the config wants batch·seq = {}",
            batch.tokens.len(),
            cfg.batch * cfg.seq
        );
        let embed_name = "embed_tokens";
        let h0 = match self.stage.store.get(embed_name) {
            Some(table) => gather_rows(table, &batch.tokens, cfg.hidden)?,
            None => self.stage.src.embed(&batch.tokens, cfg.hidden)?,
        };
        let (h, snaps) = self.stage.forward(h0, Vec::new())?;
        let head = self.head()?;
        let spec_train = self.stage.spec.clone();
        let wrt = |n: &str| spec_train.is_trainable(n);
        let hb = {
            let st = &self.stage;
            let spec = layer_spec(&st.cfg, &st.opts, &st.spec, &st.store, &st.graphs);
            head_loss_backward(&head, &h, &snaps, &batch.labels, &batch.mask, &wrt, &spec)?
        };
        for (name, g) in hb.grads {
            accumulate(&mut self.stage.grads, name, g);
        }
        for (name, v) in hb.values {
            self.head_values.entry(name).or_insert(v);
        }
        self.stage.shapes.extend(hb.shapes);
        let (d_h0, _) = self.stage.backward(hb.d_h, hb.d_snaps)?;
        if self.stage.spec.is_trainable(embed_name) {
            let table = match self.stage.store.get(embed_name) {
                Some(t) => t.clone(),
                None => self.stage.src.embed_table()?,
            };
            let mut g = vec![0f32; table.len()];
            for (r, &t) in batch.tokens.iter().enumerate() {
                let (dst, src) = (t as usize * cfg.hidden, r * cfg.hidden);
                crate::layer_exec::add_into(
                    &mut g[dst..dst + cfg.hidden],
                    &d_h0[src..src + cfg.hidden],
                );
            }
            accumulate(&mut self.stage.grads, embed_name.into(), g);
            self.stage
                .shapes
                .insert(embed_name.into(), vec![cfg.vocab, cfg.hidden]);
            self.head_values.entry(embed_name.into()).or_insert(table);
        }
        Ok(hb.loss)
    }

    /// Average, all-reduce (data-parallel), clip and apply what
    /// [`Self::accumulate`] collected. Returns the pre-clip global norm.
    pub fn apply(&mut self) -> Result<f32> {
        let n = self.stage.accumulated.max(1) as f32;
        if let Some(comm) = &self.comm {
            let mut names: Vec<String> = self.stage.grads.keys().cloned().collect();
            names.sort();
            let mut flat: Vec<f32> = names
                .iter()
                .flat_map(|k| self.stage.grads[k].iter().copied())
                .collect();
            comm.all_reduce_mean(&mut flat);
            let mut off = 0;
            for k in names {
                let g = self.stage.grads.get_mut(&k).expect("listed");
                let len = g.len();
                g.copy_from_slice(&flat[off..off + len]);
                off += len;
            }
        }
        let norm = global_norm(self.stage.grads.values()) / n;
        let clip = match self.stage.opts.clip {
            Some(c) if norm > c && norm.is_finite() => c / norm,
            _ => 1.0,
        };
        ensure!(norm.is_finite(), "non-finite gradient norm {norm}");
        for (k, v) in std::mem::take(&mut self.head_values) {
            self.stage.values.entry(k).or_insert(v);
        }
        self.stage.apply(clip / n);
        self.step += 1;
        Ok(norm)
    }

    /// One optimizer step over `micro` (gradient accumulation when longer than one).
    pub fn train_step(&mut self, micro: &[Batch]) -> Result<StreamStep> {
        let t = std::time::Instant::now();
        let mut loss = 0.0;
        for b in micro {
            loss += self.accumulate(b)?;
        }
        let grad_norm = self.apply()?;
        Ok(StreamStep {
            step: self.step,
            loss: loss / micro.len().max(1) as f32,
            grad_norm,
            ms: t.elapsed().as_secs_f64() * 1e3,
        })
    }

    /// Forward + backward without stepping: `(loss, gradients)`, gradients
    /// un-averaged. For gradient checks.
    pub fn loss_and_grads(&mut self, batch: &Batch) -> Result<(f32, Params)> {
        self.stage.zero_grads();
        let loss = self.accumulate(batch)?;
        let g = std::mem::take(&mut self.stage.grads);
        self.stage.accumulated = 0;
        Ok((loss, g))
    }

    /// Trained values so far (weights touched, or adapters).
    pub fn trained(&self) -> &Params {
        &self.stage.store
    }

    /// Save trained values + optimizer state to `dir` (`trained.safetensors`,
    /// `optimizer.safetensors`, `state.json`) — enough to resume.
    pub fn save_state(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        save_params(&dir.join("trained.safetensors"), &self.stage.store)?;
        self.stage.opt.save(&dir.join("optimizer.safetensors"))?;
        std::fs::write(
            dir.join("state.json"),
            serde_json::to_vec_pretty(&serde_json::json!({ "step": self.step }))?,
        )?;
        Ok(())
    }

    /// Resume from [`Self::save_state`].
    pub fn load_state(&mut self, dir: &Path) -> Result<()> {
        self.stage.store = load_params(&dir.join("trained.safetensors"))?;
        self.stage.opt.load(&dir.join("optimizer.safetensors"))?;
        let st: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("state.json"))?)?;
        self.step = st["step"].as_u64().unwrap_or(0) as usize;
        Ok(())
    }
}

fn gather_rows(table: &[f32], tokens: &[u32], hidden: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(tokens.len() * hidden);
    for &t in tokens {
        let r = t as usize * hidden;
        ensure!(r + hidden <= table.len(), "token {t} out of range");
        out.extend_from_slice(&table[r..r + hidden]);
    }
    Ok(out)
}

/// Write named f32 vectors (flat) to a safetensors file.
pub fn save_params(path: &Path, p: &Params) -> Result<()> {
    use safetensors::{Dtype, tensor::TensorView};
    let bufs: Vec<(String, Vec<u8>, usize)> = p
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                v.len(),
            )
        })
        .collect();
    let views: Vec<(String, TensorView<'_>)> = bufs
        .iter()
        .map(|(n, b, len)| {
            TensorView::new(Dtype::F32, vec![*len], b)
                .map(|v| (n.clone(), v))
                .map_err(|e| anyhow::anyhow!("{n}: {e:?}"))
        })
        .collect::<Result<_>>()?;
    safetensors::serialize_to_file(views, None, path)
        .map_err(|e| anyhow::anyhow!("writing {path:?}: {e:?}"))
}

/// Read what [`save_params`] wrote.
pub fn load_params(path: &Path) -> Result<Params> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {path:?}"))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| anyhow::anyhow!("{path:?}: {e:?}"))?;
    Ok(st
        .tensors()
        .into_iter()
        .map(|(n, v)| {
            (
                n,
                v.data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        })
        .collect())
}
