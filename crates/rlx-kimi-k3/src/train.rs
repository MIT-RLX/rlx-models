// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Training Kimi-K3: next-token cross-entropy, full-weight or LoRA.
//!
//! Two drivers share the pieces in this module:
//!
//! * **Whole graph** ([`build_train_graph`] + [`WholeTrainer`]) — the full decoder
//!   as one differentiable graph (tokens → embed → every layer → head → loss),
//!   trained by [`rlx_tune::Trainer`]. Data-parallel across processes comes with
//!   it (gradient all-reduce, ZeRO-1, clipping, LR schedules) through a
//!   [`rlx_tune::GradComm`]. For any config whose weights and activations fit:
//!   synthetic, truncated, or a cluster node's share.
//! * **Streamed** ([`crate::train_stream::StreamTrainer`]) — one layer resident at
//!   a time with exact layer-by-layer backprop, routed experts paged per batch.
//!   The path for the real checkpoint, and the per-stage body of pipeline
//!   training ([`crate::dist`]).
//!
//! What trains is a [`TrainSpec`]: every checkpoint weight except the routed
//! experts (full), or LoRA adapters on chosen linear modules. Graph params are
//! canonical ([`crate::common::CanonicalParams`]), so trained values export
//! straight back to checkpoint tensors ([`crate::export`]).

use crate::common::CanonicalParams;
use crate::export::{hf_name, trainable_by_default};
use crate::flow::{FlowConfig, FlowWeights, build_kimi_text_flow};
use anyhow::{Context, Result, bail, ensure};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Graph, HirGraphExt, Op, Shape};
use rlx_tune::{AdapterParam, LoraSpec, ParamSlot};
use std::collections::HashMap;
use std::path::Path;

pub type Params = HashMap<String, Vec<f32>>;

/// Linear modules LoRA targets when none are named: every attention and FFN
/// projection that is not a routed expert. Substrings of canonical param names.
pub const DEFAULT_LORA_TARGETS: &[&str] = &[
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.g_proj",
    "self_attn.o_proj",
    "self_attn.q_a_proj",
    "self_attn.q_b_proj",
    "self_attn.kv_a_proj_with_mqa",
    "self_attn.kv_b_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "shared_experts.gate_proj",
    "shared_experts.up_proj",
    "shared_experts.down_proj",
];

/// Full-weight or adapter training.
#[derive(Debug, Clone)]
pub enum TrainMode {
    /// Update the checkpoint weights themselves.
    Full,
    /// Freeze them and train LoRA adapters.
    Lora(LoraSpec),
}

/// Which params train.
#[derive(Debug, Clone)]
pub struct TrainSpec {
    pub mode: TrainMode,
    /// Full mode: also train the `[vocab, hidden]` embedding table.
    pub train_embed: bool,
    /// Full mode: also train routed-expert banks (whole-graph only — the
    /// streamed path pages experts and never holds a bank to update).
    pub train_experts: bool,
    /// Substrings; a matching param never trains.
    pub freeze: Vec<String>,
    /// Substrings; when non-empty only matching params train (e.g. `l10.`).
    pub only: Vec<String>,
}

impl TrainSpec {
    pub fn full() -> Self {
        Self {
            mode: TrainMode::Full,
            train_embed: false,
            train_experts: false,
            freeze: Vec::new(),
            only: Vec::new(),
        }
    }

    pub fn lora(rank: usize, alpha: f32, targets: &[String]) -> Self {
        let targets = if targets.is_empty() {
            DEFAULT_LORA_TARGETS.iter().map(|s| s.to_string()).collect()
        } else {
            targets.to_vec()
        };
        Self {
            mode: TrainMode::Lora(LoraSpec::new(rank, alpha, targets)),
            ..Self::full()
        }
    }

    pub fn lora_spec(&self) -> Option<&LoraSpec> {
        match &self.mode {
            TrainMode::Lora(s) => Some(s),
            TrainMode::Full => None,
        }
    }

    /// Does canonical graph param `name` train under this spec?
    pub fn is_trainable(&self, name: &str) -> bool {
        if self.freeze.iter().any(|f| name.contains(f.as_str())) {
            return false;
        }
        if !self.only.is_empty() && !self.only.iter().any(|f| name.contains(f.as_str())) {
            return false;
        }
        match &self.mode {
            TrainMode::Lora(_) => name.starts_with("lora."),
            TrainMode::Full => {
                if name == "embed_tokens" {
                    return self.train_embed;
                }
                if name.ends_with("experts.gate_up") || name.ends_with("experts.down") {
                    return self.train_experts;
                }
                trainable_by_default(name)
            }
        }
    }
}

/// Deterministic LoRA init (PEFT's): `A ~ U(±1/√in)`, `B = 0` — the adapted
/// model starts equal to the base one. Seeded per param name, so every rank and
/// every pipeline stage initialises the same adapter identically.
pub fn init_adapter(name: &str, shape: &[usize], seed: u64) -> Vec<f32> {
    let n: usize = shape.iter().product();
    if name.ends_with(".b") {
        return vec![0.0; n];
    }
    let fan_in = shape.first().copied().unwrap_or(1).max(1);
    let bound = 1.0 / (fan_in as f32).sqrt();
    let mut s = seed
        ^ name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
        });
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * bound
        })
        .collect()
}

/// Add values for any adapter in `adapters` that `store` does not hold yet.
pub fn ensure_adapters(store: &mut Params, adapters: &[AdapterParam], seed: u64) {
    for a in adapters {
        store
            .entry(a.name.clone())
            .or_insert_with(|| init_adapter(&a.name, &a.shape, seed));
    }
}

/// One training batch: `tokens`/`labels`/`mask`, each `[batch·seq]`.
#[derive(Debug, Clone)]
pub struct Batch {
    pub tokens: Vec<u32>,
    pub labels: Vec<f32>,
    pub mask: Vec<f32>,
}

/// Random `(seq + 1)`-token windows from a corpus of token sequences.
pub struct Batcher {
    seqs: Vec<Vec<u32>>,
    state: u64,
}

impl Batcher {
    /// Sequences shorter than `seq + 1` tokens are dropped.
    pub fn new(seqs: Vec<Vec<u32>>, seq: usize, seed: u64) -> Result<Self> {
        let seqs: Vec<Vec<u32>> = seqs.into_iter().filter(|s| s.len() > seq).collect();
        ensure!(!seqs.is_empty(), "no sequence is longer than {seq} tokens");
        Ok(Self {
            seqs,
            state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
        })
    }

    fn next_index(&mut self, n: usize) -> usize {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 33) as usize) % n.max(1)
    }

    pub fn next(&mut self, batch: usize, seq: usize) -> Batch {
        let mut tokens = Vec::with_capacity(batch * seq);
        let mut labels = Vec::with_capacity(batch * seq);
        for _ in 0..batch {
            let si = self.next_index(self.seqs.len());
            let len = self.seqs[si].len();
            let start = self.next_index(len - seq);
            let s = &self.seqs[si];
            tokens.extend_from_slice(&s[start..start + seq]);
            labels.extend(s[start + 1..start + seq + 1].iter().map(|&t| t as f32));
        }
        Batch {
            tokens,
            labels,
            mask: vec![1.0; batch * seq],
        }
    }
}

/// Read a corpus: a token-id file (one sequence per line, ids separated by
/// commas or whitespace) when `tokenizer` is `None`, else text tokenized with
/// that `tokenizer.json` (one document per blank-line-separated paragraph).
pub fn load_corpus(path: &Path, tokenizer: Option<&Path>) -> Result<Vec<Vec<u32>>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path:?}"))?;
    match tokenizer {
        None => text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                l.split(|c: char| c == ',' || c.is_whitespace())
                    .filter(|t| !t.is_empty())
                    .map(|t| {
                        t.parse::<u32>()
                            .with_context(|| format!("{t:?} is not a token id"))
                    })
                    .collect()
            })
            .collect(),
        Some(tk) => {
            let tok = tokenizers::Tokenizer::from_file(tk)
                .map_err(|e| anyhow::anyhow!("loading {tk:?}: {e}"))?;
            text.split("\n\n")
                .filter(|p| !p.trim().is_empty())
                .map(|p| {
                    tok.encode(p, false)
                        .map(|e| e.get_ids().to_vec())
                        .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))
                })
                .collect()
        }
    }
}

/// The whole decoder as one loss graph.
pub struct TrainGraph {
    /// Inputs `tokens`, `labels`, `label_mask` (all `[batch·seq]` f32); output
    /// the masked mean next-token loss.
    pub graph: Graph,
    /// Every param's value (adapters included).
    pub params: Params,
    pub shapes: HashMap<String, Vec<usize>>,
    /// The params that train, in a stable order.
    pub slots: Vec<ParamSlot>,
}

/// Build the whole-model loss graph for `spec`. `overrides` replaces param
/// values by canonical name (resumed weights, a loaded adapter); missing LoRA
/// adapters are initialised with `seed`.
pub fn build_train_graph(
    w: &FlowWeights,
    embed: &[f32],
    cfg: &FlowConfig,
    spec: &TrainSpec,
    overrides: &Params,
    seed: u64,
) -> Result<TrainGraph> {
    let (b, s, hidden) = (cfg.batch, cfg.seq, cfg.hidden);
    ensure!(
        embed.len() == cfg.vocab * hidden,
        "embedding table has {} values, expected vocab·hidden = {}",
        embed.len(),
        cfg.vocab * hidden
    );
    let rows = b * s;
    let (hir, params) = {
        let _canon = CanonicalParams::new();
        let mut hir = HirModule::new("kimi_train");
        let mut g = HirMut::new(&mut hir);
        let mut params = Params::new();
        let ids = g.input("tokens", Shape::new(&[rows], DType::F32));
        params.insert("embed_tokens".into(), embed.to_vec());
        let table = g.param("embed_tokens", Shape::new(&[cfg.vocab, hidden], DType::F32));
        let e = g.gather_(table, ids, 0);
        let h = g.reshape_(e, vec![b as i64, s as i64, hidden as i64]);
        let logits = build_kimi_text_flow(&mut g, &mut params, h, w, cfg)?;
        g.set_outputs(vec![logits]);
        (hir, params)
    };
    let (mut graph, mut params) = rlx_core::flow_util::graph_from_hir(hir, params)?;
    if let Some(lora) = spec.lora_spec() {
        let (g2, adapters) = rlx_tune::inject_lora(&graph, lora, rlx_tune::FuseMode::Unfused);
        ensure!(
            !adapters.is_empty(),
            "LoRA targets {:?} match no linear weight",
            lora.target_modules
        );
        graph = g2;
        let mut store: Params = overrides
            .iter()
            .filter(|(k, _)| k.starts_with("lora."))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        ensure_adapters(&mut store, &adapters, seed);
        for a in &adapters {
            params.insert(a.name.clone(), store[&a.name].clone());
        }
    }
    let logits = graph.outputs[0];
    let flat = graph.add_node(
        Op::Reshape {
            new_shape: vec![rows as i64, cfg.vocab as i64],
        },
        vec![logits],
        Shape::new(&[rows, cfg.vocab], DType::F32),
    );
    let labels = graph.input("labels", Shape::new(&[rows], DType::F32));
    let mask = graph.input("label_mask", Shape::new(&[rows], DType::F32));
    let loss = rlx_tune::cross_entropy_masked(&mut graph, flat, labels, mask);
    graph.set_outputs(vec![loss]);

    let mut shapes = HashMap::new();
    let mut slots = Vec::new();
    for n in graph.nodes() {
        if let Op::Param { name } = &n.op {
            let dims: Vec<usize> = n.shape.dims().iter().map(|d| d.unwrap_static()).collect();
            shapes.insert(name.clone(), dims);
            if spec.is_trainable(name) {
                slots.push(ParamSlot {
                    name: name.clone(),
                    node: n.id,
                });
            }
        }
    }
    for (k, v) in overrides {
        if !k.starts_with("lora.") && params.contains_key(k) {
            params.insert(k.clone(), v.clone());
        }
    }
    slots.sort_by(|a, b| a.name.cmp(&b.name));
    ensure!(
        !slots.is_empty(),
        "nothing to train: the spec selects no param"
    );
    Ok(TrainGraph {
        graph,
        params,
        shapes,
        slots,
    })
}

/// Inputs for one batch of a [`TrainGraph`].
pub fn batch_inputs(b: &Batch) -> Vec<(String, Vec<f32>)> {
    vec![
        (
            "tokens".into(),
            b.tokens.iter().map(|&t| t as f32).collect(),
        ),
        ("labels".into(), b.labels.clone()),
        ("label_mask".into(), b.mask.clone()),
    ]
}

/// Whole-graph training on top of [`rlx_tune::Trainer`] (data-parallel when
/// given a `comm` with more than one rank).
pub struct WholeTrainer<'c> {
    pub trainer: rlx_tune::Trainer<'c>,
    pub shapes: HashMap<String, Vec<usize>>,
    trainable: Vec<String>,
}

impl<'c> WholeTrainer<'c> {
    pub fn new(
        tg: TrainGraph,
        steps: usize,
        comm: Option<&'c dyn rlx_tune::GradComm>,
        cfg: &rlx_tune::DpConfig,
    ) -> Result<Self> {
        let trainable = tg.slots.iter().map(|s| s.name.clone()).collect();
        let trainer = rlx_tune::Trainer::new(tg.graph, &tg.slots, &tg.params, steps, comm, cfg)?;
        Ok(Self {
            trainer,
            shapes: tg.shapes,
            trainable,
        })
    }

    /// One optimizer step on `batch` (repeated for every gradient-accumulation
    /// micro-batch; use [`Self::step_with`] to vary them).
    pub fn step(&mut self, batch: &Batch) -> Result<rlx_tune::StepMetrics> {
        let inputs = batch_inputs(batch);
        self.trainer.step(&mut |_, _| inputs.clone())
    }

    pub fn step_with(&mut self, next: &mut impl FnMut() -> Batch) -> Result<rlx_tune::StepMetrics> {
        self.trainer.step(&mut |_, _| batch_inputs(&next()))
    }

    /// Current values of the trained params.
    pub fn trained(&self) -> Params {
        let all = self.trainer.params();
        self.trainable
            .iter()
            .filter_map(|n| all.get(n).map(|v| (n.clone(), v.clone())))
            .collect()
    }
}

/// AdamW over named host tensors, with state that can be saved and resumed —
/// [`rlx_optim::AdamW`] (pure-f32 arithmetic, PyTorch's update) in two groups:
/// matrices with decoupled weight decay, norms and biases (rank-1 checkpoint
/// tensors, conventionally exempt) without.
///
/// The streamed and pipelined trainers update params a layer at a time on
/// whichever process holds that layer, so the optimizer is keyed by param name
/// and kept wherever the param lives. Call [`Self::end_step`] once after every
/// param of an optimizer step has been updated: the bias correction counts
/// steps, not updates.
#[derive(Debug, Clone)]
pub struct AdamW {
    /// Learning rate, applied to both groups on the next update.
    pub lr: f32,
    decay: rlx_optim::AdamW,
    exempt: rlx_optim::AdamW,
}

impl AdamW {
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        let mk = |wd: f32| {
            rlx_optim::AdamW::new(lr)
                .with_weight_decay(wd)
                .with_f32_math(true)
        };
        Self {
            lr,
            decay: mk(weight_decay),
            exempt: mk(0.0),
        }
    }

    /// One update of `param` by `grad` (already averaged / clipped / scaled).
    pub fn step(&mut self, name: &str, param: &mut [f32], grad: &[f32]) {
        use rlx_optim::Optimizer;
        let exempt = param.len() <= 1 || name.contains("norm") || name.contains("bias");
        let opt = if exempt {
            &mut self.exempt
        } else {
            &mut self.decay
        };
        opt.set_lr(self.lr);
        opt.step(name, &[param.len()], param, grad);
    }

    /// Close an optimizer step (advances the bias-correction step count).
    pub fn end_step(&mut self) {
        use rlx_optim::Optimizer;
        self.decay.end_iteration();
        self.exempt.end_iteration();
    }

    /// Serialize both groups' moments and step counts.
    pub fn save(&self, path: &Path) -> Result<()> {
        use rlx_optim::Optimizer;
        use safetensors::{Dtype, tensor::TensorView};
        let groups = [
            ("decay", self.decay.state_dict()),
            ("exempt", self.exempt.state_dict()),
        ];
        let mut bufs: Vec<(String, Vec<u8>, usize)> = Vec::new();
        let mut meta = HashMap::new();
        for (g, st) in &groups {
            let st = st.as_ref().context("optimizer has no state dict")?;
            meta.insert(format!("{g}.step"), st.step.to_string());
            for (k, v) in &st.buffers {
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                bufs.push((format!("{g}/{k}"), bytes, v.len()));
            }
        }
        let views: Vec<(String, TensorView<'_>)> = bufs
            .iter()
            .map(|(n, b, len)| {
                TensorView::new(Dtype::F32, vec![*len], b)
                    .map(|v| (n.clone(), v))
                    .map_err(|e| anyhow::anyhow!("{n}: {e:?}"))
            })
            .collect::<Result<_>>()?;
        safetensors::serialize_to_file(views, Some(meta), path)
            .map_err(|e| anyhow::anyhow!("writing {path:?}: {e:?}"))
    }

    /// Restore state written by [`Self::save`] (hyperparameters stay as set).
    pub fn load(&mut self, path: &Path) -> Result<()> {
        use rlx_optim::{Optimizer, OptimizerState};
        let bytes = std::fs::read(path)?;
        let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes)
            .map_err(|e| anyhow::anyhow!("{path:?}: {e:?}"))?;
        let meta = meta.metadata().clone().unwrap_or_default();
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow::anyhow!("{path:?}: {e:?}"))?;
        for (g, opt) in [("decay", &mut self.decay), ("exempt", &mut self.exempt)] {
            let step = meta
                .get(&format!("{g}.step"))
                .with_context(|| format!("{path:?}: no {g} step count"))?
                .parse()?;
            let prefix = format!("{g}/");
            let buffers = st
                .tensors()
                .into_iter()
                .filter_map(|(n, v)| {
                    n.strip_prefix(&prefix).map(|k| {
                        (
                            k.to_string(),
                            v.data()
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect(),
                        )
                    })
                })
                .collect();
            ensure!(
                opt.load_state_dict(&OptimizerState { step, buffers }),
                "{path:?}: {g} state does not belong to AdamW"
            );
        }
        Ok(())
    }
}

/// Global L2 norm of a set of gradients.
pub fn global_norm<'a>(grads: impl IntoIterator<Item = &'a Vec<f32>>) -> f32 {
    grads
        .into_iter()
        .map(|g| g.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>())
        .sum::<f64>()
        .sqrt() as f32
}

/// Split trained params into `(checkpoint weights, LoRA adapters)`.
pub fn split_trained(trained: &Params) -> (Params, Params) {
    let mut w = Params::new();
    let mut a = Params::new();
    for (k, v) in trained {
        if k.starts_with("lora.") {
            a.insert(k.clone(), v.clone());
        } else if hf_name(k).is_some() {
            w.insert(k.clone(), v.clone());
        }
    }
    (w, a)
}

/// Fold LoRA adapters into their base weights: `W += scale · A · B`, all in the
/// graph's `[in, out]` layout. `base` maps canonical weight names to values.
pub fn merge_lora(
    base: &mut Params,
    adapters: &Params,
    shapes: &HashMap<String, Vec<usize>>,
    scale: f32,
) -> Result<()> {
    for (name, a) in adapters {
        let Some(weight) = name
            .strip_prefix("lora.")
            .and_then(|n| n.strip_suffix(".a"))
        else {
            continue;
        };
        let b = adapters
            .get(&format!("lora.{weight}.b"))
            .with_context(|| format!("adapter {weight} has no B"))?;
        let w = base
            .get_mut(weight)
            .with_context(|| format!("no base weight {weight} to merge into"))?;
        let ashape = shapes
            .get(name)
            .with_context(|| format!("no shape for {name}"))?;
        let (k, r) = (ashape[0], ashape[1]);
        let n = w.len() / k;
        if b.len() != r * n {
            bail!("{weight}: B has {} values, expected {}", b.len(), r * n);
        }
        *w = rlx_tune::fuse_lora(w, a, b, k, r, n, scale)?;
    }
    Ok(())
}

/// Learning rate at `step` (0-based) of `total`: linear warmup over `warmup`
/// steps, then cosine decay to `min_ratio · base` (constant when `cosine` is
/// false).
pub fn scheduled_lr(
    base: f32,
    step: usize,
    total: usize,
    warmup: usize,
    cosine: bool,
    min_ratio: f32,
) -> f32 {
    if warmup > 0 && step < warmup {
        return base * (step + 1) as f32 / warmup as f32;
    }
    if !cosine || total <= warmup {
        return base;
    }
    let t = ((step - warmup) as f32 / (total - warmup).max(1) as f32).min(1.0);
    let min = base * min_ratio;
    min + (base - min) * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Saving mid-run and resuming continues exactly where an uninterrupted run
    /// goes — both the decayed (matrix) and exempt (norm) groups.
    #[test]
    fn adamw_resume_is_exact() {
        let grads = |t: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * 7 + t * 3) % 11) as f32 * 0.01 - 0.05)
                .collect()
        };
        let run = |opt: &mut AdamW,
                   w: &mut Vec<f32>,
                   nrm: &mut Vec<f32>,
                   steps: std::ops::Range<usize>| {
            for t in steps {
                let (gw, gn) = (grads(t, w.len()), grads(t + 1, nrm.len()));
                opt.step("l0.self_attn.q_proj", w, &gw);
                opt.step("l0.input_ln", nrm, &gn);
                opt.end_step();
            }
        };
        let (w0, n0) = (vec![0.5f32; 6], vec![1.0f32; 3]);
        let (mut wa, mut na) = (w0.clone(), n0.clone());
        let mut a = AdamW::new(1e-2, 0.1);
        run(&mut a, &mut wa, &mut na, 0..6);

        let (mut wb, mut nb) = (w0.clone(), n0.clone());
        let mut b = AdamW::new(1e-2, 0.1);
        run(&mut b, &mut wb, &mut nb, 0..3);
        let path =
            std::env::temp_dir().join(format!("rlx-kimi-adamw-{}.safetensors", std::process::id()));
        b.save(&path).unwrap();
        let mut c = AdamW::new(1e-2, 0.1);
        c.load(&path).unwrap();
        run(&mut c, &mut wb, &mut nb, 3..6);
        let _ = std::fs::remove_file(&path);
        assert_eq!(wa, wb);
        assert_eq!(na, nb);
        // The norm group is not decayed: with zero gradients it would not move.
        let mut d = AdamW::new(1e-2, 0.5);
        let mut nz = vec![1.0f32; 3];
        d.step("l0.post_ln.norm", &mut nz, &[0.0; 3]);
        d.end_step();
        assert_eq!(nz, vec![1.0; 3]);
    }
}
