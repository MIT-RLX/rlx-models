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

//! Training: fit a LoRA adapter and the pointer head on labelled requests.
//!
//! The objective is plain cross-entropy over a question's options — which is
//! what the released kev checkpoints actually used (`perm_kl`, `ord_w`,
//! `brier_w`, `focal_gamma` and `anchor_w` are all `0` in
//! `jaredpalmer/kev-4b`'s own `training_config.json`).
//!
//! ```text
//! rows = questions, one causal row each (state ++ branch)
//!
//!   trunk (frozen f32)  --inject_lora-->  hidden [rows, seq, d]
//!        |
//!        +-- gather </opt> and <decide> rows
//!        +-- pointer head (TRAINABLE, in-graph)
//!        +-- softmax cross-entropy against the labelled option
//! ```
//!
//! # What is trainable
//!
//! The LoRA `A`/`B` pairs and the four pointer-head tensors. Base weights stay
//! frozen and out of `wrt`, so the optimizer never touches them.
//!
//! # Two layout facts that are easy to get backwards
//!
//! * **In-graph weights are `[in, out]`.** The `rlx-qwen35` builder transposes
//!   before registering, so `inject_lora` emits `A [in, r]` and `B [r, out]` —
//!   the transpose of peft's `A [r, in]` / `B [out, r]`. [`export_adapter`]
//!   transposes back, so a checkpoint written here loads in kev.
//! * **The head is stored transposed too**: `[d, dp]` in the graph versus
//!   `nn.Linear`'s `[dp, d]`.
//!
//! # Not the released recipe
//!
//! This is the training *engine*. It does not reproduce
//! `jaredpalmer/kev-4b` bit-for-bit: there is no LoRA dropout (0.05 in the
//! recipe), the LR schedule is rlx's warmup+cosine rather than OneCycle, and
//! the per-epoch `none`/distractor augmentation is not ported. Records come
//! from JSONL, not from the frozen suites' generators.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use rlx_autodiff::checkpoint::{checkpoint_backward, suggest_segments};
use rlx_autodiff::grad_with_loss;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_qwen35::{Qwen35Config, Qwen35Weights};
use rlx_runtime::{CompiledGraph, Device, Session};
use rlx_tune::adapter::{LoraInit, LoraSpec};
use rlx_tune::inject::{DropoutMask, FuseMode, inject_lora_dropout, sample_dropout_mask};
use rlx_tune::{Adam, Optimizer, ParamSlot};

use crate::api::Record;
use crate::encode::{Encoder, TokenizerLike, rows_of};
use crate::pointer::PointerHead;

/// Which projections LoRA adapts. Names are matched as substrings against the
/// graph's `Param` names, which are GGUF-style (`blk.3.attn_q.weight`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraTargets {
    /// Attention + MLP + the Gated DeltaNet projections. kev's `all`.
    ///
    /// Note `ssm_alpha`/`ssm_beta` are **fused** into one
    /// `ssm_alpha_beta.weight` matmul by the builder when both are dense, so
    /// they get a single shared adapter here rather than one each. That is a
    /// slightly different parameterization from the reference.
    All,
    /// Attention + MLP only — kev's `dense`, i.e. `all` minus the DeltaNet
    /// projections.
    Dense,
    /// Attention only.
    Attn,
    /// Just `q` and `v`, the classic minimal LoRA set.
    Qv,
}

impl LoraTargets {
    pub fn patterns(self) -> Vec<String> {
        let dense = [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ];
        let gdn = [
            "attn_qkv.weight",
            "attn_gate.weight",
            "ssm_out.weight",
            "ssm_alpha_beta.weight",
            "ssm_alpha.weight",
            "ssm_beta.weight",
        ];
        let v: Vec<&str> = match self {
            LoraTargets::All => dense.iter().chain(gdn.iter()).copied().collect(),
            LoraTargets::Dense => dense.to_vec(),
            LoraTargets::Attn => dense[..4].to_vec(),
            LoraTargets::Qv => vec!["attn_q.weight", "attn_v.weight"],
        };
        v.into_iter().map(str::to_string).collect()
    }
}

/// Training hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub targets: LoraTargets,
    pub head_dim: usize,
    pub lr: f32,
    /// `0` = same as `lr`. The head trains from scratch, so it often wants more.
    pub head_lr: f32,
    pub weight_decay: f32,
    /// Global-norm clip; `0` disables.
    pub clip_grad_norm: f32,
    /// LoRA dropout. The released recipe uses 0.05.
    pub lora_dropout: f32,
    /// Gradient checkpointing segments. `0` = pick `sqrt(n)` automatically,
    /// `1` = off. The released recipe used checkpointing (`--checkpointing 1`).
    pub checkpoint_segments: usize,
    /// bf16 autocast with fp32 master weights — the recipe's `--dtype bf16`.
    ///
    /// Compute, elementwise and reductions run BF16; **`Param` nodes and
    /// graph boundaries stay F32**, so the weights the optimizer updates and
    /// the gradients it consumes are full precision. That is exactly torch's
    /// autocast contract.
    ///
    /// Uses `PrecisionPolicy::AutoMixedBf16Safe`, not `AutoMixedBf16`: the
    /// latter is the TPU policy and puts matmul and data movement in BF16,
    /// which produces NaN on CPU (measured — see that policy's docs). The
    /// safe variant keeps those in F32.
    ///
    /// **Measured, this does not currently save memory on CPU** — 116 GB ->
    /// 134 GB on the real training graph, because the large tensors are
    /// Compute outputs that stay F32 while the inserted casts add buffers.
    /// It is correct (gradient cosine 1.00000 against the f32 run) and it is
    /// what a backend with real BF16 matmul-on-activations would need, but
    /// today it is off by default and only applied when the memory planner
    /// says it helps.
    pub bf16: bool,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        // The released recipe's shape: rank 16, alpha 32, AdamW wd 0.01.
        Self {
            lora_rank: 16,
            lora_alpha: 32.0,
            targets: LoraTargets::All,
            head_dim: crate::pointer::DEFAULT_HEAD_DIM,
            lr: 2e-5,
            head_lr: 0.0,
            weight_decay: 0.01,
            clip_grad_norm: 1.0,
            lora_dropout: 0.05,
            checkpoint_segments: 0,
            bf16: false,
            seed: 0,
        }
    }
}

/// The compiled shape a training step runs at. Records are padded into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainShape {
    /// Question rows per step.
    pub rows: usize,
    /// Row length (state + branch), padded.
    pub seq: usize,
    /// Options per question, padded.
    pub max_options: usize,
}

/// One labelled record: the request plus the correct option per question.
#[derive(Debug, Clone)]
pub struct TrainExample {
    pub record: Record,
}

/// OneCycle learning rate, matching `torch.optim.lr_scheduler.OneCycleLR`
/// with cosine annealing — the schedule the released recipe used.
///
/// Torch's phase boundaries are not the obvious ones: `step_size_up` is
/// `pct_start · total_steps − 1`, so with `pct_start = 0.1` over 20 steps the
/// peak is reached at step **1**, not step 2, and the anneal spans the
/// remaining 18. Getting that off by one shifts the whole schedule.
///
/// `div_factor = 25`, `final_div_factor = 1e4` — torch's defaults, i.e. start
/// at `max_lr/25` and end at `max_lr/25/1e4`.
pub fn one_cycle_lr(max_lr: f32, step: usize, total_steps: usize, pct_start: f32) -> f32 {
    if total_steps <= 1 {
        return max_lr;
    }
    let initial = max_lr / 25.0;
    let final_lr = initial / 1e4;
    let up = pct_start as f64 * total_steps as f64 - 1.0;
    let i = step.min(total_steps - 1) as f64;
    let anneal = |start: f64, end: f64, pct: f64| -> f32 {
        (end + (start - end) / 2.0 * ((std::f64::consts::PI * pct).cos() + 1.0)) as f32
    };
    if up > 0.0 && i <= up {
        anneal(initial as f64, max_lr as f64, i / up)
    } else {
        let down = (total_steps as f64 - up - 1.0).max(1.0);
        let pct = ((i - up) / down).clamp(0.0, 1.0);
        anneal(max_lr as f64, final_lr as f64, pct)
    }
}

/// Deterministic pseudo-random scalar in `[-1, 1)`.
fn hashed(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 40) as f32) / 8_388_608.0 - 1.0
}

fn name_seed(name: &str, salt: u64) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325u64 ^ salt, |h, b| {
        (h ^ b as u64).wrapping_mul(0x1000_0000_01b3)
    })
}

/// Input names the training graph declares beyond the trunk's own.
pub mod inputs {
    /// Flat `[rows]` index of each row's `<decide>` token into `[rows*seq, d]`.
    pub const DECIDE_IDX: &str = "kev.decide_idx";
    /// Flat `[rows*max_options]` index of each option's `</opt>` token.
    pub const OPT_IDX: &str = "kev.opt_idx";
    /// Additive `[rows, max_options]`; `-1e9` on padded option slots.
    pub const OPT_MASK: &str = "kev.opt_mask";
    /// `[rows]` correct option index per question (f32-encoded).
    pub const LABELS: &str = "kev.labels";
    /// `[rows]` 1.0 for a real question row, 0.0 for padding.
    pub const ROW_MASK: &str = "kev.row_mask";
}

/// A built training graph plus everything needed to run it.
pub struct TrainGraph {
    pub forward: Graph,
    pub params: HashMap<String, Vec<f32>>,
    /// Trainable slots, in the order their gradients come back.
    pub wrt: Vec<ParamSlot>,
    pub shape: TrainShape,
    pub hidden_size: usize,
    pub head_dim: usize,
    /// Adapter tensor names, for [`export_adapter`].
    pub adapter_names: Vec<String>,
    /// LoRA rank these adapter tensors were built at.
    pub lora_rank: usize,
    /// Dropout mask inputs, which the trainer must feed every step.
    pub dropout_masks: Vec<DropoutMask>,
    pub lora_dropout: f32,
}

/// Graph param names for the pointer head.
pub const HEAD_Q_W: &str = "kev.head.q.weight";
pub const HEAD_Q_B: &str = "kev.head.q.bias";
pub const HEAD_K_W: &str = "kev.head.k.weight";
pub const HEAD_K_B: &str = "kev.head.k.bias";

/// The bf16 autocast policy to train under, per backend.
///
/// Full `AutoMixedBf16` (BF16 in Compute and DataMovement too) now *runs*
/// everywhere — it used to produce NaN on CPU and Metal, which was a slot-layout
/// bug, not a numerical one: a bf16 tensor lives in the arena either packed at
/// 2 B/elem or widened to 4 B, and those two backends had kernels reading one
/// layout out of the other (fixed in `rlx-cpu`'s matmul dispatch and
/// `ArenaWidthPolicy::NativeHalfWidened`, and in Metal's cast lowering and
/// `NativeBf16Widened`). Forward parity against f32 is exact on CPU, Metal and
/// MLX for every op kind.
///
/// Training still takes `AutoMixedBf16Safe`, for two measured reasons:
///
/// * **Fidelity.** Under the full policy the head gradient's cosine against the
///   f32 run drops to 0.83; under the safe policy it is 1.00000.
/// * **No memory to win.** CPU and Metal have no bf16 kernels at all —
///   `rlx_runtime::cpu_low_precision::promote_to_f32` rewrites bf16 *arithmetic*
///   to f32 before planning, and bf16 activations are stored f32-wide — so the
///   full policy costs accuracy and saves nothing there.
///
/// MLX *does* run bf16 natively, and its backward now works (it used to fail on
/// `nelems doesn't match shape product` — two upstream bugs, both fixed). So MLX
/// is where the full policy could pay for itself in activation memory, and this
/// function is where that per-backend choice belongs. It is not made here yet
/// because the win has not been measured: MLX manages its own allocations, so
/// `plan_memory` cannot price it, and the fidelity cost is known (0.83).
pub fn bf16_policy_for(device: Device) -> rlx_compile::precision::PrecisionPolicy {
    let _ = device;
    rlx_compile::precision::PrecisionPolicy::AutoMixedBf16Safe
}

/// Build the trunk, inject LoRA, and append the pointer head and loss.
///
/// `device` only affects the bf16 autocast policy — see [`bf16_policy_for`].
pub fn build_train_graph_for(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    shape: TrainShape,
    tcfg: &TrainConfig,
    device: Device,
) -> Result<TrainGraph> {
    build_train_graph_inner(cfg, weights, shape, tcfg, device)
}

/// [`build_train_graph_for`] on CPU.
pub fn build_train_graph(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    shape: TrainShape,
    tcfg: &TrainConfig,
) -> Result<TrainGraph> {
    build_train_graph_inner(cfg, weights, shape, tcfg, Device::Cpu)
}

fn build_train_graph_inner(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    shape: TrainShape,
    tcfg: &TrainConfig,
    device: Device,
) -> Result<TrainGraph> {
    let TrainShape {
        rows,
        seq,
        max_options,
    } = shape;
    anyhow::ensure!(rows > 0 && seq > 0 && max_options > 0, "empty train shape");
    let d = cfg.hidden_size;
    let dp = tcfg.head_dim;

    // Trunk: row form, hidden out, no LM head. Runtime MRoPE keeps the
    // 262k-row baked table out of the parameter set.
    let (trunk, mut params, packed) = rlx_qwen35::build_qwen35_graph_sized_ext(
        cfg, weights, rows, seq, false, false, false, false, None, true, false, true, false,
    )
    .context("building the trunk forward graph")?;
    anyhow::ensure!(
        packed.is_empty(),
        "training needs dequantized f32 weights, but {} tensors came back packed",
        packed.len()
    );

    let spec = LoraSpec {
        rank: tcfg.lora_rank,
        alpha: tcfg.lora_alpha,
        dropout: tcfg.lora_dropout,
        target_modules: tcfg.targets.patterns(),
        init: LoraInit::Gaussian,
    };
    // Unfused: explicit `base + scale·((x·A)·B)` matmuls, so the backward
    // needs no unfuse pass and every backend can train it.
    let (mut g, adapters, dropout_masks) = inject_lora_dropout(&trunk, &spec, FuseMode::Unfused);
    if adapters.is_empty() {
        bail!(
            "LoRA matched no weights; targets {:?} do not appear in the graph's \
             Param names (they are GGUF-style, e.g. `blk.3.attn_q.weight`)",
            spec.target_modules
        );
    }

    let hidden = *g
        .outputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("trunk graph has no output"))?;

    // ── readout ──────────────────────────────────────────────────
    let f = DType::F32;
    let flat = g.reshape_(hidden, vec![(rows * seq) as i64, d as i64]);
    let d_idx = g.input(inputs::DECIDE_IDX, Shape::new(&[rows], f));
    let o_idx = g.input(inputs::OPT_IDX, Shape::new(&[rows * max_options], f));
    let h_dec = g.gather_(flat, d_idx, 0); // [rows, d]
    let h_opt = g.gather_(flat, o_idx, 0); // [rows*max_options, d]

    // ── pointer head (trainable) ─────────────────────────────────
    // Stored `[d, dp]` for `matmul(x, W)`; `nn.Linear` is `[dp, d]`.
    let qw = g.param(HEAD_Q_W, Shape::new(&[d, dp], f));
    let qb = g.param(HEAD_Q_B, Shape::new(&[dp], f));
    let kw = g.param(HEAD_K_W, Shape::new(&[d, dp], f));
    let kb = g.param(HEAD_K_B, Shape::new(&[dp], f));
    let q = g.matmul(h_dec, qw, Shape::new(&[rows, dp], f));
    let q = g.add(q, qb);
    let k = g.matmul(h_opt, kw, Shape::new(&[rows * max_options, dp], f));
    let k = g.add(k, kb);

    // z[r, j] = <k[r, j, :], q[r, :]> / sqrt(dp), as a batched matmul so no
    // operand has to broadcast.
    let k3 = g.reshape_(k, vec![rows as i64, max_options as i64, dp as i64]);
    let q3 = g.reshape_(q, vec![rows as i64, dp as i64, 1]);
    let zb = g.matmul(k3, q3, Shape::new(&[rows, max_options, 1], f));
    let z = g.reshape_(zb, vec![rows as i64, max_options as i64]);
    let scale = 1.0 / (dp as f32).sqrt();
    let scale_c = g.constant(scale as f64, f);
    let z = g.mul(z, scale_c);

    // Padded option slots must not win the softmax.
    let omask = g.input(inputs::OPT_MASK, Shape::new(&[rows, max_options], f));
    let logits = g.add(z, omask);

    let labels = g.input(inputs::LABELS, Shape::new(&[rows], f));
    let row_mask = g.input(inputs::ROW_MASK, Shape::new(&[rows], f));
    let loss = rlx_tune::loss::cross_entropy_masked(&mut g, logits, labels, row_mask);
    g.set_outputs(vec![loss]);

    // ── initial values ───────────────────────────────────────────
    let mut wrt = Vec::with_capacity(adapters.len() + 4);
    let mut adapter_names = Vec::with_capacity(adapters.len());
    let r = tcfg.lora_rank as f32;
    for a in &adapters {
        let n: usize = a.shape.iter().product();
        // `A` random, `B` zero: the injected forward is then exactly the base
        // model until the first step, which is what makes a zero-step run a
        // no-op and is asserted in the tests.
        let init = if a.name.ends_with(".a") {
            let s = name_seed(&a.name, tcfg.seed);
            let sd = (1.0 / r).sqrt();
            (0..n).map(|i| sd * hashed(s, i)).collect()
        } else {
            vec![0.0f32; n]
        };
        params.insert(a.name.clone(), init);
        wrt.push(ParamSlot {
            name: a.name.clone(),
            node: a.node,
        });
        adapter_names.push(a.name.clone());
    }
    let head_init = |name: &str, n: usize, scale: f32| -> Vec<f32> {
        let s = name_seed(name, tcfg.seed ^ 0x5eed);
        (0..n).map(|i| scale * hashed(s, i)).collect()
    };
    let wscale = (1.0 / d as f32).sqrt();
    params.insert(HEAD_Q_W.into(), head_init(HEAD_Q_W, d * dp, wscale));
    params.insert(HEAD_K_W.into(), head_init(HEAD_K_W, d * dp, wscale));
    params.insert(HEAD_Q_B.into(), vec![0.0; dp]);
    params.insert(HEAD_K_B.into(), vec![0.0; dp]);
    for (name, node) in [
        (HEAD_Q_W, qw),
        (HEAD_Q_B, qb),
        (HEAD_K_W, kw),
        (HEAD_K_B, kb),
    ] {
        wrt.push(ParamSlot {
            name: name.into(),
            node,
        });
    }

    // bf16 autocast, applied to the FORWARD before it is differentiated.
    // Doing it here (rather than to the combined graph) is what gives the
    // recipe's semantics: `Cast` boundaries are differentiable, so the
    // backward inherits bf16 intermediates while parameter gradients arrive
    // back in F32 for the optimizer.
    let (g, wrt) = if tcfg.bf16 {
        use rlx_compile::precision::{AutoMixedPrecision, PrecisionPolicy};
        use rlx_fusion::pass::Pass;
        let _ = PrecisionPolicy::AlwaysF32; // keep the import meaningful
        let g = AutoMixedPrecision::new(bf16_policy_for(device)).run(g);
        // The pass REBUILDS the graph, so every `NodeId` captured above is
        // stale. Re-resolve the trainable slots by name — differentiating
        // w.r.t. stale ids silently produces gradients for whatever nodes now
        // occupy those indices, which is how this first showed up: a
        // 128-element head gradient came back with 2048 elements.
        let by_name: HashMap<&str, NodeId> = g
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                rlx_ir::Op::Param { name } => Some((name.as_str(), n.id)),
                _ => None,
            })
            .collect();
        let remapped = wrt
            .iter()
            .map(|s| {
                by_name
                    .get(s.name.as_str())
                    .map(|node| ParamSlot {
                        name: s.name.clone(),
                        node: *node,
                    })
                    .ok_or_else(|| {
                        anyhow::anyhow!("{} vanished from the graph after autocast", s.name)
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        (g, remapped)
    } else {
        (g, wrt)
    };

    Ok(TrainGraph {
        lora_rank: tcfg.lora_rank,
        forward: g,
        params,
        wrt,
        shape,
        hidden_size: d,
        head_dim: dp,
        adapter_names,
        dropout_masks,
        lora_dropout: tcfg.lora_dropout,
    })
}

/// Return whichever of the plain or checkpointed backward the memory planner
/// says is smaller.
///
/// Checkpointing is a heuristic whose benefit depends on graph shape, so this
/// asks rather than assumes. Sweeping a few segment counts costs a planning
/// pass each — microseconds against a compile.
fn best_checkpointing(forward: &rlx_ir::Graph, backward: rlx_ir::Graph) -> rlx_ir::Graph {
    use rlx_compile::memory::{ArenaWidthPolicy, plan_memory_with_policy};
    let plan =
        |g: &rlx_ir::Graph| plan_memory_with_policy(g, 256, ArenaWidthPolicy::Native).arena_size;
    let mut best = plan(&backward);
    let mut chosen: Option<rlx_ir::Graph> = None;
    let suggested = suggest_segments(forward.nodes().len());
    for segs in [suggested, 4, 8, 16, 32] {
        if segs <= 1 {
            continue;
        }
        let ck = checkpoint_backward(forward, &backward, segs);
        let bytes = plan(&ck);
        if bytes < best {
            best = bytes;
            chosen = Some(ck);
        }
    }
    chosen.unwrap_or(backward)
}

/// Per-step inputs built from a batch of examples.
pub struct StepFeeds {
    pub named: Vec<(String, Vec<f32>)>,
    /// Real question rows in this batch (for reporting).
    pub active_rows: usize,
    /// Real tokens across those rows, i.e. `Σ (state + branch)`.
    ///
    /// The compiled shape is `rows × seq`, so `real_tokens / (rows × seq)` is the
    /// fraction of the step's compute that is not padding. Reported because it is
    /// the one number that says whether the shape bucketing
    /// (`KEV_TRAIN_TOKEN_BUDGET`) is worth tuning, and it is invisible otherwise.
    pub real_tokens: usize,
}

/// Pack a batch of labelled records into the fixed training shape.
///
/// Rows are filled question by question across examples; leftover rows are
/// masked out rather than dropped, so a short batch still steps.
pub fn build_feeds<T: TokenizerLike>(
    encoder: &Encoder<T>,
    cfg: &Qwen35Config,
    shape: TrainShape,
    examples: &[TrainExample],
    encode_opts: crate::encode::EncodeOpts,
) -> Result<StepFeeds> {
    let TrainShape {
        rows,
        seq,
        max_options,
    } = shape;
    let mut input_ids = vec![0f32; rows * seq];
    let mut decide_idx = vec![0f32; rows];
    let mut opt_idx = vec![0f32; rows * max_options];
    let mut opt_mask = vec![-1e9f32; rows * max_options];
    let mut labels = vec![0f32; rows];
    let mut row_mask = vec![0f32; rows];

    let mut row = 0usize;
    let mut real_tokens = 0usize;
    for ex in examples {
        let enc = encoder.encode(&ex.record, encode_opts)?;
        let (state_ids, _, branch_rows) = rows_of(&enc)?;
        let ls = state_ids.len();
        for (qi, br) in branch_rows.iter().enumerate() {
            if row >= rows {
                bail!(
                    "batch needs more than {rows} question rows; raise TrainShape::rows \
                     or send fewer records per step"
                );
            }
            let total = ls + br.ids.len();
            if total > seq {
                bail!("row of {total} tokens exceeds the compiled seq {seq}");
            }
            if br.opts.len() > max_options {
                bail!(
                    "question has {} options, compiled for {max_options}",
                    br.opts.len()
                );
            }
            let base = row * seq;
            for (j, t) in state_ids.iter().chain(br.ids.iter()).enumerate() {
                input_ids[base + j] = *t as f32;
            }
            real_tokens += total;
            decide_idx[row] = (base + ls + br.decide) as f32;
            for (j, o) in br.opts.iter().enumerate() {
                opt_idx[row * max_options + j] = (base + ls + o) as f32;
                opt_mask[row * max_options + j] = 0.0;
            }
            let label = ex.record.questions[qi].label;
            if label >= br.opts.len() {
                bail!(
                    "label {label} out of range for a question with {} options",
                    br.opts.len()
                );
            }
            labels[row] = label as f32;
            row_mask[row] = 1.0;
            row += 1;
        }
    }
    if row == 0 {
        bail!("batch contained no questions");
    }

    // Positions are contiguous within a row, and identical across rows, so one
    // table serves the batch — same argument as the inference path.
    let rot_half = cfg.rope_dim_count / 2;
    let (cos, sin) = rlx_qwen35::mrope_prefill_feeds(cfg, seq, None, rot_half);

    Ok(StepFeeds {
        named: vec![
            ("input_ids".into(), input_ids),
            ("rope_cos".into(), cos),
            ("rope_sin".into(), sin),
            (inputs::DECIDE_IDX.into(), decide_idx),
            (inputs::OPT_IDX.into(), opt_idx),
            (inputs::OPT_MASK.into(), opt_mask),
            (inputs::LABELS.into(), labels),
            (inputs::ROW_MASK.into(), row_mask),
        ],
        active_rows: row,
        real_tokens,
    })
}

/// A compiled training loop: forward + backward, host-side AdamW.
pub struct Trainer {
    compiled: CompiledGraph,
    params: HashMap<String, Vec<f32>>,
    wrt: Vec<ParamSlot>,
    adam: Adam,
    head_adam: Adam,
    head_names: Vec<String>,
    clip: f32,
    weight_decay: f32,
    /// Accumulated gradients, cleared on each optimizer step.
    accum: HashMap<String, Vec<f32>>,
    accum_count: usize,
    pub shape: TrainShape,
    pub hidden_size: usize,
    pub head_dim: usize,
    pub adapter_names: Vec<String>,
    dropout_masks: Vec<DropoutMask>,
    lora_dropout: f32,
    drop_rng: crate::pyrandom::PyRandom,
}

impl Trainer {
    /// Compile `graph`'s backward pass for `device`.
    pub fn new(graph: TrainGraph, device: Device, tcfg: &TrainConfig) -> Result<Self> {
        let wrt_ids: Vec<NodeId> = graph.wrt.iter().map(|s| s.node).collect();
        let backward = grad_with_loss(&graph.forward, &wrt_ids);
        // Gradient checkpointing, chosen by measurement rather than faith.
        //
        // As a pure graph transform it is a big win on chains and a *loss* on
        // this residual trunk — 116 GB -> 149 GB at the shapes decision-v7
        // produces, because skip connections make most nodes cross a segment
        // boundary and get kept anyway. So plan both and keep the smaller;
        // `checkpoint_segments: 0` means "try and only use it if it helps".
        let backward = match tcfg.checkpoint_segments {
            1 => backward,
            0 => best_checkpointing(&graph.forward, backward),
            n => checkpoint_backward(&graph.forward, &backward, n),
        };
        let mut compiled = Session::new(device).compile(backward);
        for (n, v) in &graph.params {
            compiled.set_param(n, v);
        }
        let head_lr = if tcfg.head_lr > 0.0 {
            tcfg.head_lr
        } else {
            tcfg.lr
        };
        // `Adam` here is plain Adam; the decoupled AdamW decay is applied in
        // `optimizer_step` so it does not enter the moment estimates.
        let mk = Adam::new;
        Ok(Self {
            compiled,
            params: graph.params,
            wrt: graph.wrt,
            adam: mk(tcfg.lr),
            head_adam: mk(head_lr),
            weight_decay: tcfg.weight_decay,
            head_names: vec![
                HEAD_Q_W.into(),
                HEAD_Q_B.into(),
                HEAD_K_W.into(),
                HEAD_K_B.into(),
            ],
            clip: tcfg.clip_grad_norm,
            accum: HashMap::new(),
            accum_count: 0,
            shape: graph.shape,
            hidden_size: graph.hidden_size,
            head_dim: graph.head_dim,
            adapter_names: graph.adapter_names,
            dropout_masks: graph.dropout_masks,
            lora_dropout: graph.lora_dropout,
            drop_rng: crate::pyrandom::PyRandom::seed_u64(tcfg.seed ^ 0xd0_9a),
        })
    }

    pub fn params(&self) -> &HashMap<String, Vec<f32>> {
        &self.params
    }

    /// The accumulated gradient of `wrt[i]`, for gradient checking.
    ///
    /// Reflects whatever has been accumulated so far, un-averaged and
    /// un-clipped — so after a single [`Self::accumulate`] it is the raw
    /// analytic gradient a finite difference can be compared against.
    pub fn grad_of(&self, i: usize) -> Option<&[f32]> {
        let name = &self.wrt.get(i)?.name;
        self.accum.get(name).map(Vec::as_slice)
    }

    /// Trainable slot names, in gradient order.
    pub fn trainable_names(&self) -> Vec<&str> {
        self.wrt.iter().map(|s| s.name.as_str()).collect()
    }

    /// Run forward + backward on one micro-batch and accumulate gradients.
    ///
    /// Returns the loss. Call [`Self::optimizer_step`] after `accum`
    /// micro-batches.
    pub fn accumulate(&mut self, feeds: &StepFeeds) -> Result<f32> {
        let seed = [1.0f32];
        // A fresh dropout mask per module per step. These inputs read ZEROS if
        // left unbound, which would silently zero the whole adapter branch —
        // so they are always fed, all-ones when dropout is off.
        let masks: Vec<(String, Vec<f32>)> = self
            .dropout_masks
            .iter()
            .map(|m| {
                let n: usize = m.shape.iter().product();
                let rng = &mut self.drop_rng;
                (
                    m.name.clone(),
                    sample_dropout_mask(n, self.lora_dropout, &mut || rng.random() as f32),
                )
            })
            .collect();
        let mut run: Vec<(&str, &[f32])> = feeds
            .named
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        for (n, v) in &masks {
            run.push((n.as_str(), v.as_slice()));
        }
        // `grad_with_loss` exposes the loss cotangent; 1.0 = ∂loss/∂loss.
        run.push(("d_output", &seed));
        let outs = self.compiled.run(&run);
        if outs.is_empty() || outs[0].is_empty() {
            bail!("backward graph produced no loss output");
        }
        let loss = outs[0][0];
        if !loss.is_finite() {
            bail!("non-finite loss: {loss}");
        }
        for (i, slot) in self.wrt.iter().enumerate() {
            let g = outs.get(1 + i).map(Vec::as_slice).unwrap_or(&[]);
            let acc = self
                .accum
                .entry(slot.name.clone())
                .or_insert_with(|| vec![0.0; g.len()]);
            if acc.len() != g.len() {
                acc.resize(g.len(), 0.0);
            }
            for (a, x) in acc.iter_mut().zip(g) {
                *a += *x;
            }
        }
        self.accum_count += 1;
        Ok(loss)
    }

    /// Average the accumulated gradients, clip, and apply AdamW.
    pub fn optimizer_step(&mut self) -> Result<f32> {
        if self.accum_count == 0 {
            bail!("optimizer_step with nothing accumulated");
        }
        let inv = 1.0 / self.accum_count as f32;
        for g in self.accum.values_mut() {
            for x in g.iter_mut() {
                *x *= inv;
            }
        }
        // Global-norm clipping over ALL trainables together, matching
        // `clip_grad_norm_` — per-tensor clipping would change the direction.
        let mut total = 0.0f64;
        for g in self.accum.values() {
            total += g.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();
        }
        let norm = total.sqrt() as f32;
        // Refuse a non-finite gradient instead of stepping on it.
        //
        // The loss can be perfectly finite while a *gradient* is not — a
        // backward-only backend bug looks exactly like that — and clipping by a
        // NaN norm quietly writes NaN into every parameter. What that produces
        // downstream is not a crash but a plausible-looking run: the loss
        // settles on `ln(n_options)`, the chance floor, and stays there. A
        // 1529-step reproduction was lost that way before this check existed
        // (Metal's Qwen3.5 trunk backward; the forward was bit-identical to
        // CPU's). Name the worst offender, since it points straight at the
        // layer whose backward is at fault.
        if !norm.is_finite() {
            let worst = self
                .accum
                .iter()
                .filter(|(_, g)| g.iter().any(|x| !x.is_finite()))
                .map(|(n, g)| {
                    (
                        n.clone(),
                        g.iter().filter(|x| !x.is_finite()).count(),
                        g.len(),
                    )
                })
                .min_by(|a, b| a.0.cmp(&b.0));
            match worst {
                Some((name, bad, len)) => bail!(
                    "non-finite gradient norm ({norm}) — the loss was finite, so this is the \
                     backward: `{name}` has {bad}/{len} non-finite entries. Try `--device cpu`; \
                     `rlx-qwen35`'s `trunk_backward_device_parity` test covers this class of bug"
                ),
                None => bail!(
                    "non-finite gradient norm ({norm}) with every accumulated gradient finite \
                     — the sum overflowed"
                ),
            }
        }
        let scale = if self.clip > 0.0 && norm > self.clip {
            self.clip / (norm + 1e-6)
        } else {
            1.0
        };

        for slot in &self.wrt {
            let Some(g) = self.accum.get(&slot.name) else {
                continue;
            };
            let Some(p) = self.params.get_mut(&slot.name) else {
                continue;
            };
            let scaled: Vec<f32> = g.iter().map(|x| x * scale).collect();
            let head = self.head_names.contains(&slot.name);
            let opt: &mut Adam = if head {
                &mut self.head_adam
            } else {
                &mut self.adam
            };
            // Decoupled (AdamW) decay: applied to the weights directly, not
            // folded into the gradient, so it never enters m/v.
            if self.weight_decay > 0.0 {
                let f = opt.lr * self.weight_decay;
                for x in p.iter_mut() {
                    *x -= f * *x;
                }
            }
            opt.step(&slot.name, p, &scaled);
            self.compiled.set_param(&slot.name, p);
        }
        for g in self.accum.values_mut() {
            g.iter_mut().for_each(|x| *x = 0.0);
        }
        self.accum_count = 0;
        Ok(norm)
    }

    /// Move both optimizers' state out, to carry across a graph rebuild.
    ///
    /// The recipe compiles one graph per (rows, seq, max_options) bucket and
    /// rebuilds the `Trainer` when the shape changes — 28 times in a full
    /// `decision-v7` epoch. The adapter and head *weights* were already carried;
    /// the Adam moments were not, so every rebuild restarted the bias correction
    /// and made the next step effectively un-normalized. With 28 of those during
    /// the OneCycle ramp the run diverged: loss 1.06 at step 50 → 4.33 at step
    /// 100 with a pre-clip grad norm of 1294 → 9.41 by step 150, which is above
    /// every `ln(n_options)` chance floor. (The comment here used to claim
    /// `Trainer::rebuild_at` carried this; no such function existed.)
    pub fn take_optimizer_state(&mut self) -> (rlx_tune::AdamState, rlx_tune::AdamState) {
        (self.adam.take_state(), self.head_adam.take_state())
    }

    /// Put back what [`take_optimizer_state`](Self::take_optimizer_state) moved.
    pub fn restore_optimizer_state(&mut self, state: (rlx_tune::AdamState, rlx_tune::AdamState)) {
        self.adam.restore_state(state.0);
        self.head_adam.restore_state(state.1);
    }

    /// Micro-batches accumulated since the last optimizer step.
    pub fn pending(&self) -> usize {
        self.accum_count
    }

    pub fn set_lr(&mut self, lr: f32, head_lr: f32) {
        self.adam.lr = lr;
        self.head_adam.lr = head_lr;
    }

    /// The trained pointer head, in the host layout [`PointerHead`] expects.
    pub fn pointer_head(&self, temperature: f32) -> Result<PointerHead> {
        let get = |n: &str| -> Result<&Vec<f32>> {
            self.params
                .get(n)
                .ok_or_else(|| anyhow::anyhow!("missing {n}"))
        };
        // Graph stores `[d, dp]`; `PointerHead` wants `nn.Linear`'s `[dp, d]`.
        let t = |v: &[f32]| -> Vec<f32> {
            let (d, dp) = (self.hidden_size, self.head_dim);
            let mut out = vec![0.0; v.len()];
            for i in 0..d {
                for j in 0..dp {
                    out[j * d + i] = v[i * dp + j];
                }
            }
            out
        };
        PointerHead::new(
            t(get(HEAD_Q_W)?),
            get(HEAD_Q_B)?.clone(),
            t(get(HEAD_K_W)?),
            get(HEAD_K_B)?.clone(),
            self.hidden_size,
            self.head_dim,
            temperature,
        )
    }
}

/// Seed a freshly-built [`TrainGraph`]'s **trainable adapter** from a checkpoint.
///
/// This is what warm-starting has to do, and the alternative is not merely
/// slower — it is wrong. Folding the checkpoint's delta into the base weights
/// and then fitting a *fresh* low-rank adapter on top trains a correct model,
/// but what gets exported afterwards is only the new delta: loading it the
/// normal way (`base + adapter`) silently drops everything the checkpoint knew.
/// Measured on kev-0.8b, total `||B||` went 86.85 -> 0.35 across a two-step
/// delta run. A rank-16 export cannot represent the sum of two rank-16 deltas,
/// so composing on export is not available either — continuing the same
/// adapter is, and it is also what peft and kev do.
///
/// Inverts [`export_adapter`]'s transpose exactly: peft holds `A [r, in]` /
/// `B [out, r]`, the graph holds `A [in, r]` / `B [r, out]`. Neither side
/// bakes in `alpha/rank` — the graph applies it as an op and
/// [`crate::lora::LoraAdapter::merge_into`] applies it on load — so this is a
/// pure re-layout with no rescaling.
pub fn restore_adapter(g: &mut TrainGraph, ad: &crate::lora::LoraAdapter) -> Result<()> {
    let rank = g.lora_rank;
    if ad.r != rank {
        bail!(
            "checkpoint adapter is rank {} but this run is rank {rank} — rerun with \
             --lora-rank {}",
            ad.r,
            ad.r
        );
    }
    let mut restored = 0usize;
    for name in g.adapter_names.clone() {
        let body = name
            .strip_prefix("lora.")
            .ok_or_else(|| anyhow::anyhow!("unexpected adapter name {name}"))?;
        let (weight, side) = body
            .rsplit_once('.')
            .ok_or_else(|| anyhow::anyhow!("unexpected adapter name {name}"))?;
        let Some(module) = crate::lora::module_path_for_gguf(weight) else {
            continue;
        };
        let Some(pair) = ad.get(&module) else {
            continue;
        };
        // peft -> graph. `rows`/`cols` are the GRAPH orientation, matching
        // `export_adapter`'s naming so the two stay legible side by side.
        let (src, rows, cols) = if side == "a" {
            (&pair.a, pair.in_features, rank)
        } else {
            (&pair.b, rank, pair.out_features)
        };
        let Some(slot) = g.params.get_mut(&name) else {
            continue;
        };
        if slot.len() != src.len() {
            bail!(
                "{name}: checkpoint has {} values, graph wants {}",
                src.len(),
                slot.len()
            );
        }
        for i in 0..rows {
            for j in 0..cols {
                slot[i * cols + j] = src[j * rows + i];
            }
        }
        restored += 1;
    }
    if restored == 0 {
        bail!("the checkpoint adapter matched none of this graph's modules");
    }
    Ok(())
}

/// Seed a freshly-built [`TrainGraph`]'s pointer head from a trained checkpoint.
///
/// `--init-from` merges the adapter into the base weights, which restores the
/// *backbone* — but the head is a separate set of parameters that
/// [`build_train_graph`] always seeds pseudo-randomly. Without this, a warm
/// start throws the trained readout away and begins with a random one, so the
/// model predicts at chance at step 0 and has to relearn the head. That defeats
/// the point of warm-starting and makes the released recipe — which is a
/// continuation round (`init_from: jaredpalmer/kev-0.8b`) — unreproducible.
///
/// Inverts the transpose [`Trainer::pointer_head`] applies: the graph stores
/// `[d, dp]` while `PointerHead` holds `nn.Linear`'s `[dp, d]`.
pub fn restore_head(g: &mut TrainGraph, head: &PointerHead) -> Result<()> {
    let (d, dp) = (g.hidden_size, g.head_dim);
    if head.hidden != d || head.head_dim != dp {
        bail!(
            "checkpoint head is {}x{} but this graph wants {d}x{dp} — refusing to \
             restore a mismatched head",
            head.hidden,
            head.head_dim
        );
    }
    // `[dp, d]` -> `[d, dp]`.
    let untranspose = |w: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0; w.len()];
        for i in 0..d {
            for j in 0..dp {
                out[i * dp + j] = w[j * d + i];
            }
        }
        out
    };
    let mut put = |name: &str, v: Vec<f32>| -> Result<()> {
        let slot = g
            .params
            .get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("graph has no {name}"))?;
        if slot.len() != v.len() {
            bail!(
                "{name}: checkpoint has {} values, graph wants {}",
                v.len(),
                slot.len()
            );
        }
        slot.copy_from_slice(&v);
        Ok(())
    };
    put(HEAD_Q_W, untranspose(&head.q_weight))?;
    put(HEAD_Q_B, head.q_bias.clone())?;
    put(HEAD_K_W, untranspose(&head.k_weight))?;
    put(HEAD_K_B, head.k_bias.clone())?;
    Ok(())
}

/// Write the adapter as peft-compatible safetensors.
///
/// Transposes back to peft's orientation (`A [r, in]`, `B [out, r]`) and names
/// tensors `base_model.model.<module>.lora_{A,B}.weight`, so the result loads
/// in kev and round-trips through [`crate::lora::LoraAdapter`].
pub fn export_adapter(
    trainer: &Trainer,
    rank: usize,
    alpha: f32,
    dir: &std::path::Path,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut owned: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    for name in &trainer.adapter_names {
        // `lora.<gguf weight name>.a` / `.b`
        let body = name
            .strip_prefix("lora.")
            .ok_or_else(|| anyhow::anyhow!("unexpected adapter name {name}"))?;
        let (weight, side) = body
            .rsplit_once('.')
            .ok_or_else(|| anyhow::anyhow!("unexpected adapter name {name}"))?;
        let module = crate::lora::module_path_for_gguf(weight)
            .ok_or_else(|| anyhow::anyhow!("no peft module path for {weight}; cannot export it"))?;
        let v = trainer
            .params
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing {name}"))?;
        // In-graph: A [in, r], B [r, out]. peft: A [r, in], B [out, r].
        let (rows, cols) = if side == "a" {
            (v.len() / rank, rank)
        } else {
            (rank, v.len() / rank)
        };
        let mut t = vec![0.0f32; v.len()];
        for i in 0..rows {
            for j in 0..cols {
                t[j * rows + i] = v[i * cols + j];
            }
        }
        let key = format!(
            "base_model.model.{module}.lora_{}.weight",
            if side == "a" { "A" } else { "B" }
        );
        owned.push((key, vec![cols, rows], t));
    }

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = owned
        .iter()
        .map(|(k, shape, data)| {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(&data[..]),
                )
            };
            Ok((
                k.clone(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    shape.clone(),
                    bytes,
                )?,
            ))
        })
        .collect::<Result<_, safetensors::SafeTensorError>>()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let blob = safetensors::serialize(views, None).map_err(|e| anyhow::anyhow!("{e}"))?;
    std::fs::write(dir.join("adapter_model.safetensors"), blob)?;

    let modules: Vec<String> = trainer
        .adapter_names
        .iter()
        .filter_map(|n| n.strip_prefix("lora.").and_then(|b| b.rsplit_once('.')))
        .filter_map(|(w, _)| crate::lora::module_path_for_gguf(w))
        .filter_map(|m| m.rsplit('.').next().map(str::to_string))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let cfg = serde_json::json!({
        "peft_type": "LORA",
        "task_type": "FEATURE_EXTRACTION",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": false,
        "use_dora": false,
        "trainable_token_indices": serde_json::Value::Null,
        "target_modules": modules,
    });
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&cfg)?,
    )?;
    Ok(())
}

/// Read labelled records from JSONL — one object per line:
///
/// ```json
/// {"state": "...", "questions": {...}, "labels": {"department": "shipping"}}
/// ```
///
/// `labels` maps a question id to the **key** of the correct option
/// (`"shipping"`, `"true"`/`"false"` for `noul`, the level index as a string
/// for `score`), matching what the response reports. Keys are resolved through
/// [`crate::api::question_keys`], so a typo is an error rather than a silent
/// class 0.
pub fn load_jsonl(path: &std::path::Path) -> Result<Vec<TrainExample>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let ex = parse_labelled(line).with_context(|| format!("{}:{}", path.display(), ln + 1))?;
        out.push(ex);
    }
    if out.is_empty() {
        bail!("{}: no records", path.display());
    }
    Ok(out)
}

/// Parse one JSONL line into a labelled record.
pub fn parse_labelled(line: &str) -> Result<TrainExample> {
    let req = crate::api::SystemOneRequest::parse(line)?;
    let (mut rec, meta) = crate::api::to_record(&req)?;

    // `labels` lives alongside the request fields; pull it out with the
    // order-preserving parser so nothing is re-sorted.
    let doc = crate::json::Json::parse(line)?;
    let labels = doc
        .get("labels")
        .ok_or_else(|| anyhow::anyhow!("record has no `labels` object"))?;
    let pairs = labels
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("`labels` must be an object"))?;

    for (i, m) in meta.iter().enumerate() {
        let want = pairs
            .iter()
            .find(|(k, _)| *k == m.id)
            .map(|(_, v)| v)
            .ok_or_else(|| anyhow::anyhow!("no label for question {:?}", m.id))?;
        let key = want
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("label for {:?} must be a string", m.id))?;
        let idx = m.keys.iter().position(|k| k == key).ok_or_else(|| {
            anyhow::anyhow!(
                "label {key:?} is not an option of question {:?} (have {:?})",
                m.id,
                m.keys
            )
        })?;
        rec.questions[i].label = idx;
    }
    Ok(TrainExample { record: rec })
}

/// Group examples into shape-homogeneous batches under a token budget.
///
/// A single global shape does not work here. `decision-v7` mixes 2-option
/// yes/no questions against banking77's 77 intents, and short states against
/// 384-token ones, so the max over the split is `rows x 832 x 78 options` —
/// which asks rlx for a 68 GB activation arena. (The reference gets away with
/// per-batch shapes because PyTorch is dynamic, and still needed gradient
/// checkpointing at this size.)
///
/// So examples are sorted by shape and packed into batches whose
/// `rows x seq` stays under `token_budget`, with one compiled graph per
/// distinct batch shape. Sorting keeps the number of recompiles small; it also
/// means the data order is **not** the reference's shuffled order, which is a
/// real deviation from the trajectory even though the data itself is identical.
///
/// Returns `(shape, example indices)` per batch.
pub fn shape_batches<T: TokenizerLike>(
    encoder: &Encoder<T>,
    examples: &[TrainExample],
    encode_opts: crate::encode::EncodeOpts,
    max_rows: usize,
    token_budget: usize,
) -> Result<Vec<(TrainShape, Vec<usize>)>> {
    let bucket = crate::model::SHAPE_BUCKET;
    // Per example: (rows needed, padded seq, options).
    let mut sized: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(examples.len());
    for (i, ex) in examples.iter().enumerate() {
        let enc = encoder.encode(&ex.record, encode_opts)?;
        let (state_ids, _, rows) = rows_of(&enc)?;
        let seq = rows
            .iter()
            .map(|r| state_ids.len() + r.ids.len())
            .max()
            .unwrap_or(0)
            .div_ceil(bucket)
            .max(1)
            * bucket;
        // Quantize the option count as well as the sequence length. Exact
        // counts give ~39 distinct shapes over 500 examples, and every distinct
        // shape costs a graph rebuild plus a full reload of the base weights.
        // Powers of two collapse that to a handful.
        let opts = rows
            .iter()
            .map(|r| r.opts.len())
            .max()
            .unwrap_or(1)
            .max(2)
            .next_power_of_two();
        sized.push((seq, opts, rows.len(), i));
    }
    // Sort by shape so a batch is homogeneous and recompiles are rare.
    sized.sort_unstable();

    let mut out: Vec<(TrainShape, Vec<usize>)> = Vec::new();
    let mut i = 0usize;
    while i < sized.len() {
        let (seq, opts, _, _) = sized[i];
        // All examples sharing this exact (seq, opts) shape.
        let mut j = i;
        while j < sized.len() && sized[j].0 == seq && sized[j].1 == opts {
            j += 1;
        }
        // Rows are padded to a constant per shape so a partial batch does not
        // become a new shape (each one costs a rebuild plus a full reload of
        // the base weights). The cap therefore has to respect the budget at
        // the PADDED row count, not the real one — checking it against the
        // real count is how an 832-token batch became 8x832 and asked for a
        // 45 GB arena.
        let rows_cap = max_rows.min(token_budget / seq.max(1)).max(1);
        let mut k = i;
        while k < j {
            let mut batch = Vec::new();
            let mut used = 0usize;
            while k < j && used + sized[k].2 <= rows_cap {
                used += sized[k].2;
                batch.push(sized[k].3);
                k += 1;
            }
            if batch.is_empty() {
                // One example already exceeds the cap: give it its own batch
                // at exactly the rows it needs.
                let rows = sized[k].2;
                out.push((
                    TrainShape {
                        rows,
                        seq,
                        max_options: opts,
                    },
                    vec![sized[k].3],
                ));
                k += 1;
                continue;
            }
            out.push((
                TrainShape {
                    rows: rows_cap,
                    seq,
                    max_options: opts,
                },
                batch,
            ));
        }
        i = j;
    }
    Ok(out)
}

/// Randomize batch order while keeping same-shape batches contiguous.
///
/// Shape-sorted order is needed to keep graph rebuilds down, but taken
/// literally it is a terrible curriculum: every 2-option yes/no question comes
/// first and every 128-option banking77 question last, so the hard examples
/// all land where OneCycle has annealed the learning rate to ~0. Measured on a
/// 1200-record run the reported loss climbed 1.36 -> 6.68 over the final third
/// purely from that ordering.
///
/// Shuffling the *groups* (and the batches within each) costs nothing — the
/// rebuild count is the number of distinct shapes either way — and spreads
/// difficulty across the schedule. It is still not the reference's per-record
/// shuffle.
pub fn shuffle_shape_groups(
    batches: Vec<(TrainShape, Vec<usize>)>,
    rng: &mut crate::pyrandom::PyRandom,
) -> Vec<(TrainShape, Vec<usize>)> {
    let mut groups: Vec<Vec<(TrainShape, Vec<usize>)>> = Vec::new();
    for b in batches {
        match groups.last_mut() {
            Some(g) if g[0].0 == b.0 => g.push(b),
            _ => groups.push(vec![b]),
        }
    }
    for g in groups.iter_mut() {
        rng.shuffle(g);
    }
    rng.shuffle(&mut groups);
    groups.into_iter().flatten().collect()
}

/// A shape that fits every record in `examples`, rounded to the bucket.
pub fn shape_for<T: TokenizerLike>(
    encoder: &Encoder<T>,
    examples: &[TrainExample],
    encode_opts: crate::encode::EncodeOpts,
    rows_per_step: usize,
) -> Result<TrainShape> {
    let mut seq = 0usize;
    let mut max_options = 0usize;
    for ex in examples {
        let enc = encoder.encode(&ex.record, encode_opts)?;
        let (state_ids, _, branch_rows) = rows_of(&enc)?;
        for br in &branch_rows {
            seq = seq.max(state_ids.len() + br.ids.len());
            max_options = max_options.max(br.opts.len());
        }
    }
    let bucket = crate::model::SHAPE_BUCKET;
    Ok(TrainShape {
        rows: rows_per_step.max(1),
        seq: seq.div_ceil(bucket).max(1) * bucket,
        max_options: max_options.max(1),
    })
}

/// Export the adapter + head into `dir` without ever leaving a torn checkpoint.
///
/// A periodic checkpoint that writes straight into the output directory is
/// worse than none during the write window: the run is killed mid-file and now
/// both the new checkpoint AND the previous one are unusable. So build the
/// whole set in a staging subdirectory, then `rename` each file into place.
/// Rename is atomic within a filesystem and staging is a child of `dir`, so a
/// kill at any instant leaves either the old file or the new one.
pub fn checkpoint_atomically(
    trainer: &Trainer,
    rank: usize,
    alpha: f32,
    meta: &crate::checkpoint::Meta,
    dir: &std::path::Path,
) -> anyhow::Result<()> {
    let staging = dir.join(".staging");
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }
    export_adapter(trainer, rank, alpha, &staging)?;
    export_head(trainer, meta, &staging)?;
    for entry in std::fs::read_dir(&staging)? {
        let entry = entry?;
        std::fs::rename(entry.path(), dir.join(entry.file_name()))?;
    }
    std::fs::remove_dir_all(&staging)?;
    Ok(())
}

/// Write the pointer head as `head.safetensors` + `head.json`.
///
/// Not `head.pt`: writing a torch pickle to be read back by our own native
/// reader would be a pointless round trip. [`crate::checkpoint::Checkpoint`]
/// accepts either form, so a checkpoint trained here loads for inference
/// directly.
///
/// Tensors use `nn.Linear`'s `[dp, d]` orientation, matching `head.pt`.
pub fn export_head(
    trainer: &Trainer,
    meta: &crate::checkpoint::Meta,
    dir: &std::path::Path,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let head = trainer.pointer_head(meta.temperature)?;
    let (dp, d) = (head.head_dim, head.hidden);
    let owned: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
        ("head.q.weight".into(), vec![dp, d], head.q_weight.clone()),
        ("head.q.bias".into(), vec![dp], head.q_bias.clone()),
        ("head.k.weight".into(), vec![dp, d], head.k_weight.clone()),
        ("head.k.bias".into(), vec![dp], head.k_bias.clone()),
    ];
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = owned
        .iter()
        .map(|(k, shape, data)| {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(&data[..]),
                )
            };
            Ok((
                k.clone(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    shape.clone(),
                    bytes,
                )?,
            ))
        })
        .collect::<Result<_, safetensors::SafeTensorError>>()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let blob = safetensors::serialize(views, None).map_err(|e| anyhow::anyhow!("{e}"))?;
    std::fs::write(dir.join("head.safetensors"), blob)?;
    std::fs::write(
        dir.join("head.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "base": meta.base,
            "base_revision": meta.base_revision,
            "lora": meta.lora,
            "head_dim": meta.head_dim,
            "option_isolation": meta.option_isolation,
            "special_embeddings": meta.special_embeddings,
            "weights_dtype": meta.weights_dtype,
            "temperature": meta.temperature,
        }))?,
    )?;
    Ok(())
}

/// The released recipe, end to end.
///
/// Reproduces `jaredpalmer/kev-*`'s training loop over a frozen suite:
/// per-epoch augmentation seeded per record, optional none-of-the-above
/// minimal pairs, synthetic oversampling, AdamW with OneCycle, global-norm
/// clipping and gradient accumulation.
///
/// Deviations from the published run, all of them things rlx does not yet
/// have rather than choices:
///
/// * **no LoRA dropout** (0.05 in the recipe) — `rlx_tune::inject_lora`
///   ignores `LoraSpec::dropout`;
/// * **f32 throughout**, where the recipe used bf16 autocast with fp32 master
///   weights and gradient checkpointing;
/// * **no soft-target loss**. `decision-v7` contains none (checked: zero
///   `target` fields), so this only matters for the later `night2` delta data.
#[derive(Debug, Clone)]
pub struct RecipeConfig {
    pub train: TrainConfig,
    pub epochs: usize,
    /// Records per micro-batch (`--batch`).
    pub records_per_step: usize,
    /// Micro-batches per optimizer step (`--accum`).
    pub accum: usize,
    pub p_none: f64,
    pub p_none_distract: f64,
    pub p_distract: f64,
    /// Fraction of records that also emit a none-present/none-absent pair.
    pub p_none_pair: f64,
    /// Oversample the synthetic policy sources this many times per epoch.
    pub synthetic_repeat: usize,
    pub one_cycle_pct_start: f32,
    /// Run seed; `source_seed(seed, "{epoch}:{id}")` seeds each record.
    pub seed: u64,
    /// Question rows the graph compiles for.
    pub rows: usize,
}

impl Default for RecipeConfig {
    fn default() -> Self {
        // jaredpalmer/kev-4b's own training_config.json.
        Self {
            train: TrainConfig::default(),
            epochs: 1,
            records_per_step: 4,
            accum: 2,
            p_none: 0.1,
            p_none_distract: 0.12,
            p_distract: 0.15,
            p_none_pair: 0.25,
            synthetic_repeat: 1,
            one_cycle_pct_start: 0.1,
            seed: 1,
            rows: 8,
        }
    }
}

/// One epoch's worth of augmented, materialized examples, in training order.
///
/// Built up front so the step loop has a fixed length (OneCycle needs to know
/// `total_steps`) and so the augmentation RNG is driven exactly once per
/// record per epoch, as in the reference.
pub fn epoch_examples(
    records: &[crate::recipe::LabelledRecord],
    cfg: &RecipeConfig,
    epoch: usize,
) -> Result<Vec<TrainExample>> {
    use crate::pyrandom::PyRandom;
    use crate::recipe::{augment, materialize, none_pair};

    let mut out = Vec::with_capacity(records.len());
    for rec in records {
        let mut rng = PyRandom::from_source_seed(cfg.seed, &format!("{epoch}:{}", rec.id));
        let aug = augment(
            rec,
            &mut rng,
            cfg.p_none,
            cfg.p_none_distract,
            cfg.p_distract,
        )?;
        out.push(TrainExample {
            record: materialize(&aug)?.record,
        });
        // `p_none_pair` draws from the SAME generator, after `augment` — the
        // order matters, so this must stay in this position.
        if cfg.p_none_pair > 0.0 && rng.random() < cfg.p_none_pair {
            for p in none_pair(rec, &mut rng)? {
                out.push(TrainExample {
                    record: materialize(&p)?.record,
                });
            }
        }
    }
    Ok(out)
}

/// Apply `--synthetic_repeat`: append extra copies of the synthetic pools.
pub fn with_synthetic_repeat(
    records: &[crate::recipe::LabelledRecord],
    repeat: usize,
) -> Vec<crate::recipe::LabelledRecord> {
    let mut out = records.to_vec();
    for _ in 1..repeat.max(1) {
        out.extend(
            records
                .iter()
                .filter(|r| crate::recipe::is_synthetic(&r.source))
                .cloned(),
        );
    }
    out
}
