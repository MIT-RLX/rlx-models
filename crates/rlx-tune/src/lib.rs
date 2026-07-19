// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Generic LoRA / DoRA fine-tuning for RLX models — model-agnostic and
//! data-parallel out of the box.
//!
//! Host-side, model-agnostic pieces of fine-tuning:
//!   - [`adapter`] — LoRA/DoRA specs and the host-side [`adapter::fuse_lora`]
//!     merge (the forward pass uses rlx's first-class `LoraMatMul` op, which
//!     lowers on every backend).
//!   - [`dataset`] — text / chat / completions JSONL loaders with prompt
//!     masking (mirrors mlx-lm's `tuner/datasets.py`).
//!   - [`inject`] — graph-rewrite `inject_lora` over a model's forward graph.
//!
//! Training ([`trainer`]): [`train`] is the minimal loop; [`Trainer`] and
//! [`train_dp`] / [`train_dp_with`] add **data parallelism** with a single
//! [`DpConfig`] — ZeRO-1 optimizer-state sharding, comm/compute overlap,
//! mixed-precision reduction, global-norm clipping, LR schedules, gradient
//! accumulation, world-size-agnostic [`Checkpoint`]ing, and per-step timing.
//!
//! Distributed bring-up is zero-config: [`from_env`] builds the collective from
//! `RANK`/`WORLD` env vars, or [`cluster::launch_or_join`] self-spawns a local
//! `--nnodes N` cluster. Builds on `rlx-autodiff` / `rlx-optim` and
//! `rlx-driver`'s collectives.
//!
//! ```
//! # fn main() -> anyhow::Result<()> {
//! let comm = rlx_tune::from_env()?;                 // Some(..) iff WORLD > 1
//! let cfg = rlx_tune::DpConfig::new(2e-4).shard().overlap().bf16();
//! // then: train_dp(graph, &wrt, &mut params, &inputs, steps, comm.as_deref(), &cfg,
//! //                |m| println!("{m}"))?;   — see the `data_parallel` example
//! # let _ = (comm, cfg);
//! # Ok(()) }
//! ```

pub mod adapter;
pub mod cluster;
pub mod dataset;
pub mod distributed;
pub mod dwq;
pub mod inject;
pub mod loss;
pub mod resident;
pub mod trainer;

pub use adapter::{DoraSpec, LoraInit, LoraSpec, column_norms, fuse_dora, fuse_lora};
pub use dataset::{
    Example, Tokenized, Turn, load_jsonl, parse_line, tokenize_completion, tokenize_text,
};
pub use distributed::{GradComm, ProcessGroupGradComm, RdmaGradComm, ReduceDtype, from_env};
pub use dwq::{DwqResult, dwq_heal_linear};
pub use inject::{AdapterParam, FuseMode, inject_dora, inject_lora};
pub use loss::{cross_entropy_masked, dpo_loss_from_margin};
pub use resident::ResidentTrainer;
pub use rlx_runtime::Device;
pub use trainer::{
    Adam, AdamConfig, Checkpoint, DpConfig, LrSchedule, Optimizer, ParamSlot, StepMetrics, Trainer,
    lora_linear, train, train_dp, train_dp_with,
};
