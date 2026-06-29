// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Generic LoRA / DoRA fine-tuning for RLX models.
//!
//! This crate provides the host-side, model-agnostic pieces of fine-tuning:
//!   - [`adapter`] — LoRA/DoRA specs and the host-side [`adapter::fuse_lora`]
//!     merge (the forward pass uses rlx's first-class `LoraMatMul` op, which
//!     lowers on every backend).
//!   - [`dataset`] — text / chat / completions JSONL loaders with prompt
//!     masking (mirrors mlx-lm's `tuner/datasets.py`).
//!
//! The graph-rewrite injection (`inject_lora` over a model forward graph) and
//! the `Trainer` loop (compile `grad_with_loss` once, accumulate, step via
//! `rlx-optim`) build on these and on `rlx-autodiff` / `rlx-optim`.

pub mod adapter;
pub mod dataset;
pub mod distributed;
pub mod dwq;
pub mod inject;
pub mod loss;
pub mod trainer;

pub use adapter::{DoraSpec, LoraInit, LoraSpec, column_norms, fuse_dora, fuse_lora};
pub use dataset::{
    Example, Tokenized, Turn, load_jsonl, parse_line, tokenize_completion, tokenize_text,
};
pub use distributed::{GradComm, RdmaGradComm};
pub use dwq::{DwqResult, dwq_heal_linear};
pub use inject::{AdapterParam, FuseMode, inject_dora, inject_lora};
pub use loss::{cross_entropy_masked, dpo_loss_from_margin};
pub use trainer::{Adam, Optimizer, ParamSlot, lora_linear, train};
