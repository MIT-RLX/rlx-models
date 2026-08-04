// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Unified LLM benchmark harness for RLX.
//!
//! One driver, three dimensions, model-agnostic over
//! [`rlx_runtime::lm::LmRunner`] — the seam every rlx LM crate already
//! implements — so a new model joins the leaderboard by adding one small
//! [`adapters`] entry, not a new harness:
//!
//! - **speed** ([`speed`]) — prefill & decode tok/s, time-to-first-token, peak
//!   RSS.
//! - **quality** ([`quality`]) — MMLU (multiple-choice log-likelihood) and
//!   GSM8K (generative exact-match), reusing [`rlx_eval`]'s scoring primitives.
//! - **parity** ([`parity`]) — argmax agreement + logit cosine against a
//!   reference dump (mlx-lm / HuggingFace).
//!
//! Every dimension emits both a machine-readable `LLMBENCH …` line and a row
//! for the markdown [`report`] leaderboard.
//!
//! ```no_run
//! use rlx_llm_bench::{BenchModel, speed, quality};
//! # fn demo(mut model: BenchModel) -> anyhow::Result<()> {
//! let sp = speed::run_speed(&mut model, &speed::SpeedConfig::default())?;
//! println!("{} decode tok/s", sp.decode_toks_s);
//! # Ok(()) }
//! ```

pub mod adapters;
pub mod metrics;
pub mod mock;
pub mod model;
pub mod parity;
pub mod quality;
pub mod report;
pub mod speed;

pub use model::BenchModel;
pub use parity::{ParityResult, ReferenceDump, run_parity};
pub use quality::{Gsm8kResult, MmluResult, QualityRow};
pub use report::{BenchRow, Report};
pub use speed::{SpeedConfig, SpeedResult};

// Re-export the reused rlx-eval data types so downstream code has one import
// surface for the whole harness.
pub use rlx_eval::{McItem, McResult, PerplexityConfig};
