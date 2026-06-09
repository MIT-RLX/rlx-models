// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// TIDE parity layer for [`crate::qwen35`] MoE runners.
// Reference: `/Users/Shared/TIDE` (ims-kdks/TIDE) — `model/modeling_llada2_moe.py`.

//! # TIDE (MoE diffusion-LLM expert offload)
//!
//! Mirrors the public API of [TIDE](https://github.com/ims-kdks/TIDE):
//!
//! - [`PredictiveOffloadInfo`] / [`enable_predictive_expert_offload`]
//! - [`refresh_experts`] (`jump_steps` τ, prefill-block refresh)
//! - [`TideOffloadStats`] / [`aggregate_offload_stats`]
//! - [`BlockDenoiseConfig`] + [`num_transfer_tokens_schedule`] (LLaDA2 `generate`)
//!
//! RLX wires this to [`crate::qwen35::Qwen35Runner`] (MoE AR) and [`TideRunner`] /
//! [`crate::LLaDA2Runner`] (LLaDA2 block diffusion).

mod diffusion;
mod generate;
mod llada2_config;
mod moe_state;
mod offload;
mod refresh;
mod runner;
mod stats;

pub use diffusion::{
    BlockDenoiseConfig, BlockDenoiseLoop, BlockDiffusionForward, BlockForwardOutput,
};
pub use generate::{
    BlockDenoiseSampler, BlockDenoiseStepStats, DenoiseStepCtx, GenerateConfig, GenerateForward,
    generate, num_transfer_tokens_schedule, run_block_diffusion,
};
pub use llada2_config::LLaDA2MoeConfig;
pub use moe_state::MoeOffloadState;
pub use offload::{
    PredictiveOffloadInfo, PredictiveOffloadParams, device_memory_for_offload,
    enable_predictive_expert_offload, gpu_expert_budget_from_device_memory,
};
pub use refresh::refresh_experts;
pub use runner::{TideRunner, preview_predictive_offload};
pub use stats::{TideOffloadStats, aggregate_offload_stats};
