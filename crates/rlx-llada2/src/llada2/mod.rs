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

//! LLaDA2 MoE diffusion language model (TIDE reference: `/Users/Shared/TIDE`).
//!
//! - [`LLaDA2MoeConfig`] — HF / TIDE `config.json`
//! - [`build_llada2_forward_graph`] — bidirectional attention + MoE FFN
//! - [`LLaDA2Runner`] — multi-backend forward + [`generate`] + TIDE offload
//!   (standard backends: CPU, Metal, MLX, CUDA, ROCm, WGPU, Vulkan)
//!
//! ## PyTorch parity checklist
//!
//! | Component | Status |
//! |-----------|--------|
//! | RMSNorm, fused QKV, QK-norm, partial RoPE | Graph |
//! | Bidirectional attention + `head_dim^-0.5` scale | [`Op::Attention`] B,H,S,D |
//! | Group-limited sigmoid gate + expert bias routing | [`gate_op`] |
//! | SwiGLU dense + MoE + shared expert | Graph |
//! | Block-diffusion mask + `generate` loop | [`generate`] |
//! | Temperature / top-k / top-p sampling | [`sampling`] |
//! | TIDE expert offload | [`moe_store`] + runtime pools |
//!
//! Validate against PyTorch:
//! - Component parity: `tests/llada2_numerical_parity.rs`
//! - Full e2e (weights): `LLADA2_MODEL_DIR=… cargo test --test llada2_e2e_parity`
//! - CLI: `cargo run -p rlx-models --example llada2_run -- --model-dir … --device metal`

pub mod builder;
pub mod capabilities;
pub mod compile_util;
pub mod config;
#[cfg(feature = "hf-download")]
pub mod download;
#[cfg(feature = "hf-download")]
pub use download::{DEFAULT_HF_REPO, download_llada2_mini};
pub mod gate;
pub mod gate_op;
pub mod generate;
pub mod load;
pub mod mask;
pub mod moe_offload;
pub mod moe_store;
pub mod rope;
pub mod runner;
pub mod sampling;
pub mod synth;
pub mod weights;

pub use builder::build_llada2_forward_graph;
pub use capabilities::{default_memory_budget_bytes, validate_device};
pub use compile_util::{compile_llada2_built, llada2_profile};
pub use config::LLaDA2MoeConfig;
pub use generate::{GenerateConfig, GenerateForward, generate};
pub use load::{load_llada2_from_dir, load_llada2_partial};
pub use mask::block_diffusion_attention_mask;
pub use moe_store::{
    apply_moe_store_to_compiled, build_moe_expert_store, moe_host_bind_from_store,
    moe_layer_indices,
};
pub use runner::{LLaDA2Runner, LLaDA2RunnerBuilder, LLaDA2RunnerForward};
pub use sampling::sample_logits;
pub use weights::LLaDA2Weights;
