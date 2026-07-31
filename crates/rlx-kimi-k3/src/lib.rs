// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! # rlx-kimi-k3 — Kimi-K3 (Moonshot AI) for RLX
//!
//! Kimi-K3 is a multimodal MoE model whose text backbone (`KimiLinear`) interleaves
//! two attention mechanisms across its 93 layers:
//!
//! * **KDA** (Kimi Delta Attention) — a gated delta-net linear attention with a
//!   short causal conv on q/k/v, L2-normed keys, and a gated-RMSNorm output. Runs
//!   on RLX's [`rlx_ir::op::Op::GatedDeltaNet`].
//! * **MLA** (NoPE) — DeepSeek-style multi-head latent attention with **no** rotary
//!   embedding (`mla_use_nope`) and an optional sigmoid output gate.
//!
//! The FFN is a **LatentMoE**: 896 routed experts (16 active, MXFP4-packed) with a
//! shared latent up/down projection, plus 2 always-on shared experts, gated by a
//! sigmoid `noaux_tc` grouped-topk router. Activation is the custom **situ** GLU.
//!
//! Layers are stitched with **Attention Residuals** (a block-residual refreshed
//! every `attn_res_block_size` layers). A ViT vision tower + patch-merger projector
//! feed image/video embeddings into the text stream (multimodal wrapper).

pub mod common;
pub mod config;
pub mod flow;
pub mod kda;
pub mod mla;
pub mod moe;
pub mod vision;
pub mod wrapper;

pub use config::{KimiK3Config, KimiLinearConfig, KimiVisionConfig, LinearAttnConfig};
