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

//! # rlx-ling — Ling 3.0 / BailingMoeV3 (inclusionAI) for RLX
//!
//! [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) is a
//! `bailing_hybrid` decoder: ~7.9 B total parameters, ~0.6 B active. Every layer
//! is MoE except the first, and attention alternates between two mechanisms:
//!
//! * **KDA** — Kimi Delta Attention, a gated delta-net linear attention with a
//!   4-tap causal short conv on q/k/v, L2-normed q/k, a per-channel log-decay
//!   gate and a gated output RMSNorm. 18 of 24 layers. Rides
//!   [`rlx_ir::op::Op::GatedDeltaNet`] with `gate_per_channel`.
//! * **MLA** — DeepSeek-style multi-head latent attention (low-rank Q and KV, one
//!   decoupled interleaved-RoPE head) plus a Bailing-specific sigmoid output
//!   gate. Every `layer_group_size`-th layer: 3, 7, 11, 15, 19, 23.
//!
//! The FFN is the fine-grained `noaux_tc` MoE — 128 routed experts, 8 active,
//! grouped top-k over 8 groups, one always-on shared expert — shared verbatim
//! with [`rlx_deepseek::moe`], which in turn reuses rlx-llada2's
//! `group_limited_gate` op.
//!
//! ## Usage
//!
//! ```no_run
//! use rlx_core::flow_util::compile_built;
//! use rlx_core::weight_map::WeightMap;
//! use rlx_ling::{LingConfig, build_ling_text_flow, prepare_checkpoint};
//! use rlx_runtime::Device;
//! # fn main() -> anyhow::Result<()> {
//! let dir = std::path::Path::new("/weights/Ling-3.0-tiny");
//! let cfg = LingConfig::from_file(dir.join("config.json"))?;
//! let mut wm = WeightMap::from_safetensors_dir(dir)?;
//! prepare_checkpoint(&cfg, &mut wm)?;
//!
//! let seq = 8;
//! let built = build_ling_text_flow(&cfg, &mut wm, seq, true)?;
//! let mut compiled = compile_built(built, Device::Cpu)?;
//!
//! let (cos, sin) = cfg.rope_tables(seq);
//! let ids: Vec<f32> = vec![1.0; seq];
//! let logits = compiled.run(&[
//!     ("input_ids", ids.as_slice()),
//!     ("rope_cos", cos.as_slice()),
//!     ("rope_sin", sin.as_slice()),
//! ]);
//! # let _ = logits;
//! # Ok(())
//! # }
//! ```
//!
//! ## Status
//!
//! Architecture is complete and verified against the published
//! `modeling_bailing_moe_v3.py`, the FLA kernels it calls, and every tensor shape
//! in the checkpoint index. Tests cover the graph end to end on synthetic
//! weights. A real-weight run has **not** been done: `WeightMap` is f32, so the
//! routed experts alone would be ~27.8 GB resident. That needs the paged/packed
//! expert path (as in `rlx_kimi_k3::moe`), which is not wired here yet.

pub mod config;
pub mod flow;
pub mod flow_decode;
pub mod kda;
pub mod mla;
pub mod quant;
pub mod streaming;
pub mod weights;

pub use config::{AttnGate, AttnKind, LingConfig};
pub use flow::{EMBED_KEY, build_ling_text_flow};
pub use flow_decode::{
    DecodeNames, DecodeSession, ScanState, build_ling_decode_flow, build_ling_decode_flow_with,
};
pub use streaming::{load_and_compile, load_without_experts};
pub use weights::prepare_checkpoint;

/// HuggingFace repo this crate targets.
pub const HF_MODEL_ID: &str = "inclusionAI/Ling-3.0-tiny";
/// `config.json` `model_type` this crate claims.
pub const MODEL_TYPE: &str = "bailing_hybrid";
