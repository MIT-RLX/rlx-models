// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Convenience re-exports for `rlx-kimi-k3` users.
//!
//! ```rust,ignore
//! use rlx_kimi_k3::prelude::*;
//!
//! // Config + weights straight off a HuggingFace-style checkpoint dir:
//! let cfg = KimiK3Config::load("/path/to/kimi-k3/config.json")?;
//! let mut loader = CheckpointLoader::open("/path/to/kimi-k3")?;
//!
//! // Compile the text backbone and generate:
//! let flow = build_kimi_text_flow(&cfg.text, &fcfg, &weights)?;
//! let ids = run_generate(&mut compiled, &prompt, &cfg.text, &fcfg, Device::Metal)?;
//! ```
//!
//! `rlx-kimi-k3` exposes the model as composable pieces rather than a single
//! facade, so the prelude gathers the whole toolkit: config + checkpoint I/O,
//! the text-flow graph builders, the generation/decode entry points, the
//! per-layer building blocks (KDA / MLA / LatentMoE), the ViT + multimodal
//! fusion path, and the quantization knobs.

// ── Config + architecture ────────────────────────────────────────
pub use crate::config::{KimiK3Config, KimiLinearConfig, KimiVisionConfig, LinearAttnConfig};

// ── Checkpoint I/O (safetensors + MXFP4 experts) ──────────────────
pub use crate::loader::{
    CheckpointLoader, ExpertPacked, load_expert_ranges, load_expert_ranges_packed, warm_ranges,
};

// ── Text backbone: graph builders + weight containers ─────────────
pub use crate::flow::{
    AttnWeights, FfnWeights, FlowConfig, FlowWeights, LayerWeights, build_head,
    build_kimi_text_flow, build_kimi_text_stage, build_layer_decode_step, build_layer_pre_ffn,
};

// ── Generation + incremental decode ──────────────────────────────
pub use crate::runner::{
    DecodeState, apply_head, apply_head_cached, decode_forward, decode_forward_range,
    load_layer_backbone, run_generate, run_moe_paged, run_prefix_logits, run_speculative_generate,
};

// ── Layer building blocks: KDA (linear attn) ─────────────────────
pub use crate::kda::{KdaDims, KdaWeights, build_kda_decode_step, build_kda_layer};

// ── Layer building blocks: MLA (NoPE latent attn) ────────────────
pub use crate::mla::{MlaDims, MlaWeights, build_mla_decode_step, build_mla_layer};

// ── Layer building blocks: LatentMoE FFN ─────────────────────────
pub use crate::moe::{
    DenseMlpWeights, MoeDims, MoeWeights, build_dense_mlp, build_latent_moe,
    build_moe_experts_paged, build_moe_experts_paged_packed, build_moe_route,
};

// ── Vision tower + multimodal fusion ─────────────────────────────
pub use crate::vision::{
    VisionBlockWeights, VisionDims, VisionWeights, build_vision, patch_embed, vision_rope_2d,
};
pub use crate::wrapper::merge_text_and_vision_embds;

// ── Quantization policy + helpers ────────────────────────────────
pub use crate::common::{
    QuantPolicy, WeightQuant, fake_quant_weight, is_quant_hotspot, quant_policy, resolve_quant,
    weight_quant,
};

// ── Devices (so callers don't need to import upstream) ────────────
pub use rlx_runtime::Device;
