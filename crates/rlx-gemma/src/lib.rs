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

//! Gemma / Gemma 2 / Gemma 3 / Gemma 4 causal LMs for RLX.
//!
//! See [`README.md`](https://github.com/anthropics/rlx-models/blob/main/crates/rlx-gemma/README.md)
//! for a tour. The short version:
//!
//! - [`GemmaConfig`] round-trips the HF `config.json` (including the
//!   nested `text_config` of the Gemma 4 unified layout) and exposes
//!   per-layer accessors (`layer_head_dim(i)`, `layer_n_rot(i)`,
//!   `layer_rope_theta(i)`, `is_full_attention_layer(i)`).
//! - [`GemmaFlow`] builds the prefill + decode graphs. The same flow
//!   covers every Gemma variant by reading the per-layer accessors:
//!   no separate Gemma-4 code path.
//! - [`GemmaRunner`] is the high-level text-only runner (compile,
//!   load weights, generate). Use [`GemmaMultimodalRunner`] for the
//!   Gemma 4 vision + audio pipeline.
//!
//! ## Module layout
//!
//! | Module | Purpose |
//! |---|---|
//! | [`config`] | `GemmaConfig`, arch detection, per-layer dispatch helpers |
//! | [`flow`] | Tier-0 prefill + decode graph assembly (uses `rlx-flow` blocks) |
//! | [`builder`] | Thin compile-time wrappers around [`flow`] for legacy call sites |
//! | [`generator`] | Cached single-batch generator (prefill → decode loop) |
//! | [`runner`] | High-level runner over a weights path / device |
//! | [`rope`] | RoPE inv-freq + cos/sin table construction |
//! | [`multimodal`] | Vision/audio projector HIR builders, image/WAV loaders, tokenizer placeholders |
//! | [`multimodal_runner`] | Compile + run the projectors on a device, splice into LM embeddings |
//! | [`cli`] | `rlx-gemma` binary entry point |

pub mod builder;
pub mod capabilities;
pub mod cli;
pub mod config;
pub mod flow;
pub mod gemma4_audio;
pub mod gemma4_e2b_mm;
pub mod gemma4_vision;
pub mod gemma_e2b;
pub mod generator;
pub mod multimodal;
pub mod multimodal_cli;
pub mod multimodal_embed;
pub mod multimodal_flow;
pub mod multimodal_mask;
pub mod multimodal_runner;
pub mod packed_session;
pub mod prelude;
pub mod prompt;
pub mod qat;
pub mod qat_loader;
pub mod rope;
pub mod runner;
pub mod unified_preprocess;
pub mod unified_projector;

// ── Graph builders (thin wrappers over `flow`) ─────────────────────
pub use builder::{
    PackedDecodeLmOutput, build_gemma_decode_graph_sized, build_gemma_decode_graph_sized_ext,
    build_gemma_decode_graph_sized_packed, build_gemma_decode_graph_sized_packed_ext,
    build_gemma_decode_hir_dynamic_ext, build_gemma_decode_hir_sized,
    build_gemma_decode_hir_sized_ext, build_gemma_graph_sized, build_gemma_graph_sized_last_logits,
    build_gemma_graph_sized_packed, build_gemma_graph_sized_packed_ext,
    build_gemma_prefill_hir_dynamic_ext, drain_gemma_packed_weights,
    precompute_packed_decode_tied_lm_head,
};

// ── Config + GGUF / HF parsing ────────────────────────────────────
pub use config::{GemmaArch, GemmaConfig, gemma_cfg_from_gguf};
pub use gemma_e2b::{compile_e2b_prefill, resolve_e2b_device, with_e2b_metal_guard};

// ── Flow assembly (tier-0 reference, supports every Gemma variant) ─
pub use flow::{
    GEMMA_PROFILE_FILE, GemmaDecodeOpts, GemmaFlow, GemmaMode, GemmaPrefillOpts,
    build_gemma_decode_built, build_gemma_decode_flow, build_gemma_decode_graph,
    build_gemma_prefill_built, build_gemma_prefill_flow, gemma_profile_near_weights,
};

// ── Generator + runner (high-level text inference) ────────────────
pub use generator::{GemmaGenerator, decode_profile_for_device};
pub use packed_session::prefill_bucket_len;
pub use runner::{GemmaConfigSource, GemmaRunner, GemmaRunnerBuilder};

// ── Tokenizer re-exports (shared with rlx-qwen35) ─────────────────
pub use prompt::{decode_token_auto, encode_chat_prompt_auto};
pub use rlx_qwen35::{decode_ids_auto, encode_prompt, encode_prompt_auto, resolve_tokenizer_path};

// ── Multimodal: configs + builders + preprocessing ────────────────
pub use multimodal::{
    AUDIO_MARKER, AudioProjectionInputs, GemmaAudioConfig, GemmaMultimodalConfig,
    GemmaVisionConfig, IMAGE_MARKER, ImageNormalize, MediaSlot, ProjectionGraph,
    VisionProjectionInputs, VisionProjectionLearnedQueriesInputs, build_audio_projection_graph,
    build_audio_projection_hir, build_vision_projection_graph, build_vision_projection_hir,
    build_vision_projection_learned_queries_graph, build_vision_projection_learned_queries_hir,
    expand_media_placeholders, extract_image_patches, extract_image_patches_normalized,
    frame_audio_samples, fuse_multimodal_embeddings, load_image_patches,
    load_image_patches_normalized, load_wav_mono_16khz, parse_wav_16khz_mono, resample_linear,
    tokenize_with_media,
};

// ── Multimodal runner (compile + run projectors on a device) ──────
pub use multimodal_embed::{build_multimodal_inputs_embeds, embed_token_ids_scaled};
pub use multimodal_runner::{GemmaMultimodalRunner, MultimodalWeights, ProjectorLayout};
pub use unified_preprocess::{UnifiedImageBatch, load_unified_image};
pub use unified_projector::{
    build_unified_audio_graph, build_unified_vision_graph, is_unified_vision_weights,
};

// ── Multimodal CLI entry point ────────────────────────────────────
pub use multimodal_cli::run as run_multimodal_cli;

#[cfg(feature = "parity-llama")]
pub use rlx_qwen35::llama_reference;
