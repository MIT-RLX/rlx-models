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

//! # rlx-diffusiongemma — DiffusionGemma for RLX
//!
//! [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
//! is a **discrete text diffusion** model: 25.2 B total parameters, 3.8 B active,
//! that denoises a whole block of tokens at once instead of emitting them one at
//! a time.
//!
//! ## Shape of the model
//!
//! The backbone is Gemma 4's MoE stack — 30 layers, 128 experts with 8 active
//! plus one always-on shared expert, a 5:1 sliding:full attention pattern — but
//! it is wired as an **encoder/decoder pair that shares all its weights**:
//!
//! * The **encoder** runs causally over the prompt and leaves a KV cache behind.
//! * The **decoder** ("denoiser") holds a fixed 256-token *canvas* and attends
//!   bidirectionally over `[cache ; canvas]`. It never writes to the cache, so
//!   the same cache is reused across every denoising step of a block.
//!
//! The only weight the two stacks do not share is the per-layer `layer_scalar`.
//!
//! Three details are easy to miss and each one silently corrupts output:
//!
//! * Full-attention layers have **no `v_proj`** — V is the *pre-`k_norm`* K
//!   projection — and use a different geometry from sliding layers (16×512 with
//!   2 KV heads vs 16×256 with 8).
//! * Attention `scaling` is **1.0**, not `1/sqrt(head_dim)`, because Q and K are
//!   RMS-normed per head.
//! * The FFN is a **two-branch** block: the router scores the raw residual while
//!   the experts consume a separately normalized copy. See [`layer`].
//!
//! The vision tower shares the `scaling = 1.0` convention — see [`vision`].
//!
//! ## Generation
//!
//! Each block starts from a canvas of uniform random tokens and is denoised for
//! up to `max_denoising_steps`. Every step the model rescores the whole canvas;
//! the [`sampler`] accepts the lowest-entropy positions whose joint mutual
//! information stays under `entropy_bound`, re-noises the rest, and feeds the
//! step's soft embeddings back in as a self-conditioning signal. That is what
//! yields ~15–20 tokens per forward pass.
//!
//! ## Modules
//!
//! | module | role |
//! |---|---|
//! | [`config`] | `config.json` / `generation_config.json`, and the per-layer geometry helpers |
//! | [`attention`], [`moe`], [`layer`] | one transformer layer: attention, the 128-expert MoE, the two-branch FFN |
//! | [`flow`] | the two graphs — causal encoder (taps K/V) and bidirectional denoiser |
//! | [`vision`] | Gemma 4's `gemma4_vision` tower and the projector into LM space |
//! | [`preprocess`] | `Gemma4ImageProcessor`: aspect-preserving resize, patchify, pad |
//! | [`prompt`] | the canonical chat template and image-token expansion |
//! | [`sampler`], [`generate`] | entropy-bounded acceptance and the block-diffusion loop |
//! | [`weights`] | checkpoint adaptation (expert banks → `GroupedMatMul` layout) |
//!
//! ## Usage
//!
//! ```no_run
//! use rlx_core::flow_util::compile_built;
//! use rlx_core::weight_map::WeightMap;
//! use rlx_diffusiongemma::{
//!     DiffusionGemmaConfig, EncoderCacheLens, build_decoder_flow, build_encoder_flow,
//!     prepare_checkpoint,
//! };
//! use rlx_runtime::Device;
//! # fn main() -> anyhow::Result<()> {
//! let dir = std::path::Path::new("/weights/diffusiongemma-26B-A4B-it");
//! let cfg = DiffusionGemmaConfig::from_file(dir.join("config.json"))?;
//! let mut wm = WeightMap::from_safetensors_dir(dir)?;
//! prepare_checkpoint(&cfg, &mut wm)?;
//!
//! // Both stacks read the same tensors, so the builders borrow rather than
//! // consume the map.
//! let prompt_len = 32;
//! let encoder = build_encoder_flow(&cfg, &wm, prompt_len)?;
//! let cache = EncoderCacheLens::for_prompt(&cfg.text_config, prompt_len);
//! let decoder = build_decoder_flow(&cfg, &wm, cfg.canvas_length, cache)?;
//!
//! let mut enc = compile_built(encoder, Device::Cpu)?;
//! let mut dec = compile_built(decoder, Device::Cpu)?;
//! # let _ = (&mut enc, &mut dec);
//! # Ok(())
//! # }
//! ```
//!
//! [`BlockDiffusion`] drives the denoising loop over that pair, including the
//! encoder re-run that appends each finished canvas to the context. For images,
//! [`preprocess_image`] → [`build_vision_flow`] → [`merge_multimodal_embeds`]
//! → [`build_encoder_flow_embeds`] replaces the token-id entry point.
//! [`DecoderOutputs::Reduced`] moves the per-step entropy / argmax / sampling
//! reduction into the graph, so the `[canvas, vocab]` logit block never crosses
//! the device boundary.
//!
//! The model is **text + images only** — the reference raises
//! `NotImplementedError` for audio and video.
//!
//! ## Status
//!
//! Both graphs, the vision tower, the image processor, the chat template, the
//! checkpoint adapter and the sampler are implemented and covered by tests.
//! Numeric parity is checked at three levels: against a PyTorch transcription of
//! `modeling_diffusion_gemma.py` at tiny sizes, against the real checkpoint's
//! shard headers for every tensor name and shape (no download), and against
//! torch on the **real trained weights** for the vision tower and one text
//! layer, on CPU and Metal. See the crate README for the measured numbers.
//!
//! Not done: a whole-model forward pass. That is a *format* problem rather than
//! a correctness one — as f32 the routed experts alone are 91.4 GB, but a
//! precision sweep on real weights puts Q8_0 at 24.3 GB and Q4_0 at 12.8 GB for
//! ~1e-3 of cosine, so what is missing is a packed loader feeding
//! [`rlx_ir::op::Op::DequantGroupedMatMul`].

pub mod attention;
pub mod config;
pub mod flow;
pub mod generate;
pub mod layer;
pub mod moe;
pub mod preprocess;
pub mod prompt;
pub mod sampler;
pub mod vision;
pub mod weights;

pub use config::{
    DiffusionGemmaConfig, DiffusionGenerationConfig, LayerType, RopeKind, TextConfig, VisionConfig,
};
pub use flow::{
    ARGMAX_OUTPUT, CANVAS_INPUT, DecoderOutputs, EMBED_KEY, ENTROPY_OUTPUT, EncoderCacheLens,
    INPUTS_EMBEDS_INPUT, SAMPLED_OUTPUT, SC_SIGNAL_INPUT, SOFT_EMBED_OUTPUT, TEMPERATURE_INPUT,
    build_decoder_flow, build_decoder_flow_with, build_encoder_flow, build_encoder_flow_embeds,
    enc_k_name, enc_v_name,
};
pub use generate::{BlockDiffusion, DenoiseOutcome};
pub use preprocess::{ImagePreprocessConfig, PreprocessedImage, preprocess_image};
pub use prompt::{ChatMessage, ChatOptions, ContentPart, Role, format_chat};
pub use sampler::{EntropyBoundSampler, Rng, StableAndConfident, StepScores};
pub use vision::{
    build_vision_flow, grid_positions, merge_multimodal_embeds, vision_pool_matrix,
    vision_rope_tables,
};
pub use weights::prepare_checkpoint;

/// HuggingFace repo this crate targets.
pub const HF_MODEL_ID: &str = "google/diffusiongemma-26B-A4B-it";
/// `config.json` `model_type` this crate claims.
pub const MODEL_TYPE: &str = "diffusion_gemma";
/// `text_config.model_type`.
pub const TEXT_MODEL_TYPE: &str = "diffusion_gemma_text";
