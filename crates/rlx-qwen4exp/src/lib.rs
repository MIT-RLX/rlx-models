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

//! # rlx-qwen4exp — Qwen3.8-Flash-Next (`qwen4exp`) for RLX
//!
//! [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is the
//! first checkpoint of the Qwen4 architecture — its own model card calls it
//! "A Preview of the Qwen4 Architecture". GGUF converters emit it as
//! `general.architecture = qwen4exp`; HF calls it
//! `Qwen4ExpForConditionalGeneration` / `model_type = qwen4_exp`.
//!
//! ## Architecture
//!
//! 48 decoder layers, `hidden = 2560`, `vocab = 248320`, `512 × 56 B`:
//!
//! | Piece | Where | Notes |
//! |---|---|---|
//! | [`hc`] — hyper-connections | every layer, twice, plus the output | 4 parallel residual streams; **replaces every layer norm** |
//! | [`gdn`] — GatedDeltaNet | 36 layers | Qwen3.5's linear attention, with a sigmoid output gate |
//! | [`attention`] — gated GQA | layers 3, 7, … 47 | 24 q / 2 kv heads × 256, partial M-RoPE over 64 dims |
//! | [`indexer`] — QSA | on every attention layer | top-512 of 4-token mean-pooled key blocks, plus each query's tail |
//! | [`moe`] — 512-expert MoE | every layer | 10 active + 1 shared, softmax top-k, `moe_intermediate_size = 640` |
//! | [`ple`] — n-gram embeddings | layer 1 only | 51 B hashed parameters; the bulk of the checkpoint |
//! | [`mtp`] — multi-token prediction | `blk.48`, its own GGUF | one draft layer fed by the trunk's residual |
//! | `vit` — the vision tower | `mmproj-*.gguf`, feature `vision` | a Qwen3-VL ViT: dual patch conv, axial 2-D RoPE, merge-block projector |
//! | `vision` — the multimodal seam | host-side, feature `vision` | `inputs_embeds` splice, the PLE image-token substitution, 3-D M-RoPE |
//! | [`decode`] — incremental decode | one token (or one chunk) per `run()` | KV cache + carried GatedDeltaNet / PLE state |
//! | [`runner`] — the ready-to-run loop | GGUF in, logits out | compile once, chunked prefill into one-token decode, [`rlx_runtime::LmRunner`] |
//! | [`ple::GgufPleRows`] — the n-gram table, lazily | 28 GB on disk, 9 rows per step | reads only the quantization blocks a row overlaps |
//! | [`paged_moe`] — experts off disk | 512 per layer, 10 fire | router graph first, then the fired slices into their arena slots; **quantized banks dequantize per expert** |
//!
//! Two structural surprises worth stating up front:
//!
//! * **There is no `output_norm`.** The final hyper-connection mixer
//!   (`output_hc_*`) carries it, and the checkpoint has no separate norm
//!   tensor — nor any `attn_norm` / `ffn_norm` per layer.
//! * **`per_layer_token_embd` is bigger than the rest of the model.**
//!   `[160, 320001536]`, addressed by a hash of the trailing token window
//!   rather than by token id, so [`ple`] takes it as a graph input gathered on
//!   the host instead of as a parameter.
//!
//! ## Usage
//!
//! ```no_run
//! use rlx_core::flow_util::compile_built;
//! use rlx_core::weight_map::WeightMap;
//! use rlx_qwen4exp::{Qwen4ExpConfig, build_qwen4exp_text_flow};
//! use rlx_runtime::Device;
//! # fn main() -> anyhow::Result<()> {
//! let path = "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf";
//! // The first shard of an unsloth split carries all metadata and no tensors.
//! let cfg = Qwen4ExpConfig::from_gguf_path(path)?;
//!
//! // A GGUF loads through the format registry — `WeightMap::from_file` is the
//! // safetensors path. `_dequant_all` lands the K-quants as f32; for the
//! // packed path use `build_qwen4exp_text_flow_with_source` with
//! // `rlx_core::flow_bridge::PackedWeightLoaderSource`.
//! let mut loader = rlx_core::weight_loader::load_from_path(path)?;
//! let mut weights = WeightMap::from_weight_loader_dequant_all(loader.as_mut())?;
//! let built = build_qwen4exp_text_flow(&cfg, &mut weights, 64, true)?;
//! let mut compiled = compile_built(built, Device::Cpu)?;
//! let logits = compiled.run(&[("input_ids", &[1.0f32; 64][..])]);
//! # let _ = logits;
//! # Ok(())
//! # }
//! ```
//!
//! ## Running it
//!
//! | what | how |
//! |---|---|
//! | **just run it** | [`runner::Qwen4ExpRunner::open`] → `generate_greedy` / `prefill_logits` + `decode_logits` |
//! | whole-prompt prefill | [`build_qwen4exp_text_flow`] |
//! | chunked prefill | [`decode::build_qwen4exp_step_flow`] with `chunk > 1` |
//! | incremental decode | the same builder with `chunk = 1`, driven by [`decode::DecodeSession`] |
//! | speculative drafting | the trunk at [`OutputKind::HcResidual`], then [`mtp::build_qwen4exp_mtp_flow`] and [`mtp::build_qwen4exp_head_flow`] |
//! | multimodal | `vision::VisionEncoder` + `vision::assemble_multimodal` + [`FlowOptions::host_embed`] (feature `vision`) |
//! | packed (no dequant) | [`flow::build_qwen4exp_text_flow_with_source`] over a `PackedWeightLoaderSource` |
//!
//! ## Status
//!
//! Complete against `modular_qwen4_exp.py`, llama.cpp's
//! `src/models/qwen4exp.cpp` and `conversion/qwen4exp.py`, and every tensor
//! shape and metadata key in the published GGUF conversions.
//!
//! Validated on **real published weights** through range-sliced subsets
//! (`just qwen4exp-real`, `just qwen4exp-vision-real`): the converter's sign and
//! offset conventions, both block kinds end to end, the QSA short-circuit
//! against the selection machinery it skips, **decode reproducing prefill**,
//! and the vision tower on 3 real mmproj blocks.
//!
//! On synthetic weights and header data — all of it over the one fixture in
//! [`synth`], whose shapes `tests/tensor_manifest.rs` checks against the real
//! checkpoint, so no test carries a second copy of the tensor contract: config
//! parsing from both GGUF metadata
//! and HF `config.json` (and that the two agree); the PLE hash constants against
//! the values the checkpoint publishes; the QSA visibility algebra; the n-gram
//! EOS reset and its chunk-boundary equivalence; the tensor manifest against the
//! real 1224-tensor list in both directions for all 48 blocks; the MTP block's
//! contract against its own `mtp-*.gguf`; the residual seam reproducing the
//! trunk's logits; chunked prefill and right-padded chunks reproducing a
//! single-shot prefill; the mmproj contract against the published
//! `mmproj-F16.gguf` header; and the **vision tower against `rlx-qwen35`'s
//! independent implementation of the same architecture**, element for element.
//!
//! Not done:
//!
//! * **No end-to-end generation on the full checkpoint.** Every piece is here
//!   and tested against the path it replaces — [`runner`] drives the loop
//!   (bit-for-bit against the explicit one), [`ple::GgufPleRows`] reads the
//!   28 GB n-gram table 90 bytes at a time, and [`paged_moe`] runs a layer from
//!   the experts that fire rather than all 512, **on the real checkpoint's own
//!   IQ2_XXS / IQ4_NL banks**, against the resident layer. What is missing is
//!   the assembly: a runner that drives all 48 layers with paged MoE, and the
//!   hours of disk it would read. Nothing here has done that.
//! * **Decode and chunked prefill stop at the QSA budget.** Past 2048 cached
//!   positions the indexer genuinely selects, which needs a position-ordered
//!   indexer-key cache; [`decode`] refuses rather than quietly running dense
//!   attention.
//! * **Padding must be on the right, and the batch is 1.** That is exact (see
//!   [`decode`]); left padding and multi-sequence packing would need the
//!   reference's per-query visibility walk, which [`indexer`] does not emit.
//! * **No tokenizer or chat template.** The GGUF carries a `qwen35`-pre BPE
//!   vocabulary and a chat template with this model's `reasoning_effort`
//!   system injection; neither is wired here, so [`runner`] — like
//!   [`rlx_runtime::LmRunner`] itself — is defined over token ids.
//! * **The vision tower has no deepstack path and no video.**
//!   `deepstack_visual_indexes` is `[]` in the published checkpoint, so
//!   `vit` refuses a mmproj that declares one rather than dropping the
//!   features; a video's temporal patch window would need the two patch
//!   kernels fed *different* frames instead of the same one.

pub mod attention;
pub mod common;
pub mod config;
pub mod decode;
pub mod flow;
pub mod gdn;
pub mod hc;
pub mod indexer;
pub mod moe;
pub mod mtp;
pub mod paged_moe;
pub mod ple;
pub mod rope;
pub mod runner;
pub mod synth;
#[cfg(feature = "vision")]
pub mod vision;
#[cfg(feature = "vision")]
pub mod vit;

pub use attention::{AttnCache, AttnCacheOut, AttnDims};
pub use config::{
    ACCEPTED_ARCHES, HF_MODEL_TYPE, HF_TEXT_MODEL_TYPE, LayerKind, PleConstants, PleHash,
    Qwen4ExpConfig,
};
pub use decode::{
    ATTN_MASK_INPUT, DecodeLayout, DecodeNames, DecodeSession, ScanState,
    build_qwen4exp_decode_flow, build_qwen4exp_decode_flow_with, build_qwen4exp_step_flow,
    build_qwen4exp_step_flow_with,
};
pub use flow::{
    EMBED_KEY, FlowOptions, OutputKind, ROPE_COS_INPUT, ROPE_SIN_INPUT, build_qwen4exp_text_flow,
    build_qwen4exp_text_flow_with_options, build_qwen4exp_text_flow_with_source,
    is_text_flow_weight,
};
pub use gdn::{GdnDims, GdnState, GdnStateOut};
pub use hc::HcDims;
pub use indexer::QsaDims;
pub use moe::MoeDims;
pub use mtp::{
    HC_RESIDUAL_INPUT, MtpDims, MtpOut, build_qwen4exp_head_flow, build_qwen4exp_mtp_flow,
    build_qwen4exp_mtp_flow_with, emit_mtp_block, mtp_dims,
};
pub use ple::{
    DensePleTable, PLE_EMBED_INPUT, PleDims, PleRowSource, ngram_embeddings, ngram_rows,
    ngram_rows_with_history, ple_embed_input, prepare_ple_embeddings,
};

/// HuggingFace repo this crate targets.
pub const HF_MODEL_ID: &str = "Qwen/Qwen3.8-Flash-Next";
/// Pre-quantized GGUF conversions.
pub const GGUF_MODEL_ID: &str = "unsloth/Qwen3.8-Flash-Next-GGUF";
