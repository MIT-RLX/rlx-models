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

//! # rlx-glm5next — GLM-5.3-Flash (`glm5next`) for RLX
//!
//! [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) — 320 B total /
//! 18 B active — is a hybrid linear/sparse-attention MoE. GGUF converters emit
//! it as `general.architecture = glm5next`; HF calls it
//! `Glm5NextForConditionalGeneration` / `model_type = glm5_next`.
//!
//! ## Architecture
//!
//! 45 decoder layers (plus one MTP block, `blk.45`), `hidden = 4096`,
//! `vocab = 154880`:
//!
//! | Piece | Where | Notes |
//! |---|---|---|
//! | [`kda`] — Kimi Delta Attention | 34 layers | gated delta-net linear attention; the model's only positional signal |
//! | [`mla`] — NoPE latent attention | layers 3, 7, … 43 | `q_lora 1536`, `kv_lora 512`, head dim 256 |
//! | [`indexer`] — DSA lightning indexer | on every MLA layer | k-pool compressed top-2048 key selection |
//! | [`mhc`] — hyper-connections | every layer, twice | 4 parallel residual streams, Sinkhorn-projected mixing |
//! | [`moe`] — 288-expert MoE | layers 3.. | 8 active + 1 shared, `noaux_tc` sigmoid gate, clamped SwiGLU |
//! | [`decode`] — incremental decode | one token per `run()` | latent KV cache + carried KDA conv/scan state |
//!
//! **The text model has no RoPE at all.** `qk_rope_head_dim = 0` and the
//! reference config rejects anything else, so position information reaches the
//! sparse-attention layers only through the KDA layers beneath them.
//!
//! ## Usage
//!
//! ```no_run
//! use rlx_core::flow_util::compile_built;
//! use rlx_core::weight_map::WeightMap;
//! use rlx_glm5next::{Glm5NextConfig, build_glm5next_text_flow};
//! use rlx_runtime::Device;
//! # fn main() -> anyhow::Result<()> {
//! let path = "GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf";
//! // The first shard of an unsloth split carries all metadata and no tensors.
//! let cfg = Glm5NextConfig::from_gguf_path(path)?;
//!
//! // A GGUF loads through the format registry — `WeightMap::from_file` is the
//! // safetensors path. `_dequant_all` because this crate has no packed-matmul
//! // lowering yet, so the K-quants have to land as f32.
//! let mut loader = rlx_core::weight_loader::load_from_path(path)?;
//! let mut weights = WeightMap::from_weight_loader_dequant_all(loader.as_mut())?;
//! let built = build_glm5next_text_flow(&cfg, &mut weights, 64, true)?;
//! let mut compiled = compile_built(built, Device::Cpu)?;
//! let logits = compiled.run(&[("input_ids", &[1.0f32; 64][..])]);
//! # let _ = logits;
//! # Ok(())
//! # }
//! ```
//!
//! ## Status
//!
//! The text architecture is complete and matches `modeling_glm5_next.py` and
//! every tensor shape in the published GGUF. Covered by tests on synthetic
//! weights: config parsing from both GGUF metadata and HF `config.json`, the
//! mHC Sinkhorn projection, the indexer's pool/tail visibility algebra, an
//! end-to-end tiny-model graph run, and single-token decode reproducing prefill
//! for both scan-state modes. Both layer kinds are additionally validated on
//! **real published weights** — `scripts/glm5next_subset.py` range-fetches a
//! 337 MB subset of one shard rather than the 93 GB model; see
//! `tests/real_weights.rs`.
//!
//! Not done:
//!
//! * **No whole-model run.** Individual blocks are covered on real weights and
//!   everything — projections *and* routed expert banks — runs packed (see
//!   [`flow::build_glm5next_text_flow_with_source`]). What is left is scale, not
//!   a missing capability, and both halves of the scale story now exist:
//!   [`pipeline`] splits the stack across machines, and `MoeDims::paged` runs a
//!   layer against a `seq * top_k` gather from
//!   [`rlx_distributed::ExpertPager`] instead of the 288-expert banks. What is
//!   [`paged_moe::PagedMoeLayer`] joins them: per token it routes on the host,
//!   pages the fired experts off disk, and runs the slot-indexed graph —
//!   reproducing the resident layer (`tests/paged_moe.rs`). What is left is to
//!   run that loop for all 45 layers on the published checkpoint, which is a
//!   matter of driving it rather than of a missing capability.
//! * **Decode stops at `index_topk`.** [`decode`] carries a KV cache and the
//!   KDA recurrent state, and `tests/decode_equivalence.rs` shows stepping it
//!   reproduces prefill — but only while DSA selection is the identity. Past
//!   2048 tokens the builder refuses rather than quietly running dense
//!   attention; that needs a cached indexer, which is not emitted.
//! * **The DSA indexer assumes an unpadded batch-1 prefill** — see
//!   [`indexer`] for exactly which reference branches that drops.
//! * **The MTP block and the vision tower** (`mmproj-*.gguf`,
//!   `clip.projector_type = glm5next`) are parsed but not built.

pub mod common;
pub mod config;
pub mod decode;
pub mod flow;
pub mod indexer;
pub mod kda;
pub mod mhc;
pub mod mla;
pub mod moe;
pub mod paged_moe;
pub mod pipeline;

pub use config::{ACCEPTED_ARCHES, AttnKind, Glm5NextConfig, IndexerKind};
pub use decode::{
    DecodeLayout, DecodeNames, DecodeSession, ScanState, build_glm5next_decode_flow,
    build_glm5next_decode_flow_with,
};
pub use flow::{
    BlockSpec, EMBED_KEY, STREAM_INPUT, block_weight_filter, build_glm5next_block_with_source,
    build_glm5next_text_flow,
};
pub use indexer::IndexerDims;
pub use kda::KdaDims;
pub use mhc::MhcDims;
pub use mla::MlaDims;
pub use moe::MoeDims;
pub use paged_moe::{BANKS, PagedMoeLayer, PhaseTimes};
pub use pipeline::{Glm5NextPipelineStage, block_spec_for};

/// HuggingFace repo this crate targets.
pub const HF_MODEL_ID: &str = "zai-org/GLM-5.3-Flash";
/// Pre-quantized GGUF conversions.
pub const GGUF_MODEL_ID: &str = "unsloth/GLM-5.3-Flash-GGUF";
