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

//! [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR) —
//! SAM-ViT-B + CLIP-L/14-224 "deep encoder" vision tower, a linear
//! `2048 → 1280` projector, and a DeepSeek-V2-style Mixture-of-Experts
//! decoder (dense early layers, routed + shared experts afterwards, rolling
//! sliding-window attention) for one-shot long-document OCR.
//!
//! ## Components
//!
//! 1. **Config** — HF `config.json` parsing ([`config`])
//! 2. **Preprocessing** — Base/Gundam/Multi image modes, EXIF-correct
//!    loading, dynamic tiling ([`preprocess`])
//! 3. **Weights** — mmap-backed safetensors access, HF key helpers ([`weights`])
//! 4. **Decode guards** — sliding-window n-gram repeat blocking ([`ngram`]),
//!    sampling ([`generation`])
//! 5. **Vision tower** — SAM-ViT-B ([`sam_tower`]) + CLIP-L/14-224
//!    ([`clip_tower`]) "deep encoder", fused and projected by [`deep_encoder`]
//!    / [`projector`]
//! 6. **Decoder** — eager host MoE ([`lm_flow`]) and compiled device MoE
//!    ([`lm_graph`] / [`lm_device`]), using [`embed`] for inputs-embeds fusion
//!    and [`rswa`] for the sliding-window mask
//! 7. **Session** — [`runner`] wires host vision + compiled LM for generation;
//!    [`infer`] is the high-level single/multi/PDF session API; [`cli`] is
//!    the command-line front end

pub mod cli;
pub mod clip_tower;
pub mod compile_support;
pub mod config;
pub mod deep_encoder;
pub mod device;
pub mod embed;
pub mod expert_pack;
pub mod fixtures;
pub mod generation;
pub mod host_math;
pub mod hub;
pub mod infer;
pub mod lm_device;
pub mod lm_flow;
pub mod lm_graph;
pub mod lm_precision;
pub mod ngram;
pub mod nn;
pub mod preprocess;
pub mod projector;
pub mod rswa;
pub mod runner;
pub mod sam_tower;
pub mod weights;

#[cfg(feature = "hf-download")]
pub mod download;

#[cfg(feature = "tokenizer")]
pub mod tokenizer;

pub use clip_tower::ClipTower;
pub use config::{
    BOS_TOKEN_ID, ClipTowerConfig, DOWNSAMPLE_RATIO, EOS_TOKEN_ID, IMAGE_TOKEN_ID, PAD_TOKEN_ID,
    PATCH_SIZE, ProjectorConfig, SamTowerConfig, UnlimitedOcrConfig, UnlimitedOcrVisionConfig,
    num_queries,
};
pub use deep_encoder::DeepEncoder;
pub use device::{pick_auto_device, resolve_device};
#[cfg(feature = "hf-download")]
pub use download::{
    fetch_default, fetch_unlimited_ocr, read_snapshot_pointer, snapshot_pointer_path,
};
pub use embed::{argmax_token, fuse_inputs_embeds};
pub use expert_pack::PackedLmWeights;
pub use fixtures::{
    SAMPLE_IMAGE_REL, probe_image_path, require_model_dir, require_probe_image, resolve_image_path,
    sample_image_path,
};
pub use generation::{SampleOpts, sample_token};
pub use hub::{
    default_hf_cache_dir, default_model_dir, hf_snapshot_dir, is_hub_model_id, resolve_weights_path,
};
pub use infer::{InferenceOptions, InputKind, OcrResult, UnlimitedOcrSession};
pub use lm_device::CompiledLm;
pub use lm_flow::{KvCache, LmFlow, MoeLayerShape};
pub use lm_graph::{
    build_unlimited_ocr_decode_built, build_unlimited_ocr_decode_built_from_pack,
    build_unlimited_ocr_prefill_built, build_unlimited_ocr_prefill_built_from_pack,
    compute_rope_slice,
};
pub use lm_precision::{
    LmWeightPrecision, ResolvedLmPrecision, available_ram_bytes, estimate_pack_compile_need,
    estimate_packed_lm_bytes, resolve_lm_precision, resolve_lm_precision_with_ram,
};
pub use ngram::SlidingWindowNoRepeatNgramProcessor;
pub use preprocess::{
    ImageMode, PreprocessedBatch, PreprocessedImage, base_image_tokens, gundam_tile_tokens,
    load_image_exif_corrected, pad_to_square, pdf_to_page_images, preprocess_batch, preprocess_one,
    preprocess_path, preprocess_paths,
};
pub use projector::Projector;
pub use rswa::{build_rswa_mask, within_window};
pub use runner::{RunnerOptions, UnlimitedOcrRunner};
pub use sam_tower::SamTower;
pub use weights::{UnlimitedOcrWeightPrefix, UnlimitedOcrWeightStore};

#[cfg(feature = "tokenizer")]
pub use tokenizer::{build_prompt_ids, decode, encode, load_tokenizer};

pub const FAMILY: &str = "Unlimited-OCR";
pub const HF_MODEL_ID: &str = UnlimitedOcrConfig::HF_MODEL_ID;
