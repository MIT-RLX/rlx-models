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

//! Qwen2.5-VL runner — vision encoder + Qwen2.5 dense LM with multimodal RoPE.
//!
//! Target checkpoint: **Qwen2.5-VL-7B-Instruct** (paper baseline for AIF).
//! Weights ship as separate LM + mmproj GGUFs (llama.cpp `convert_hf_to_gguf.py
//! --mmproj`).
//!
//! ## Status
//!
//! | Component | Status |
//! |-----------|--------|
//! | LM (Qwen2 / 2.5 dense via `rlx-qwen3`) | wired — text-only generate |
//! | mmproj config + weight load | wired |
//! | Image preprocess (smart resize) | wired |
//! | Vision HIR (window attn + vision mRoPE + SiLU merger) | wired |
//! | Runtime mRoPE LM prefill/decode | wired (`lm_flow`) |
//! | Multimodal prompt assembly + MRoPE sections | wired (host) |
//! | AIF decode mask (block visual KV keys) | wired (`aif`) |
//! | Native AIF probe (Eq. 2 + decode-step, graph Q/K) | wired (`aif::native_probe`) |
//! | VLMEvalKit datasets + scoring in Rust | wired (`eval::vlmevalkit`) |
//!
//! Paper eval uses VLMEvalKit prompts on Qwen2.5-VL-7B; this crate is the
//! RLX inference path toward that comparison.

pub mod aif;
pub mod chat_template;
pub mod cli;
pub mod config;
pub mod decode_side;
pub mod eval;
pub mod lm;
pub mod lm_flow;
pub mod mrope;
pub mod multimodal;
pub mod prefill_side;
pub mod probe;
pub mod runner;
pub mod synth;

#[cfg(feature = "tokenizer")]
pub mod tokenizer;

pub mod vision;

pub use aif::{
    AifConfig, AifDynamicsMode, AifLiteConfig, AifProbe, MASK_RATIO_CANDIDATES,
    NativePrefillProbeInputs, VisionKeySpan, block_highest_entropy_keys, block_lowest_scored_keys,
    compute_dynamics_eq2_prefill, compute_mu, compute_token_entropies, decode_mask_row_causal,
    distribution_entropy, dynamics_from_graph_qk_decode_step, dynamics_from_graph_qk_layers,
    native_prefill_probe, select_adaptive_mask_ratio,
};
#[allow(deprecated)]
pub use aif::{adaptive_mask_ratio, legacy_adaptive_mask_ratio, mu_distribution_entropy};
pub use chat_template::{
    DEFAULT_SYSTEM, expand_media_marker, qwen25_vl_chatml, user_turn_with_media,
    vlmevalkit_chat_prompt, vlmevalkit_user_text,
};
pub use config::{Qwen25VlConfig, Qwen25VlHfConfig, Qwen25VlLmConfig};
pub use eval::{
    EvalSummary, VlmevalkitDataset, VlmevalkitMetric, VlmevalkitRecord, VlmevalkitReport,
    VlmevalkitSample, VqaSample, infer_dataset, load_vlmevalkit_dataset, load_vqa_jsonl,
    normalize_answer, normalized_exact_match, sample_question_text, score_prediction,
};
pub use lm::{load_lm_config_from_gguf, qwen25_vl_lm_from_gguf};
pub use lm_flow::{
    Qwen25VlPrefillOpts, build_qwen25_vl_decode_built, build_qwen25_vl_prefill_mrope_built,
    mrope_decode_feeds,
};
pub use mrope::{
    build_mrope_tables, build_multimodal_mrope_sections, image_chunk_n_pos, image_decoder_pos,
    mrope_prefill_feeds, mrope_row_for_sections, mrope_sections4, mrope_slice_at_pos,
    text_section_pos,
};
pub use multimodal::{
    IMAGE_PAD, MEDIA_MARKER, MultimodalPrefill, MultimodalPrompt, VISION_END, VISION_START,
    assemble_from_token_ids, merge_text_and_vision_embd,
};
pub use probe::{
    load_probe_reference_dump, load_probe_sample, run_hf_python_probe, sanitize_sample_id,
};
pub use runner::{Qwen25VlRunner, Qwen25VlRunnerBuilder};
#[cfg(feature = "tokenizer")]
pub use tokenizer::{decode_token, encode_prompt, load_tokenizer, resolve_tokenizer_path};
pub use vision::{
    MmProjConfig, MmProjWeights, Qwen25VlVisionEncoder, VisionEncodeOutput, load_vision_encoder,
    load_vision_weights,
};

pub const FAMILY: &str = "Qwen2.5-VL";

/// Accepted `general.architecture` tags for the **LM** GGUF.
pub const ACCEPTED_LM_ARCHES: &[&str] = &["qwen2", "qwen25", "qwen2_5", "qwen2.5", "qwen2vl"];

/// Accepted mmproj `clip.projector_type` values.
pub const ACCEPTED_MMPROJ_TYPES: &[&str] = &["qwen2.5vl_merger", "qwen2vl_merger"];

/// HF `model_type` values for sidecar `config.json`.
pub const ACCEPTED_HF_MODEL_TYPES: &[&str] = &["qwen2_5_vl", "qwen2_vl"];
