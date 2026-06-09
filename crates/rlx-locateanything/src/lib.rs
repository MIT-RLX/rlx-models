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

//! NVIDIA [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) —
//! MoonViT vision encoder + MLP projector + Qwen2.5-3B-Instruct with parallel
//! box decoding (MTP).
//!
//! Full runbook: [README.md](README.md).
//!
//! ## Quick start
//!
//! ```no_run
//! use rlx_locateanything::{LocateAnythingSession, fixtures};
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut session = LocateAnythingSession::open_default()?;
//! let out = session.ground_path(fixtures::sample_image_path(), "person")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Components
//!
//! 1. **MoonViT** — compiled encoder ([`moonvit_flow`], [`moonvit::MoonVitCache`])
//! 2. **Projector** — compiled `mlp1` ([`projector`])
//! 3. **Language model** — Qwen2.5 ([`lm_flow`], [`runner`], [`infer::LocateAnythingSession`])

pub mod cli;
pub mod compile_support;
pub mod config;
pub mod device;
pub mod embed;
pub mod fixtures;
pub mod generation;
pub mod hub;
pub mod infer;
pub mod kv_buckets;
pub mod lm_flow;
pub mod load;
pub mod mask;
pub mod moonvit;
pub mod moonvit_flow;
pub mod mtp;
pub mod output;
pub mod parse;
pub mod preprocess;
pub mod processor_prompt;
pub mod projector;
pub mod prompts;
pub mod rope2d;
pub mod runner;
pub mod session_cache;
pub mod weights;

#[cfg(feature = "hf-download")]
pub mod download;

#[cfg(feature = "tokenizer")]
pub mod tokenizer;

pub use compile_support::{
    lm_active_extent_enabled, lm_decode_compile_options, lm_gpu_kv_enabled, lm_host_device,
    locateanything_host_device, locateanything_uses_cpu_host, metal_lm_compile_guard,
    moonvit_use_decomposed_rope, vision_encode_device,
};
pub use config::{LocateAnythingConfig, LocateAnythingTextConfig, MoonVitConfig};
pub use device::{pick_auto_device, resolve_device};
#[cfg(feature = "hf-download")]
pub use download::{
    fetch_default, fetch_locateanything, read_snapshot_pointer, snapshot_pointer_path,
};
pub use embed::{fuse_inputs_embeds, fuse_inputs_embeds_from_store};
pub use fixtures::{
    SAMPLE_IMAGE_REL, probe_image_path, require_model_dir, require_probe_image, resolve_image_path,
    sample_image_path,
};
pub use generation::{GenerationMode, SampleOpts, TokenIds};
pub use hub::{
    default_hf_cache_dir, default_model_dir, hf_snapshot_dir, is_hub_model_id, resolve_weights_path,
};
pub use infer::{GroundingResult, InferenceOptions, LocateAnythingSession, PromptStyle};
pub use load::{
    LocateAnythingWeightStore, PREFIX_LANGUAGE_MODEL, PREFIX_PROJECTOR, PREFIX_VISION,
    load_language_model_weights, load_projector_weights, load_vision_weights, resolve_model_dir,
};
pub use moonvit::{MoonVitCache, encode_image, load_moonvit_weights};
pub use moonvit_flow::build_moonvit_built;
pub use output::print_grounding;
pub use parse::{
    GroundingParse, ParsedBox, ParsedPoint, parse_boxes, parse_grounding, parse_points, parse_refs,
};
pub use preprocess::preprocess_image;
pub use processor_prompt::ProcessorPromptConfig;
pub use runner::{GenerateProfile, LocateAnythingRunner, LocateAnythingRunnerBuilder};
pub use weights::LocateAnythingWeightPrefix;

pub const FAMILY: &str = "LocateAnything";
pub const HF_MODEL_ID: &str = LocateAnythingConfig::HF_MODEL_ID;
