// RLX models — model reference / resolution / HuggingFace download.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Model reference parsing, local resolution, and HuggingFace download for
//! RLX models.
//!
//! This crate fills the gap between "the user typed something" and "here are
//! the local files to load". It has three layers:
//!
//! - [`ModelRef`] — parse `org/repo`, `org/repo:Q4_K_M`, `org/repo@rev`, and
//!   HuggingFace repo URLs into a repo id + optional revision + optional quant
//!   selector.
//! - [`quant`] — pick the best-matching GGUF file from a repo's file listing,
//!   honoring an explicit selector or an ordered default preference.
//! - [`Resolver`] — resolve an input to a [`ResolvedModel`] (concrete local
//!   files + [`ModelFormat`]), reading a local path directly or downloading
//!   from HuggingFace into a configurable cache directory.
//!
//! # Example
//!
//! ```no_run
//! use rlx_model_hub::Resolver;
//!
//! let resolver = Resolver::new();
//! // Downloads (or reuses the cache for) the best Q4_K_M GGUF in the repo.
//! let model = resolver.resolve("unsloth/Qwen3-8B-GGUF:Q4_K_M")?;
//! println!("{:?} -> {:?}", model.format, model.files);
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod model_ref;
pub mod quant;
pub mod resolve;

pub use model_ref::{
    ModelRef, ModelRefParseError, SplitGgufShard, format_canonical_ref, format_model_ref,
    gguf_matches_quant_selector, is_quant_like_selector, normalize_gguf_distribution_id,
    parse_model_ref, quant_selector_from_gguf_file, split_gguf_shard_info,
};
pub use quant::{DEFAULT_QUANT_PREFERENCE, gguf_shard_group, is_gguf_file, select_gguf};
pub use resolve::{ModelFormat, ResolvedModel, Resolver, default_cache_dir};
