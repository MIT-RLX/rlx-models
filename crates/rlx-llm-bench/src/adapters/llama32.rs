// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Llama-family adapter: build a [`BenchModel`] around
//! [`rlx_llama32::Llama32Runner`].
//!
//! Covers every arch that runs on the llama32 packed GGUF path — plain Llama,
//! Granite, Phi, Cohere/Command-R, OLMo-2, GLM-4, and Muse Glimmer
//! (`DenseArch::MuseGlimmer`). The runner reads `general.architecture` from the
//! GGUF and selects its own per-arch block deltas, so nothing here is
//! arch-specific beyond the stop-token defaults.

use anyhow::{Result, anyhow};
use rlx_runtime::lm::LmRunner;
use tokenizers::Tokenizer;

use rlx_llama32::{Llama32Runner, SampleOpts};

use super::{BuildSpec, device_label};
use crate::model::BenchModel;

/// Stop ids for a GGUF checkpoint, read from its own tokenizer metadata
/// (`tokenizer.ggml.{eos,eot}_token_id`) rather than hardcoded per family —
/// Llama-3 uses 128001/128009, Muse Glimmer 200001/200008, and a guess that
/// misses simply never stops, inflating the decode measurement.
///
/// Falls back to the Llama-3 pair for non-GGUF checkpoints.
fn default_eos(weights: &std::path::Path, vocab: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();
    if weights
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
    {
        if let Ok(f) = rlx_gguf::GgufFile::from_path(weights) {
            for key in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"] {
                if let Some(v) = f.metadata.get(key).and_then(rlx_gguf::MetaValue::as_u32) {
                    ids.push(v);
                }
            }
        }
    }
    if ids.is_empty() {
        ids.extend([128001u32, 128009]);
    }
    ids.sort_unstable();
    ids.dedup();
    ids.retain(|&id| (id as usize) < vocab);
    ids
}

fn load_tokenizer(path: &std::path::Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(|e| anyhow!("loading tokenizer {}: {e}", path.display()))
}

/// Construct a Llama-family [`BenchModel`] from `spec`.
pub fn build(spec: &BuildSpec) -> Result<BenchModel> {
    let mut builder = Llama32Runner::builder()
        .weights(spec.weights.clone())
        .device(spec.device)
        .max_seq(spec.max_seq)
        .sample(SampleOpts::greedy());
    if spec.force_f32 {
        // Quality tasks drive prefill_logits/decode_logits, which only exist on
        // the F32 generator path.
        builder = builder.packed_weights(false);
    }
    let runner = builder
        .build()
        .map_err(|e| anyhow!("building Llama32 runner: {e}"))?;

    let vocab = runner.vocab_size();
    let tokenizer = match &spec.tokenizer {
        Some(p) => Some(load_tokenizer(p)?),
        None => None,
    };
    let eos = if spec.eos_ids.is_empty() {
        default_eos(&spec.weights, vocab)
    } else {
        spec.eos_ids.clone()
    };
    let name = spec.name.clone().unwrap_or_else(|| {
        spec.weights
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "llama32".to_string())
    });

    Ok(BenchModel::new(
        name,
        device_label(spec.device),
        Box::new(runner),
        tokenizer,
        eos,
    ))
}
