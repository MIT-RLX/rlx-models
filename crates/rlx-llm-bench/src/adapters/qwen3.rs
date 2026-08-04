// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Qwen3 adapter: build a [`BenchModel`] around [`rlx_qwen3::Qwen3Runner`].

use anyhow::{Result, anyhow};
use rlx_runtime::lm::LmRunner;
use tokenizers::Tokenizer;

use rlx_qwen3::{Qwen3Runner, SampleOpts};

use super::{BuildSpec, device_label};
use crate::model::BenchModel;

/// Well-known Qwen3 EOS ids (`<|endoftext|>`, `<|im_end|>`), filtered to the
/// model's actual vocab so a tiny/off-family checkpoint doesn't get bogus ids.
fn default_eos(vocab: usize) -> Vec<u32> {
    [151643u32, 151645]
        .into_iter()
        .filter(|&id| (id as usize) < vocab)
        .collect()
}

fn load_tokenizer(path: &std::path::Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(|e| anyhow!("loading tokenizer {}: {e}", path.display()))
}

/// Construct a Qwen3 [`BenchModel`] from `spec`.
pub fn build(spec: &BuildSpec) -> Result<BenchModel> {
    let mut builder = Qwen3Runner::builder()
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
        .map_err(|e| anyhow!("building Qwen3 runner: {e}"))?;

    let vocab = runner.vocab_size();
    let tokenizer = match &spec.tokenizer {
        Some(p) => Some(load_tokenizer(p)?),
        None => None,
    };
    let eos = if spec.eos_ids.is_empty() {
        default_eos(vocab)
    } else {
        spec.eos_ids.clone()
    };
    let name = spec.name.clone().unwrap_or_else(|| {
        spec.weights
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "qwen3".to_string())
    });

    Ok(BenchModel::new(
        name,
        device_label(spec.device),
        Box::new(runner),
        tokenizer,
        eos,
    ))
}
