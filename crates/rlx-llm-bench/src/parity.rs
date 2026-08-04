// RLX models — LLM benchmark harness.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Parity dimension: does this runner agree with a reference?
//!
//! A [`ReferenceDump`] is a JSON `{prompt_ids, logits?, argmax?}`. It can come
//! from an external oracle (mlx-lm / HuggingFace `prefill` dump) **or** from a
//! prior rlx run on another backend (save CPU logits, then compare the GPU run
//! against them) — so the same machinery covers both "matches PyTorch" and
//! "matches our own CPU baseline" checks.
//!
//! Compared signals:
//! - **argmax agreement** — greedy next-token id matches (always available).
//! - **logit cosine** — cosine of the two last-position logit vectors (only
//!   when the dump carries `logits`).

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::metrics::{argmax, cosine};
use crate::model::BenchModel;

/// A reference forward result to compare against.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceDump {
    /// Prompt token ids fed to both the reference and this runner.
    pub prompt_ids: Vec<u32>,
    /// Reference last-position logits `[vocab]`, if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logits: Option<Vec<f32>>,
    /// Reference greedy next-token id, if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub argmax: Option<u32>,
}

impl ReferenceDump {
    /// Build a dump from a logit vector (fills `argmax` from it).
    pub fn from_logits(prompt_ids: Vec<u32>, logits: Vec<f32>) -> Self {
        let am = argmax(&logits) as u32;
        Self {
            prompt_ids,
            logits: Some(logits),
            argmax: Some(am),
        }
    }

    /// Load from a JSON file.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading reference dump {}", path.display()))?;
        let dump: ReferenceDump = serde_json::from_str(&text)
            .with_context(|| format!("parsing reference dump {}", path.display()))?;
        if dump.prompt_ids.is_empty() {
            bail!("{}: reference dump has empty prompt_ids", path.display());
        }
        Ok(dump)
    }

    /// Write to a JSON file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string(self)?;
        std::fs::write(path, text)
            .with_context(|| format!("writing reference dump {}", path.display()))?;
        Ok(())
    }
}

/// Result of a parity comparison.
#[derive(Debug, Clone)]
pub struct ParityResult {
    pub our_argmax: u32,
    pub ref_argmax: Option<u32>,
    /// `Some(true/false)` when the reference carried an argmax; else `None`.
    pub argmax_match: Option<bool>,
    /// Cosine of last-position logits, when the reference carried logits.
    pub cosine: Option<f32>,
    pub vocab: usize,
}

impl ParityResult {
    pub fn bench_line(&self, name: &str, device: &str) -> String {
        let cos = self
            .cosine
            .map(|c| format!("{c:.6}"))
            .unwrap_or_else(|| "na".into());
        let am = match self.argmax_match {
            Some(true) => "yes",
            Some(false) => "no",
            None => "na",
        };
        format!(
            "LLMBENCH kind=parity model={name} device={device} argmax_match={am} \
             our_argmax={} cosine={cos}",
            self.our_argmax
        )
    }
}

/// Compare this runner's forward against `dump`.
pub fn run_parity(model: &mut BenchModel, dump: &ReferenceDump) -> Result<ParityResult> {
    let vocab = model.vocab_size();
    // Shared with MC scoring: F32 `prefill_logits` else the packed forward.
    let ours = model.context_last_logits(&dump.prompt_ids)?;
    let our_argmax = argmax(&ours[..vocab]) as u32;

    let argmax_match = dump.argmax.map(|r| r == our_argmax);
    let cosine = dump.logits.as_ref().map(|ref_logits| {
        let n = ref_logits.len().min(ours.len());
        cosine(&ours[..n], &ref_logits[..n])
    });

    Ok(ParityResult {
        our_argmax,
        ref_argmax: dump.argmax,
        argmax_match,
        cosine,
        vocab,
    })
}
