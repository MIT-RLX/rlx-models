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

//! Neutrino-8B ([FermionResearch/Neutrino-8B](https://huggingface.co/FermionResearch/Neutrino-8B)).
//!
//! Neutrino is a **Qwen3-8B** derivative (Apache-2.0, from Qwen/Qwen3-8B) that
//! Fermion Research put through ternary QAT and shipped as a sub-2-bit TRTC v4
//! container. The `gguf/neutrino-8b-fv5.gguf` release stores it in two custom
//! ggml types added by the `fermion-fv5` llama.cpp fork:
//!
//! * **`FV5`** (ggml 43) — every transformer linear, five-value ternary
//!   `{0, ±s_lo, ±s_hi}` at 3.25 bpw.
//! * **`FV5B`** (ggml 44) — the untied int8 `token_embd` / `output` rows.
//!
//! The topology is **stock Qwen3** (`general.architecture = qwen3`; per-head
//! Q/K RMSNorm, biasless QKV, 36 layers, hidden 4096, 32 Q / 8 KV heads,
//! head_dim 128, RoPE θ 1e6, RMS eps 1e-6, vocab 151936), so this crate is a
//! thin wrapper over [`rlx_qwen3::Qwen3Runner`]. All FV5-specific work lives in
//! `rlx-gguf` (block decode) and the `rlx-cpu` `DequantMatMul` dispatch; here we
//! only validate the GGUF is a genuine Qwen3-topology FV5 pack, force the packed
//! path (so the 8 B stays ~4 GB instead of expanding to ~32 GB of F32), and add
//! a text/ids CLI.
//!
//! # Parity
//!
//! The fork's numerics policy is `vec_dot_type = F32`: activations are **never**
//! quantized at runtime, so the only difference from an fp32 expansion of the
//! container is float summation order. RLX matches this exactly — the FV5/FV5B
//! blocks decode to the identical f32 weights and feed an f32 matmul. The
//! reference CPU gate is greedy token-identity vs that fp32 expansion; the ids
//! CLI path here ([`cli_run`] with `--prompt-ids`) is the same argmax harness.

use anyhow::{Context, Result, anyhow, bail};
use std::path::{Path, PathBuf};

pub use rlx_qwen3::{Precision, Qwen3Config, Qwen3ConfigSource, Qwen3Runner, Qwen3RunnerBuilder};

/// GGUF `general.architecture` tags accepted as Neutrino (stock Qwen3 topology).
pub const ACCEPTED_ARCHES: &[&str] = &["qwen3"];

/// Default system prompt baked into Neutrino's chat template.
pub const DEFAULT_SYSTEM_PROMPT: &str =
    "You are Neutrino-1, a large language model made by Fermion Research.";

/// Stop tokens from `generation_config.json`: `<|im_end|>` (151645) and
/// `<|endoftext|>` (151643).
pub const EOS_TOKENS: &[u32] = &[151645, 151643];

/// What a `--weights` GGUF header says about a Neutrino pack (metadata-only —
/// no tensor-data slurp).
#[derive(Debug, Clone)]
pub struct NeutrinoInfo {
    /// `general.architecture` (expected `qwen3`).
    pub arch: String,
    /// `general.name`, if present.
    pub name: Option<String>,
    /// Number of `FV5` (ggml 43) tensors — the transformer linears.
    pub fv5_tensors: usize,
    /// Number of `FV5B` (ggml 44) tensors — int8 embed / lm_head.
    pub fv5b_tensors: usize,
}

impl NeutrinoInfo {
    /// True when the pack carries Fermion FV5/FV5B weights (vs a vanilla Qwen3
    /// GGUF that merely shares the `qwen3` arch tag).
    pub fn is_fermion_fv5(&self) -> bool {
        self.fv5_tensors > 0 || self.fv5b_tensors > 0
    }
}

/// Read a GGUF header and classify it. Errors when the arch isn't a Qwen3
/// topology; succeeds (with `is_fermion_fv5() == false`) for a plain Qwen3 GGUF
/// so the wrapper can still run one, but callers can warn on that.
pub fn detect(weights: &Path) -> Result<NeutrinoInfo> {
    use rlx_gguf::{GgmlType, GgufFile, MetaValue};
    let raw = GgufFile::header_from_path(weights)
        .with_context(|| format!("rlx-neutrino: reading GGUF header {weights:?}"))?;
    let arch = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .ok_or_else(|| anyhow!("rlx-neutrino: {weights:?} missing general.architecture"))?
        .to_string();
    if !ACCEPTED_ARCHES.contains(&arch.as_str()) {
        bail!(
            "rlx-neutrino: {weights:?} has general.architecture=`{arch}`, expected one of \
             {ACCEPTED_ARCHES:?} (Neutrino is stock Qwen3 topology)"
        );
    }
    let name = raw
        .metadata
        .get("general.name")
        .and_then(MetaValue::as_str)
        .map(str::to_string);
    let mut fv5 = 0usize;
    let mut fv5b = 0usize;
    for t in raw.tensors.values() {
        match t.dtype {
            GgmlType::FV5 => fv5 += 1,
            GgmlType::FV5B => fv5b += 1,
            _ => {}
        }
    }
    Ok(NeutrinoInfo {
        arch,
        name,
        fv5_tensors: fv5,
        fv5b_tensors: fv5b,
    })
}

/// Neutrino-8B runner — a validated [`Qwen3Runner`] pinned to the packed FV5
/// path.
pub struct NeutrinoRunner {
    inner: Qwen3Runner,
    info: NeutrinoInfo,
}

impl NeutrinoRunner {
    pub fn builder() -> NeutrinoRunnerBuilder {
        NeutrinoRunnerBuilder::default()
    }

    /// The parsed Qwen3 config (dims, RoPE, GQA, …) read from GGUF metadata.
    pub fn config(&self) -> &Qwen3Config {
        self.inner.config()
    }

    /// Header classification for the loaded pack.
    pub fn info(&self) -> &NeutrinoInfo {
        &self.info
    }

    /// The underlying Qwen3 runner (for callers that want the full API).
    pub fn inner(&self) -> &Qwen3Runner {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut Qwen3Runner {
        &mut self.inner
    }

    /// Full next-token logits over the vocab for `prompt_ids` (packed forward).
    pub fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.inner.predict_logits(prompt_ids)
    }

    /// Deterministic greedy (argmax) generation — the reference `fermion-greedy`
    /// harness. Re-runs the packed prefill against the growing history each step
    /// (so `max_seq` must be ≥ `prompt_ids.len() + n_new`).
    ///
    /// When `stop` is non-empty, generation halts (without emitting) on the
    /// first stop id. An empty `stop` runs a fixed `n_new` steps and emits every
    /// argmax — the faithful parity mode, matching the fork's free-run gate.
    /// Ties resolve to the lowest vocab index (`argmax` picks the first max),
    /// identical to the fork's `argmax2`.
    pub fn generate_greedy(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        stop: &[u32],
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let mut history = prompt_ids.to_vec();
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let logits = self.predict_logits(&history)?;
            let next = argmax(&logits) as u32;
            if stop.contains(&next) {
                break;
            }
            on_token(next);
            history.push(next);
            out.push(next);
        }
        Ok(out)
    }
}

/// Lowest-index argmax (first maximum, matching the fork's `argmax2`).
fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Builder for [`NeutrinoRunner`]. Same surface as [`Qwen3RunnerBuilder`] for the
/// bits Neutrino needs, plus GGUF validation and forced packed weights.
pub struct NeutrinoRunnerBuilder {
    weights: Option<PathBuf>,
    max_seq: Option<usize>,
    packed_weights: Option<bool>,
    inner: Qwen3RunnerBuilder,
}

impl Default for NeutrinoRunnerBuilder {
    fn default() -> Self {
        Self {
            weights: None,
            max_seq: None,
            packed_weights: None,
            inner: Qwen3Runner::builder(),
        }
    }
}

impl NeutrinoRunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        let p: PathBuf = path.into();
        self.weights = Some(p.clone());
        self.inner = self.inner.weights(p);
        self
    }

    /// Prefill/history bucket. Packed greedy decode re-prefills the growing
    /// history, so this must be ≥ `prompt_len + tokens_to_generate`.
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }

    /// Override the packed-weights decision. Defaults to `Some(true)` — the 8 B
    /// FV5 pack would otherwise expand to ~32 GB of F32 weights.
    pub fn packed_weights(mut self, on: bool) -> Self {
        self.packed_weights = Some(on);
        self
    }

    pub fn build(self) -> Result<NeutrinoRunner> {
        let weights = self
            .weights
            .clone()
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?;
        let info = detect(&weights)?;
        // Force packed unless the caller explicitly opted out.
        let packed = self.packed_weights.unwrap_or(true);
        let mut inner = self.inner.packed_weights(packed);
        if let Some(n) = self.max_seq {
            inner = inner.max_seq(n);
        }
        let inner = inner
            .build()
            .context("rlx-neutrino: building underlying Qwen3Runner")?;
        Ok(NeutrinoRunner { inner, info })
    }
}

// ─── CLI ──────────────────────────────────────────────────────────────

fn parse_ids(s: &str) -> Result<Vec<u32>> {
    s.split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| {
            t.trim()
                .parse::<u32>()
                .with_context(|| format!("--prompt-ids: bad token id {t:?}"))
        })
        .collect()
}

const HELP: &str = "\
rlx-neutrino — Neutrino-8B (Qwen3 topology, FV5 ternary weights)

USAGE:
    rlx-neutrino --weights <neutrino-8b-fv5.gguf> [--prompt \"text\" | --prompt-ids 1,2,3] [OPTIONS]

OPTIONS:
    --weights <PATH>       GGUF path (required)
    --prompt <TEXT>        Text prompt (chat-templated unless --no-chat); needs the `tokenizer` feature
    --prompt-ids <IDS>     Comma-separated u32 token ids (bypasses tokenizer/chat — the parity harness)
    --system <TEXT>        System prompt (default: Neutrino-1 identity)
    --no-chat              Encode --prompt as raw text (no ChatML wrapping / generation prompt)
    -n, --max-tokens <N>   Tokens to generate (default 128)
    --max-seq <N>          Prefill bucket (default: prompt_len + max_tokens + 8)
    --tokenizer <PATH>     Explicit tokenizer.json (default: GGUF-embedded BPE)
    --raw                  Print generated token ids instead of decoded text
    -h, --help             Show this help

Greedy (argmax) decoding only — the deterministic mode the FV5 correctness gate certifies.
FV5 is a CPU-only ggml type; this runs on CPU regardless of build features.
";

/// CLI entry point (`rlx-run neutrino …` / `rlx-neutrino …`).
pub fn cli_run(args: &[String]) -> Result<()> {
    if args.iter().any(|a| a == "-h" || a == "--help") || args.is_empty() {
        print!("{HELP}");
        return Ok(());
    }

    let mut weights: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut prompt_ids: Option<String> = None;
    let mut system: Option<String> = None;
    let mut no_chat = false;
    let mut max_tokens = 128usize;
    let mut max_seq: Option<usize> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut raw = false;

    let mut i = 0;
    let need = |args: &[String], i: &mut usize, flag: &str| -> Result<String> {
        *i += 1;
        args.get(*i)
            .cloned()
            .ok_or_else(|| anyhow!("{flag} needs a value"))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(need(args, &mut i, "--weights")?.into()),
            "--prompt" => prompt = Some(need(args, &mut i, "--prompt")?),
            "--prompt-ids" => prompt_ids = Some(need(args, &mut i, "--prompt-ids")?),
            "--system" => system = Some(need(args, &mut i, "--system")?),
            "--no-chat" => no_chat = true,
            "-n" | "--max-tokens" => {
                max_tokens = need(args, &mut i, "--max-tokens")?
                    .parse()
                    .context("--max-tokens: expected integer")?;
            }
            "--max-seq" => {
                max_seq = Some(
                    need(args, &mut i, "--max-seq")?
                        .parse()
                        .context("--max-seq: expected integer")?,
                );
            }
            "--tokenizer" => tokenizer = Some(need(args, &mut i, "--tokenizer")?.into()),
            "--raw" => raw = true,
            other if other.starts_with("--weights=") => {
                weights = Some(other["--weights=".len()..].into());
            }
            other => bail!("unknown arg {other:?} (see --help)"),
        }
        i += 1;
    }

    let weights = weights.ok_or_else(|| anyhow!("--weights <PATH> is required"))?;
    let info = detect(&weights)?;
    if !info.is_fermion_fv5() {
        eprintln!(
            "[rlx-neutrino] warning: {weights:?} has no FV5/FV5B tensors — this looks like a \
             vanilla Qwen3 GGUF, not a Fermion Neutrino pack. Running it anyway."
        );
    } else {
        eprintln!(
            "[rlx-neutrino] {} — {} FV5 + {} FV5B tensors (Qwen3 topology)",
            info.name.as_deref().unwrap_or("Neutrino"),
            info.fv5_tensors,
            info.fv5b_tensors
        );
    }

    // Resolve the prompt into token ids.
    let (input_ids, ids_mode) = match (prompt_ids, prompt) {
        (Some(ids), _) => (parse_ids(&ids)?, true),
        (None, Some(text)) => (encode_text(&weights, tokenizer.as_deref(), &system, &text, no_chat)?, false),
        (None, None) => bail!("provide --prompt \"text\" or --prompt-ids 1,2,3"),
    };
    if input_ids.is_empty() {
        bail!("empty prompt");
    }

    let seq = max_seq.unwrap_or(input_ids.len() + max_tokens + 8);
    let mut runner = NeutrinoRunner::builder()
        .weights(&weights)
        .max_seq(seq)
        .build()?;

    // ids mode = faithful parity harness (fixed steps, emit every argmax).
    // text mode = stop on EOS.
    let stop: &[u32] = if ids_mode { &[] } else { EOS_TOKENS };
    let mut generated: Vec<u32> = Vec::new();
    let started = std::time::Instant::now();
    runner.generate_greedy(&input_ids, max_tokens, stop, |tok| generated.push(tok))?;
    let dt = started.elapsed().as_secs_f64();

    if raw || ids_mode {
        let s = generated
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!("{s}");
    } else {
        let text = decode_text(&weights, tokenizer.as_deref(), &generated)?;
        println!("{text}");
    }
    eprintln!(
        "[rlx-neutrino] {} tokens in {:.1}s ({:.2} tok/s)",
        generated.len(),
        dt,
        generated.len() as f64 / dt.max(1e-9),
    );
    Ok(())
}

#[cfg(feature = "tokenizer")]
fn encode_text(
    weights: &Path,
    tokenizer: Option<&Path>,
    system: &Option<String>,
    text: &str,
    no_chat: bool,
) -> Result<Vec<u32>> {
    if no_chat {
        return rlx_qwen35::encode_prompt_auto(weights, tokenizer, text);
    }
    let sys = system.as_deref().unwrap_or(DEFAULT_SYSTEM_PROMPT);
    let messages = vec![
        rlx_qwen35::ChatMessage::system(sys),
        rlx_qwen35::ChatMessage::user(text),
    ];
    rlx_qwen35::encode_chat_auto(weights, tokenizer, &messages)
}

#[cfg(not(feature = "tokenizer"))]
fn encode_text(
    _weights: &Path,
    _tokenizer: Option<&Path>,
    _system: &Option<String>,
    _text: &str,
    _no_chat: bool,
) -> Result<Vec<u32>> {
    bail!("text prompts need the `tokenizer` feature; use --prompt-ids, or rebuild with --features tokenizer")
}

#[cfg(feature = "tokenizer")]
fn decode_text(weights: &Path, tokenizer: Option<&Path>, ids: &[u32]) -> Result<String> {
    rlx_qwen35::decode_ids_auto(weights, tokenizer, ids, true)
}

#[cfg(not(feature = "tokenizer"))]
fn decode_text(_weights: &Path, _tokenizer: Option<&Path>, _ids: &[u32]) -> Result<String> {
    bail!("detokenize needs the `tokenizer` feature; use --raw to print token ids")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_first_max_on_ties() {
        assert_eq!(argmax(&[0.1, 0.9, 0.9, 0.2]), 1);
        assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
        assert_eq!(argmax(&[5.0]), 0);
    }

    #[test]
    fn parse_ids_ok_and_err() {
        assert_eq!(parse_ids("1, 2 ,3").unwrap(), vec![1, 2, 3]);
        assert_eq!(parse_ids("").unwrap(), Vec::<u32>::new());
        assert!(parse_ids("1,x,3").is_err());
    }

    #[test]
    fn eos_tokens_match_generation_config() {
        assert_eq!(EOS_TOKENS, &[151645, 151643]);
    }
}
