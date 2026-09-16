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

//! [`S1Runner`] — the text-normalizer front end over [`Qwen3Runner`].
//!
//! S1-mini's topology is stock Qwen3-0.6B (its `config.json` is byte-identical
//! to `Qwen/Qwen3-0.6B`), so there is no new architecture here. What this crate
//! adds is the part that is actually easy to get wrong: the exact prompt
//! protocol, greedy-only decoding, the two-id stop set, a `max_new_tokens`
//! sized to the input, and chunking for transcripts past the model's
//! dictation-length design point.

use crate::prompt::{Controls, chunk_transcript, recommended_max_new_tokens, render_prompt};
use anyhow::{Context as _, Result, anyhow, bail};
use rlx_qwen3::{Precision, Qwen3Config, Qwen3ConfigSource, Qwen3Runner, SampleOpts};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

/// Stop ids from `generation_config.json`: `<|im_end|>` (151645) and
/// `<|endoftext|>` (151643).
pub const EOS_TOKENS: &[u32] = &[151645, 151643];

/// Default prefill bucket, in tokens. Prompt lengths round up to a multiple of
/// this before the prefill graph is compiled, so one compiled shape serves a
/// span of transcript lengths instead of one exact length. 64 costs at most 63
/// padded positions (~30 ms of prefill at this model size) and saves a 5–30 s
/// recompile every time an utterance has a length the runner hasn't seen.
pub const DEFAULT_PREFILL_BUCKET: usize = 64;

/// Default transcript budget per pass, in tokens. The model card asks for
/// single passes under roughly 1,000 tokens; 896 leaves room for the ~60-token
/// system + control-line prefix inside that ceiling.
pub const DEFAULT_CHUNK_TOKENS: usize = 896;

/// Topology the released S1-mini checkpoint has, i.e. Qwen3-0.6B's.
pub const REFERENCE_SHAPE: S1Shape = S1Shape {
    num_hidden_layers: 28,
    hidden_size: 1024,
    intermediate_size: 3072,
    num_attention_heads: 16,
    num_key_value_heads: 8,
    head_dim: 128,
    vocab_size: 151_936,
};

/// The topology fields that identify an S1-mini checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S1Shape {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
}

impl S1Shape {
    pub fn from_config(cfg: &Qwen3Config) -> Self {
        Self {
            num_hidden_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            vocab_size: cfg.vocab_size,
        }
    }

    /// True when this is the released S1-mini / Qwen3-0.6B topology.
    pub fn matches_reference(&self) -> bool {
        *self == REFERENCE_SHAPE
    }
}

// ─── Builder ──────────────────────────────────────────────────────────

/// Builder for [`S1Runner`].
#[derive(Debug, Clone, Default)]
pub struct S1RunnerBuilder {
    weights: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    config: Option<PathBuf>,
    device: Option<Device>,
    precision: Option<Precision>,
    max_seq: Option<usize>,
    max_new_tokens: Option<usize>,
    chunk_tokens: Option<usize>,
    packed_weights: Option<bool>,
    prefer_gguf: Option<String>,
    prefill_bucket: Option<usize>,
    strict_shape: bool,
}

impl S1RunnerBuilder {
    /// Weights: the `superwhisper/s1-mini` directory (safetensors), a
    /// `.safetensors` file, or an `s1-mini-*.gguf` from `superwhisper/s1-mini-GGUF`.
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights = Some(path.into());
        self
    }

    /// Explicit `tokenizer.json`. Default: resolved next to the weights, or the
    /// GGUF-embedded BPE.
    pub fn tokenizer(mut self, path: impl Into<PathBuf>) -> Self {
        self.tokenizer = Some(path.into());
        self
    }

    /// Explicit `config.json` (safetensors only; GGUF reads its metadata).
    pub fn config(mut self, path: impl Into<PathBuf>) -> Self {
        self.config = Some(path.into());
        self
    }

    /// Inference device. Default [`Device::Cpu`] — at 0.6B the card's own claim
    /// is that CPU is comfortable.
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn precision(mut self, p: Precision) -> Self {
        self.precision = Some(p);
        self
    }

    /// Decode-bucket ceiling. Default: `chunk_tokens` + its recommended
    /// `max_new_tokens` + the prompt prefix, so a full-budget chunk never falls
    /// off the compiled ladder.
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }

    /// Fixed generation ceiling. Default: `1.3 × prompt_tokens + 32`, per the
    /// model card — output length tracks input length, so a per-call ceiling is
    /// both cheaper and safer than a flat 1024.
    pub fn max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = Some(n);
        self
    }

    /// Transcript budget per pass. `0` disables chunking entirely (one pass, no
    /// matter how long the transcript).
    pub fn chunk_tokens(mut self, n: usize) -> Self {
        self.chunk_tokens = Some(n);
        self
    }

    /// Keep K-quant GGUF weights packed in the arena. Default: let
    /// [`Qwen3Runner`] auto-decide (it prefers the F32 path for a model this
    /// small, which is also the faster one).
    pub fn packed_weights(mut self, on: bool) -> Self {
        self.packed_weights = Some(on);
        self
    }

    /// When `weights` is a directory of `.gguf` files, prefer names containing
    /// this substring. Default `q4_k_m` — the build the card's 94.8% number was
    /// measured on.
    pub fn prefer_gguf_quant(mut self, sub: impl Into<String>) -> Self {
        self.prefer_gguf = Some(sub.into());
        self
    }

    /// Round prompt lengths up to a multiple of `step` before compiling the
    /// prefill graph. Default [`DEFAULT_PREFILL_BUCKET`]; `0` or `1` disables.
    ///
    /// This matters more here than for a chat model. `rlx-qwen3`'s prefill
    /// compile cache keys on the exact prompt length, and transcript lengths
    /// vary continuously — so without bucketing a dictation pipeline recompiles
    /// the prefill graph on nearly every utterance (5–30 s on Metal), while the
    /// forward pass itself is ~0.2 s. Bucketing collapses that to a handful of
    /// shapes; output is token-identical either way.
    pub fn prefill_bucket(mut self, step: usize) -> Self {
        self.prefill_bucket = Some(step);
        self
    }

    /// Error (instead of warning) when the checkpoint isn't the S1-mini
    /// topology. Off by default so a future S1 size still runs.
    pub fn strict_shape(mut self, on: bool) -> Self {
        self.strict_shape = on;
        self
    }

    pub fn build(self) -> Result<S1Runner> {
        let weights_in = self
            .weights
            .clone()
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?;
        let prefer_gguf = self.prefer_gguf.clone().unwrap_or_else(|| "q4_k_m".into());
        // Resolve a directory down to the concrete weights file *here*, not just
        // inside the Qwen3 builder, because the tokenizer bridge keys off this
        // path too: given `s1-mini-GGUF/` it has to see `.../s1-mini-q4_k_m.gguf`
        // to fall back to the GGUF-embedded vocab. Handed the bare directory it
        // finds no `tokenizer.json`, can't tell the path is a GGUF, and errors.
        let weights = rlx_cli::resolve_weights_cli(
            &weights_in,
            &rlx_cli::WeightsResolveCli {
                prefer_gguf: Some(prefer_gguf.clone()),
                gguf_index: None,
            },
        )
        .with_context(|| format!("rlx-s1: resolving weights {weights_in:?}"))?;
        let chunk_tokens = self.chunk_tokens.unwrap_or(DEFAULT_CHUNK_TOKENS);
        // A full-budget chunk plus its generation ceiling plus the prompt
        // prefix — so the decode-bucket ladder covers the worst case we can
        // actually produce.
        let max_seq = self
            .max_seq
            .unwrap_or_else(|| chunk_tokens + recommended_max_new_tokens(chunk_tokens) + 128);

        let mut b = Qwen3Runner::builder()
            .weights(weights.clone())
            .device(self.device.unwrap_or(Device::Cpu))
            .max_seq(max_seq)
            .precision(self.precision.unwrap_or_default())
            .stream(true)
            // Greedy, always. Normalization is a deterministic transformation;
            // `generation_config.json` ships `do_sample: false` and the card is
            // explicit that sampling only adds variance.
            .sample(SampleOpts::greedy())
            .prefill_bucket(self.prefill_bucket.unwrap_or(DEFAULT_PREFILL_BUCKET))
            .prefer_gguf_quant(prefer_gguf);
        if let Some(p) = self.config.clone() {
            b = b.config(Qwen3ConfigSource::JsonFile(p));
        }
        if let Some(on) = self.packed_weights {
            b = b.packed_weights(on);
        }
        let inner = b.build().context("rlx-s1: building the Qwen3 runner")?;

        let shape = S1Shape::from_config(inner.config());
        if !shape.matches_reference() {
            let msg = format!(
                "rlx-s1: {weights:?} is not the S1-mini topology \
                 (got {shape:?}, expected {REFERENCE_SHAPE:?}). The prompt protocol in this \
                 crate is trained into the released checkpoint and means nothing to other weights."
            );
            if self.strict_shape {
                bail!("{msg}");
            }
            eprintln!("[rlx-s1] warning: {msg}");
        }

        Ok(S1Runner {
            inner,
            weights,
            tokenizer: self.tokenizer,
            max_new_tokens: self.max_new_tokens,
            chunk_tokens,
            shape,
        })
    }
}

// ─── Runner ───────────────────────────────────────────────────────────

/// S1-mini text normalizer.
///
/// Build once, call [`S1Runner::normalize`] many times. Decoding is greedy and
/// not configurable — see [`S1RunnerBuilder::build`].
pub struct S1Runner {
    inner: Qwen3Runner,
    weights: PathBuf,
    tokenizer: Option<PathBuf>,
    max_new_tokens: Option<usize>,
    chunk_tokens: usize,
    shape: S1Shape,
}

impl S1Runner {
    pub fn builder() -> S1RunnerBuilder {
        S1RunnerBuilder::default()
    }

    /// The parsed Qwen3 config behind the runner.
    pub fn config(&self) -> &Qwen3Config {
        self.inner.config()
    }

    /// Topology of the loaded checkpoint.
    pub fn shape(&self) -> S1Shape {
        self.shape
    }

    pub fn device(&self) -> Device {
        self.inner.device()
    }

    /// The underlying Qwen3 runner, for callers that want the full API.
    pub fn inner(&self) -> &Qwen3Runner {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut Qwen3Runner {
        &mut self.inner
    }

    /// Transcript budget per pass (0 = chunking disabled).
    pub fn chunk_tokens(&self) -> usize {
        self.chunk_tokens
    }

    /// The weights path this runner was built from.
    pub fn weights_path(&self) -> &Path {
        &self.weights
    }

    /// The explicit `tokenizer.json` override, if one was given.
    pub fn tokenizer_path(&self) -> Option<&Path> {
        self.tokenizer.as_deref()
    }

    // ── Tokenizer bridge ──────────────────────────────────────────────

    /// Tokenize `text` with the checkpoint's Qwen3 BPE. Special tokens in the
    /// string (`<|im_start|>`, `<think>`, …) are matched as single ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        encode_text(&self.weights, self.tokenizer.as_deref(), text)
    }

    /// Detokenize, dropping special tokens.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        decode_text(&self.weights, self.tokenizer.as_deref(), ids)
    }

    /// Token count of `text` under the checkpoint's tokenizer.
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.encode(text)?.len())
    }

    /// The full prompt ids for one pass — the exact input the model sees.
    pub fn encode_prompt(&self, transcript: &str, controls: Controls) -> Result<Vec<u32>> {
        self.encode(&render_prompt(controls, transcript))
    }

    // ── Chunking ──────────────────────────────────────────────────────

    /// Split `transcript` into per-pass chunks at sentence (then word)
    /// boundaries. Returns a single chunk when it already fits, and an empty
    /// vector for whitespace-only input.
    pub fn chunk(&self, transcript: &str) -> Result<Vec<String>> {
        chunk_transcript(transcript, self.chunk_tokens, |s| self.count_tokens(s))
    }

    // ── Normalization ─────────────────────────────────────────────────

    /// Normalize with the default controls
    /// (`[Styling: semi-formal] [Structure: prose] [Context: general]`).
    pub fn normalize(&mut self, transcript: &str) -> Result<String> {
        self.normalize_with(transcript, Controls::new())
    }

    /// Normalize under explicit controls, chunking when the transcript exceeds
    /// the per-pass budget.
    ///
    /// An empty result is a valid answer, not a failure: filler-only input
    /// ("um") normalizes to the empty string by design.
    pub fn normalize_with(&mut self, transcript: &str, controls: Controls) -> Result<String> {
        let parts = self.normalize_chunks_with(transcript, controls)?;
        Ok(join_chunk_outputs(&parts, controls))
    }

    /// Like [`normalize_with`](Self::normalize_with) but returns one cleaned
    /// string per chunk, so the caller can join them their own way.
    pub fn normalize_chunks_with(
        &mut self,
        transcript: &str,
        controls: Controls,
    ) -> Result<Vec<String>> {
        let chunks = self.chunk(transcript)?;
        let mut out = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            out.push(self.normalize_pass(chunk, controls, |_| {})?);
        }
        Ok(out)
    }

    /// Single pass over `transcript` — no chunking, no join. `on_token` sees
    /// every generated id as it is produced (stop tokens are not emitted).
    pub fn normalize_pass(
        &mut self,
        transcript: &str,
        controls: Controls,
        mut on_token: impl FnMut(u32),
    ) -> Result<String> {
        // Nothing to normalize is not a degenerate prompt to run — it's the
        // empty string, same as the answer for filler-only input.
        if transcript.trim().is_empty() {
            return Ok(String::new());
        }
        let prompt = render_prompt(controls, transcript);
        let ids = self.encode(&prompt)?;
        if ids.is_empty() {
            return Ok(String::new());
        }
        let n_new = self
            .max_new_tokens
            .unwrap_or_else(|| recommended_max_new_tokens(ids.len()));

        let mut generated: Vec<u32> = Vec::new();
        let mut sink = |tok: u32| -> bool {
            if EOS_TOKENS.contains(&tok) {
                return false;
            }
            on_token(tok);
            generated.push(tok);
            true
        };

        // One batched prefill over the whole prompt, then decode.
        //
        // Reusing a KV snapshot of the fixed system prefix (via
        // `Qwen3Runner::cache_prefix`) looks attractive here — the prefix really
        // is byte-identical on every call — but it loses badly for this model:
        // the suffix it would have to replay *is the transcript*, and the reuse
        // path folds a suffix in one token at a time (`feed_continuation` →
        // `step_cached`), turning a single batched prefill into N sequential
        // decode steps. Measured on CPU with the card's own example: 38.5 s
        // cached vs 18.8 s plain, for the same 10 output tokens. The prompt
        // prefix is ~60 tokens and the transcript is unbounded, so the trade
        // never comes out ahead.
        self.inner.generate_stoppable(&ids, n_new, &mut sink)?;

        let text = self.decode(&generated)?;
        Ok(text.trim().to_string())
    }
}

/// Join per-chunk outputs with [`Controls::chunk_separator`], dropping the empty
/// ones — a filler-only chunk normalizes to nothing and must not leave a stray
/// separator behind.
pub fn join_chunk_outputs(parts: &[String], controls: Controls) -> String {
    parts
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(controls.chunk_separator())
}

// ─── Tokenizer bridge (feature-gated) ─────────────────────────────────

/// Tokenize with the checkpoint's BPE without borrowing a runner. Same bridge
/// as [`S1Runner::encode`]; useful when a `&mut S1Runner` is already live (a
/// streaming callback, say).
pub fn encode_ids(weights: &Path, tokenizer: Option<&Path>, text: &str) -> Result<Vec<u32>> {
    encode_text(weights, tokenizer, text)
}

/// Detokenize (special tokens dropped) without borrowing a runner.
pub fn decode_ids(weights: &Path, tokenizer: Option<&Path>, ids: &[u32]) -> Result<String> {
    decode_text(weights, tokenizer, ids)
}

#[cfg(feature = "tokenizer")]
fn encode_text(weights: &Path, tokenizer: Option<&Path>, text: &str) -> Result<Vec<u32>> {
    rlx_qwen35::encode_prompt_auto(weights, tokenizer, text)
}

#[cfg(not(feature = "tokenizer"))]
fn encode_text(_weights: &Path, _tokenizer: Option<&Path>, _text: &str) -> Result<Vec<u32>> {
    bail!("rlx-s1: text input needs the `tokenizer` feature (on by default)")
}

#[cfg(feature = "tokenizer")]
fn decode_text(weights: &Path, tokenizer: Option<&Path>, ids: &[u32]) -> Result<String> {
    rlx_qwen35::decode_ids_auto(weights, tokenizer, ids, /*skip_special*/ true)
}

#[cfg(not(feature = "tokenizer"))]
fn decode_text(_weights: &Path, _tokenizer: Option<&Path>, _ids: &[u32]) -> Result<String> {
    bail!("rlx-s1: detokenization needs the `tokenizer` feature (on by default)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::{Context, Structure, Styling};

    #[test]
    fn eos_tokens_match_generation_config() {
        assert_eq!(EOS_TOKENS, &[151645, 151643]);
    }

    #[test]
    fn reference_shape_matches_published_config_json() {
        // superwhisper/s1-mini config.json, which is byte-identical to
        // Qwen/Qwen3-0.6B's.
        assert_eq!(REFERENCE_SHAPE.num_hidden_layers, 28);
        assert_eq!(REFERENCE_SHAPE.hidden_size, 1024);
        assert_eq!(REFERENCE_SHAPE.intermediate_size, 3072);
        assert_eq!(REFERENCE_SHAPE.num_attention_heads, 16);
        assert_eq!(REFERENCE_SHAPE.num_key_value_heads, 8);
        assert_eq!(REFERENCE_SHAPE.head_dim, 128);
        assert_eq!(REFERENCE_SHAPE.vocab_size, 151_936);
        assert!(REFERENCE_SHAPE.matches_reference());
    }

    #[test]
    fn shape_mismatch_is_detected() {
        let mut other = REFERENCE_SHAPE;
        other.num_hidden_layers = 36;
        assert!(!other.matches_reference());
    }

    #[test]
    fn chunk_join_separators_follow_the_controls() {
        let parts = vec!["One.".to_string(), "Two.".to_string()];
        assert_eq!(join_chunk_outputs(&parts, Controls::new()), "One. Two.");
        let lists = Controls::new().structure(Structure::Lists);
        assert_eq!(join_chunk_outputs(&parts, lists), "One.\nTwo.");
        let email = Controls::new().context(Context::Email);
        assert_eq!(join_chunk_outputs(&parts, email), "One.\n\nTwo.");
    }

    #[test]
    fn chunk_join_drops_empty_passes() {
        // Filler-only chunks normalize to "" and must not leave stray
        // separators in the joined result.
        let parts = vec![String::new(), "Real text.".into(), "  ".into()];
        assert_eq!(join_chunk_outputs(&parts, Controls::new()), "Real text.");
        assert_eq!(join_chunk_outputs(&[], Controls::new()), "");
    }

    #[test]
    fn default_max_seq_covers_a_full_budget_chunk() {
        let b = S1Runner::builder().weights("/nonexistent");
        let chunk = b.chunk_tokens.unwrap_or(DEFAULT_CHUNK_TOKENS);
        let max_seq = chunk + recommended_max_new_tokens(chunk) + 128;
        assert!(max_seq >= chunk * 2, "decode ladder must cover in + out");
    }

    #[test]
    fn builder_requires_weights() {
        assert!(S1Runner::builder().build().is_err());
    }

    #[test]
    fn styling_axis_is_complete() {
        assert_eq!(Styling::ALL.len(), 4);
        assert_eq!(Structure::ALL.len(), 2);
        assert_eq!(Context::ALL.len(), 2);
    }
}
