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

//! Transformers-style one-liner inference for TinyLlama.
//!
//! `TinyLlamaRunner` speaks token ids; this module wraps it with the pieces
//! `huggingface/transformers` gives you for free — auto-download, a cached
//! tokenizer, chat templates, EOS-aware decoding, and streaming — so text
//! goes in and text comes out:
//!
//! ```no_run
//! use rlx_tinyllama::pipeline::{TextGeneration, GenerationConfig};
//!
//! // Downloads from the Hub on first use (or point at a local dir / .gguf / .safetensors).
//! let mut pipe = TextGeneration::from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")?;
//!
//! // Raw completion — text in, text out.
//! let out = pipe.generate("Once upon a time", &GenerationConfig::default())?;
//! println!("{out}");
//!
//! // Chat — applies the model's chat template for you.
//! use rlx_tinyllama::pipeline::ChatMessage;
//! let reply = pipe.chat(&[ChatMessage::user("Name three primary colors.")],
//!                       &GenerationConfig::default())?;
//! println!("{reply}");
//! # anyhow::Ok(())
//! ```
//!
//! Mirrors `transformers`' `pipeline("text-generation", model=…)`.

use anyhow::{Context, Result, anyhow};
use rlx_llama32::SampleOpts;
use rlx_runtime::Device;
use rlx_text::chat::{ChatRenderOptions, ChatTemplate};
use rlx_text::{TokenizerHandle, incremental_emit, load_tokenizer};
use serde_json::Value as Json;
use std::path::{Path, PathBuf};

pub use rlx_text::chat::ChatMessage;

use crate::TinyLlamaRunner;

/// TinyLlama's Zephyr-style chat template, used when a checkpoint ships no
/// `chat_template` (e.g. a bare GGUF). Kept deliberately whitespace-clean.
const TINYLLAMA_CHAT_TEMPLATE: &str = concat!(
    "{% for message in messages %}",
    "{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + eos_token + '\n' }}",
    "{% elif message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + eos_token + '\n' }}",
    "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + eos_token + '\n' }}",
    "{% endif %}{% endfor %}",
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
);

/// Decoding knobs, mirroring `transformers.GenerationConfig` (the subset that
/// maps onto RLX's sampler). All fields have HF-compatible defaults.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to sample (generation also stops at EOS).
    pub max_new_tokens: usize,
    /// Softmax temperature. `0.0` ⇒ greedy/argmax (deterministic).
    pub temperature: f32,
    /// Nucleus (top-p) cutoff. `1.0` ⇒ disabled.
    pub top_p: f32,
    /// Top-k cutoff. `None` ⇒ disabled.
    pub top_k: Option<u32>,
    /// Repetition penalty (`> 1.0` discourages repeats). `1.0` ⇒ disabled.
    pub repetition_penalty: f32,
    /// Drop special tokens (BOS/EOS/role markers) from the returned text.
    pub skip_special_tokens: bool,
    /// Let the tokenizer add its special tokens (e.g. BOS) when encoding the
    /// prompt. Matches `transformers`' default of `add_special_tokens=True`.
    pub add_special_tokens: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: 1.0,
            skip_special_tokens: true,
            add_special_tokens: true,
        }
    }
}

impl GenerationConfig {
    /// Greedy defaults (same as [`Default`]). Deterministic.
    pub fn greedy() -> Self {
        Self::default()
    }

    /// Temperature + nucleus sampling preset.
    pub fn sampling(temperature: f32, top_p: f32) -> Self {
        Self {
            temperature,
            top_p,
            ..Self::default()
        }
    }

    /// Builder-style override for the new-token budget.
    pub fn with_max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }

    /// Builder-style temperature override.
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Translate into RLX sampler options. `temperature <= 0` selects greedy
    /// (deterministic argmax); otherwise temperature + top-p/top-k sampling.
    pub fn to_sample_opts(&self) -> SampleOpts {
        if self.temperature <= 0.0 {
            return SampleOpts {
                repetition_penalty: self.repetition_penalty,
                ..SampleOpts::greedy()
            };
        }
        SampleOpts {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k.map(|k| k as usize).unwrap_or(0),
            repetition_penalty: self.repetition_penalty,
            greedy: false,
            ..SampleOpts::greedy()
        }
    }
}

/// Loaded, ready-to-run TinyLlama text-generation pipeline: weights +
/// tokenizer + chat template in one handle.
pub struct TextGeneration {
    runner: TinyLlamaRunner,
    tokenizer: TokenizerHandle,
    chat_template: ChatTemplate,
    eos_ids: Vec<u32>,
}

impl TextGeneration {
    /// Load a model by Hugging Face id (downloaded and cached on first use)
    /// or by local path (a directory, a `.safetensors` file, or a `.gguf`
    /// file). Runs on CPU; use [`from_pretrained_on`](Self::from_pretrained_on)
    /// or [`builder`](Self::builder) to pick a device.
    pub fn from_pretrained(model: &str) -> Result<Self> {
        Self::builder(model).build()
    }

    /// Like [`from_pretrained`](Self::from_pretrained) but on a chosen device.
    pub fn from_pretrained_on(model: &str, device: Device) -> Result<Self> {
        Self::builder(model).device(device).build()
    }

    /// Start a builder for finer control (device, `max_seq`, packed GGUF, an
    /// explicit tokenizer path).
    pub fn builder(model: &str) -> TextGenerationBuilder {
        TextGenerationBuilder::new(model)
    }

    /// Generate a completion for `prompt` (text in, text out). Stops at the
    /// model's EOS or after `cfg.max_new_tokens`, whichever comes first.
    pub fn generate(&mut self, prompt: &str, cfg: &GenerationConfig) -> Result<String> {
        self.generate_stream(prompt, cfg, |_| {})
    }

    /// Streaming variant of [`generate`](Self::generate): `on_piece` is called
    /// with each newly-stable chunk of text as it is produced. Returns the
    /// full generated text as well.
    pub fn generate_stream(
        &mut self,
        prompt: &str,
        cfg: &GenerationConfig,
        on_piece: impl FnMut(&str),
    ) -> Result<String> {
        let ids = self
            .tokenizer
            .encode(prompt, cfg.add_special_tokens)
            .context("encoding prompt")?;
        self.run(&ids, cfg, on_piece)
    }

    /// Apply the chat template to `messages` and generate the assistant's
    /// reply. Mirrors `tokenizer.apply_chat_template(...)` + `model.generate`.
    pub fn chat(&mut self, messages: &[ChatMessage], cfg: &GenerationConfig) -> Result<String> {
        self.chat_stream(messages, cfg, |_| {})
    }

    /// Streaming variant of [`chat`](Self::chat).
    pub fn chat_stream(
        &mut self,
        messages: &[ChatMessage],
        cfg: &GenerationConfig,
        on_piece: impl FnMut(&str),
    ) -> Result<String> {
        let prompt = self.apply_chat_template(messages)?;
        // The chat template already injects role markers / EOS; the tokenizer
        // still contributes BOS via `add_special_tokens`.
        let ids = self
            .tokenizer
            .encode(&prompt, cfg.add_special_tokens)
            .context("encoding chat prompt")?;
        self.run(&ids, cfg, on_piece)
    }

    /// Render `messages` through the chat template into the raw prompt string
    /// (with a trailing generation prompt), without generating.
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        self.chat_template.render_with_options(
            messages,
            ChatRenderOptions::user_turn(true),
        )
    }

    /// The cached tokenizer (encode/decode without re-reading `tokenizer.json`).
    pub fn tokenizer(&self) -> &TokenizerHandle {
        &self.tokenizer
    }

    /// End-of-sequence ids that halt generation.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    /// Mutable access to the underlying typed runner for advanced control.
    pub fn runner_mut(&mut self) -> &mut TinyLlamaRunner {
        &mut self.runner
    }

    // Shared decode loop: prefill `prompt_ids`, sample up to `max_new_tokens`,
    // stream stable text out, and stop on EOS. Returns only the generated
    // span (not the prompt).
    //
    // The detokenizer is *seeded* with the prompt ids so incremental decoding
    // sees the prompt as left-context. This matters for SentencePiece (Llama):
    // the leading space of the first generated token is decided by context, so
    // decoding generated ids in isolation drops it ("colors are" + "red" →
    // "arered"). Emitting only the text past the prompt keeps that boundary
    // space correct.
    fn run(
        &mut self,
        prompt_ids: &[u32],
        cfg: &GenerationConfig,
        mut on_piece: impl FnMut(&str),
    ) -> Result<String> {
        self.runner.set_sample(cfg.to_sample_opts());

        // Disjoint field borrows: `runner` (mut) vs `tokenizer`/`eos_ids` (shared).
        let tokenizer = &self.tokenizer;
        let eos_ids = &self.eos_ids;
        let skip = cfg.skip_special_tokens;

        let mut ids = prompt_ids.to_vec();
        // Byte offset already accounted for = the decoded prompt length.
        let mut emitted = tokenizer.decode(&ids, skip).context("decoding prompt")?.len();
        let mut generated = String::new();
        let mut decode_err: Option<anyhow::Error> = None;

        self.runner
            .generate_until(prompt_ids, cfg.max_new_tokens, |tok| {
                ids.push(tok);
                match incremental_emit(tokenizer, &ids, emitted, skip) {
                    Ok((delta, new_emitted)) => {
                        emitted = new_emitted;
                        if !delta.is_empty() {
                            generated.push_str(&delta);
                            on_piece(&delta);
                        }
                    }
                    Err(e) => {
                        decode_err = Some(e);
                        return false;
                    }
                }
                !eos_ids.contains(&tok)
            })
            .context("generation")?;

        if let Some(e) = decode_err {
            return Err(e);
        }

        // Flush any tail held back past `emitted` (e.g. a completed multi-byte
        // char whose replacement run was pending).
        let full = tokenizer.decode(&ids, skip).context("decoding output")?;
        if emitted < full.len() {
            let tail = &full[emitted..];
            generated.push_str(tail);
            on_piece(tail);
        }
        Ok(generated)
    }
}

/// Builder for [`TextGeneration`].
pub struct TextGenerationBuilder {
    model: String,
    device: Device,
    max_seq: usize,
    tokenizer_path: Option<PathBuf>,
    packed_weights: Option<bool>,
}

impl TextGenerationBuilder {
    fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            device: Device::Cpu,
            max_seq: 2048,
            tokenizer_path: None,
            packed_weights: None,
        }
    }

    /// Execution device (default [`Device::Cpu`]).
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// KV-cache / compile sequence cap (default 2048, TinyLlama's max context).
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = n;
        self
    }

    /// Explicit `tokenizer.json` path (otherwise resolved next to the weights).
    pub fn tokenizer(mut self, path: impl Into<PathBuf>) -> Self {
        self.tokenizer_path = Some(path.into());
        self
    }

    /// Force packed-GGUF decode on/off (default: auto — packed for large GGUF).
    pub fn packed_weights(mut self, on: bool) -> Self {
        self.packed_weights = Some(on);
        self
    }

    /// Resolve the source, load weights + tokenizer + chat template, and
    /// build the pipeline.
    pub fn build(self) -> Result<TextGeneration> {
        let source = resolve_source(&self.model)?;

        let mut rb = TinyLlamaRunner::builder()
            .weights(&source.weights)
            .device(self.device)
            .max_seq(self.max_seq);
        if let Some(p) = self.packed_weights {
            rb = rb.packed_weights(p);
        }
        let runner = rb.build()?;

        let tokenizer_path = self
            .tokenizer_path
            .or(source.tokenizer_json.clone())
            .ok_or_else(|| {
                anyhow!(
                    "no tokenizer.json found for {:?}. Pass one via \
                     TextGeneration::builder(..).tokenizer(path), or place \
                     tokenizer.json next to the weights.",
                    source.weights
                )
            })?;
        let tokenizer = load_tokenizer(&tokenizer_path)
            .with_context(|| format!("loading tokenizer {tokenizer_path:?}"))?;

        let chat_template = load_chat_template(&source)?;
        let eos_ids = collect_eos_ids(&source, &tokenizer, &chat_template);

        Ok(TextGeneration {
            runner,
            tokenizer,
            chat_template,
            eos_ids,
        })
    }
}

/// Resolved on-disk layout: the weights file to load plus, when known, the
/// sibling tokenizer / config files used for chat templates and EOS ids.
struct ResolvedSource {
    weights: PathBuf,
    tokenizer_json: Option<PathBuf>,
    config_json: Option<PathBuf>,
    tokenizer_config_json: Option<PathBuf>,
    generation_config_json: Option<PathBuf>,
    is_gguf: bool,
}

fn resolve_source(model: &str) -> Result<ResolvedSource> {
    let path = Path::new(model);
    if path.exists() {
        return resolve_local(path);
    }
    // Not a local path — treat as a Hugging Face repo id.
    resolve_hf(model)
}

fn resolve_local(path: &Path) -> Result<ResolvedSource> {
    let weights = if path.is_dir() {
        find_weights_in_dir(path)?
    } else {
        path.to_path_buf()
    };
    let dir = weights.parent().unwrap_or(Path::new("."));
    let is_gguf = has_ext(&weights, "gguf");
    Ok(ResolvedSource {
        tokenizer_json: exists_opt(dir.join("tokenizer.json")),
        config_json: exists_opt(dir.join("config.json")),
        tokenizer_config_json: exists_opt(dir.join("tokenizer_config.json")),
        generation_config_json: exists_opt(dir.join("generation_config.json")),
        is_gguf,
        weights,
    })
}

fn find_weights_in_dir(dir: &Path) -> Result<PathBuf> {
    let canonical = dir.join("model.safetensors");
    if canonical.is_file() {
        return Ok(canonical);
    }
    let mut first_st: Option<PathBuf> = None;
    let mut first_gguf: Option<PathBuf> = None;
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {dir:?}"))? {
        let p = entry?.path();
        if has_ext(&p, "safetensors") && first_st.is_none() {
            first_st = Some(p);
        } else if has_ext(&p, "gguf") && first_gguf.is_none() {
            first_gguf = Some(p);
        }
    }
    first_st.or(first_gguf).ok_or_else(|| {
        anyhow!("no .safetensors or .gguf weights found in directory {dir:?}")
    })
}

#[cfg(feature = "hf-download")]
fn resolve_hf(model: &str) -> Result<ResolvedSource> {
    use crate::download::default_hf_cache_dir;

    let cache = default_hf_cache_dir();
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache)
        .build()
        .context("hf_hub ApiBuilder")?;
    let repo = api.model(model.to_string());

    // config.json anchors the snapshot dir and is required to validate dims.
    let config = repo
        .get("config.json")
        .with_context(|| format!("downloading config.json from {model}"))?;
    let dir = config
        .parent()
        .context("snapshot dir")?
        .to_path_buf();

    // Best-effort auxiliary files.
    let tokenizer_json = repo.get("tokenizer.json").ok();
    let tokenizer_config_json = repo.get("tokenizer_config.json").ok();
    let generation_config_json = repo.get("generation_config.json").ok();
    let _ = repo.get("special_tokens_map.json");

    // Weights: index → shards, else single-file.
    let weights = download_safetensors(&repo, &dir)?;

    Ok(ResolvedSource {
        weights,
        tokenizer_json,
        config_json: Some(config),
        tokenizer_config_json,
        generation_config_json,
        is_gguf: false,
    })
}

#[cfg(feature = "hf-download")]
fn download_safetensors(repo: &hf_hub::api::sync::ApiRepo, dir: &Path) -> Result<PathBuf> {
    if let Ok(index) = repo.get("model.safetensors.index.json") {
        let text = std::fs::read_to_string(&index)?;
        let parsed: Json = serde_json::from_str(&text).context("parse safetensors index")?;
        if let Some(map) = parsed.get("weight_map").and_then(|m| m.as_object()) {
            let mut shards: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            shards.sort();
            shards.dedup();
            for shard in &shards {
                repo.get(shard)
                    .with_context(|| format!("downloading shard {shard}"))?;
            }
            if let Some(first) = shards.first() {
                return Ok(dir.join(first));
            }
        }
    }
    let single = repo
        .get("model.safetensors")
        .context("downloading model.safetensors")?;
    Ok(single)
}

#[cfg(not(feature = "hf-download"))]
fn resolve_hf(model: &str) -> Result<ResolvedSource> {
    anyhow::bail!(
        "{model:?} is not a local path, and Hugging Face download is disabled. \
         Rebuild with `--features hf-download` (or `pipeline`), or pass a local \
         directory / .safetensors / .gguf path."
    )
}

fn load_chat_template(source: &ResolvedSource) -> Result<ChatTemplate> {
    // 1. GGUF metadata.
    if source.is_gguf {
        if let Ok(t) = ChatTemplate::from_gguf(&source.weights) {
            return Ok(t);
        }
    }
    // 2. tokenizer_config.json `chat_template` (HF safetensors convention).
    if let Some(cfg) = &source.tokenizer_config_json {
        if let Some(json) = read_json(cfg) {
            if let Some(src) = json.get("chat_template").and_then(Json::as_str) {
                let bos = json_token(json.get("bos_token"));
                let eos = json_token(json.get("eos_token"));
                return ChatTemplate::from_source(src)
                    .map(|t| t.with_tokens(bos, eos.or_else(|| Some("</s>".to_string()))));
            }
        }
    }
    // 3. Built-in TinyLlama fallback.
    ChatTemplate::from_source(TINYLLAMA_CHAT_TEMPLATE)
        .map(|t| t.with_tokens(Some("<s>".to_string()), Some("</s>".to_string())))
}

fn collect_eos_ids(
    source: &ResolvedSource,
    tokenizer: &TokenizerHandle,
    chat_template: &ChatTemplate,
) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();
    let mut push = |id: u32| {
        if !ids.contains(&id) {
            ids.push(id);
        }
    };

    for path in [&source.generation_config_json, &source.config_json]
        .into_iter()
        .flatten()
    {
        if let Some(json) = read_json(path) {
            push_eos_from_json(json.get("eos_token_id"), &mut push);
        }
    }

    // The chat template's EOS string → id, and the classic Llama </s>.
    if let Some(eos) = chat_template.eos_token() {
        if let Some(id) = tokenizer.raw().token_to_id(eos) {
            push(id);
        }
    }
    if let Some(id) = tokenizer.raw().token_to_id("</s>") {
        push(id);
    }
    ids
}

fn push_eos_from_json(v: Option<&Json>, push: &mut impl FnMut(u32)) {
    match v {
        Some(Json::Number(n)) => {
            if let Some(u) = n.as_u64() {
                push(u as u32);
            }
        }
        Some(Json::Array(arr)) => {
            for item in arr {
                if let Some(u) = item.as_u64() {
                    push(u as u32);
                }
            }
        }
        _ => {}
    }
}

// ── small helpers ────────────────────────────────────────────────────

fn has_ext(path: &Path, ext: &str) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case(ext))
        .unwrap_or(false)
}

fn exists_opt(path: PathBuf) -> Option<PathBuf> {
    path.is_file().then_some(path)
}

fn read_json(path: &Path) -> Option<Json> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// HF special-token fields are either a bare string or `{ "content": "…" }`.
fn json_token(v: Option<&Json>) -> Option<String> {
    match v? {
        Json::String(s) => Some(s.clone()),
        Json::Object(map) => map.get("content").and_then(Json::as_str).map(str::to_string),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_config_defaults_are_greedy() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.temperature, 0.0);
        assert!(cfg.to_sample_opts().greedy);
        assert_eq!(cfg.max_new_tokens, 256);
        assert!(cfg.skip_special_tokens);
    }

    #[test]
    fn sampling_preset_is_not_greedy() {
        let opts = GenerationConfig::sampling(0.7, 0.95).to_sample_opts();
        assert!(!opts.greedy);
        assert!((opts.temperature - 0.7).abs() < 1e-6);
        assert!((opts.top_p - 0.95).abs() < 1e-6);
    }

    #[test]
    fn builtin_chat_template_renders_zephyr_format() {
        let t = ChatTemplate::from_source(TINYLLAMA_CHAT_TEMPLATE)
            .unwrap()
            .with_tokens(Some("<s>".into()), Some("</s>".into()));
        let out = t
            .render_with_options(
                &[
                    ChatMessage::system("Be brief."),
                    ChatMessage::user("Hi"),
                ],
                ChatRenderOptions::user_turn(true),
            )
            .unwrap();
        assert_eq!(
            out,
            "<|system|>\nBe brief.</s>\n<|user|>\nHi</s>\n<|assistant|>\n"
        );
    }

    #[test]
    fn json_token_handles_string_and_object() {
        let s = serde_json::json!("</s>");
        assert_eq!(json_token(Some(&s)).as_deref(), Some("</s>"));
        let o = serde_json::json!({ "content": "<eos>", "lstrip": false });
        assert_eq!(json_token(Some(&o)).as_deref(), Some("<eos>"));
        assert_eq!(json_token(None), None);
    }

    #[test]
    fn eos_json_parses_scalar_and_array() {
        let mut got: Vec<u32> = Vec::new();
        push_eos_from_json(Some(&serde_json::json!(2)), &mut |id| got.push(id));
        push_eos_from_json(Some(&serde_json::json!([2, 32000])), &mut |id| got.push(id));
        assert_eq!(got, vec![2, 2, 32000]);
    }

    #[test]
    fn has_ext_is_case_insensitive() {
        assert!(has_ext(Path::new("m.GGUF"), "gguf"));
        assert!(has_ext(Path::new("m.safetensors"), "safetensors"));
        assert!(!has_ext(Path::new("m.bin"), "gguf"));
    }
}
