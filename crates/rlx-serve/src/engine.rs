// RLX models — OpenAI-compatible server.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! The generation engine: the host-driven decode loop and the seam the HTTP
//! routes talk to.
//!
//! Everything here is **host-side** — it consumes the raw `Vec<f32>` logits
//! that `LmRunner::{prefill_logits,decode_logits}` return, so it runs
//! identically on any `Device` (CPU / Metal / MLX). Sampling, `logit_bias`,
//! log-probs, EOS, and multi-token stop strings are all decided here.

use crate::stop;
use anyhow::Result;
use rlx_core::lm_host_driven::HostDriven;
use rlx_core::prompt_cache::PromptCache;
use rlx_qwen3::SampleOpts;
use rlx_qwen3::sampling::{apply_logit_bias, sample_token_with_history, softmax_logits};
use rlx_runtime::lm::{LmRunner, SessionSnapshot};
use rlx_text::{ChatTemplate, StreamingDetokenizer, TokenizerHandle};
use std::sync::Mutex;

/// A single generation request, already tokenized and mapped to rlx types.
#[derive(Debug, Clone)]
pub struct GenRequest {
    pub prompt_ids: Vec<u32>,
    pub opts: SampleOpts,
    pub bias: Vec<(u32, f32)>,
    pub max_tokens: usize,
    pub stop_ids: Vec<u32>,
    pub stop_strings: Vec<String>,
    /// `Some(k)` ⇒ include the chosen token's log-prob and the top-`k`
    /// alternatives.
    pub want_logprobs: Option<usize>,
}

/// Why generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

/// Log-prob detail for one generated token.
#[derive(Debug, Clone)]
pub struct TokenLogprob {
    pub token: u32,
    pub logprob: f32,
    /// `(token_id, logprob)` for the top-k alternatives.
    pub top: Vec<(u32, f32)>,
}

/// One item in the generation stream.
#[derive(Debug, Clone)]
pub enum StreamItem {
    Token {
        id: u32,
        text: String,
        logprob: Option<TokenLogprob>,
    },
    Done {
        finish_reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Error(String),
}

/// A chat message (role + content) before templating.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
    /// Tool calls this turn made, when `role` is `assistant`.
    ///
    /// A tool-use conversation is multi-turn: the model calls, the caller runs the
    /// tool, and the result comes back as a `tool` turn. For that to read as a
    /// continuation the model's own call has to still be in the transcript —
    /// without it the model is handed a result for a call it has no record of
    /// making, and typically calls the same tool again. Tool-capable chat
    /// templates render these into the assistant turn.
    ///
    /// `arguments` stays a JSON value: templates `tojson` it themselves, so
    /// pre-stringifying would double-encode.
    pub tool_calls: Vec<ChatToolCall>,
    /// Which tool produced this turn's content, when `role` is `tool`.
    pub tool_name: Option<String>,
}

/// A tool call attached to an assistant turn.
#[derive(Debug, Clone)]
pub struct ChatToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

impl ChatTurn {
    /// A turn with no tool structure.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_name: None,
        }
    }
}

/// `/v1/models` entry.
#[derive(Debug, Clone)]
pub struct ModelCard {
    pub id: String,
}

/// Chosen-token log-prob + top-k alternatives from a logits row.
pub fn top_logprobs(logits: &[f32], chosen: u32, k: usize) -> TokenLogprob {
    let probs = softmax_logits(logits);
    let lp = |p: f32| if p > 0.0 { p.ln() } else { f32::NEG_INFINITY };
    let chosen_lp = lp(probs.get(chosen as usize).copied().unwrap_or(0.0));
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top = idx
        .iter()
        .take(k)
        .map(|&i| (i as u32, lp(probs[i])))
        .collect();
    TokenLogprob {
        token: chosen,
        logprob: chosen_lp,
        top,
    }
}

/// The host-driven decode loop. Prefills, then samples one token at a time —
/// applying `logit_bias`, computing log-probs, detokenizing, and checking
/// EOS / stop strings — pushing [`StreamItem`]s to `emit`. `emit` returns
/// `false` to abort early (e.g. the client disconnected).
/// Make `runner` drivable by [`run_generation`] whatever its family has
/// implemented.
///
/// [`run_generation`] needs the host-driven pair — `prefill_logits` then
/// `decode_logits` — because sampling, logit bias, log-probs and stop handling
/// all live above the model. The trait's default for that pair is *unsupported*
/// and most families in this workspace have not overridden it, so serving them
/// fails on the first request rather than running slowly.
///
/// This wraps such a runner in [`HostDriven`], which answers each step by
/// re-running `predict_logits` over the whole context. The result is correct
/// and **quadratic**: prefer a family that implements the pair natively, which
/// this detects and passes through untouched on the first prefill.
pub fn host_drivable(runner: Box<dyn LmRunner>) -> Box<dyn LmRunner> {
    Box::new(HostDriven::new(runner))
}

pub fn run_generation(
    runner: &mut dyn LmRunner,
    req: &GenRequest,
    detok: &mut dyn FnMut(u32) -> Result<String>,
    emit: &mut dyn FnMut(StreamItem) -> bool,
) -> Result<()> {
    let logits = runner.prefill_logits(&req.prompt_ids)?;
    decode_loop(runner, req, logits, detok, emit)
}

/// The decode half of [`run_generation`]: sample from a pre-computed prefill
/// `logits`, then loop. Split out so the engine can prefill with prompt-cache
/// reuse and snapshot the prompt KV before decoding.
pub fn decode_loop(
    runner: &mut dyn LmRunner,
    req: &GenRequest,
    mut logits: Vec<f32>,
    detok: &mut dyn FnMut(u32) -> Result<String>,
    emit: &mut dyn FnMut(StreamItem) -> bool,
) -> Result<()> {
    let prompt_tokens = req.prompt_ids.len();
    let mut history = req.prompt_ids.clone();
    let mut emitted = String::new();
    let mut completion = 0usize;
    let mut finish = FinishReason::Length;

    for step in 0..req.max_tokens {
        apply_logit_bias(&mut logits, &req.bias);
        let tok = sample_token_with_history(&logits, &req.opts, &history, step as u64);

        // EOS id ends generation without emitting any text for the token.
        if req.stop_ids.contains(&tok) {
            finish = FinishReason::Stop;
            break;
        }

        let logprob = req.want_logprobs.map(|k| top_logprobs(&logits, tok, k));
        let delta = detok(tok)?;
        completion += 1;

        // Stop strings are matched on decoded text and may span tokens.
        let candidate = format!("{emitted}{delta}");
        if let Some(cut) = stop::first_stop(&candidate, &req.stop_strings) {
            let start = emitted.len().min(cut);
            let visible = &candidate[start..cut];
            if !visible.is_empty()
                && !emit(StreamItem::Token {
                    id: tok,
                    text: visible.to_string(),
                    logprob,
                })
            {
                return Ok(());
            }
            finish = FinishReason::Stop;
            break;
        }

        if !emit(StreamItem::Token {
            id: tok,
            text: delta,
            logprob,
        }) {
            return Ok(()); // client disconnected
        }
        emitted = candidate;
        history.push(tok);
        logits = runner.decode_logits(tok)?;
    }

    emit(StreamItem::Done {
        finish_reason: finish,
        prompt_tokens,
        completion_tokens: completion,
    });
    Ok(())
}

/// One image, decoded and ready for a vision tower.
///
/// 8-bit RGB, row-major, three bytes per pixel — the shape every vision
/// preprocessor in this workspace takes. Decoding happens at the edge, so the
/// engine never has to know PNG from JPEG.
#[derive(Debug, Clone)]
pub struct VisionImage {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl VisionImage {
    /// Whether the buffer is the size the dimensions claim.
    ///
    /// Checked rather than trusted: a short buffer reaching a vision tower is
    /// an out-of-bounds read deep inside a kernel, where the error says
    /// nothing about the image that caused it.
    pub fn is_consistent(&self) -> bool {
        self.width > 0 && self.height > 0 && self.rgb.len() == self.width * self.height * 3
    }
}

/// The server-facing seam. Hides how many runners / sessions back the model.
pub trait Engine: Send + Sync {
    fn model_cards(&self) -> Vec<ModelCard>;
    /// Encode chat messages (via the model's chat template) to token ids.
    fn encode_chat(&self, turns: &[ChatTurn]) -> Result<Vec<u32>>;
    /// As [`Self::encode_chat`], but also describing callable tools to the model.
    ///
    /// `tools` are OpenAI-style function specs. Tool-capable chat templates have
    /// a `tools` branch that renders them into a system block; without it the
    /// model is never told the tools exist, answers in prose, and the caller
    /// cannot tell "unsupported" from "the model chose not to call anything".
    ///
    /// Defaults to ignoring them, which is correct for an engine whose template
    /// has no tools branch — pair the override with [`Self::supports_tools`] so a
    /// caller can be told rather than guess.
    fn encode_chat_with_tools(
        &self,
        turns: &[ChatTurn],
        tools: &[serde_json::Value],
    ) -> Result<Vec<u32>> {
        let _ = tools;
        self.encode_chat(turns)
    }
    /// As [`Self::encode_chat_with_tools`], but also saying whether the model
    /// should reason before answering.
    ///
    /// A reasoning model's template gates its thought channel on
    /// `enable_thinking`. Left alone it renders whatever the template defaults
    /// to, which is what every caller got before this existed.
    fn encode_chat_full(
        &self,
        turns: &[ChatTurn],
        tools: &[serde_json::Value],
        think: Option<bool>,
    ) -> Result<Vec<u32>> {
        let _ = think;
        self.encode_chat_with_tools(turns, tools)
    }

    /// Whether this engine's chat template can be told about tools.
    fn supports_tools(&self) -> bool {
        false
    }

    /// Whether this engine's chat template has a thought channel to gate.
    fn supports_thinking(&self) -> bool {
        false
    }

    /// Whether this engine has a vision tower and can be given images.
    fn supports_vision(&self) -> bool {
        false
    }

    /// Render chat turns to the prompt *text* a multimodal entry point takes.
    ///
    /// The text path exists because a vision model does not take token ids: the
    /// image's embeddings are spliced into the sequence where its placeholder
    /// sits, so the splice has to happen before tokenizing is finished. An
    /// engine with no vision tower never needs this.
    fn render_chat(
        &self,
        turns: &[ChatTurn],
        tools: &[serde_json::Value],
        think: Option<bool>,
    ) -> Result<String> {
        let _ = (turns, tools, think);
        anyhow::bail!("this engine renders prompts to token ids, not to text")
    }

    /// Run a request whose prompt carries images.
    ///
    /// Refusing by default is deliberate. The alternative — dropping the images
    /// and running the text — answers as though the picture had been looked at,
    /// which is the worst way to be wrong about an image.
    fn run_vision(
        &self,
        req: &GenRequest,
        prompt: &str,
        images: &[VisionImage],
        emit: &mut dyn FnMut(StreamItem) -> bool,
    ) -> Result<()> {
        let _ = (req, prompt, images, emit);
        anyhow::bail!("this model has no vision tower, so it cannot be shown an image")
    }
    /// Encode a raw completion prompt to token ids.
    fn encode_text(&self, text: &str) -> Result<Vec<u32>>;

    /// Encode text that *continues* an already-templated prompt.
    ///
    /// Distinct from `encode_text` because that one starts a sequence and adds
    /// whatever special tokens the tokenizer prepends; appending those again in
    /// the middle of a rendered chat prompt would put a second start marker
    /// inside the assistant's turn. The default is `encode_text`, which is
    /// right for the many tokenizers that add nothing.
    fn encode_continuation(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_text(text)
    }
    /// EOS token ids that terminate generation.
    fn eos_ids(&self) -> Vec<u32>;
    /// Decode a single token id to its text (for logprob alternatives).
    fn decode_token(&self, id: u32) -> String;
    /// Run one request to completion, pushing items to `emit`. Blocking.
    fn run(&self, req: &GenRequest, emit: &mut dyn FnMut(StreamItem) -> bool);
}

/// Phase-0 engine: a single runner behind a `Mutex` (requests serialize).
/// Weight-sharing pools / continuous batching slot in behind this same trait
/// without changing the routes.
pub struct SingleEngine {
    runner: Mutex<Box<dyn LmRunner>>,
    tokenizer: TokenizerHandle,
    chat_template: Option<ChatTemplate>,
    eos_ids: Vec<u32>,
    model_id: String,
    /// Optional prompt-prefix KV cache (longest-prefix reuse). `None` ⇒ off.
    cache: Option<Mutex<PromptCache>>,
}

impl SingleEngine {
    pub fn new(
        runner: Box<dyn LmRunner>,
        tokenizer: TokenizerHandle,
        chat_template: Option<ChatTemplate>,
        eos_ids: Vec<u32>,
        model_id: String,
    ) -> Self {
        Self {
            runner: Mutex::new(runner),
            tokenizer,
            chat_template,
            eos_ids,
            model_id,
            cache: None,
        }
    }

    /// Enable the prompt-prefix KV cache, bounded to `cap_bytes` of KV data.
    pub fn with_prompt_cache(mut self, cap_bytes: usize) -> Self {
        self.cache = Some(Mutex::new(PromptCache::new(cap_bytes)));
        self
    }
}

impl Engine for SingleEngine {
    fn model_cards(&self) -> Vec<ModelCard> {
        vec![ModelCard {
            id: self.model_id.clone(),
        }]
    }

    fn encode_chat(&self, turns: &[ChatTurn]) -> Result<Vec<u32>> {
        let text = match &self.chat_template {
            Some(t) => {
                let msgs: Vec<rlx_text::ChatMessage> = turns
                    .iter()
                    .map(|m| rlx_text::ChatMessage {
                        role: m.role.clone(),
                        content: m.content.clone(),
                    })
                    .collect();
                t.render(&msgs, true)?
            }
            // No template: concatenate roles plainly.
            None => turns
                .iter()
                .map(|m| format!("{}: {}\n", m.role, m.content))
                .collect::<String>(),
        };
        self.encode_text(&text)
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, true)
    }

    fn eos_ids(&self) -> Vec<u32> {
        self.eos_ids.clone()
    }

    fn decode_token(&self, id: u32) -> String {
        self.tokenizer.decode(&[id], false).unwrap_or_default()
    }

    fn run(&self, req: &GenRequest, emit: &mut dyn FnMut(StreamItem) -> bool) {
        let mut runner = self.runner.lock().expect("runner mutex poisoned");

        // Prefill, reusing the longest cached prefix of this prompt if any.
        let reuse = self.cache.as_ref().and_then(|c| {
            c.lock()
                .expect("cache mutex poisoned")
                .longest_prefix(&req.prompt_ids)
        });
        let prefill = match reuse {
            Some((p, entry)) if p > 0 => {
                let snap = SessionSnapshot {
                    kv: entry.kv,
                    tokens: entry.tokens,
                };
                runner.prefill_logits_reusing(&req.prompt_ids, &snap, p)
            }
            _ => runner.prefill_logits(&req.prompt_ids),
        };
        let logits = match prefill {
            Ok(l) => l,
            Err(e) => {
                emit(StreamItem::Error(e.to_string()));
                return;
            }
        };

        // Snapshot the prompt-only KV (cache covers exactly the prompt now)
        // so future requests sharing this prefix can skip its prefill.
        if let Some(c) = &self.cache
            && let Some(snap) = runner.export_session()
        {
            c.lock()
                .expect("cache mutex poisoned")
                .insert(snap.tokens, snap.kv);
        }

        let mut detok = StreamingDetokenizer::new(&self.tokenizer, true);
        let mut detok_fn = |tok: u32| detok.push(tok);
        if let Err(e) = decode_loop(runner.as_mut(), req, logits, &mut detok_fn, emit) {
            emit(StreamItem::Error(e.to_string()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A runner that emits a fixed token sequence via one-hot logits, so the
    /// host-side loop can be tested with no model or tokenizer.
    struct MockRunner {
        vocab: usize,
        seq: Vec<u32>,
        pos: usize,
    }
    impl MockRunner {
        fn onehot(&self, tok: u32) -> Vec<f32> {
            let mut v = vec![0.0f32; self.vocab];
            v[tok as usize] = 10.0;
            v
        }
    }
    impl LmRunner for MockRunner {
        fn family(&self) -> &'static str {
            "mock"
        }
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn predict_logits(&mut self, _p: &[u32]) -> Result<Vec<f32>> {
            Ok(vec![0.0; self.vocab])
        }
        fn prefill_logits(&mut self, _p: &[u32]) -> Result<Vec<f32>> {
            self.pos = 0;
            Ok(self.onehot(self.seq[0]))
        }
        fn decode_logits(&mut self, _t: u32) -> Result<Vec<f32>> {
            self.pos += 1;
            Ok(self.onehot(self.seq[self.pos.min(self.seq.len() - 1)]))
        }
    }

    fn collect(req: GenRequest, mut runner: MockRunner) -> (Vec<u32>, String, FinishReason) {
        let mut ids = Vec::new();
        let mut text = String::new();
        let mut reason = FinishReason::Length;
        let mut detok = |tok: u32| Ok(format!("[{tok}]"));
        let mut emit = |item: StreamItem| {
            match item {
                StreamItem::Token { id, text: t, .. } => {
                    ids.push(id);
                    text.push_str(&t);
                }
                StreamItem::Done { finish_reason, .. } => reason = finish_reason,
                StreamItem::Error(e) => panic!("{e}"),
            }
            true
        };
        run_generation(&mut runner, &req, &mut detok, &mut emit).unwrap();
        (ids, text, reason)
    }

    fn req(max: usize) -> GenRequest {
        GenRequest {
            prompt_ids: vec![1, 2, 3],
            opts: SampleOpts::greedy(),
            bias: vec![],
            max_tokens: max,
            stop_ids: vec![],
            stop_strings: vec![],
            want_logprobs: None,
        }
    }

    /// A family with only `predict_logits` cannot be served, and `host_drivable`
    /// is what makes it servable.
    ///
    /// This is the gap itself: `run_generation` needs `prefill_logits`, the
    /// trait's default for it is an error, and eight of the ten model families
    /// in this workspace have not overridden it — so the first request fails
    /// rather than running slowly.
    #[test]
    fn host_drivable_makes_a_predict_only_runner_servable() {
        struct PredictOnly {
            next: std::cell::Cell<u32>,
        }
        impl LmRunner for PredictOnly {
            fn family(&self) -> &'static str {
                "predict-only"
            }
            fn vocab_size(&self) -> usize {
                16
            }
            fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
                // Argmax walks forward with the context length, so losing
                // context shows up as a repeated token.
                let mut v = vec![0.0f32; 16];
                let pick = (prompt_ids.len() % 16) as u32;
                self.next.set(pick);
                v[pick as usize] = 1.0;
                Ok(v)
            }
        }

        // Unwrapped: the first call fails, and says why.
        let mut bare: Box<dyn LmRunner> = Box::new(PredictOnly {
            next: std::cell::Cell::new(0),
        });
        let err = run_generation(
            bare.as_mut(),
            &req(3),
            &mut |_| Ok(String::new()),
            &mut |_| true,
        )
        .expect_err("a predict-only family cannot be served");
        assert!(
            format!("{err:#}").contains("unsupported"),
            "expected the trait's unsupported error, got: {err:#}"
        );

        // Wrapped: the same request runs, and the ids follow the context length.
        let mut wrapped = host_drivable(Box::new(PredictOnly {
            next: std::cell::Cell::new(0),
        }));
        let mut ids = Vec::new();
        run_generation(
            wrapped.as_mut(),
            &req(3),
            &mut |_| Ok(String::new()),
            &mut |item| {
                if let StreamItem::Token { id, .. } = item {
                    ids.push(id);
                }
                true
            },
        )
        .expect("the wrapped runner serves");
        assert_eq!(
            ids,
            vec![3, 4, 5],
            "prompt is 3 long, so context grows 3,4,5"
        );
    }

    #[test]
    fn generates_until_length() {
        let runner = MockRunner {
            vocab: 16,
            seq: vec![5, 6, 7, 8],
            pos: 0,
        };
        let (ids, text, reason) = collect(req(3), runner);
        assert_eq!(ids, vec![5, 6, 7]);
        assert_eq!(text, "[5][6][7]");
        assert_eq!(reason, FinishReason::Length);
    }

    #[test]
    fn stops_on_eos_id() {
        let runner = MockRunner {
            vocab: 16,
            seq: vec![5, 6, 9, 8],
            pos: 0,
        };
        let mut r = req(10);
        r.stop_ids = vec![9];
        let (ids, _t, reason) = collect(r, runner);
        assert_eq!(ids, vec![5, 6]); // 9 is EOS, not emitted
        assert_eq!(reason, FinishReason::Stop);
    }

    #[test]
    fn stops_on_stop_string() {
        // Tokens decode to "[5][6]..." — stop at the literal "[6]".
        let runner = MockRunner {
            vocab: 16,
            seq: vec![5, 6, 7],
            pos: 0,
        };
        let mut r = req(10);
        r.stop_strings = vec!["[6]".to_string()];
        let (_ids, text, reason) = collect(r, runner);
        assert_eq!(text, "[5]"); // truncated before the stop string
        assert_eq!(reason, FinishReason::Stop);
    }

    #[test]
    fn logprobs_are_returned() {
        let runner = MockRunner {
            vocab: 16,
            seq: vec![5],
            pos: 0,
        };
        let mut r = req(1);
        r.want_logprobs = Some(3);
        let mut got = None;
        let mut detok = |tok: u32| Ok(format!("[{tok}]"));
        let mut emit = |item: StreamItem| {
            if let StreamItem::Token { logprob, .. } = item {
                got = logprob;
            }
            true
        };
        run_generation(&mut { runner }, &r, &mut detok, &mut emit).unwrap();
        let lp = got.expect("logprob present");
        assert_eq!(lp.token, 5);
        assert!(lp.logprob <= 0.0);
        assert_eq!(lp.top.len(), 3);
        assert_eq!(lp.top[0].0, 5); // argmax is the one-hot token
    }
}
