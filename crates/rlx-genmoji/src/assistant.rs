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

//! The Open Genmoji prompt assistant.
//!
//! The emoji LoRA alone turns `flying pig` into a pig on a white background,
//! not an emoji: FLUX needs every feature spelled out (colors, an enlarged
//! head, facing the viewer, "3D lighting. no cast shadows."). Open Genmoji —
//! like the vendor's Genmoji, whose only language-model asset is "rephrase this
//! text" — has an LLM rewrite the user's words first, steered by a
//! *metaprompt* plus a few worked examples (`metaprompt/<name>.md` + `.json`,
//! fetched by `just fetch-genmoji`).
//!
//! The rewrite runs on a **local rlx language model** ([`LocalAssistant`],
//! default Qwen3.5-0.8B — `just fetch-qwen35-08b-instruct`) rather than the
//! reference's LM Studio server. The model is resolved through the shared
//! [`rlx_core::lm_registry`]: `config.json`'s `model_type` names the runner
//! family, and whichever constructor is registered for it builds the model.
//! This crate registers Qwen3.5 and Qwen3; a binary that links more families
//! (e.g. calls `rlx_models::run::register_builtin_lm_runners()`) can use any
//! of them with no change here. [`HttpAssistant`] remains for an
//! OpenAI-compatible endpoint (e.g. `rlx-openai`) that is already running.

use anyhow::{Context, Result, anyhow, bail};
use rlx_runtime::{Device, LmRunner};
use rlx_text::chat::{ChatMessage, ChatRenderOptions, ChatTemplate};
use std::path::{Path, PathBuf};

/// Default local assistant model (`just fetch-qwen35-08b-instruct`).
pub const DEFAULT_ASSISTANT_WEIGHTS: &str = "weights/lm/qwen3.5-0.8b";
/// Sampling temperature for [`HttpAssistant`] (the reference's 0.7).
pub const DEFAULT_TEMPERATURE: f32 = 0.7;

/// A metaprompt and its few-shot conversation.
#[derive(Debug, Clone)]
pub struct Metaprompt {
    pub name: String,
    pub text: String,
    pub history: Vec<ChatMessage>,
}

impl Metaprompt {
    /// Load `<dir>/<name>.md` and, if present, `<dir>/<name>.json`
    /// (`{"messages": [{role, content}, …]}`). `name` may also be a path to an `.md`.
    pub fn load(dir: &Path, name: &str) -> Result<Self> {
        let md = if name.ends_with(".md") {
            PathBuf::from(name)
        } else {
            dir.join(format!("{name}.md"))
        };
        let text = std::fs::read_to_string(&md)
            .with_context(|| format!("metaprompt {} (run `just fetch-genmoji`)", md.display()))?;
        let json = md.with_extension("json");
        let history = if json.is_file() {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&json)?)
                .with_context(|| format!("parsing {}", json.display()))?;
            v["messages"]
                .as_array()
                .context("few-shot history needs a `messages` array")?
                .iter()
                .map(|m| -> Result<ChatMessage> {
                    Ok(ChatMessage {
                        role: m["role"].as_str().context("message role")?.to_string(),
                        content: m["content"]
                            .as_str()
                            .context("message content")?
                            .to_string(),
                    })
                })
                .collect::<Result<_>>()?
        } else {
            Vec::new()
        };
        Ok(Self {
            name: md
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            text,
            history,
        })
    }

    /// The conversation sent to the LLM: few-shot history, then the
    /// metaprompt with the user's description appended (reference wording).
    pub fn messages(&self, user_prompt: &str) -> Vec<ChatMessage> {
        let mut m = self.history.clone();
        m.push(ChatMessage {
            role: "user".into(),
            content: format!("{}\n\nUSER PROMPT: \"{user_prompt}\"", self.text),
        });
        m
    }
}

/// Post-process an assistant reply the way the reference does (drop code
/// fences and newlines), also dropping any `<think>…</think>` block.
pub fn clean_reply(reply: &str) -> String {
    let mut s = reply.to_string();
    while let (Some(a), Some(b)) = (s.find("<think>"), s.find("</think>")) {
        if b < a {
            break;
        }
        s.replace_range(a..b + "</think>".len(), "");
    }
    s.replace("```", "").replace('\n', "").trim().to_string()
}

/// Something that rewrites a short description into a FLUX prompt.
pub trait PromptAssistant {
    fn complete(&mut self, messages: &[ChatMessage]) -> Result<String>;

    /// Log-probability (normalized over `choices`) of each of `choices` as the
    /// reply's first word, from the next-token logits; `None` when the backend
    /// cannot score.
    fn score(&mut self, messages: &[ChatMessage], choices: &[&str]) -> Result<Option<Vec<f32>>> {
        let _ = (messages, choices);
        Ok(None)
    }
}

/// An OpenAI-compatible `/v1/chat/completions` endpoint.
pub struct HttpAssistant {
    pub url: String,
    pub model: Option<String>,
    pub temperature: f32,
    pub max_tokens: usize,
}

impl HttpAssistant {
    pub fn new(url: &str) -> Self {
        let url = url.trim_end_matches('/');
        let url = if url.ends_with("/chat/completions") {
            url.to_string()
        } else if url.ends_with("/v1") {
            format!("{url}/chat/completions")
        } else {
            format!("{url}/v1/chat/completions")
        };
        Self {
            url,
            model: None,
            temperature: DEFAULT_TEMPERATURE,
            max_tokens: 256,
        }
    }
}

impl PromptAssistant for HttpAssistant {
    fn complete(&mut self, messages: &[ChatMessage]) -> Result<String> {
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();
        let mut body = serde_json::json!({
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": false,
        });
        if let Some(m) = &self.model {
            body["model"] = serde_json::Value::String(m.clone());
        }
        let resp = ureq::post(&self.url)
            .set("Content-Type", "application/json")
            .send_json(body)
            .map_err(|e| anyhow!("prompt assistant at {} failed: {e}", self.url))?;
        let v: serde_json::Value = resp.into_json()?;
        v["choices"][0]["message"]["content"]
            .as_str()
            .map(str::to_string)
            .with_context(|| format!("unexpected chat response: {v}"))
    }
}

/// Prefill width for the assistant: the metaprompt with its few-shot history
/// is ~880 tokens.
const ASSISTANT_PREFILL: usize = 1024;

/// Register the families this crate links (Qwen3.5, Qwen3) with
/// [`rlx_core::lm_registry`], keeping any constructor already registered for
/// the same name (an application's may be richer).
pub fn register_builtin_assistants() {
    use rlx_core::lm_registry::{
        LmBuildOptions, LmRunnerRegistration, lm_runner_is_registered, register_lm_runner,
    };
    fn qwen35(p: &Path, o: &LmBuildOptions) -> Result<Box<dyn LmRunner>> {
        // Text only (the HF dir also carries a vision tower), last-position
        // logits, and the prefill compiled once at the width the metaprompt
        // needs (~880 tokens) instead of growing 256 → 1024 on first use.
        let gguf = p.extension().is_some_and(|e| e == "gguf");
        let mut b = rlx_qwen35::Qwen35Runner::builder()
            .weights(p)
            .device(o.device_or_auto())
            .max_seq_opt(o.max_seq)
            .skip_auto_mmproj(true)
            .last_logits_only(true);
        if gguf {
            b = b.packed_weights(true);
        }
        if let Some(n) = o.max_seq {
            b = b.prefill_seq(ASSISTANT_PREFILL.min(n));
        }
        Ok(Box::new(Qwen35Assistant::new(b.build()?)))
    }
    fn qwen3(p: &Path, o: &LmBuildOptions) -> Result<Box<dyn LmRunner>> {
        Ok(Box::new(
            rlx_qwen3::Qwen3Runner::builder()
                .weights(p)
                .device(o.device_or_auto())
                .max_seq_opt(o.max_seq)
                .build()?,
        ))
    }
    for reg in [
        LmRunnerRegistration::new("qwen35", qwen35),
        LmRunnerRegistration::new("qwen3", qwen3),
    ] {
        if !lm_runner_is_registered(reg.runner) {
            register_lm_runner(reg);
        }
    }
}

/// Qwen3.5 scoring through the prefill graph generation already holds. Its
/// own `predict_logits` compiles a third graph with its own copy of the
/// parameters: +2.7 GB of GPU memory for the 0.8B assistant, which keeps
/// FLUX's denoiser from fitting beside it. Prefill logits are the correctly
/// scaled ones (only the decode path's differ), so the scores are unchanged.
///
/// Generation reuses the prompt prefix it shares with the previous call (the
/// metaprompt and its few-shot history, ~880 tokens): that prefix is folded
/// into a decode cache once, and each later rewrite restores the snapshot and
/// feeds only its own tokens, instead of re-running the whole prompt.
struct Qwen35Assistant {
    runner: rlx_qwen35::Qwen35Runner,
    /// Tokens the snapshot holds, plus the one still pending (`generate_continue`
    /// takes it first), so `prefix.len() - 1` tokens are folded.
    prefix: Vec<u32>,
    snapshot: Option<rlx_qwen35::Qwen35DecodeCache>,
    last: Vec<u32>,
}

/// Shortest shared prefix worth a snapshot.
const MIN_REUSED_PREFIX: usize = 64;

impl Qwen35Assistant {
    fn new(runner: rlx_qwen35::Qwen35Runner) -> Self {
        Self {
            runner,
            prefix: Vec::new(),
            snapshot: None,
            last: Vec::new(),
        }
    }
}

impl LmRunner for Qwen35Assistant {
    fn family(&self) -> &'static str {
        "qwen35"
    }

    fn vocab_size(&self) -> usize {
        LmRunner::vocab_size(&self.runner)
    }

    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.runner.prefill_get_last_logits(prompt_ids)
    }

    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let shared = self
            .last
            .iter()
            .zip(prompt_ids)
            .take_while(|(a, b)| a == b)
            .count();
        self.last = prompt_ids.to_vec();
        let reusable = |p: &[u32]| p.len() >= MIN_REUSED_PREFIX && prompt_ids.len() > p.len();
        if !(self.snapshot.is_some() && prompt_ids.starts_with(&self.prefix))
            && reusable(&prompt_ids[..shared])
        {
            // Decode-path prefill: bit-identical to decoding the prefix, which
            // is what `generate_continue` resumes from.
            self.runner.exact_prefill_seed(&prompt_ids[..shared - 1])?;
            self.snapshot = self.runner.decode_cache_checkpoint();
            self.prefix = prompt_ids[..shared].to_vec();
        }
        if self.snapshot.is_some() && prompt_ids.starts_with(&self.prefix) && reusable(&self.prefix)
        {
            self.runner.restore_decode_cache(self.snapshot.clone());
            return self.runner.generate_continue(
                &prompt_ids[self.prefix.len() - 1..],
                n_new,
                rlx_qwen35::SampleOpts::greedy(),
                &mut *on_token,
            );
        }
        LmRunner::generate(&mut self.runner, prompt_ids, n_new, on_token)
    }
}

/// Build a runner for the HF checkpoint in `dir` via [`rlx_core::lm_registry`],
/// with its weights from `gguf` when given (same family, packed: a Q8_0 file
/// keeps the weights at ~1 B each instead of f32).
pub fn open_lm(
    dir: &Path,
    gguf: Option<&Path>,
    device: Device,
    max_seq: usize,
) -> Result<Box<dyn LmRunner>> {
    register_builtin_assistants();
    let cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("{}/config.json", dir.display()))?,
    )?;
    let model_type = cfg["model_type"]
        .as_str()
        .or_else(|| cfg["text_config"]["model_type"].as_str())
        .context("config.json has no model_type")?;
    let runner = rlx_core::model_registry::runner_for_hf_model_type(model_type)
        .with_context(|| format!("no rlx runner family for model_type `{model_type}`"))?;
    let opts = rlx_core::lm_registry::LmBuildOptions::new(device).max_seq(max_seq);
    rlx_core::lm_registry::build_lm_runner(runner, gguf.unwrap_or(dir), &opts)
}

/// A local rlx language model as the assistant: any [`LmRunner`] plus the
/// checkpoint's own tokenizer and chat template (thinking turned off).
/// Decoding is greedy, so a description always rewrites the same way.
pub struct LocalAssistant {
    runner: Box<dyn LmRunner>,
    tokenizer: tokenizers::Tokenizer,
    template: ChatTemplate,
    eos: Vec<u32>,
    max_seq: usize,
    pub max_new_tokens: usize,
}

impl LocalAssistant {
    /// `dir` holds the weights, `config.json`, `tokenizer.json` and a
    /// `tokenizer_config.json` (or `chat_template.jinja`) with the chat template.
    pub fn open(dir: &Path, device: Device) -> Result<Self> {
        Self::open_with(dir, None, device)
    }

    /// [`Self::open`] with the weights from a GGUF of the same model; the
    /// tokenizer and chat template still come from `dir`.
    pub fn open_with(dir: &Path, gguf: Option<&Path>, device: Device) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("{}: {e}", dir.join("tokenizer.json").display()))?;
        let tcfg: serde_json::Value = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let source = match tcfg["chat_template"].as_str() {
            Some(t) => t.to_string(),
            None => std::fs::read_to_string(dir.join("chat_template.jinja"))
                .with_context(|| format!("no chat template in {}", dir.display()))?,
        };
        let template = ChatTemplate::from_source(source)?;
        let mut eos: Vec<u32> = Vec::new();
        if let Ok(g) = std::fs::read_to_string(dir.join("generation_config.json")) {
            let g: serde_json::Value = serde_json::from_str(&g)?;
            match &g["eos_token_id"] {
                serde_json::Value::Array(a) => {
                    eos.extend(a.iter().filter_map(|v| v.as_u64().map(|v| v as u32)))
                }
                v => eos.extend(v.as_u64().map(|v| v as u32)),
            }
        }
        for t in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>"] {
            eos.extend(tokenizer.token_to_id(t));
        }
        // Metaprompt + history (~880 tokens) + the request + the reply.
        let max_seq = 1536;
        let runner = open_lm(dir, gguf, device, max_seq)
            .with_context(|| format!("loading the assistant LM from {}", dir.display()))?;
        Ok(Self {
            runner,
            tokenizer,
            template,
            eos,
            max_seq,
            max_new_tokens: 160,
        })
    }
}

impl PromptAssistant for LocalAssistant {
    fn score(&mut self, messages: &[ChatMessage], choices: &[&str]) -> Result<Option<Vec<f32>>> {
        let text = self.template.render_with_options(
            messages,
            ChatRenderOptions {
                add_generation_prompt: true,
                enable_thinking: false,
                ..ChatRenderOptions::default()
            },
        )?;
        let ids = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("assistant tokenize: {e}"))?
            .get_ids()
            .to_vec();
        let logits = self.runner.predict_logits(&ids)?;
        let mut scores = Vec::with_capacity(choices.len());
        for c in choices {
            // Score every spelling the reply could open with; the best one counts.
            let cap = format!("{}{}", c[..1].to_ascii_uppercase(), &c[1..]);
            let mut l = f32::NEG_INFINITY;
            for v in [c.to_string(), cap, format!(" {c}")] {
                let tok = self
                    .tokenizer
                    .encode(v, false)
                    .map_err(|e| anyhow!("{e}"))?;
                if let Some(&first) = tok.get_ids().first() {
                    l = l.max(
                        logits
                            .get(first as usize)
                            .copied()
                            .unwrap_or(f32::NEG_INFINITY),
                    );
                }
            }
            scores.push(l);
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = m + scores.iter().map(|l| (l - m).exp()).sum::<f32>().ln();
        Ok(Some(scores.iter().map(|l| l - lse).collect()))
    }

    fn complete(&mut self, messages: &[ChatMessage]) -> Result<String> {
        let text = self.template.render_with_options(
            messages,
            ChatRenderOptions {
                add_generation_prompt: true,
                enable_thinking: false,
                ..ChatRenderOptions::default()
            },
        )?;
        let ids = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("assistant tokenize: {e}"))?
            .get_ids()
            .to_vec();
        if ids.len() + self.max_new_tokens > self.max_seq {
            bail!(
                "assistant prompt is {} tokens; the local context is {}",
                ids.len(),
                self.max_seq
            );
        }
        let mut out = Vec::new();
        let eos = self.eos.clone();
        self.runner.generate(&ids, self.max_new_tokens, &mut |t| {
            if eos.contains(&t) {
                return false;
            }
            out.push(t);
            true
        })?;
        self.tokenizer
            .decode(&out, true)
            .map_err(|e| anyhow!("assistant detokenize: {e}"))
    }
}

/// What an emoji depicts — decides its add-on phrases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    Object,
    Animal,
    Person,
}

impl Category {
    /// Parse a one-word answer (first match of object / animal / person).
    pub fn parse(reply: &str) -> Option<Self> {
        let r = reply.to_lowercase();
        let pos = |w: &str| r.find(w);
        [
            (
                pos("person").or(pos("human")).or(pos("people")),
                Category::Person,
            ),
            (pos("animal"), Category::Animal),
            (pos("object"), Category::Object),
        ]
        .into_iter()
        .filter_map(|(p, c)| p.map(|p| (p, c)))
        .min_by_key(|(p, _)| *p)
        .map(|(_, c)| c)
    }

    /// The metaprompt's add-on rules, applied by code rather than left to a
    /// small model: "cute." for anything not an object or a human; "enlarged
    /// head in cartoon style." only for animals; "head is turned towards
    /// viewer." only for humans and animals; "detailed texture." only for
    /// objects. People also get the vendor's own framing — every person Genmoji is a
    /// head-and-shoulders portrait.
    pub fn addons(self) -> &'static [&'static str] {
        match self {
            Category::Object => &["detailed texture."],
            Category::Animal => &[
                "cute.",
                "enlarged head in cartoon style.",
                "head is turned towards viewer.",
            ],
            Category::Person => &[
                "head and shoulders portrait.",
                "head is turned towards viewer.",
            ],
        }
    }
}

/// Phrases [`assemble`] owns: removed from a rewrite before the right ones go back.
const MANAGED: &[&str] = &[
    "cute",
    "enlarged head in cartoon style",
    "head is turned towards viewer",
    "head turned towards viewer",
    "detailed texture",
    "3d lighting",
    "no cast shadows",
    "head and shoulders portrait",
];

/// Places and backdrops a rewrite may invent ("owl in dark forest"). An
/// emoji is a subject on white; a scene in the prompt renders a scene, which
/// no matte turns back into a sticker.
const SETTINGS: &[&str] = &[
    "background",
    "forest",
    "woods",
    "jungle",
    "ocean",
    "sea",
    "water",
    "river",
    "lake",
    "sky",
    "night",
    "room",
    "field",
    "meadow",
    "garden",
    "park",
    "desert",
    "beach",
    "city",
    "street",
    "mountains",
    "snow",
    "grass",
    "branch",
    "tree",
    "landscape",
    "scene",
];
const SETTING_PREPS: &[&str] = &[
    "in", "on", "at", "under", "inside", "against", "among", "over",
];

/// Drop "in/on/… [up to two words] SETTING" from `phrase` unless the user
/// asked for that setting.
fn strip_settings(phrase: &str, user_prompt: &str) -> String {
    let user = user_prompt.to_lowercase();
    let words: Vec<&str> = phrase.split_whitespace().collect();
    let bare = |w: &str| {
        w.trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase()
    };
    let mut out: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < words.len() {
        if SETTING_PREPS.contains(&bare(words[i]).as_str()) {
            // The setting must end the clause: "in space suit" is not a place.
            let ends = |j: usize| {
                words.get(j + 1).is_none_or(|n| {
                    ["with", "and"].contains(&bare(n).as_str()) || words[j].ends_with(',')
                })
            };
            let hit = (i + 1..(i + 4).min(words.len())).find(|&j| {
                let w = bare(words[j]);
                SETTINGS.contains(&w.as_str()) && !user.contains(&w) && ends(j)
            });
            if let Some(j) = hit {
                i = j + 1;
                continue;
            }
        }
        out.push(words[i]);
        i += 1;
    }
    out.join(" ").trim_end_matches(',').to_string()
}

/// Genmoji-style prompt assembly: the rewrite's description sentences (minus
/// any setting the user did not ask for), then the category's add-ons, then
/// the fixed style suffix.
pub fn assemble(rewrite: &str, category: Category, user_prompt: &str) -> String {
    let keep: Vec<String> = rewrite
        .split('.')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .filter(|p| !MANAGED.contains(&p.to_lowercase().as_str()))
        .map(|p| strip_settings(p, user_prompt))
        .filter(|p| !p.is_empty())
        .map(|p| format!("{p}."))
        .collect();
    let mut parts = keep;
    if parts
        .first()
        .is_none_or(|p| !p.to_lowercase().starts_with("emoji of"))
    {
        parts.insert(0, "emoji of".into());
    }
    parts.extend(category.addons().iter().map(|s| s.to_string()));
    parts.push("3D lighting.".into());
    parts.push("no cast shadows.".into());
    parts.join(" ")
}

/// Ask the assistant what `user_prompt` depicts.
pub fn classify(assistant: &mut dyn PromptAssistant, user_prompt: &str) -> Result<Category> {
    classify_with(assistant, user_prompt, &mut None)
}

/// [`classify`], reusing the calibration scores in `baseline` (they depend
/// only on the assistant, so a resident caller computes them once).
pub fn classify_with(
    assistant: &mut dyn PromptAssistant,
    user_prompt: &str,
    baseline: &mut Option<Vec<f32>>,
) -> Result<Category> {
    const CHOICES: [(&str, Category); 3] = [
        ("object", Category::Object),
        ("animal", Category::Animal),
        ("person", Category::Person),
    ];
    let ask = |subject: &str| {
        [ChatMessage {
            role: "user".into(),
            content: format!(
                "Classify the main subject of an emoji. Answer with one word: object, animal, or person.\n\
                 A person is a human being in any job or role; an animal is a living creature or a \
                 character based on one; an object is anything else.\n\
                 taco -> object\nfireplace -> object\nrobber -> person\nteddy bear -> animal\n\
                 dog chef -> animal\nastronaut -> person\nrainbow -> object\n\
                 {subject} ->"
            ),
        }]
    };
    let msgs = ask(user_prompt);
    let names = CHOICES.map(|c| c.0);
    // Contextual calibration: subtract what the model answers for a
    // content-free subject, which removes its prior toward one label.
    if baseline.is_none() {
        *baseline = assistant.score(&ask("N/A"), &names)?;
    }
    if let (Some(s), Some(base)) = (assistant.score(&msgs, &names)?, baseline.as_deref()) {
        let cal: Vec<f32> = s.iter().zip(base).map(|(a, b)| a - b).collect();
        if rlx_ir::env::flag("RLX_GENMOJI_VERBOSE") {
            eprintln!("[rlx-genmoji] {user_prompt:?} raw {s:.2?} calibrated {cal:.2?}");
        }
        let k = (0..cal.len())
            .max_by(|&a, &b| cal[a].total_cmp(&cal[b]))
            .unwrap_or(0);
        return Ok(CHOICES[k].1);
    }
    let reply = assistant.complete(&msgs)?;
    Ok(Category::parse(&clean_reply(&reply)).unwrap_or(Category::Object))
}

/// Rewrite `user_prompt` with `assistant` under `metaprompt`.
pub fn rewrite(
    assistant: &mut dyn PromptAssistant,
    metaprompt: &Metaprompt,
    user_prompt: &str,
) -> Result<String> {
    let reply = assistant.complete(&metaprompt.messages(user_prompt))?;
    let cleaned = clean_reply(&reply);
    if cleaned.is_empty() {
        bail!("the prompt assistant returned nothing for {user_prompt:?}");
    }
    Ok(cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_reply_matches_reference_and_drops_thinking() {
        assert_eq!(
            clean_reply("```\nemoji of pig. cute.\n```"),
            "emoji of pig. cute."
        );
        assert_eq!(
            clean_reply("<think>\nhmm\n</think>\n\nemoji of cat."),
            "emoji of cat."
        );
    }

    #[test]
    fn messages_append_the_user_prompt_after_history() {
        let mp = Metaprompt {
            name: "t".into(),
            text: "META".into(),
            history: vec![ChatMessage {
                role: "user".into(),
                content: "x".into(),
            }],
        };
        let m = mp.messages("flying pig");
        assert_eq!(m.len(), 2);
        assert_eq!(m[1].content, "META\n\nUSER PROMPT: \"flying pig\"");
    }

    #[test]
    fn assembly_replaces_misplaced_addons() {
        let r = "emoji of a cozy wooden fireplace with glowing orange flames. detailed texture. head is turned towards viewer. 3D lighting. no cast shadows.";
        assert_eq!(
            assemble(r, Category::Object, "fireplace"),
            "emoji of a cozy wooden fireplace with glowing orange flames. detailed texture. 3D lighting. no cast shadows."
        );
        let r = "emoji of flying pink pig. white wings. cute.";
        assert_eq!(
            assemble(r, Category::Animal, "flying pig"),
            "emoji of flying pink pig. white wings. cute. enlarged head in cartoon style. head is turned towards viewer. 3D lighting. no cast shadows."
        );
        let r = "emoji of owl in dark forest with glowing eyes. cute.";
        assert_eq!(
            assemble(r, Category::Animal, "owl"),
            "emoji of owl with glowing eyes. cute. enlarged head in cartoon style. head is turned towards viewer. 3D lighting. no cast shadows."
        );
        let strip = |p: &str, u: &str| strip_settings(p, u);
        assert_eq!(
            strip("emoji of a clear blue fish swimming in the ocean", "fish"),
            "emoji of a clear blue fish swimming"
        );
        assert_eq!(
            strip("emoji of astronaut on the moon", "astronaut on the moon"),
            "emoji of astronaut on the moon"
        );
        assert_eq!(
            strip("emoji of fox in the snow", "fox in the snow"),
            "emoji of fox in the snow"
        );
        assert_eq!(
            strip(
                "emoji of teddy bear in space suit",
                "teddy bear in space suit"
            ),
            "emoji of teddy bear in space suit"
        );
        assert_eq!(
            strip("emoji of frog on a tree branch", "frog"),
            "emoji of frog"
        );
        assert_eq!(Category::parse("Person."), Some(Category::Person));
        assert_eq!(
            Category::parse("an animal, not an object"),
            Some(Category::Animal)
        );
    }

    #[test]
    fn endpoint_urls_normalize() {
        assert_eq!(
            HttpAssistant::new("http://h:1234").url,
            "http://h:1234/v1/chat/completions"
        );
        assert_eq!(
            HttpAssistant::new("http://h:1234/v1/").url,
            "http://h:1234/v1/chat/completions"
        );
        assert_eq!(
            HttpAssistant::new("http://h/v1/chat/completions").url,
            "http://h/v1/chat/completions"
        );
    }
}
