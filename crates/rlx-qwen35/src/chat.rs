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

//! ChatML prompt formatting for Qwen3.5 / Qwen3.6.
//!
//! HF checkpoints ship a Jinja chat template in `tokenizer_config.json`; the
//! Rust `tokenizers` crate does not apply it, so we emit the canonical ChatML
//! wire format and tokenize that string.
//!
//! **Thinking:** when [`ChatFormatOpts::enable_thinking`] is false, the
//! assistant turn is primed with [`EMPTY_THINK_BLOCK`] — the same hard switch
//! as HF `apply_chat_template(..., enable_thinking=False)`. CLI: `--no-think`
//! / `--fast`. Use [`split_thinking`] to separate `<think>…</think>` from the
//! answer for display.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

use super::tokenizer::{encode_prompt, resolve_tokenizer_path};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// Empty think block used to disable reasoning (HF hard switch).
pub const EMPTY_THINK_BLOCK: &str = "<think>\n\n</think>\n\n";

/// Open think prefix when thinking is enabled (HF `enable_thinking=True`).
pub const OPEN_THINK_PREFIX: &str = "<think>\n";

/// Early-stop cue when a thinking budget is exhausted mid-`<think>`.
pub const THINK_BUDGET_CLOSE: &str = "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n";

/// Options for [`format_chatml_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatFormatOpts {
    /// When `false`, prime the assistant turn with [`EMPTY_THINK_BLOCK`]
    /// so the model skips chain-of-thought (faster, fewer tokens).
    pub enable_thinking: bool,
}

impl Default for ChatFormatOpts {
    fn default() -> Self {
        Self {
            enable_thinking: true,
        }
    }
}

/// Conversation role for [`ChatMessage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// One turn in a ChatML conversation.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

/// Format messages as ChatML ending with an open assistant turn.
///
/// Thinking enabled (default) — matches Qwen3 `enable_thinking=True`.
pub fn format_chatml(messages: &[ChatMessage]) -> String {
    format_chatml_with(messages, ChatFormatOpts::default())
}

/// Format ChatML with explicit thinking control.
pub fn format_chatml_with(messages: &[ChatMessage], opts: ChatFormatOpts) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(IM_START);
        out.push_str(msg.role.as_str());
        out.push('\n');
        out.push_str(&msg.content);
        out.push_str(IM_END);
        out.push('\n');
    }
    out.push_str(IM_START);
    out.push_str("assistant");
    out.push('\n');
    if opts.enable_thinking {
        // Match HF chat_template: assistant turn already opens `<think>\n`
        // so generation continues inside the think block.
        out.push_str(OPEN_THINK_PREFIX);
    } else {
        out.push_str(EMPTY_THINK_BLOCK);
    }
    out
}

/// Convenience: system (optional) + user prompt → ChatML messages.
pub fn messages_from_prompt(system: Option<&str>, user: &str) -> Vec<ChatMessage> {
    let mut msgs = Vec::new();
    if let Some(s) = system {
        if !s.is_empty() {
            msgs.push(ChatMessage::system(s));
        }
    }
    msgs.push(ChatMessage::user(user));
    msgs
}

/// Parse a JSON array of `{ "role": "...", "content": "..." }`.
pub fn parse_messages_json(raw: &str) -> Result<Vec<ChatMessage>> {
    serde_json::from_str(raw).context("parse --messages-json")
}

/// Split model output into `(thinking, answer)`.
///
/// If no `</think>` is present, the whole string is treated as the answer
/// (thinking `None`). Leading `<think>` without a close keeps thinking as
/// the interior (may be incomplete) and an empty answer.
pub fn split_thinking(text: &str) -> (Option<String>, String) {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let Some(open_at) = text.find(OPEN) else {
        return (None, text.to_string());
    };
    let after_open = open_at + OPEN.len();
    let Some(rel_close) = text[after_open..].find(CLOSE) else {
        let think = text[after_open..].trim().to_string();
        return (Some(think), String::new());
    };
    let close_at = after_open + rel_close;
    let think = text[after_open..close_at].trim().to_string();
    let answer = text[close_at + CLOSE.len()..].trim_start().to_string();
    (Some(think), answer)
}

/// Encode a ChatML conversation to token ids.
#[cfg(feature = "qwen35-tokenizer")]
pub fn encode_chat(tokenizer_path: &Path, messages: &[ChatMessage]) -> Result<Vec<u32>> {
    encode_prompt(tokenizer_path, &format_chatml(messages))
}

#[cfg(not(feature = "qwen35-tokenizer"))]
pub fn encode_chat(_tokenizer_path: &Path, _messages: &[ChatMessage]) -> Result<Vec<u32>> {
    anyhow::bail!("tokenizer support not compiled in — rebuild with feature `qwen35-tokenizer`")
}

/// Resolve tokenizer next to weights and encode a chat conversation.
/// Falls back to the GGUF-embedded BPE when no `tokenizer.json` is present
/// (same policy as [`crate::encode_prompt_auto`]).
pub fn encode_chat_auto(
    weights: &Path,
    explicit_tokenizer: Option<&Path>,
    messages: &[ChatMessage],
) -> Result<Vec<u32>> {
    encode_chat_auto_with(
        weights,
        explicit_tokenizer,
        messages,
        ChatFormatOpts::default(),
    )
}

/// Like [`encode_chat_auto`] with thinking options.
pub fn encode_chat_auto_with(
    weights: &Path,
    explicit_tokenizer: Option<&Path>,
    messages: &[ChatMessage],
    opts: ChatFormatOpts,
) -> Result<Vec<u32>> {
    let formatted = format_chatml_with(messages, opts);
    if let Some(path) = resolve_tokenizer_path(weights, explicit_tokenizer) {
        return encode_prompt(&path, &formatted);
    }
    let is_gguf = weights
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false);
    if is_gguf {
        return crate::encode_prompt_from_gguf(weights, &formatted);
    }
    anyhow::bail!(
        "no tokenizer found for {:?}. Pass --tokenizer <path> or place \
         tokenizer.json next to the GGUF",
        weights
    )
}

/// Resolve special-token ids used for stop / thinking-budget control.
#[derive(Debug, Clone, Default)]
pub struct SpecialTokenIds {
    pub think_open: Option<u32>,
    pub think_close: Option<u32>,
    pub im_end: Option<u32>,
    pub eos: Option<u32>,
}

impl SpecialTokenIds {
    /// Best-effort encode of known markers against the model tokenizer.
    pub fn resolve(weights: &Path, explicit_tokenizer: Option<&Path>) -> Self {
        let enc = |s: &str| -> Option<u32> {
            let ids = crate::encode_prompt_auto(weights, explicit_tokenizer, s).ok()?;
            // Prefer a single-id encode; otherwise take the first id.
            ids.first().copied()
        };
        // `<think></think>` is often two specials; encode separately.
        let pair = crate::encode_prompt_auto(weights, explicit_tokenizer, "<think></think>").ok();
        let (think_open, think_close) = match pair.as_deref() {
            Some([a, b]) => (Some(*a), Some(*b)),
            Some([a]) => (Some(*a), None),
            _ => (enc("<think>"), enc("</think>")),
        };
        Self {
            think_open,
            think_close,
            im_end: enc(IM_END),
            eos: enc("<|endoftext|>"),
        }
    }

    pub fn is_stop(&self, tok: u32) -> bool {
        self.im_end == Some(tok) || self.eos == Some(tok)
    }
}

/// Tracks think-block state during decode for `--thinking-budget`.
#[derive(Debug, Clone)]
pub struct ThinkingBudgetWatch {
    pub specials: SpecialTokenIds,
    pub budget: usize,
    in_think: bool,
    think_tokens: usize,
    pub closed: bool,
    pub budget_hit: bool,
}

impl ThinkingBudgetWatch {
    pub fn new(specials: SpecialTokenIds, budget: usize) -> Self {
        Self {
            specials,
            budget,
            in_think: false,
            think_tokens: 0,
            closed: false,
            budget_hit: false,
        }
    }

    /// Like [`Self::new`], but already inside an open `<think>` (HF primes
    /// `<think>\n` into the assistant turn when thinking is enabled).
    pub fn new_already_thinking(specials: SpecialTokenIds, budget: usize) -> Self {
        Self {
            specials,
            budget,
            in_think: true,
            think_tokens: 0,
            closed: false,
            budget_hit: false,
        }
    }

    /// Observe one generated token. Returns `false` when the thinking
    /// budget is exhausted while still inside `<think>` (caller should
    /// stop and continue after a forced close).
    pub fn observe(&mut self, tok: u32) -> bool {
        if self.specials.is_stop(tok) {
            return false;
        }
        if self.specials.think_open == Some(tok) {
            self.in_think = true;
            return true;
        }
        if self.specials.think_close == Some(tok) {
            self.in_think = false;
            self.closed = true;
            return true;
        }
        if self.in_think {
            self.think_tokens += 1;
            if self.think_tokens >= self.budget {
                self.budget_hit = true;
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_format_matches_qwen_convention() {
        let msgs = messages_from_prompt(Some("Be brief."), "Hi");
        let s = format_chatml(&msgs);
        assert!(s.starts_with("<|im_start|>system\nBe brief.<|im_end|>"));
        assert!(s.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(s.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn think_enabled_primes_open_block() {
        let msgs = messages_from_prompt(None, "Hi");
        let s = format_chatml_with(
            &msgs,
            ChatFormatOpts {
                enable_thinking: true,
            },
        );
        assert!(s.ends_with(OPEN_THINK_PREFIX));
        assert!(!s.contains("</think>"));
    }

    #[test]
    fn no_think_primes_empty_block() {
        let msgs = messages_from_prompt(None, "Hi");
        let s = format_chatml_with(
            &msgs,
            ChatFormatOpts {
                enable_thinking: false,
            },
        );
        assert!(s.ends_with(EMPTY_THINK_BLOCK));
        assert!(s.contains("</think>"));
    }

    #[test]
    fn split_thinking_extracts_answer() {
        let (t, a) = split_thinking("<think>\nreason\n</think>\n\nParis");
        assert_eq!(t.as_deref(), Some("reason"));
        assert_eq!(a, "Paris");
    }

    #[test]
    fn messages_json_roundtrip() {
        let raw = r#"[{"role":"user","content":"hello"}]"#;
        let msgs = parse_messages_json(raw).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, ChatRole::User);
        assert_eq!(msgs[0].content, "hello");
    }
}
