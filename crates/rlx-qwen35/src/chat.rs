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

//! ChatML prompt formatting for Qwen3.5 / Qwen3.6 CLI use.
//!
//! HuggingFace Qwen checkpoints ship a Jinja chat template in
//! `tokenizer_config.json`; the Rust `tokenizers` crate does not expose
//! that yet, so we format the canonical ChatML wire format here and
//! tokenize the resulting string.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

use super::tokenizer::{encode_prompt, resolve_tokenizer_path};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "";

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

/// Format messages as a ChatML prompt ending with an open assistant turn.
///
/// Example:
/// ```text
/// <|im_start|>system
/// You are helpful.
/// <|im_start|>user
/// Hello
/// <|im_start|>assistant
/// ```
pub fn format_chatml(messages: &[ChatMessage]) -> String {
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
pub fn encode_chat_auto(
    weights: &Path,
    explicit_tokenizer: Option<&Path>,
    messages: &[ChatMessage],
) -> Result<Vec<u32>> {
    let path = resolve_tokenizer_path(weights, explicit_tokenizer).ok_or_else(|| {
        anyhow::anyhow!(
            "no tokenizer found for {:?}. Pass --tokenizer <path> or place \
             tokenizer.json next to the GGUF",
            weights
        )
    })?;
    encode_chat(&path, messages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_format_matches_qwen_convention() {
        let msgs = messages_from_prompt(Some("Be brief."), "Hi");
        let s = format_chatml(&msgs);
        assert!(s.starts_with("<|im_start|>system\nBe brief."));
        assert!(s.contains("<|im_start|>user\nHi"));
        assert!(s.ends_with("<|im_start|>assistant\n"));
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
