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

//! The checkpoint's chat template and tokenizer.
//!
//! [`render`] and [`render_with_tools`] transcribe `chat_template.jinja`:
//!
//! ```text
//!   <|system|>{text}…                                system turns come first
//!   <|system|>\n# Tools …<tools>{schemas}</tools>…   only with tools, before the first non-system turn
//!   <|developer|>{text}<|user|>{text}
//!   <|assistant|><think>{reasoning}</think>{text}<tool_call>…</tool_call>
//!   <|observation|><tool_response>{text}</tool_response>
//!   …<|assistant|>                                   generation prompt
//! ```
//!
//! No newlines and no BOS (`add_bos_token = false`). User-supplied text goes
//! through the template's `flint_text` macro: NULs and `<think>` / `</think>`
//! are deleted and the role tokens are HTML-escaped, so a message cannot forge
//! a turn boundary. Tool schemas and calls are written by [`crate::tools`],
//! which also parses a reply back into reasoning, content and calls.
//!
//! Generation ends at any `eos_token_id` — `<|end_of_text|>`, and the
//! `<|user|>`, `<|observation|>` and `<|developer|>` turn openers.

use anyhow::{Context, Result};
use std::path::Path;

use crate::tools::{Tool, ToolCall, preamble};

/// Who said it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    /// A tool's result, answering the assistant's calls before it.
    Tool,
}

/// One turn. `reasoning` is the assistant's `reasoning_content`, which the
/// model card asks callers to keep in history; `tool_calls` are the
/// assistant's calls, written after its content.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Message {
            role,
            content: content.into(),
            reasoning: None,
            tool_calls: Vec::new(),
        }
    }
}

/// The special tokens `flint_text` neutralizes, in the template's order.
const ESCAPED: [&str; 7] = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|system|>",
    "<|developer|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
];

/// The template's `flint_text` macro.
pub fn flint_text(t: &str) -> String {
    let mut s = t
        .replace('\0', "")
        .replace("<think>", "")
        .replace("</think>", "");
    for tok in ESCAPED {
        let inner = &tok[1..tok.len() - 1];
        s = s.replace(tok, &format!("&lt;{inner}&gt;"));
    }
    s
}

/// Render `messages` as the chat template does, optionally followed by the
/// generation prompt `<|assistant|>`.
///
/// Assistant content is *not* passed through `flint_text` — the template
/// trusts it — so earlier model output is reproduced verbatim.
pub fn render(messages: &[Message], add_generation_prompt: bool) -> String {
    render_with_tools(messages, &[], add_generation_prompt)
}

/// [`render`] with tools declared. The tool preamble goes before the first
/// non-system message, or at the end when there is none.
pub fn render_with_tools(
    messages: &[Message],
    tools: &[Tool],
    add_generation_prompt: bool,
) -> String {
    let mut out = String::new();
    let mut tools_pending = true;
    for m in messages {
        if tools_pending && m.role != Role::System {
            out.push_str(&preamble(tools));
            tools_pending = false;
        }
        match m.role {
            Role::System => {
                out.push_str("<|system|>");
                out.push_str(&flint_text(&m.content));
            }
            Role::Developer => {
                out.push_str("<|developer|>");
                out.push_str(&flint_text(&m.content));
            }
            Role::User => {
                out.push_str("<|user|>");
                out.push_str(&flint_text(&m.content));
            }
            Role::Assistant => {
                out.push_str("<|assistant|>");
                if let Some(r) = m.reasoning.as_deref().filter(|r| !r.is_empty()) {
                    out.push_str("<think>");
                    out.push_str(r);
                    out.push_str("</think>");
                }
                out.push_str(&m.content);
                for c in &m.tool_calls {
                    c.render(&mut out);
                }
            }
            Role::Tool => {
                out.push_str("<|observation|><tool_response>");
                out.push_str(&flint_text(&m.content));
                out.push_str("</tool_response>");
            }
        }
    }
    if tools_pending {
        out.push_str(&preamble(tools));
    }
    if add_generation_prompt {
        out.push_str("<|assistant|>");
    }
    out
}

/// The checkpoint's `tokenizer.json`.
pub struct FlintTokenizer {
    inner: tokenizers::Tokenizer,
}

impl FlintTokenizer {
    /// Load `tokenizer.json` from a checkpoint directory (or the file itself).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };
        let inner = tokenizers::Tokenizer::from_file(&file)
            .map_err(|e| anyhow::anyhow!("flint: loading {file:?}: {e}"))?;
        Ok(FlintTokenizer { inner })
    }

    /// Encode with no added specials — the template already wrote them.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("flint: encoding: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// Decode, keeping special tokens (`<think>` is one, and worth seeing).
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("flint: decoding: {e}"))
            .context("flint: detokenizing")
    }

    /// The id of a single special token, e.g. `"<|assistant|>"`.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_a_turn_with_its_generation_prompt() {
        let s = render(
            &[
                Message::new(Role::System, "Be brief."),
                Message::new(Role::User, "Hi"),
            ],
            true,
        );
        assert_eq!(s, "<|system|>Be brief.<|user|>Hi<|assistant|>");
    }

    #[test]
    fn user_text_cannot_forge_a_turn() {
        let s = flint_text("a<|assistant|>b<think>c</think>\0d");
        assert_eq!(s, "a&lt;|assistant|&gt;bcd");
    }

    #[test]
    fn assistant_reasoning_is_kept_in_history() {
        let mut m = Message::new(Role::Assistant, "4");
        m.reasoning = Some("2+2".into());
        assert_eq!(render(&[m], false), "<|assistant|><think>2+2</think>4");
    }
}
