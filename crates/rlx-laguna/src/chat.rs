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

//! Tokenizer + chat template for Laguna (HF `tokenizer.json` + Jinja).

use anyhow::{Context, Result, bail};
use rlx_text::chat::{ChatMessage, ChatRenderOptions, ChatTemplate};
use rlx_text::{TokenizerHandle, load_tokenizer};
use std::path::{Path, PathBuf};

/// Poolside default EOS string (matches tokenizer special token id 2).
pub const EOS_TOKEN: &str = "〈|EOS|〉";

/// Simplified Jinja used when the HF `chat_template.jinja` cannot be compiled
/// (e.g. unsupported `{% generation %}` tags under minijinja).
const FALLBACK_CHAT_TEMPLATE: &str = r#"{%- set enable_thinking = enable_thinking | default(false) -%}
{{- bos_token if bos_token else "〈|EOS|〉" -}}
{%- set system_message = "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations." -%}
{%- if messages and messages[0].role == "system" -%}
  {%- set system_message = messages[0].content -%}
  {%- set messages = messages[1:] -%}
{%- endif -%}
{%- if system_message -%}
  {{- "<system>" + system_message + "</system>\n" -}}
{%- endif -%}
{%- for message in messages -%}
  {%- set content = message.content if message.content is string else "" -%}
  {%- if message.role == "user" -%}
    {{- "<user>" + content + "</user>\n" -}}
  {%- elif message.role == "assistant" -%}
    {{- "<assistant></think>" + content + "</assistant>\n" -}}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
  {{- "<assistant>" -}}
  {%- if enable_thinking -%}{{- "<think>" -}}{%- else -%}{{- "</think>" -}}{%- endif -%}
{%- endif -%}"#;

/// Loaded tokenizer + chat template for Laguna prompts.
pub struct LagunaChat {
    pub tokenizer: TokenizerHandle,
    pub template: ChatTemplate,
    pub tokenizer_dir: PathBuf,
    pub used_fallback_template: bool,
}

impl LagunaChat {
    /// Load from a directory containing `tokenizer.json` and optionally
    /// `chat_template.jinja`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let tok_path = dir.join("tokenizer.json");
        if !tok_path.is_file() {
            bail!("missing tokenizer.json under {}", dir.display());
        }
        let tokenizer = load_tokenizer(&tok_path)
            .with_context(|| format!("load tokenizer {}", tok_path.display()))?;

        let jinja_path = dir.join("chat_template.jinja");
        let (template, used_fallback) = if jinja_path.is_file() {
            let raw = std::fs::read_to_string(&jinja_path)
                .with_context(|| format!("read {}", jinja_path.display()))?;
            match try_compile_template(&raw) {
                Ok(t) => (t, false),
                Err(e) => {
                    eprintln!(
                        "rlx-laguna: chat_template.jinja failed to compile ({e:#}); \
                         using simplified in-crate template"
                    );
                    (compile_fallback()?, true)
                }
            }
        } else {
            (compile_fallback()?, true)
        };

        Ok(Self {
            tokenizer,
            template,
            tokenizer_dir: dir.to_path_buf(),
            used_fallback_template: used_fallback,
        })
    }

    /// Render chat turns and encode to token ids (no extra BOS from tokenizer).
    pub fn encode_chat(&self, messages: &[ChatMessage], enable_thinking: bool) -> Result<Vec<u32>> {
        let text = self
            .template
            .render_with_options(
                messages,
                ChatRenderOptions {
                    add_generation_prompt: true,
                    enable_thinking,
                },
            )
            .or_else(|e| {
                if self.used_fallback_template {
                    return Err(e);
                }
                // HF templates can still fail at render (exotic filters); fall back once.
                eprintln!(
                    "rlx-laguna: chat template render failed ({e:#}); using simplified in-crate template"
                );
                compile_fallback()?.render_with_options(
                    messages,
                    ChatRenderOptions {
                        add_generation_prompt: true,
                        enable_thinking,
                    },
                )
            })?;
        self.encode_text(&text)
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, false)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.tokenizer.decode(ids, skip_special)
    }

    pub fn decode_token(&self, id: u32) -> String {
        self.tokenizer.decode(&[id], false).unwrap_or_default()
    }
}

fn compile_fallback() -> Result<ChatTemplate> {
    Ok(ChatTemplate::from_source(FALLBACK_CHAT_TEMPLATE)?
        .with_tokens(Some(EOS_TOKEN.into()), Some(EOS_TOKEN.into())))
}

fn try_compile_template(raw: &str) -> Result<ChatTemplate> {
    // Strip HF `{% generation %}` … `{% endgeneration %}` wrappers if present —
    // minijinja does not define that tag.
    let cleaned = strip_generation_tags(raw);
    Ok(ChatTemplate::from_source(cleaned)?
        .with_tokens(Some(EOS_TOKEN.into()), Some(EOS_TOKEN.into())))
}

fn strip_generation_tags(src: &str) -> String {
    let mut out = src.to_string();
    for tag in [
        "{%- generation -%}",
        "{% generation %}",
        "{%- endgeneration -%}",
        "{% endgeneration %}",
        "{%- generation %}",
        "{% endgeneration -%}",
    ] {
        out = out.replace(tag, "");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_template_renders_user_turn() {
        let t = compile_fallback().unwrap();
        let text = t
            .render_with_options(
                &[ChatMessage::user("Say hello")],
                ChatRenderOptions {
                    add_generation_prompt: true,
                    enable_thinking: false,
                },
            )
            .unwrap();
        assert!(text.contains("<user>Say hello</user>"));
        assert!(text.contains("<assistant>"));
        assert!(text.contains("</think>"));
    }

    #[test]
    fn strip_generation_tags_removes_wrappers() {
        let s = "{%- generation -%}hello{%- endgeneration -%}";
        assert_eq!(strip_generation_tags(s), "hello");
    }
}
