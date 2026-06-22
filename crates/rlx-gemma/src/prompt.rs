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

//! Chat-template prompt encoding for Gemma GGUF / HF checkpoints.

use anyhow::Result;
use rlx_cli::{ChatMessage, auto_chat_template};
use rlx_qwen35::{decode_ids_auto, encode_prompt_auto};
use std::path::Path;

/// Render the embedded GGUF chat template (when present) and tokenize.
///
/// Falls back to raw `user` text when no template is embedded or rendering
/// fails. Gemma 4 thinking models (e.g. the Composer/Fable coder fine-tune)
/// rely on the template to open the thought channel before generation.
pub fn encode_chat_prompt_auto(
    weights: &Path,
    tokenizer: Option<&Path>,
    system: Option<&str>,
    user: &str,
    use_chat_template: bool,
) -> Result<Vec<u32>> {
    let text = if use_chat_template {
        match auto_chat_template(weights) {
            Ok(tmpl) => {
                let mut msgs = Vec::new();
                if let Some(s) = system {
                    msgs.push(ChatMessage::system(s.to_string()));
                }
                msgs.push(ChatMessage::user(user.to_string()));
                tmpl.render(&msgs, true)?
            }
            Err(_) => user.to_string(),
        }
    } else {
        user.to_string()
    };
    encode_prompt_auto(weights, tokenizer, &text)
}

/// Decode one generated token id to a UTF-8 fragment (best-effort streaming).
pub fn decode_token_auto(weights: &Path, tokenizer: Option<&Path>, tok: u32) -> Result<String> {
    decode_ids_auto(weights, tokenizer, &[tok], false)
}
