// RLX models — fine-tuning.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Dataset loading for fine-tuning: text / chat / completions JSONL, with
//! prompt masking. Mirrors mlx-lm's `tuner/datasets.py`.

use anyhow::{Context, Result, bail};
use std::path::Path;

/// One chat turn.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

/// A training example, format-detected from the JSONL line.
#[derive(Debug, Clone, PartialEq)]
pub enum Example {
    /// `{"text": "..."}` — raw text, the whole thing is a label.
    Text(String),
    /// `{"messages": [{"role","content"}, ...]}` — chat.
    Chat(Vec<Turn>),
    /// `{"prompt": "...", "completion": "..."}` — only the completion is a label.
    Completion { prompt: String, completion: String },
}

/// Parse one JSONL line into an [`Example`], auto-detecting the format.
pub fn parse_line(line: &str) -> Result<Example> {
    let v: serde_json::Value =
        serde_json::from_str(line).with_context(|| format!("parsing JSONL line: {line}"))?;
    let obj = v.as_object().context("JSONL line is not an object")?;

    if let Some(t) = obj.get("text").and_then(|t| t.as_str()) {
        return Ok(Example::Text(t.to_string()));
    }
    if let Some(msgs) = obj.get("messages").and_then(|m| m.as_array()) {
        let turns = msgs
            .iter()
            .map(|m| {
                let o = m.as_object().context("message is not an object")?;
                Ok(Turn {
                    role: o
                        .get("role")
                        .and_then(|r| r.as_str())
                        .context("message missing role")?
                        .to_string(),
                    content: o
                        .get("content")
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_string(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(Example::Chat(turns));
    }
    if let (Some(p), Some(c)) = (
        obj.get("prompt").and_then(|p| p.as_str()),
        obj.get("completion").and_then(|c| c.as_str()),
    ) {
        return Ok(Example::Completion {
            prompt: p.to_string(),
            completion: c.to_string(),
        });
    }
    bail!("unrecognized JSONL record (need `text`, `messages`, or `prompt`+`completion`)")
}

/// Load and parse a `.jsonl` file (blank lines skipped).
pub fn load_jsonl(path: &Path) -> Result<Vec<Example>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path:?}"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(parse_line)
        .collect()
}

/// A tokenized example: ids plus a parallel mask where `true` ⇒ the position
/// contributes to the loss (a "label") and `false` ⇒ it's context (ignored).
#[derive(Debug, Clone, PartialEq)]
pub struct Tokenized {
    pub input_ids: Vec<u32>,
    pub label_mask: Vec<bool>,
}

/// Tokenize a completion example, masking the prompt so only completion
/// tokens train. `encode(text)` returns token ids for `text`.
pub fn tokenize_completion(
    prompt: &str,
    completion: &str,
    encode: &mut dyn FnMut(&str) -> Vec<u32>,
    mask_prompt: bool,
) -> Tokenized {
    let p = encode(prompt);
    let c = encode(completion);
    let mut input_ids = Vec::with_capacity(p.len() + c.len());
    let mut label_mask = Vec::with_capacity(p.len() + c.len());
    for id in p {
        input_ids.push(id);
        label_mask.push(!mask_prompt);
    }
    for id in c {
        input_ids.push(id);
        label_mask.push(true);
    }
    Tokenized {
        input_ids,
        label_mask,
    }
}

/// Tokenize a raw-text example (every token is a label).
pub fn tokenize_text(text: &str, encode: &mut dyn FnMut(&str) -> Vec<u32>) -> Tokenized {
    let ids = encode(text);
    let n = ids.len();
    Tokenized {
        input_ids: ids,
        label_mask: vec![true; n],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_text_chat_completion() {
        assert_eq!(
            parse_line(r#"{"text":"hello"}"#).unwrap(),
            Example::Text("hello".into())
        );
        let chat = parse_line(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert_eq!(
            chat,
            Example::Chat(vec![Turn {
                role: "user".into(),
                content: "hi".into()
            }])
        );
        let comp = parse_line(r#"{"prompt":"Q","completion":"A"}"#).unwrap();
        assert_eq!(
            comp,
            Example::Completion {
                prompt: "Q".into(),
                completion: "A".into()
            }
        );
    }

    #[test]
    fn unknown_record_errors() {
        assert!(parse_line(r#"{"foo":1}"#).is_err());
    }

    #[test]
    fn completion_masks_prompt_tokens() {
        // Encoder: one token per char (id = byte value).
        let mut enc = |s: &str| s.bytes().map(|b| b as u32).collect::<Vec<_>>();
        let t = tokenize_completion("ab", "cd", &mut enc, true);
        assert_eq!(t.input_ids, vec![97, 98, 99, 100]);
        // prompt masked out, completion trained.
        assert_eq!(t.label_mask, vec![false, false, true, true]);

        let t2 = tokenize_completion("ab", "cd", &mut enc, false);
        assert_eq!(t2.label_mask, vec![true, true, true, true]);
    }

    #[test]
    fn text_trains_all_positions() {
        let mut enc = |s: &str| s.bytes().map(|b| b as u32).collect::<Vec<_>>();
        let t = tokenize_text("abc", &mut enc);
        assert_eq!(t.label_mask, vec![true, true, true]);
    }
}
