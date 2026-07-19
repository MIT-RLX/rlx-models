// RLX models — OpenAI-compatible server.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! OpenAI-compatible request/response wire types for `/v1/chat/completions`,
//! `/v1/completions`, and `/v1/models`.

use crate::engine::{ChatTurn, TokenLogprob};
use crate::sampling_map::SamplingParams;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// `stop` may be a single string or a list.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopField::One(s) => vec![s],
            StopField::Many(v) => v,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessageIn {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessageIn>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopField>,
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>,
    #[serde(default)]
    pub logprobs: bool,
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    /// OpenAI-style tool/function definitions. Validated by rlx-guardrails
    /// (reserved-name collisions rejected). Accepted for API compatibility;
    /// the model only *uses* them once the chat template gains tool support.
    #[serde(default)]
    pub tools: Option<Vec<rlx_guardrails::ToolDef>>,
    /// OpenAI `tool_choice` (`"auto"` / `"none"` / a named tool). Passed
    /// through; kept as raw JSON since it's advisory to the model.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Optional context budget (tokens). When set, older non-system messages
    /// are dropped (via rlx-guardrails compaction) to fit before templating.
    #[serde(default)]
    pub max_context_tokens: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopField>,
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>,
    #[serde(default)]
    pub logprobs: Option<usize>,
}

impl ChatCompletionRequest {
    pub fn sampling(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
        }
    }
    pub fn turns(&self) -> Vec<ChatTurn> {
        self.messages
            .iter()
            .map(|m| ChatTurn {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect()
    }

    /// Reject tool definitions that collide with rlx's reserved (`_rlx_*`)
    /// tool names. Returns a client-facing message on violation.
    pub fn validate_tools(&self) -> Result<(), String> {
        if let Some(tools) = &self.tools {
            if rlx_guardrails::tools_collide_with_reserved(tools) {
                return Err("tool definitions collide with reserved rlx_* tool names".to_string());
            }
        }
        Ok(())
    }

    /// Messages as [`ChatTurn`]s, compacted to `max_context_tokens` when set:
    /// oldest non-system turns are dropped (tool results first) to fit the
    /// budget. Without a budget this is exactly [`turns`](Self::turns).
    pub fn compacted_turns(&self) -> Vec<ChatTurn> {
        match self.max_context_tokens {
            Some(budget) => {
                let msgs: Vec<rlx_guardrails::Message> = self
                    .messages
                    .iter()
                    .map(|m| rlx_guardrails::Message::new(m.role.clone(), m.content.clone()))
                    .collect();
                rlx_guardrails::compact_messages(msgs, budget)
                    .into_iter()
                    .map(|m| ChatTurn {
                        role: m.role,
                        content: m.content,
                    })
                    .collect()
            }
            None => self.turns(),
        }
    }
    pub fn want_logprobs(&self) -> Option<usize> {
        if self.logprobs {
            Some(self.top_logprobs.unwrap_or(0).max(1))
        } else {
            None
        }
    }
}

impl CompletionRequest {
    pub fn sampling(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
        }
    }
}

// ─── responses ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TopLogprobEntry {
    pub token: String,
    pub logprob: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct LogprobEntry {
    pub token: String,
    pub logprob: f32,
    pub top_logprobs: Vec<TopLogprobEntry>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct LogprobsBlock {
    pub content: Vec<LogprobEntry>,
}

/// Render an engine [`TokenLogprob`] into the OpenAI wire shape. `chosen_text`
/// is the chosen token's decoded text; `decode` maps alternative ids to text.
pub fn logprob_entry(
    lp: &TokenLogprob,
    chosen_text: &str,
    decode: &dyn Fn(u32) -> String,
) -> LogprobEntry {
    LogprobEntry {
        token: chosen_text.to_string(),
        logprob: lp.logprob,
        top_logprobs: lp
            .top
            .iter()
            .map(|&(id, lp)| TopLogprobEntry {
                token: decode(id),
                logprob: lp,
            })
            .collect(),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionOut {
    pub name: String,
    /// JSON-encoded arguments string (OpenAI convention).
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCallOut {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: FunctionOut,
}

#[derive(Debug, Clone, Serialize)]
pub struct RespMessage {
    pub role: &'static str,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallOut>>,
}

/// Convert parsed [`rlx_text::ToolCall`]s into OpenAI `tool_calls`.
pub fn tool_calls_out(calls: &[rlx_text::ToolCall], id_prefix: &str) -> Vec<ToolCallOut> {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| ToolCallOut {
            id: format!("{id_prefix}-{i}"),
            kind: "function",
            function: FunctionOut {
                name: c.name.clone(),
                arguments: c.arguments.to_string(),
            },
        })
        .collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: RespMessage,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogprobsBlock>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

// streaming chunk
#[derive(Debug, Clone, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

// completions (legacy)
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

// models
#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_chat_request_with_defaults() {
        let json = r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "m");
        assert_eq!(req.messages.len(), 1);
        assert!(!req.stream);
        assert!(req.want_logprobs().is_none());
    }

    #[test]
    fn stop_field_accepts_string_or_array() {
        let one: StopField = serde_json::from_str(r#""END""#).unwrap();
        assert_eq!(one.into_vec(), vec!["END"]);
        let many: StopField = serde_json::from_str(r#"["a","b"]"#).unwrap();
        assert_eq!(many.into_vec(), vec!["a", "b"]);
    }

    #[test]
    fn logprobs_request_maps_top_k() {
        let json = r#"{"model":"m","messages":[],"logprobs":true,"top_logprobs":5}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.want_logprobs(), Some(5));
    }

    fn chat(v: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn reserved_tool_name_is_rejected() {
        let req = chat(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "_rlx_respond"}}],
        }));
        assert!(req.validate_tools().is_err());
    }

    #[test]
    fn ordinary_tools_validate_ok() {
        let req = chat(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object"}}
            }],
            "tool_choice": "auto",
        }));
        assert!(req.validate_tools().is_ok());
        assert!(req.tools.is_some());
        assert!(req.tool_choice.is_some());
    }

    #[test]
    fn compaction_keeps_system_and_newest_within_budget() {
        let req = chat(serde_json::json!({
            "model": "m",
            "max_context_tokens": 20,
            "messages": [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": "aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa"},
                {"role": "assistant", "content": "bbbb bbbb bbbb bbbb bbbb bbbb bbbb"},
                {"role": "user", "content": "hi"}
            ],
        }));
        let turns = req.compacted_turns();
        assert!(turns.iter().any(|t| t.role == "system"), "system survives");
        assert!(turns.len() < 4, "some older turn was dropped to fit budget");
        assert_eq!(turns.last().unwrap().content, "hi", "newest turn kept");
    }

    #[test]
    fn no_budget_is_passthrough() {
        let req = chat(serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"}
            ],
        }));
        assert_eq!(req.compacted_turns().len(), 2);
    }
}
