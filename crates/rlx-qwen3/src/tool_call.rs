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

//! Function / **tool calling** for the Qwen3 chat flow (Hermes format).
//!
//! Qwen3 (like Hermes / most modern chat models) requests a tool by emitting a
//! JSON object inside `<tool_call>…</tool_call>` tags. This module is the
//! host-side glue: declare tools ([`ToolSpec`]) → inject them into the system
//! prompt ([`render_tools_system`]) → parse the model's requests out of its
//! output ([`parse_tool_calls`]) → feed each result back ([`render_tool_response`]).
//! Model-agnostic and text-only — it slots into any generation loop.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A tool/function the model may call: name + human description + a JSON-Schema
/// `parameters` object (OpenAI/Qwen "function" shape).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// JSON-Schema for the arguments object (`{"type":"object","properties":…}`).
    pub parameters: Value,
}

impl ToolSpec {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// A parsed tool call the model requested.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    /// Call arguments (usually a JSON object). Defaults to `null` when absent.
    #[serde(default)]
    pub arguments: Value,
}

/// Render the `# Tools` system-prompt block Qwen3 expects: the function schemas
/// inside `<tools>…</tools>` plus the instruction to reply with `<tool_call>`
/// JSON. Prepend/merge this into the system message before generation.
pub fn render_tools_system(tools: &[ToolSpec]) -> String {
    let mut s = String::from(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
    );
    for t in tools {
        let obj = serde_json::json!({
            "type": "function",
            "function": { "name": t.name, "description": t.description, "parameters": t.parameters }
        });
        s.push_str(&serde_json::to_string(&obj).unwrap_or_default());
        s.push('\n');
    }
    s.push_str(
        "</tools>\n\nFor each function call, return a json object with function name and \
         arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );
    s
}

/// Extract every `<tool_call>…</tool_call>` JSON object from model output. Robust
/// to surrounding prose/whitespace and to multiple calls; a block whose JSON
/// doesn't parse (or lacks a `name`) is skipped. Returns them in emission order.
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(o) = rest.find(OPEN) {
        let after = &rest[o + OPEN.len()..];
        let Some(c) = after.find(CLOSE) else { break };
        let body = after[..c].trim();
        if let Ok(call) = serde_json::from_str::<ToolCall>(body) {
            if !call.name.is_empty() {
                out.push(call);
            }
        }
        rest = &after[c + CLOSE.len()..];
    }
    out
}

/// Whether the output contains at least one (well-formed) tool call.
pub fn has_tool_call(text: &str) -> bool {
    !parse_tool_calls(text).is_empty()
}

/// Format a tool's result to feed back as the next turn's content — the
/// `<tool_response>…</tool_response>` block Qwen3 reads (wrap in a `tool`/`user`
/// chat turn at the call site).
pub fn render_tool_response(result: &Value) -> String {
    format!(
        "<tool_response>\n{}\n</tool_response>",
        serde_json::to_string(result).unwrap_or_default()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_single_tool_call() {
        let out = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \
                   \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        let calls = parse_tool_calls(out);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], "Paris");
        assert!(has_tool_call(out));
    }

    #[test]
    fn parses_multiple_and_skips_garbage() {
        let out = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\
                   <tool_call>not json</tool_call>\
                   <tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>";
        let calls = parse_tool_calls(out);
        assert_eq!(
            calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(calls[1].arguments["x"], 1);
    }

    #[test]
    fn no_calls_in_plain_text() {
        assert!(parse_tool_calls("just a normal answer").is_empty());
        assert!(!has_tool_call("just a normal answer"));
    }

    #[test]
    fn renders_tools_system_and_response() {
        let tools = vec![ToolSpec::new(
            "get_weather",
            "Get weather for a city",
            json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}),
        )];
        let sys = render_tools_system(&tools);
        assert!(
            sys.contains("<tools>") && sys.contains("get_weather") && sys.contains("<tool_call>")
        );
        let resp = render_tool_response(&json!({"temp_c": 18}));
        assert!(resp.contains("<tool_response>") && resp.contains("temp_c"));
    }
}
