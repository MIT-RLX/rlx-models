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

//! Parse Fara `<tool_call>` blocks (JSON or MagenticLite XML-ish).

use anyhow::{Result, anyhow};
use serde_json::Value;

/// One parsed computer-use (or other) tool invocation from the model.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    /// `computer_use` action string when present (`left_click`, `terminate`, …).
    pub fn action(&self) -> Option<&str> {
        self.arguments
            .get("action")
            .and_then(|v| v.as_str())
            .or_else(|| {
                self.arguments
                    .get("arguments")
                    .and_then(|a| a.get("action"))
                    .and_then(|v| v.as_str())
            })
    }

    pub fn is_terminate(&self) -> bool {
        self.action() == Some("terminate")
    }
}

/// Extract every `<tool_call>…</tool_call>` body from assistant text.
pub fn extract_tool_call_bodies(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        let Some(end) = after.find("</tool_call>") else {
            break;
        };
        out.push(after[..end].trim().to_string());
        rest = &after[end + "</tool_call>".len()..];
    }
    out
}

/// Parse a single tool-call body (JSON object or `<function=…>` form).
pub fn parse_tool_call_body(body: &str) -> Result<ToolCall> {
    let trimmed = body.trim();
    if trimmed.starts_with('{') {
        let v: Value = serde_json::from_str(trimmed)
            .map_err(|e| anyhow!("tool_call JSON: {e}"))?;
        return tool_call_from_json(&v);
    }
    parse_function_xml(trimmed)
}

/// Parse all tool calls in an assistant reply (order preserved).
pub fn parse_tool_calls(text: &str) -> Result<Vec<ToolCall>> {
    let bodies = extract_tool_call_bodies(text);
    let mut out = Vec::with_capacity(bodies.len());
    for b in bodies {
        out.push(parse_tool_call_body(&b)?);
    }
    Ok(out)
}

/// Text before the first `<tool_call>` (chain-of-thought / commentary).
pub fn text_before_tool_calls(text: &str) -> &str {
    text.split("<tool_call>")
        .next()
        .unwrap_or(text)
        .trim_end()
}

/// Wrap a tool observation for the next user turn.
pub fn format_tool_response(content: &str) -> String {
    format!("<tool_response>\n{content}\n</tool_response>")
}

fn tool_call_from_json(v: &Value) -> Result<ToolCall> {
    let name = v
        .get("name")
        .and_then(|x| x.as_str())
        .unwrap_or("computer_use")
        .to_string();
    let arguments = v
        .get("arguments")
        .cloned()
        .or_else(|| v.get("parameters").cloned())
        .unwrap_or_else(|| v.clone());
    Ok(ToolCall { name, arguments })
}

fn parse_function_xml(body: &str) -> Result<ToolCall> {
    // <function=computer_use>\n<parameter=action>\nleft_click\n</parameter>…
    let Some(fn_start) = body.find("<function=") else {
        return Err(anyhow!(
            "tool_call body is neither JSON nor <function=…>: {body}"
        ));
    };
    let after_fn = &body[fn_start + "<function=".len()..];
    let name_end = after_fn
        .find('>')
        .ok_or_else(|| anyhow!("malformed <function=…>"))?;
    let name = after_fn[..name_end].trim().to_string();
    let mut args = serde_json::Map::new();
    let mut rest = after_fn;
    while let Some(p) = rest.find("<parameter=") {
        let after = &rest[p + "<parameter=".len()..];
        let Some(gt) = after.find('>') else {
            break;
        };
        let key = after[..gt].trim().to_string();
        let val_region = &after[gt + 1..];
        let Some(end) = val_region.find("</parameter>") else {
            break;
        };
        let val = val_region[..end].trim().to_string();
        let json_val = serde_json::from_str::<Value>(&val).unwrap_or(Value::String(val));
        args.insert(key, json_val);
        rest = &val_region[end + "</parameter>".len()..];
    }
    Ok(ToolCall {
        name,
        arguments: Value::Object(args),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_json_tool_call() {
        let text = r#"Thinking…
<tool_call>
{"name":"computer_use","arguments":{"action":"left_click","coordinate":[100,200]}}
</tool_call>"#;
        let calls = parse_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "computer_use");
        assert_eq!(calls[0].action(), Some("left_click"));
        assert_eq!(text_before_tool_calls(text), "Thinking…");
    }

    #[test]
    fn parse_terminate_and_xml_form() {
        let text = r#"<tool_call>
<function=computer_use>
<parameter=action>
terminate
</parameter>
<parameter=answer>
done
</parameter>
</function>
</tool_call>"#;
        let calls = parse_tool_calls(text).unwrap();
        assert!(calls[0].is_terminate());
        assert_eq!(
            calls[0].arguments.get("answer").and_then(|v| v.as_str()),
            Some("done")
        );
    }
}
