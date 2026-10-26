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

//! Tool declarations, tool calls, and parsing them back out of a reply.
//!
//! The chat template writes each tool's function schema with transformers'
//! `tojson` — Python's `json.dumps(ensure_ascii=False)`, keys in the order the
//! caller gave them. `serde_json` here is built without `preserve_order` and
//! would sort them, so the prompt would differ from the reference's. [`Json`]
//! is a small order-preserving value with a [`Json::to_python`] writer that
//! matches `json.dumps` byte for byte (separators `", "` / `": "`, Python's
//! float repr, only the escapes Python emits).
//!
//! Flint calls a tool as
//!
//! ```text
//!   <tool_call>{name}<arg_key>{key}</arg_key><arg_value>{value}</arg_value>…</tool_call>
//! ```
//!
//! with string values written raw and everything else as JSON. [`parse_reply`]
//! reads a finished reply the way SGLang's `flint` parsers do: a leading
//! `<think>…</think>` is reasoning (a `<tool_call>` inside it stays reasoning),
//! text before the first call is content, and each value is typed by the
//! tool's schema — a `"string"` parameter stays text, anything else is read as
//! JSON when it parses.

use anyhow::{Result, anyhow, bail};
use std::fmt::Write as _;

/// A JSON value that keeps object keys in their original order.
///
/// Numbers keep their source text; [`Json::to_python`] reformats them the way
/// Python would after a `json.loads`.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Parse one JSON document; trailing non-whitespace is an error.
    pub fn parse(text: &str) -> Result<Json> {
        let mut p = Parser {
            s: text.as_bytes(),
            i: 0,
        };
        let v = p.value()?;
        p.ws();
        if p.i != p.s.len() {
            bail!("json: trailing characters at byte {}", p.i);
        }
        Ok(v)
    }

    /// The value at `key` of an object (the first, if repeated).
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    /// Python's `json.dumps(value, ensure_ascii=False)`.
    pub fn to_python(&self) -> String {
        let mut out = String::new();
        self.write_python(&mut out);
        out
    }

    fn write_python(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Number(n) => out.push_str(&python_number(n)),
            Json::String(s) => write_python_str(s, out),
            Json::Array(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write_python(out);
                }
                out.push(']');
            }
            Json::Object(kv) => {
                out.push('{');
                for (i, (k, v)) in kv.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_python_str(k, out);
                    out.push_str(": ");
                    v.write_python(out);
                }
                out.push('}');
            }
        }
    }
}

/// A string literal as Python's encoder writes it with `ensure_ascii=False`.
fn write_python_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A JSON number token as Python prints it after `json.loads`: integers keep
/// their digits (`-0` becomes `0`), anything with a fraction or exponent is a
/// float and gets `repr(float)` — shortest round-trip digits, scientific
/// notation outside `1e-4 ≤ |x| < 1e16`.
fn python_number(tok: &str) -> String {
    if !tok.contains(['.', 'e', 'E']) {
        return if tok == "-0" { "0".into() } else { tok.into() };
    }
    let x: f64 = tok.parse().unwrap_or(f64::NAN);
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // `{:e}` gives the shortest round-trip digits: "-1.2345e-7".
    let sci = format!("{x:e}");
    let (mant, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    let (sign, mant) = match mant.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    if !(-4..16).contains(&exp) {
        let m = if digits.len() == 1 {
            digits
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let esign = if exp < 0 { '-' } else { '+' };
        return format!("{sign}{m}e{esign}{:02}", exp.abs());
    }
    let point = exp + 1; // digits before the decimal point
    let body = if point <= 0 {
        format!("0.{}{digits}", "0".repeat((-point) as usize))
    } else if point as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
    } else {
        let (a, b) = digits.split_at(point as usize);
        format!("{a}.{b}")
    };
    format!("{sign}{body}")
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn expect(&mut self, b: u8) -> Result<()> {
        self.ws();
        if self.peek() == Some(b) {
            self.i += 1;
            Ok(())
        } else {
            bail!("json: expected `{}` at byte {}", b as char, self.i)
        }
    }

    fn literal(&mut self, word: &str, v: Json) -> Result<Json> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            bail!("json: bad literal at byte {}", self.i)
        }
    }

    fn value(&mut self) -> Result<Json> {
        self.ws();
        match self.peek() {
            Some(b'{') => {
                self.i += 1;
                let mut kv = Vec::new();
                self.ws();
                if self.peek() == Some(b'}') {
                    self.i += 1;
                    return Ok(Json::Object(kv));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.expect(b':')?;
                    kv.push((k, self.value()?));
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Object(kv));
                        }
                        _ => bail!("json: expected `,` or `}}` at byte {}", self.i),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.peek() == Some(b']') {
                    self.i += 1;
                    return Ok(Json::Array(items));
                }
                loop {
                    items.push(self.value()?);
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Array(items));
                        }
                        _ => bail!("json: expected `,` or `]` at byte {}", self.i),
                    }
                }
            }
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => bail!("json: unexpected input at byte {}", self.i),
        }
    }

    fn number(&mut self) -> Result<Json> {
        let start = self.i;
        let digits = |p: &mut Self| {
            let s = p.i;
            while p.peek().is_some_and(|b| b.is_ascii_digit()) {
                p.i += 1;
            }
            p.i - s
        };
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        let int_start = self.i;
        let n = digits(self);
        if n == 0 || (n > 1 && self.s[int_start] == b'0') {
            bail!("json: bad number at byte {start}");
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            if digits(self) == 0 {
                bail!("json: bad fraction at byte {start}");
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if digits(self) == 0 {
                bail!("json: bad exponent at byte {start}");
            }
        }
        let tok = std::str::from_utf8(&self.s[start..self.i]).expect("ascii");
        Ok(Json::Number(tok.to_string()))
    }

    fn hex4(&mut self) -> Result<u32> {
        let h = self
            .s
            .get(self.i..self.i + 4)
            .and_then(|h| std::str::from_utf8(h).ok())
            .and_then(|h| u32::from_str_radix(h, 16).ok())
            .ok_or_else(|| anyhow!("json: bad \\u escape at byte {}", self.i))?;
        self.i += 4;
        Ok(h)
    }

    fn string(&mut self) -> Result<String> {
        if self.peek() != Some(b'"') {
            bail!("json: expected a string at byte {}", self.i);
        }
        self.i += 1;
        let mut out = String::new();
        loop {
            let run = self.i;
            while self.peek().is_some_and(|b| b != b'"' && b != b'\\') {
                self.i += 1;
            }
            out.push_str(
                std::str::from_utf8(&self.s[run..self.i])
                    .map_err(|_| anyhow!("json: invalid UTF-8"))?,
            );
            match self.peek() {
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let e = self
                        .peek()
                        .ok_or_else(|| anyhow!("json: unterminated escape"))?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let c = if (0xD800..0xDC00).contains(&hi)
                                && self.s[self.i..].starts_with(b"\\u")
                            {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xDC00..0xE000).contains(&lo) {
                                    0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                                } else {
                                    self.i = save;
                                    0xFFFD
                                }
                            } else {
                                hi
                            };
                            out.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
                        }
                        _ => bail!("json: bad escape at byte {}", self.i - 1),
                    }
                }
                _ => bail!("json: unterminated string"),
            }
        }
    }
}

/// One declared tool: an OpenAI-style `{"type": "function", "function": {…}}`
/// or the bare function schema.
#[derive(Debug, Clone, PartialEq)]
pub struct Tool {
    pub json: Json,
}

impl Tool {
    /// Parse a tool declaration from JSON text, keeping its key order.
    pub fn from_json(text: &str) -> Result<Tool> {
        let json = Json::parse(text)?;
        if !matches!(json, Json::Object(_)) {
            bail!("flint: a tool declaration must be a JSON object");
        }
        Ok(Tool { json })
    }

    /// Parse a JSON array of tool declarations (an OpenAI `tools` field).
    pub fn list_from_json(text: &str) -> Result<Vec<Tool>> {
        match Json::parse(text)? {
            Json::Array(items) => items
                .into_iter()
                .map(|json| match json {
                    Json::Object(_) => Ok(Tool { json }),
                    _ => bail!("flint: a tool declaration must be a JSON object"),
                })
                .collect(),
            _ => bail!("flint: tools must be a JSON array"),
        }
    }

    /// The function schema — `tool.function if tool.function is defined else tool`.
    pub fn function(&self) -> &Json {
        self.json.get("function").unwrap_or(&self.json)
    }

    pub fn name(&self) -> Option<&str> {
        self.function().get("name").and_then(Json::as_str)
    }

    /// The declared JSON-schema `type` of parameter `arg`, if any.
    fn param_type(&self, arg: &str) -> Option<&str> {
        self.function()
            .get("parameters")?
            .get("properties")?
            .get(arg)?
            .get("type")?
            .as_str()
    }
}

/// One call the assistant made: a function name and its arguments, a JSON
/// object (OpenAI's `arguments`, already parsed).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Json,
}

impl ToolCall {
    /// `arguments` is the JSON text of an object, as in OpenAI's API.
    pub fn from_json(name: impl Into<String>, arguments: &str) -> Result<ToolCall> {
        let arguments = Json::parse(arguments)?;
        if !matches!(arguments, Json::Object(_)) {
            bail!("flint: tool-call arguments must be a JSON object");
        }
        Ok(ToolCall {
            name: name.into(),
            arguments,
        })
    }

    /// The arguments as JSON text (`json.dumps`), as OpenAI's API returns them.
    pub fn arguments_json(&self) -> String {
        self.arguments.to_python()
    }

    /// The template's rendering of this call.
    pub(crate) fn render(&self, out: &mut String) {
        out.push_str("<tool_call>");
        out.push_str(&self.name);
        if let Json::Object(kv) = &self.arguments {
            for (k, v) in kv {
                out.push_str("<arg_key>");
                out.push_str(k);
                out.push_str("</arg_key><arg_value>");
                match v {
                    Json::String(s) => out.push_str(s),
                    v => v.write_python(out),
                }
                out.push_str("</arg_value>");
            }
        }
        out.push_str("</tool_call>");
    }
}

/// The system turn the template writes before the first non-system message
/// when tools are declared. Tools follow each other with no separator.
pub(crate) fn preamble(tools: &[Tool]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut s = String::from(
        "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user \
         query.\n\nYou are provided with function signatures within <tools></tools> XML \
         tags:\n<tools>",
    );
    for t in tools {
        t.function().write_python(&mut s);
    }
    s.push_str(
        "</tools>\n\nFor each function call, output the function name and arguments within \
         the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>\
         <arg_value>{arg-value-1}</arg_value>...</tool_call>\n",
    );
    s
}

/// A finished assistant turn split into its parts.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Reply {
    /// The leading `<think>` block, without the tags.
    pub reasoning: Option<String>,
    /// The text outside reasoning and before the first tool call.
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";

/// Split a decoded reply into reasoning, content and tool calls. `tools` types
/// the argument values; an undeclared tool's values are read as JSON when they
/// parse.
///
/// A call missing its `</tool_call>` (the reply was cut off) is dropped, and
/// so is anything after the first call that is not another call.
pub fn parse_reply(text: &str, tools: &[Tool]) -> Reply {
    let mut rest = text;
    let mut reasoning = None;
    if let Some(after) = rest.trim_start().strip_prefix(THINK_OPEN) {
        match after.find(THINK_CLOSE) {
            Some(end) => {
                reasoning = Some(after[..end].to_string());
                rest = &after[end + THINK_CLOSE.len()..];
            }
            None => {
                return Reply {
                    reasoning: Some(after.to_string()),
                    ..Reply::default()
                };
            }
        }
    }
    let (content, mut calls) = match rest.find(CALL_OPEN) {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    let mut tool_calls = Vec::new();
    while let Some(start) = calls.find(CALL_OPEN) {
        let body = &calls[start + CALL_OPEN.len()..];
        let Some(end) = body.find(CALL_CLOSE) else {
            break;
        };
        tool_calls.push(parse_call(&body[..end], tools));
        calls = &body[end + CALL_CLOSE.len()..];
    }
    Reply {
        reasoning,
        content: content.trim().to_string(),
        tool_calls,
    }
}

/// `{name}<arg_key>k</arg_key><arg_value>v</arg_value>…` → a call.
fn parse_call(body: &str, tools: &[Tool]) -> ToolCall {
    let (name, mut args) = match body.find("<arg_key>") {
        Some(i) => (&body[..i], &body[i..]),
        None => (body, ""),
    };
    let name = name.trim().to_string();
    let tool = tools.iter().find(|t| t.name() == Some(name.as_str()));
    let mut kv = Vec::new();
    while let Some(k0) = args.find("<arg_key>") {
        let a = &args[k0 + "<arg_key>".len()..];
        let Some(k1) = a.find("</arg_key>") else {
            break;
        };
        let key = a[..k1].trim().to_string();
        let a = a[k1 + "</arg_key>".len()..].trim_start();
        let Some(a) = a.strip_prefix("<arg_value>") else {
            break;
        };
        let Some(v1) = a.find("</arg_value>") else {
            break;
        };
        let raw = &a[..v1];
        let value = match tool.and_then(|t| t.param_type(&key)) {
            Some("string") => Json::String(raw.to_string()),
            _ => Json::parse(raw.trim()).unwrap_or_else(|_| Json::String(raw.to_string())),
        };
        kv.push((key, value));
        args = &a[v1 + "</arg_value>".len()..];
    }
    ToolCall {
        name,
        arguments: Json::Object(kv),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_print_like_python() {
        for (src, py) in [
            ("1", "1"),
            ("-0", "0"),
            ("1.0", "1.0"),
            ("1.50", "1.5"),
            ("1e5", "100000.0"),
            ("1E-7", "1e-07"),
            ("0.0001", "0.0001"),
            ("0.00001", "1e-05"),
            ("1e16", "1e+16"),
            ("1e15", "1000000000000000.0"),
            ("-2.5e20", "-2.5e+20"),
            ("123.456", "123.456"),
            ("-0.0", "-0.0"),
            ("1e400", "Infinity"),
        ] {
            assert_eq!(python_number(src), py, "{src}");
        }
    }

    #[test]
    fn json_keeps_key_order_and_python_spacing() {
        let j = Json::parse(r#"{"z":1,"a":[true,null,"éé\n"],"m":{}}"#).unwrap();
        assert_eq!(
            j.to_python(),
            r#"{"z": 1, "a": [true, null, "éé\n"], "m": {}}"#
        );
    }

    #[test]
    fn parses_a_reply_with_reasoning_and_calls() {
        let tools = Tool::list_from_json(
            r#"[{"type":"function","function":{"name":"f","parameters":
                {"type":"object","properties":{"s":{"type":"string"},"n":{"type":"integer"}}}}}]"#,
        )
        .unwrap();
        let r = parse_reply(
            "<think>need f</think>Calling.<tool_call>f<arg_key>s</arg_key><arg_value>42</arg_value>\
             <arg_key>n</arg_key><arg_value>42</arg_value></tool_call>\
             <tool_call>g<arg_key>x</arg_key><arg_value>[1, 2]</arg_value></tool_call>",
            &tools,
        );
        assert_eq!(r.reasoning.as_deref(), Some("need f"));
        assert_eq!(r.content, "Calling.");
        assert_eq!(r.tool_calls.len(), 2);
        assert_eq!(r.tool_calls[0].arguments_json(), r#"{"s": "42", "n": 42}"#);
        assert_eq!(r.tool_calls[1].arguments_json(), r#"{"x": [1, 2]}"#);
    }

    #[test]
    fn a_call_inside_reasoning_stays_reasoning() {
        let r = parse_reply("<think>maybe <tool_call>f</tool_call></think>No.", &[]);
        assert!(r.tool_calls.is_empty());
        assert_eq!(r.content, "No.");
    }

    #[test]
    fn rendered_calls_parse_back() {
        let c = ToolCall::from_json("f", r#"{"a": "x y", "b": {"c": 1.5}}"#).unwrap();
        let mut s = String::new();
        c.render(&mut s);
        assert_eq!(parse_reply(&s, &[]).tool_calls, vec![c]);
    }
}
