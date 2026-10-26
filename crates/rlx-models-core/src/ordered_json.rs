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

//! An order-preserving JSON value with Python's text semantics.
//!
//! Prompt-building code ported from Python (decision models such as
//! `rlx-kev` and `rlx-d1`) needs JSON to behave the way CPython's `json`
//! module does, because the JSON *is* the prompt:
//!
//! * **Key order.** `serde_json::Value` backs objects with a `BTreeMap` unless
//!   the workspace turns on `preserve_order`, so every object comes out sorted
//!   by key. Python dicts keep insertion order; option order and state field
//!   order both change a decision model's answer.
//! * **Number parsing.** Without serde_json's `float_roundtrip`, a decimal can
//!   parse one ULP away from the correctly rounded value, and Python's
//!   `repr` of that neighbour is a different string. [`Json::parse`] uses its
//!   own reader, whose floats go through `str::parse::<f64>` (correctly
//!   rounded, like `float()`), and which — like `json.loads` — reads a number
//!   with no `.`/`e` as an integer, accepts `NaN`/`Infinity`, and lets a
//!   repeated key overwrite the value in its first position.
//! * **Number printing.** [`python_float_repr`] is `repr(float)`: shortest
//!   round-trip digits, positional for `1e-4 <= |x| < 1e16`, otherwise
//!   `1e-05` / `1.5e+16`.
//! * **`json.dumps`.** [`Json::to_python_json`] reproduces
//!   `json.dumps(v, ensure_ascii=False, indent=…)` byte for byte.

use std::fmt::Write as _;

use serde::de::{Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Serialize, Serializer};

/// A JSON value whose objects keep their document order.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Json {
    #[default]
    Null,
    Bool(bool),
    /// A JSON integer. Kept apart from [`Json::Float`] because Python prints
    /// `3` and `3.0` differently and both reach the prompt.
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

/// A [`Json::parse`] failure: what went wrong and the byte offset.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub msg: String,
    pub offset: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON: {} at byte {}", self.msg, self.offset)
    }
}

impl std::error::Error for ParseError {}

impl Json {
    /// Parse JSON text as Python's `json.loads` would (see the module docs).
    pub fn parse(text: &str) -> Result<Self, ParseError> {
        let mut p = Parser {
            s: text.as_bytes(),
            i: 0,
        };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.i != p.s.len() {
            return Err(p.err("trailing data"));
        }
        Ok(v)
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    /// `None`, an empty string, or a whitespace-only string.
    pub fn is_blank(&self) -> bool {
        match self {
            Json::Null => true,
            Json::Str(s) => s.trim().is_empty(),
            _ => false,
        }
    }

    pub fn is_container(&self) -> bool {
        matches!(self, Json::Array(_) | Json::Object(_))
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(f) => Some(f),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The value under `key` in an object (the first match).
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?
            .iter()
            .find_map(|(k, v)| (k == key).then_some(v))
    }

    /// Python `str()` of a JSON scalar; empty for `None` and containers.
    ///
    /// Booleans print `True`/`False` and floats print as [`python_float_repr`].
    pub fn scalar_str(&self) -> String {
        match self {
            Json::Null => String::new(),
            Json::Bool(b) => if *b { "True" } else { "False" }.to_string(),
            Json::Int(i) => i.to_string(),
            Json::UInt(u) => u.to_string(),
            Json::Float(f) => python_float_repr(*f),
            Json::Str(s) => s.clone(),
            _ => String::new(),
        }
    }

    /// Python `str()` of the value as `json.loads` would have built it —
    /// what an f-string interpolates: `None`, `True`/`False`, a float's
    /// [`python_float_repr`], a string verbatim, a list or dict as its
    /// `repr` (`[1, 'a', None]`, `{'k': 2.0}`).
    pub fn python_str(&self) -> String {
        match self {
            Json::Str(s) => s.clone(),
            other => {
                let mut out = String::new();
                python_repr(other, &mut out);
                out
            }
        }
    }

    /// Python truthiness: `None`, `False`, zero, and empty strings,
    /// lists and dicts are false.
    pub fn is_truthy(&self) -> bool {
        match self {
            Json::Null => false,
            Json::Bool(b) => *b,
            Json::Int(i) => *i != 0,
            Json::UInt(u) => *u != 0,
            Json::Float(f) => *f != 0.0,
            Json::Str(s) => !s.is_empty(),
            Json::Array(a) => !a.is_empty(),
            Json::Object(o) => !o.is_empty(),
        }
    }

    /// `json.dumps(self, ensure_ascii=False, indent=indent)`.
    ///
    /// `indent = None` gives Python's compact default (`", "` / `": "`
    /// separators on one line).
    pub fn to_python_json(&self, indent: Option<usize>) -> String {
        let mut out = String::new();
        dump(self, indent, 0, &mut out);
        out
    }
}

/// Python `repr(float)`: the shortest digits that round-trip, positional when
/// the decimal exponent is in `-4..16`, scientific (`1e-05`, `1.5e+16`)
/// otherwise; `inf`, `-inf`, `nan` for the non-finite values.
pub fn python_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // Rust's `{:e}` without a precision prints the shortest round-trip
    // digits, which is exactly what Python's repr starts from.
    let sci = format!("{:e}", x.abs());
    let (mant, exp) = sci
        .split_once('e')
        .expect("LowerExp always has an exponent");
    let exp: i32 = exp.parse().expect("LowerExp exponent is an integer");
    let digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    let sign = if x < 0.0 { "-" } else { "" };
    // value = 0.DIGITS x 10^decpt
    let decpt = exp + 1;
    let n = digits.len() as i32;
    if (-4 < decpt) && (decpt <= 16) {
        let body = if decpt <= 0 {
            format!("0.{}{digits}", "0".repeat((-decpt) as usize))
        } else if decpt >= n {
            format!("{digits}{}.0", "0".repeat((decpt - n) as usize))
        } else {
            let (a, b) = digits.split_at(decpt as usize);
            format!("{a}.{b}")
        };
        return format!("{sign}{body}");
    }
    let (first, rest) = digits.split_at(1);
    let mantissa = if rest.is_empty() {
        first.to_string()
    } else {
        format!("{first}.{rest}")
    };
    let e = decpt - 1;
    let esign = if e < 0 { '-' } else { '+' };
    format!("{sign}{mantissa}e{esign}{:02}", e.abs())
}

/// Python `repr()` of a JSON value. Strings take Python's quote choice
/// (single quotes unless the text has a `'` and no `"`) and its escapes for
/// `\`, the quote, `\n`/`\r`/`\t` and the other C0/C1 controls (`\x..`);
/// other non-printable code points (rare format characters) pass through
/// where Python would escape them.
fn python_repr(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("None"),
        Json::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
        Json::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Json::UInt(u) => {
            let _ = write!(out, "{u}");
        }
        Json::Float(f) => out.push_str(&python_float_repr(*f)),
        Json::Str(s) => {
            let q = if s.contains('\'') && !s.contains('"') {
                '"'
            } else {
                '\''
            };
            out.push(q);
            for c in s.chars() {
                match c {
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if c == q => {
                        out.push('\\');
                        out.push(c);
                    }
                    c if (c as u32) < 0x20 || (0x7f..0xa0).contains(&(c as u32)) => {
                        let _ = write!(out, "\\x{:02x}", c as u32);
                    }
                    c => out.push(c),
                }
            }
            out.push(q);
        }
        Json::Array(items) => {
            out.push('[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                python_repr(x, out);
            }
            out.push(']');
        }
        Json::Object(fields) => {
            out.push('{');
            for (i, (k, x)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                python_repr(&Json::Str(k.clone()), out);
                out.push_str(": ");
                python_repr(x, out);
            }
            out.push('}');
        }
    }
}

/// `json.dumps`'s float: `repr`, except the non-finite values, which Python
/// writes as the JavaScript literals.
fn json_float(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else {
        python_float_repr(x)
    }
}

/// Python's `json` string encoder with `ensure_ascii=False`: `"` and `\`
/// escaped, the C0 controls as `\n`/`\t`/… or `\u00xx`, everything else
/// (non-ASCII and DEL included) verbatim.
fn dump_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn dump(v: &Json, indent: Option<usize>, level: usize, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Json::UInt(u) => {
            let _ = write!(out, "{u}");
        }
        Json::Float(f) => out.push_str(&json_float(*f)),
        Json::Str(s) => dump_str(s, out),
        Json::Array(items) if items.is_empty() => out.push_str("[]"),
        Json::Object(fields) if fields.is_empty() => out.push_str("{}"),
        Json::Array(items) => {
            out.push('[');
            for (i, x) in items.iter().enumerate() {
                separator(indent, level + 1, i == 0, out);
                dump(x, indent, level + 1, out);
            }
            closing(indent, level, out);
            out.push(']');
        }
        Json::Object(fields) => {
            out.push('{');
            for (i, (k, x)) in fields.iter().enumerate() {
                separator(indent, level + 1, i == 0, out);
                dump_str(k, out);
                out.push_str(": ");
                dump(x, indent, level + 1, out);
            }
            closing(indent, level, out);
            out.push('}');
        }
    }
}

fn separator(indent: Option<usize>, level: usize, first: bool, out: &mut String) {
    match indent {
        Some(n) => {
            if !first {
                out.push(',');
            }
            out.push('\n');
            out.push_str(&" ".repeat(n * level));
        }
        None if !first => out.push_str(", "),
        None => {}
    }
}

fn closing(indent: Option<usize>, level: usize, out: &mut String) {
    if let Some(n) = indent {
        out.push('\n');
        out.push_str(&" ".repeat(n * level));
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn err(&self, msg: &str) -> ParseError {
        ParseError {
            msg: msg.to_string(),
            offset: self.i,
        }
    }

    fn ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.s.get(self.i) {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.s[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Json, ParseError> {
        match self.s.get(self.i) {
            None => Err(self.err("unexpected end")),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') if self.eat("true") => Ok(Json::Bool(true)),
            Some(b'f') if self.eat("false") => Ok(Json::Bool(false)),
            Some(b'n') if self.eat("null") => Ok(Json::Null),
            Some(b'N') if self.eat("NaN") => Ok(Json::Float(f64::NAN)),
            Some(b'I') if self.eat("Infinity") => Ok(Json::Float(f64::INFINITY)),
            Some(b'-') if self.eat("-Infinity") => Ok(Json::Float(f64::NEG_INFINITY)),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.err("unexpected character")),
        }
    }

    fn object(&mut self) -> Result<Json, ParseError> {
        self.i += 1;
        let mut fields: Vec<(String, Json)> = Vec::new();
        self.ws();
        if self.eat("}") {
            return Ok(Json::Object(fields));
        }
        loop {
            self.ws();
            if self.s.get(self.i) != Some(&b'"') {
                return Err(self.err("expected a key"));
            }
            let k = self.string()?;
            self.ws();
            if !self.eat(":") {
                return Err(self.err("expected ':'"));
            }
            self.ws();
            let v = self.value()?;
            // A repeated key keeps its first position and takes the last
            // value, as a Python dict built from the pairs does.
            match fields.iter_mut().find(|(key, _)| *key == k) {
                Some(slot) => slot.1 = v,
                None => fields.push((k, v)),
            }
            self.ws();
            if self.eat(",") {
                continue;
            }
            if self.eat("}") {
                return Ok(Json::Object(fields));
            }
            return Err(self.err("expected ',' or '}'"));
        }
    }

    fn array(&mut self) -> Result<Json, ParseError> {
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.eat("]") {
            return Ok(Json::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            if self.eat(",") {
                continue;
            }
            if self.eat("]") {
                return Ok(Json::Array(items));
            }
            return Err(self.err("expected ',' or ']'"));
        }
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        let h = self
            .s
            .get(self.i..self.i + 4)
            .and_then(|b| std::str::from_utf8(b).ok())
            .and_then(|t| u32::from_str_radix(t, 16).ok())
            .ok_or_else(|| self.err("bad \\u escape"))?;
        self.i += 4;
        Ok(h)
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let start = self.i;
            while let Some(&b) = self.s.get(self.i) {
                if b == b'"' || b == b'\\' {
                    break;
                }
                self.i += 1;
            }
            out.push_str(
                std::str::from_utf8(&self.s[start..self.i]).map_err(|_| self.err("bad UTF-8"))?,
            );
            match self.s.get(self.i) {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(_) => {
                    self.i += 1;
                    let e = *self.s.get(self.i).ok_or_else(|| self.err("bad escape"))?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut c = self.hex4()?;
                            if (0xd800..0xdc00).contains(&c) && self.eat("\\u") {
                                let lo = self.hex4()?;
                                if (0xdc00..0xe000).contains(&lo) {
                                    c = 0x10000 + ((c - 0xd800) << 10) + (lo - 0xdc00);
                                } else {
                                    // Not a pair: keep both halves apart.
                                    out.push(char::REPLACEMENT_CHARACTER);
                                    c = lo;
                                }
                            }
                            out.push(char::from_u32(c).unwrap_or(char::REPLACEMENT_CHARACTER));
                        }
                        _ => return Err(self.err("bad escape")),
                    }
                }
            }
        }
    }

    fn number(&mut self) -> Result<Json, ParseError> {
        let start = self.i;
        let mut integral = true;
        if self.s.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        while let Some(&b) = self.s.get(self.i) {
            match b {
                b'0'..=b'9' => {}
                b'.' | b'e' | b'E' | b'+' | b'-' => integral = false,
                _ => break,
            }
            self.i += 1;
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).expect("ASCII digits");
        if integral {
            if let Ok(v) = text.parse::<i64>() {
                return Ok(Json::Int(v));
            }
            if let Ok(v) = text.parse::<u64>() {
                return Ok(Json::UInt(v));
            }
        }
        // Integers past u64 fall back to the nearest float; Python would keep
        // the exact integer.
        text.parse::<f64>()
            .map(Json::Float)
            .map_err(|_| ParseError {
                msg: format!("bad number `{text}`"),
                offset: start,
            })
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Json;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_unit<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_none<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Json, D::Error> {
                Json::deserialize(d)
            }
            fn visit_bool<E>(self, v: bool) -> Result<Json, E> {
                Ok(Json::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Json, E> {
                Ok(Json::Int(v))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Json, E> {
                Ok(Json::UInt(v))
            }
            fn visit_f64<E>(self, v: f64) -> Result<Json, E> {
                Ok(Json::Float(v))
            }
            fn visit_str<E>(self, v: &str) -> Result<Json, E> {
                Ok(Json::Str(v.to_string()))
            }
            fn visit_string<E>(self, v: String) -> Result<Json, E> {
                Ok(Json::Str(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Json, A::Error> {
                let mut out = Vec::with_capacity(a.size_hint().unwrap_or(0));
                while let Some(x) = a.next_element()? {
                    out.push(x);
                }
                Ok(Json::Array(out))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<Json, A::Error> {
                let mut out = Vec::with_capacity(m.size_hint().unwrap_or(0));
                while let Some((k, v)) = m.next_entry::<String, Json>()? {
                    out.push((k, v));
                }
                Ok(Json::Object(out))
            }
        }
        d.deserialize_any(V)
    }
}

impl Serialize for Json {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Json::Null => s.serialize_unit(),
            Json::Bool(b) => s.serialize_bool(*b),
            Json::Int(i) => s.serialize_i64(*i),
            Json::UInt(u) => s.serialize_u64(*u),
            Json::Float(f) => s.serialize_f64(*f),
            Json::Str(v) => s.serialize_str(v),
            Json::Array(items) => {
                let mut seq = s.serialize_seq(Some(items.len()))?;
                for x in items {
                    seq.serialize_element(x)?;
                }
                seq.end()
            }
            Json::Object(fields) => {
                let mut map = s.serialize_map(Some(fields.len()))?;
                for (k, v) in fields {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

impl From<&str> for Json {
    fn from(v: &str) -> Self {
        Json::Str(v.to_string())
    }
}

impl From<String> for Json {
    fn from(v: String) -> Self {
        Json::Str(v)
    }
}

impl From<f64> for Json {
    fn from(v: f64) -> Self {
        Json::Float(v)
    }
}

impl From<i64> for Json {
    fn from(v: i64) -> Self {
        Json::Int(v)
    }
}

impl From<usize> for Json {
    fn from(v: usize) -> Self {
        Json::UInt(v as u64)
    }
}

impl From<bool> for Json {
    fn from(v: bool) -> Self {
        Json::Bool(v)
    }
}

impl<T: Into<Json>> From<Option<T>> for Json {
    fn from(v: Option<T>) -> Self {
        v.map_or(Json::Null, Into::into)
    }
}

impl<T: Into<Json>> From<Vec<T>> for Json {
    fn from(v: Vec<T>) -> Self {
        Json::Array(v.into_iter().map(Into::into).collect())
    }
}

/// Build an object literal, preserving the written order.
pub fn obj<const N: usize>(fields: [(&str, Json); N]) -> Json {
    Json::Object(
        fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        // Expected strings are CPython 3.13 `repr(float(x))`.
        for (x, want) in [
            (0.1, "0.1"),
            (1.0, "1.0"),
            (2.5, "2.5"),
            (1e-05, "1e-05"),
            (0.0001, "0.0001"),
            (1.5e-7, "1.5e-07"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (1234567890123456.0, "1234567890123456.0"),
            (12345678901234567.0, "1.2345678901234568e+16"),
            (-3.25, "-3.25"),
            (123.456, "123.456"),
            (1e22, "1e+22"),
            (5e-324, "5e-324"),
            (f64::MAX, "1.7976931348623157e+308"),
            (-0.0, "-0.0"),
            (f64::INFINITY, "inf"),
        ] {
            assert_eq!(python_float_repr(x), want, "repr({x:e})");
        }
    }

    #[test]
    fn parse_keeps_order_kinds_and_last_duplicate() {
        let v = Json::parse(r#"{"b": 1, "a": 2.0, "b": [true, null], "c": 1e2}"#).unwrap();
        assert_eq!(
            v,
            Json::Object(vec![
                ("b".into(), Json::Array(vec![Json::Bool(true), Json::Null])),
                ("a".into(), Json::Float(2.0)),
                ("c".into(), Json::Float(100.0)),
            ])
        );
        assert!(Json::parse("[1,]").is_err());
        assert!(Json::parse("1 2").is_err());
    }

    #[test]
    fn python_str_matches_python() {
        // python3 -c 'import json; print(str(json.loads(t)))'
        let v = Json::parse(
            r#"[null, true, 1, 2.0, 1e-05, "it's", "say \"hi\"", "a\tb", {"k": [false]}]"#,
        )
        .unwrap();
        assert_eq!(
            v.python_str(),
            r#"[None, True, 1, 2.0, 1e-05, "it's", 'say "hi"', 'a\tb', {'k': [False]}]"#
        );
        assert_eq!(Json::Null.python_str(), "None");
        assert_eq!(Json::Str("x\ny".into()).python_str(), "x\ny");
        assert!(!Json::Str(String::new()).is_truthy());
        assert!(Json::Float(0.5).is_truthy());
    }

    #[test]
    fn dumps_matches_python_indent_and_escapes() {
        let v = Json::parse(
            r#"{"t": "a\"b\\c\nd\te\u0001é😀", "n": [1, 2.0, 1e-05], "e": [], "o": {}, "z": null, "f": false}"#,
        )
        .unwrap();
        // python3 -c 'import json; print(json.dumps(v, ensure_ascii=False, indent=2))'
        let want = "{\n  \"t\": \"a\\\"b\\\\c\\nd\\te\\u0001é😀\",\n  \"n\": [\n    1,\n    2.0,\n    1e-05\n  ],\n  \"e\": [],\n  \"o\": {},\n  \"z\": null,\n  \"f\": false\n}";
        assert_eq!(v.to_python_json(Some(2)), want);
        assert_eq!(
            Json::parse(r#"{"a": [1, {"b": 2}]}"#)
                .unwrap()
                .to_python_json(None),
            r#"{"a": [1, {"b": 2}]}"#
        );
    }
}
