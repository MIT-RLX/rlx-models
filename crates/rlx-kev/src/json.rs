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

//! An order-preserving JSON value.
//!
//! `serde_json::Value` backs objects with a `BTreeMap` unless the workspace
//! turns on `preserve_order`, so every object it parses comes out sorted by
//! key. For this crate that is not a cosmetic detail:
//!
//! * `criteria` is the option list, and **option order changes the answer** —
//!   options interact before the softmax by design.
//! * A `state` given as an object is flattened to text field by field, so key
//!   order is literally part of the prompt.
//! * The response lists answers in the order the questions were asked.
//!
//! Python dicts preserve insertion order, so the reference implementation gets
//! all three for free. This type does the same rather than making the whole
//! workspace's `serde_json` behave differently.

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
    /// A JSON integer. Kept apart from [`Json::Float`] because Python's
    /// `str()` prints `3` and `3.0` differently and both reach the prompt.
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Parse from JSON text. Use this rather than going through
    /// `serde_json::Value`, which sorts object keys.
    pub fn parse(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    /// `true` for `null` and for the empty string — the two things
    /// `option_text` treats as "no description".
    pub fn is_blank(&self) -> bool {
        match self {
            Json::Null => true,
            Json::Str(s) => s.is_empty(),
            _ => false,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(o) => Some(o),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Python `str()` for a JSON scalar.
    ///
    /// Two deliberate differences from Rust's `Display`: booleans print
    /// `True`/`False`, and an integral float keeps its `.0`. Very large
    /// magnitudes still print positionally where Python would switch to
    /// exponent form; no record in the training context reaches that.
    pub fn scalar_str(&self) -> String {
        match self {
            Json::Null => String::new(),
            Json::Bool(b) => if *b { "True" } else { "False" }.to_string(),
            Json::Int(i) => i.to_string(),
            Json::UInt(u) => u.to_string(),
            Json::Float(f) if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 => {
                format!("{f:.1}")
            }
            Json::Float(f) => {
                let mut s = String::new();
                let _ = write!(s, "{f}");
                s
            }
            Json::Str(s) => s.clone(),
            _ => String::new(),
        }
    }

    fn is_container(&self) -> bool {
        matches!(self, Json::Array(_) | Json::Object(_))
    }
}

/// Flatten `string | object | array` into the text the model sees.
///
/// Field names are kept as labels. A verbatim port of `kev/api.py::render`.
pub fn render(v: &Json, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        Json::Array(items) => items
            .iter()
            .map(|x| format!("{pad}- {}", render(x, indent + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        Json::Object(fields) => fields
            .iter()
            .map(|(k, x)| {
                if x.is_container() {
                    format!("{pad}{k}:\n{}", render(x, indent + 1))
                } else {
                    format!("{pad}{k}: {}", x.scalar_str())
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        scalar => scalar.scalar_str(),
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
