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

//! Bounding boxes in the prompt, BFL `flux3/bbox.py`.
//!
//! A caption may end in a JSON table of boxes, `[y_min, x_min, y_max, x_max]`
//! on a 0-1000 canvas, each element cited in the caption as `<id>`. The
//! upsampler plans such a table (its `*_bbox` system prompts) around boxes
//! the user locked, and [`reconcile`] re-imposes the locked ones on its
//! answer.
//!
//! The DiT reads the table as text, so it is written exactly as Python's
//! `json.dumps(rows, ensure_ascii=False)` writes it: `", "` and `": "`
//! separators, keys in insertion order, floats as Python prints them. That
//! is why rows are [`Json`] (an ordered value) rather than `serde_json`'s
//! map, which sorts keys here.

use anyhow::{Result, bail};
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use std::fmt::Write as _;

/// Boxes are on a `0..=SCALE` canvas.
pub const SCALE: f64 = 1000.0;

/// A JSON value that keeps object keys in order and integers apart from floats.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Int(i) => Some(*i as f64),
            Json::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Python truthiness (`if box.get("bbox"):`).
    fn truthy(&self) -> bool {
        match self {
            Json::Null => false,
            Json::Bool(b) => *b,
            Json::Int(i) => *i != 0,
            Json::Float(f) => *f != 0.0,
            Json::Str(s) => !s.is_empty(),
            Json::List(l) => !l.is_empty(),
            Json::Object(o) => !o.is_empty(),
        }
    }

    /// Set `key`, in place when present, else appended (Python dict order).
    fn set(&mut self, key: &str, value: Json) {
        if let Json::Object(kv) = self {
            match kv.iter_mut().find(|(k, _)| k == key) {
                Some((_, v)) => *v = value,
                None => kv.push((key.to_string(), value)),
            }
        }
    }

    pub fn parse(text: &str) -> Result<Json> {
        Ok(serde_json::from_str(text)?)
    }

    /// `json.dumps(self, ensure_ascii=False)`.
    pub fn dumps(&self) -> String {
        let mut s = String::new();
        self.write(&mut s);
        s
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Json::Float(f) => out.push_str(&py_float(*f)),
            Json::Str(s) => write_str(s, out),
            Json::List(l) => {
                out.push('[');
                for (i, v) in l.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Object(kv) => {
                out.push('{');
                for (i, (k, v)) in kv.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_str(k, out);
                    out.push_str(": ");
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_str(s: &str, out: &mut String) {
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

/// Python's `repr(float)`: shortest round trip, `.0` on integers, and
/// exponents (`1e-05`, `1e+16`) outside `[1e-4, 1e16)`.
pub fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let a = f.abs();
    if a != 0.0 && !(1e-4..1e16).contains(&a) {
        let s = format!("{f:e}");
        let (m, e) = s.split_once('e').expect("exponent form");
        let e: i32 = e.parse().expect("exponent");
        return format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    let s = format!("{f:?}");
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}

impl<'de> serde::Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Json;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("JSON")
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_bool<E: de::Error>(self, b: bool) -> std::result::Result<Json, E> {
                Ok(Json::Bool(b))
            }
            fn visit_i64<E: de::Error>(self, i: i64) -> std::result::Result<Json, E> {
                Ok(Json::Int(i))
            }
            fn visit_u64<E: de::Error>(self, u: u64) -> std::result::Result<Json, E> {
                Ok(i64::try_from(u).map_or(Json::Float(u as f64), Json::Int))
            }
            fn visit_f64<E: de::Error>(self, f: f64) -> std::result::Result<Json, E> {
                Ok(Json::Float(f))
            }
            fn visit_str<E: de::Error>(self, s: &str) -> std::result::Result<Json, E> {
                Ok(Json::Str(s.to_string()))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> std::result::Result<Json, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = a.next_element()? {
                    v.push(x);
                }
                Ok(Json::List(v))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> std::result::Result<Json, A::Error> {
                let mut kv: Vec<(String, Json)> = Vec::new();
                while let Some((k, v)) = a.next_entry::<String, Json>()? {
                    // A repeated key keeps its first position and the last value (Python).
                    match kv.iter_mut().find(|(kk, _)| *kk == k) {
                        Some((_, old)) => *old = v,
                        None => kv.push((k, v)),
                    }
                }
                Ok(Json::Object(kv))
            }
        }
        d.deserialize_any(V)
    }
}

/// Python's `round` (half to even).
pub fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// An element id: `[A-Za-z][A-Za-z0-9_]*`.
pub fn is_id(s: &str) -> bool {
    let mut c = s.chars();
    c.next().is_some_and(|f| f.is_ascii_alphabetic())
        && c.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn obj(kv: Vec<(&str, Json)>) -> Json {
    Json::Object(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// User boxes (0-1 coordinates, the `--boxes` file) → upsampler rows on the
/// 0-1000 canvas. Each box has `bbox`, or `src_bbox` / `tgt_bbox` with `from`.
pub fn to_canvas(boxes: &[Json]) -> Result<Vec<Json>> {
    let grid = |b: Option<&Json>| -> Result<Json> {
        match b {
            Some(Json::List(l)) if !l.is_empty() => {
                if l.len() != 4 {
                    bail!(
                        "a box is [y_min, x_min, y_max, x_max], got {}",
                        Json::List(l.clone()).dumps()
                    );
                }
                let mut out = Vec::with_capacity(4);
                for v in l {
                    let v = v
                        .as_f64()
                        .ok_or_else(|| anyhow::anyhow!("box coordinate {}", v.dumps()))?;
                    out.push(Json::Int(round_half_even(v.clamp(0.0, 1.0) * SCALE) as i64));
                }
                Ok(Json::List(out))
            }
            _ => Ok(Json::Null),
        }
    };
    let text = |v: Option<&Json>, default: String| -> String {
        match v {
            Some(j) if j.truthy() => match j {
                Json::Str(s) => s.clone(),
                other => other.dumps(),
            },
            _ => default,
        }
    };
    let mut rows = Vec::with_capacity(boxes.len());
    for (i, b) in boxes.iter().enumerate() {
        let id = text(b.get("id"), format!("obj_{}", i + 1));
        let desc = text(b.get("desc"), String::new());
        if b.get("bbox").is_some_and(Json::truthy) {
            rows.push(obj(vec![
                ("id", Json::Str(id)),
                ("bbox", grid(b.get("bbox"))?),
                ("desc", Json::Str(desc)),
            ]));
        } else if b.get("src_bbox").is_some_and(Json::truthy)
            || b.get("tgt_bbox").is_some_and(Json::truthy)
        {
            rows.push(obj(vec![
                ("id", Json::Str(id)),
                ("from", Json::Str(text(b.get("from"), String::new()))),
                ("src_bbox", grid(b.get("src_bbox"))?),
                ("tgt_bbox", grid(b.get("tgt_bbox"))?),
                ("desc", Json::Str(desc)),
            ]));
        } else {
            bail!("box {i} has neither bbox nor src_bbox/tgt_bbox");
        }
    }
    Ok(rows)
}

/// `(caption, rows)`: the last `[` from which the rest parses as a JSON list.
pub fn split(text: &str) -> (String, Option<Vec<Json>>) {
    for (i, _) in text.char_indices().rev().filter(|(_, c)| *c == '[') {
        if let Ok(Json::List(rows)) = Json::parse(&text[i..]) {
            return (text[..i].trim_end().to_string(), Some(rows));
        }
    }
    (text.trim().to_string(), None)
}

/// A well-formed box on the 0-1000 canvas, as given.
pub fn valid_box(b: Option<&Json>) -> Option<Vec<Json>> {
    let Json::List(l) = b? else {
        return None;
    };
    if l.len() != 4 {
        return None;
    }
    let mut v = [0f64; 4];
    for (i, x) in l.iter().enumerate() {
        v[i] = x.as_f64().filter(|f| f.is_finite())?;
    }
    let [y1, x1, y2, x2] = v;
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // max <= 1 means 0-1 coordinates, not the 0-1000 canvas
    if y1 < 0.0 || x1 < 0.0 || y2 <= y1 || x2 <= x1 || y2 > SCALE || x2 > SCALE || max <= 1.0 {
        return None;
    }
    Some(l.clone())
}

fn is_row(r: &Json) -> bool {
    r.get("id").and_then(Json::as_str).is_some_and(is_id)
        && ["bbox", "src_bbox", "tgt_bbox"]
            .iter()
            .any(|k| valid_box(r.get(k)).is_some())
}

/// `(prompt, rows)` when the prompt already carries a box table on the
/// 0-1000 canvas — `"<caption> [rows]"` or the upsampler's own
/// `ORIGINAL PROMPT: … LOCKED BOUNDING-BOX ANNOTATIONS …` input.
pub fn detect(prompt: &str) -> Option<(String, Vec<Json>)> {
    let (text, rows) = split(prompt);
    let rows = rows?;
    if rows.is_empty() || !rows.iter().all(is_row) {
        return None;
    }
    const HEAD: &str = "ORIGINAL PROMPT:\n";
    const LOCK: &str = "\n\nLOCKED BOUNDING-BOX ANNOTATIONS ";
    let text = match text
        .strip_prefix(HEAD)
        .and_then(|r| r.find(LOCK).map(|i| (r, i)))
    {
        Some((rest, i)) => rest[..i].to_string(),
        None => text,
    };
    Some((text, rows))
}

/// An upsampler answer, not a request: its whole box table is cited as `<id>`s.
pub fn is_finished(prompt: &str) -> bool {
    detect(prompt).is_some_and(|(caption, rows)| {
        rows.iter().all(|r| {
            r.get("id")
                .and_then(Json::as_str)
                .is_some_and(|id| caption.contains(&format!("<{id}>")))
        })
    })
}

/// The upsampler's input for `prompt` with `rows` locked.
pub fn upsampling_input(prompt: &str, rows: &[Json]) -> String {
    format!(
        "ORIGINAL PROMPT:\n{}\n\n\
         LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at \
         the top-left). This is an authoritative JSON list of dictionaries; preserve every supplied identity, \
         provenance value, null, and coordinate exactly:\n{}",
        prompt.trim(),
        Json::List(rows.to_vec()).dumps()
    )
}

/// A request planned for the upsampler.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// System prompt key: `t2i`, `i2i`, `t2i_bbox` or `i2i_bbox`.
    pub task: String,
    /// The user text.
    pub text: String,
    /// Rows to re-impose on the answer.
    pub locked: Option<Vec<Json>>,
}

/// `--boxes` wins; otherwise boxes found in the prompt lock themselves.
pub fn plan(prompt: &str, boxes: Option<&[Json]>, n_images: usize) -> Result<Plan> {
    let task = if n_images > 0 { "i2i" } else { "t2i" };
    let (prompt, locked) = match boxes {
        Some(b) if !b.is_empty() => (prompt.to_string(), to_canvas(b)?),
        _ => match detect(prompt) {
            Some((p, rows)) => (p, rows),
            None => {
                return Ok(Plan {
                    task: task.into(),
                    text: prompt.into(),
                    locked: None,
                });
            }
        },
    };
    Ok(Plan {
        task: format!("{task}_bbox"),
        text: upsampling_input(&prompt, &locked),
        locked: Some(locked),
    })
}

fn is_origin(s: &str) -> bool {
    s == "target"
        || s.strip_prefix("ref_image_")
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// A row the upsampler added, kept only if cited as `<id>` and well formed.
fn valid_added(row: &Json, caption: &str) -> Option<Json> {
    let id = row.get("id")?.as_str()?;
    let desc = row.get("desc")?.as_str()?;
    if !is_id(id) || !caption.contains(&format!("<{id}>")) || desc.trim().is_empty() {
        return None;
    }
    let desc = py_strip(desc);
    if let Some(b) = valid_box(row.get("bbox")) {
        return Some(obj(vec![
            ("id", Json::Str(id.into())),
            ("bbox", Json::List(b)),
            ("desc", Json::Str(desc)),
        ]));
    }
    let origin = row.get("from").and_then(Json::as_str)?;
    let present = |k: &str| row.get(k).is_some_and(|v| *v != Json::Null);
    let (src, tgt) = (
        valid_box(row.get("src_bbox")),
        valid_box(row.get("tgt_bbox")),
    );
    if !is_origin(origin)
        || (present("src_bbox") && src.is_none())
        || (present("tgt_bbox") && tgt.is_none())
        || (src.is_none() && tgt.is_none())
        || (origin == "target") == src.is_some()
    {
        return None;
    }
    let boxed = |b: Option<Vec<Json>>| b.map_or(Json::Null, Json::List);
    Some(obj(vec![
        ("id", Json::Str(id.into())),
        ("from", Json::Str(origin.into())),
        ("src_bbox", boxed(src)),
        ("tgt_bbox", boxed(tgt)),
        ("desc", Json::Str(desc)),
    ]))
}

/// Python `str.strip()` (whitespace by `str.isspace`).
fn py_strip(s: &str) -> String {
    s.trim_matches(char::is_whitespace).to_string()
}

fn box_of(r: &Json) -> Option<Vec<f64>> {
    let b = ["bbox", "tgt_bbox", "src_bbox"]
        .iter()
        .find_map(|k| r.get(k).filter(|v| v.truthy()))?;
    match b {
        Json::List(l) => l.iter().map(Json::as_f64).collect(),
        _ => None,
    }
}

fn iou(a: &[f64], b: &[f64]) -> f64 {
    let inter =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// The upsampler's answer with the locked boxes re-imposed: a row
/// re-planning a locked box under its own id (IoU ≥ 0.9) takes the locked
/// id; locked rows are kept verbatim with the upsampler's desc; its own
/// valid rows follow.
pub fn reconcile(text: &str, locked: &[Json]) -> String {
    let (mut caption, rows) = split(text);
    let mut rows = rows.unwrap_or_default();
    let id_of = |r: &Json| r.get("id").and_then(Json::as_str).map(str::to_string);
    let locked_ids: Vec<String> = locked.iter().filter_map(id_of).collect();
    let row_ids: Vec<Option<String>> = rows.iter().map(id_of).collect();
    let mut renamed: Vec<(String, String)> = Vec::new();
    for lock in locked {
        let Some(lid) = id_of(lock) else { continue };
        let lock_box = lock
            .get("bbox")
            .filter(|v| v.truthy())
            .or_else(|| lock.get("tgt_bbox").filter(|v| v.truthy()))
            .or_else(|| lock.get("src_bbox").filter(|v| v.truthy()));
        let Some(Json::List(lb)) = lock_box else {
            continue;
        };
        let Some(lb) = lb.iter().map(Json::as_f64).collect::<Option<Vec<f64>>>() else {
            continue;
        };
        if row_ids.iter().any(|r| r.as_deref() == Some(&lid)) {
            continue;
        }
        for row in &rows {
            let Some(rid) = id_of(row) else { continue };
            if renamed.iter().any(|(o, _)| *o == rid) || locked_ids.contains(&rid) {
                continue;
            }
            let picked = ["bbox", "tgt_bbox", "src_bbox"]
                .iter()
                .find_map(|k| row.get(k).filter(|v| v.truthy()));
            if valid_box(picked).is_some() && box_of(row).is_some_and(|b| iou(&b, &lb) >= 0.9) {
                renamed.push((rid, lid.clone()));
                break;
            }
        }
    }
    for row in &mut rows {
        if let Some(rid) = id_of(row)
            && let Some((_, new)) = renamed.iter().find(|(o, _)| *o == rid)
        {
            row.set("id", Json::Str(new.clone()));
        }
    }
    for (old, new) in &renamed {
        caption = caption.replace(&format!("<{old}>"), &format!("<{new}>"));
    }
    // `by_id`: the last row with an id wins, as a Python dict comprehension.
    let by_id = |id: &str| rows.iter().rev().find(|r| id_of(r).as_deref() == Some(id));
    let mut out: Vec<Json> = Vec::new();
    for lock in locked {
        let mut l = lock.clone();
        let desc = id_of(lock)
            .and_then(|id| by_id(&id))
            .and_then(|r| r.get("desc"))
            .and_then(Json::as_str)
            .map(py_strip)
            .filter(|d| !d.is_empty());
        let desc = desc.map(Json::Str).unwrap_or_else(|| {
            lock.get("desc")
                .cloned()
                .unwrap_or(Json::Str(String::new()))
        });
        l.set("desc", desc);
        out.push(l);
    }
    for row in &rows {
        if let Some(a) = valid_added(row, &caption) {
            let aid = id_of(&a);
            if !out.iter().any(|r| id_of(r) == aid) {
                out.push(a);
            }
        }
    }
    format!("{caption} {}", Json::List(out).dumps())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dumps_like_python() {
        let v = Json::parse(r#"{"b": [1, 2.0, 1e-05, 0.5], "a": "é\n\"q\"", "n": null}"#).unwrap();
        assert_eq!(
            v.dumps(),
            r#"{"b": [1, 2.0, 1e-05, 0.5], "a": "é\n\"q\"", "n": null}"#
        );
        assert_eq!(py_float(1e16), "1e+16");
        assert_eq!(py_float(123.456), "123.456");
    }

    #[test]
    fn canvas_rows_round_half_even() {
        let boxes = Json::parse(r#"[{"bbox": [0.0005, 0.2, 1.3, 0.5], "desc": "x"}]"#).unwrap();
        let Json::List(b) = boxes else { unreachable!() };
        let rows = to_canvas(&b).unwrap();
        assert_eq!(
            Json::List(rows).dumps(),
            r#"[{"id": "obj_1", "bbox": [0, 200, 1000, 500], "desc": "x"}]"#
        );
    }

    #[test]
    fn split_takes_the_last_parsable_list() {
        let (c, rows) = split(
            r#"A sign [SALE] reads <s_1>. [{"id": "s_1", "bbox": [1, 2, 300, 400], "desc": "a [b]"}]"#,
        );
        assert_eq!(c, "A sign [SALE] reads <s_1>.");
        assert_eq!(rows.unwrap().len(), 1);
        assert!(detect("A caption ending in [SALE]").is_none());
    }
}
