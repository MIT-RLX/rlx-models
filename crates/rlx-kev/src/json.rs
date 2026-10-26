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

//! An order-preserving JSON value — now [`rlx_core::ordered_json`], shared
//! with the other decision-model crates.
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

pub use rlx_core::ordered_json::{Json, obj};

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
