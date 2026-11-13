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

//! Kraken's `PytorchCodec`: code points ↔ label sequences.
//!
//! The model metadata stores `c2l` as a JSON object mapping each output
//! string to one or more labels (`{" ": [1], "a": [2], …}`); label 0 is the
//! CTC blank. Decoding follows `PytorchCodec.decode`: single-label codes
//! first, then multi-label codes in the file's order, skipping a label
//! nothing decodes.

use anyhow::{Context, Result, anyhow};
use rlx_core::ordered_json::Json;
use std::collections::HashMap;

/// The label codec of a recognition model.
#[derive(Debug, Clone, Default)]
pub struct Codec {
    single: HashMap<usize, String>,
    /// Multi-label codes in metadata order.
    multi: Vec<(Vec<usize>, String)>,
    max_label: usize,
}

impl Codec {
    /// Parse the `codec` metadata value.
    pub fn from_json(text: &str) -> Result<Self> {
        let j = Json::parse(text).map_err(|e| anyhow!("codec JSON: {e:?}"))?;
        Self::from_value(&j)
    }

    /// Build from an already-parsed `c2l` object.
    pub fn from_value(j: &Json) -> Result<Self> {
        let obj = j
            .as_object()
            .ok_or_else(|| anyhow!("codec is not a JSON object"))?;
        let mut c = Codec::default();
        for (s, labels) in obj {
            let ls = labels
                .as_array()
                .ok_or_else(|| anyhow!("codec entry {s:?} is not a list"))?
                .iter()
                .map(|l| match l {
                    Json::Int(i) => usize::try_from(*i).context("negative label"),
                    Json::UInt(u) => usize::try_from(*u).context("label overflow"),
                    other => Err(anyhow!("codec entry {s:?}: label {other:?}")),
                })
                .collect::<Result<Vec<usize>>>()?;
            c.max_label = c.max_label.max(ls.iter().copied().max().unwrap_or(0));
            match ls.as_slice() {
                [l] => {
                    c.single.insert(*l, s.clone());
                }
                _ => c.multi.push((ls, s.clone())),
            }
        }
        Ok(c)
    }

    /// Highest label used; the network has `max_label + 1` classes.
    pub fn max_label(&self) -> usize {
        self.max_label
    }

    /// Labels (blank-free) → text.
    pub fn decode(&self, labels: &[usize]) -> String {
        let mut out = String::new();
        let mut i = 0;
        while i < labels.len() {
            if let Some(s) = self.single.get(&labels[i]) {
                out.push_str(s);
                i += 1;
                continue;
            }
            match self
                .multi
                .iter()
                .find(|(code, _)| labels[i..].starts_with(code))
            {
                Some((code, s)) => {
                    out.push_str(s);
                    i += code.len();
                }
                // Kraken (non-strict) advances one label.
                None => i += 1,
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_and_multi_label_codes() {
        let c = Codec::from_json(r#"{" ": [1], "a": [2], "ch": [3, 4], "b": [5]}"#).unwrap();
        assert_eq!(c.max_label(), 5);
        assert_eq!(c.decode(&[2, 1, 3, 4, 5]), "a chb");
        // A lone 3 decodes to nothing and is skipped.
        assert_eq!(c.decode(&[3, 2]), "a");
    }
}
