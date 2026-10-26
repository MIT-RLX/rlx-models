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

//! Mistral's Tekken tokenizer (`tekken.json`) for plain text, as
//! mistral_common's `Tekkenizer.encode(…, bos=False, eos=False)`: the
//! config's pattern splits the text, each piece byte-pair merged by rank
//! (the lowest-ranked adjacent merge first), each rank offset by the
//! special tokens. Ranks past the vocabulary the model uses are dropped.

use std::collections::HashMap;

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use regex::Regex;
use serde_json::Value;

pub struct Tekken {
    pat: Regex,
    ranks: HashMap<Vec<u8>, u32>,
    offset: u32,
}

impl Tekken {
    pub fn load(path: &str) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
        let v: Value = serde_json::from_str(&raw).map_err(|e| format!("parse {path}: {e}"))?;
        let c = &v["config"];
        let pat = Regex::new(c["pattern"].as_str().ok_or("tekken: no pattern")?)
            .map_err(|e| format!("tekken pattern: {e}"))?;
        let specials = c["default_num_special_tokens"].as_u64().unwrap_or(1000) as u32;
        let vocab_size = c["default_vocab_size"].as_u64().unwrap_or(131_072) as u32;
        let inner = vocab_size - specials;
        let mut ranks = HashMap::new();
        for t in v["vocab"].as_array().ok_or("tekken: no vocab")? {
            let rank = t["rank"].as_u64().ok_or("tekken: a rank")? as u32;
            if rank >= inner {
                continue;
            }
            let bytes = B64
                .decode(t["token_bytes"].as_str().ok_or("tekken: token_bytes")?)
                .map_err(|e| format!("tekken: {e}"))?;
            ranks.insert(bytes, rank);
        }
        Ok(Self {
            pat,
            ranks,
            offset: specials,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut out = Vec::new();
        for m in self.pat.find_iter(text) {
            let piece = m
                .map_err(|e| format!("tekken split: {e}"))?
                .as_str()
                .as_bytes();
            for r in self.bpe(piece)? {
                out.push(r + self.offset);
            }
        }
        Ok(out)
    }

    fn bpe(&self, piece: &[u8]) -> Result<Vec<u32>, String> {
        if let Some(&r) = self.ranks.get(piece) {
            return Ok(vec![r]);
        }
        let mut parts: Vec<Vec<u8>> = piece.iter().map(|&b| vec![b]).collect();
        while parts.len() > 1 {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..parts.len() - 1 {
                let mut merged = parts[i].clone();
                merged.extend_from_slice(&parts[i + 1]);
                if let Some(&r) = self.ranks.get(&merged)
                    && best.is_none_or(|(_, b)| r < b)
                {
                    best = Some((i, r));
                }
            }
            let Some((i, _)) = best else { break };
            let next = parts.remove(i + 1);
            parts[i].extend(next);
        }
        parts
            .iter()
            .map(|p| {
                self.ranks
                    .get(p)
                    .copied()
                    .ok_or_else(|| format!("tekken: no rank for {p:?}"))
            })
            .collect()
    }
}
