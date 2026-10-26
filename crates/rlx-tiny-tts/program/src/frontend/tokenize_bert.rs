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

//! bert-base-uncased WordPiece segmentation through the host's tokenizer
//! (`tokenizer.encode`), the ids read back as tokens from its vocabulary —
//! `tokenizer.tokenize(text)`, as rlx-inflect-nano's does it.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rlx_program::{Message, host};
use serde_json::json;

/// The tokenizer's file, as the package names it.
const TOKENIZER: &str = "sidecar:frontend/bert/tokenizer.json";

pub struct BertTokenizer {
    tokens: HashMap<u64, String>,
}

impl BertTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let tok: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let tokens = tok["model"]["vocab"]
            .as_object()
            .context("tokenizer.json has no model.vocab")?
            .iter()
            .filter_map(|(k, v)| Some((v.as_u64()?, k.clone())))
            .collect();
        Ok(Self { tokens })
    }

    /// WordPiece tokens, no specials.
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        let m = host(
            "tokenizer.encode",
            Message::body(json!({"tokenizer": TOKENIZER, "text": text})),
        )
        .map_err(|e| anyhow!("bert encode: {e}"))?;
        Ok(m.body["ids"]
            .as_array()
            .context("no ids")?
            .iter()
            .filter_map(|v| self.tokens.get(&v.as_u64()?).cloned())
            .filter(|t| !matches!(t.as_str(), "[CLS]" | "[SEP]" | "[PAD]"))
            .collect())
    }
}
