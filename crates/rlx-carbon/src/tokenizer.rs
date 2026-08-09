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

//! Native port of Carbon's `HybridDNATokenizer` (`tokenizer.py`).
//!
//! Text is encoded with the base Qwen3 byte-level BPE from `tokenizer.json`;
//! nucleotide runs inside `<dna>…</dna>` are encoded with the algorithmic
//! DNA 6-mer table in [`crate::dna_config`]. Decode is the inverse, splitting
//! ids at the DNA boundary (`dna_start_id`) and dispatching each run to the
//! matching detokenizer.

use crate::dna_config::{DnaConfig, parse_dna_region, split_by_dna_tags};
use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// Carbon-500M generation `eos`/`pad` id (`<|endoftext|>`).
const DEFAULT_EOS_ID: u32 = 151643;

/// Hybrid Qwen3-BPE + DNA-6mer tokenizer.
pub struct HybridDnaTokenizer {
    base: Tokenizer,
    dna: DnaConfig,
    eos_id: u32,
    pad_id: u32,
}

impl HybridDnaTokenizer {
    /// Load from a Carbon model directory (`tokenizer.json` + `dna_config.json`
    /// + optional `generation_config.json`). Missing `dna_config.json` falls
    /// back to the Carbon defaults ([`DnaConfig::default`]).
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let tok_path = dir.join("tokenizer.json");
        let base = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("load base tokenizer {}: {e}", tok_path.display()))?;

        let dna_path = dir.join("dna_config.json");
        let dna = if dna_path.is_file() {
            DnaConfig::from_file(&dna_path)?
        } else {
            DnaConfig::default()
        };

        let (eos_id, pad_id) = read_eos_pad(&dir.join("generation_config.json"));
        Ok(Self {
            base,
            dna,
            eos_id,
            pad_id,
        })
    }

    /// Construct from an already-parsed base tokenizer + DNA config.
    pub fn new(base: Tokenizer, dna: DnaConfig) -> Self {
        Self {
            base,
            dna,
            eos_id: DEFAULT_EOS_ID,
            pad_id: DEFAULT_EOS_ID,
        }
    }

    pub fn dna_config(&self) -> &DnaConfig {
        &self.dna
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    pub fn is_eos(&self, id: u32) -> bool {
        id == self.eos_id
    }

    /// Full model vocabulary (base BPE + DNA range).
    pub fn vocab_size(&self) -> usize {
        self.dna.total_vocab_size()
    }

    /// Encode text/DNA to token ids using the tokenizer's default
    /// `auto_dna_tags` behavior. No BOS/EOS is added (matches Carbon/Qwen3).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_opt(text, None)
    }

    /// Encode with an explicit `auto_dna_tags` override: when `Some(true)` and
    /// the input has no `<dna>` tag, the whole input is treated as one DNA
    /// region (`<dna>…</dna>`). `None` uses the config default.
    pub fn encode_opt(&self, text: &str, auto_dna_tags: Option<bool>) -> Result<Vec<u32>> {
        let use_auto = auto_dna_tags.unwrap_or(self.dna.auto_dna_tags);
        let wrapped;
        let text = if use_auto && !text.contains("<dna>") {
            wrapped = format!("<dna>{text}</dna>");
            wrapped.as_str()
        } else {
            text
        };

        let mut ids = Vec::new();
        for (content, is_dna) in split_by_dna_tags(text) {
            if is_dna {
                let region = parse_dna_region(&content);
                if region.has_start {
                    ids.push(self.dna.begin_id());
                }
                if !region.content.is_empty() {
                    ids.extend(self.dna.process_dna_sequence_ids(&region.content));
                }
                if region.has_end {
                    ids.push(self.dna.end_id());
                }
            } else {
                let enc = self
                    .base
                    .encode(content.as_str(), false)
                    .map_err(|e| anyhow::anyhow!("base BPE encode: {e}"))?;
                ids.extend_from_slice(enc.get_ids());
            }
        }
        Ok(ids)
    }

    /// Decode token ids to text, dispatching DNA ids to the 6-mer table and
    /// base ids to the Qwen3 BPE. Mirrors `HybridDNATokenizer.decode`.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        // Python drops eos/pad up front when skipping specials.
        let filtered: Vec<u32>;
        let ids: &[u32] = if skip_special_tokens {
            filtered = ids
                .iter()
                .copied()
                .filter(|&t| t != self.eos_id && t != self.pad_id)
                .collect();
            &filtered
        } else {
            ids
        };

        let begin = self.dna.begin_id();
        let end = self.dna.end_id();
        let oov = self.dna.oov_id();

        let mut out = String::new();
        let mut i = 0;
        while i < ids.len() {
            let tid = ids[i];
            if tid == begin {
                i += 1;
                let mut dna_seq = String::new();
                while i < ids.len() && ids[i] != end {
                    if let Some(tok) = self.dna.id_to_dna_token(ids[i]) {
                        dna_seq.push_str(&tok);
                    }
                    i += 1;
                }
                if skip_special_tokens {
                    out.push_str(&dna_seq);
                } else {
                    out.push_str("<dna>");
                    out.push_str(&dna_seq);
                    if i < ids.len() && ids[i] == end {
                        out.push_str("</dna>");
                        i += 1;
                    }
                }
            } else if self.dna.is_dna_id(tid) {
                let is_special = tid == begin || tid == end || tid == oov;
                if !(skip_special_tokens && is_special) {
                    if let Some(tok) = self.dna.id_to_dna_token(tid) {
                        out.push_str(&tok);
                    }
                }
                i += 1;
            } else {
                let start = i;
                while i < ids.len() && !self.dna.is_dna_id(ids[i]) {
                    i += 1;
                }
                let text_ids = &ids[start..i];
                if !text_ids.is_empty() {
                    let s = self
                        .base
                        .decode(text_ids, skip_special_tokens)
                        .map_err(|e| anyhow::anyhow!("base BPE decode: {e}"))?;
                    out.push_str(&s);
                }
            }
        }
        Ok(out)
    }
}

/// Read `eos_token_id` / `pad_token_id` from `generation_config.json`, falling
/// back to Carbon's `<|endoftext|>` id for either that is absent/null.
fn read_eos_pad(path: &Path) -> (u32, u32) {
    let parse = || -> Option<(Option<u32>, Option<u32>)> {
        let data = std::fs::read_to_string(path).ok()?;
        let v: serde_json::Value = serde_json::from_str(&data).ok()?;
        let eos = v
            .get("eos_token_id")
            .and_then(|x| x.as_u64())
            .map(|x| x as u32);
        let pad = v
            .get("pad_token_id")
            .and_then(|x| x.as_u64())
            .map(|x| x as u32);
        Some((eos, pad))
    };
    match parse() {
        Some((eos, pad)) => (eos.unwrap_or(DEFAULT_EOS_ID), pad.unwrap_or(DEFAULT_EOS_ID)),
        None => (DEFAULT_EOS_ID, DEFAULT_EOS_ID),
    }
}

/// Resolve a Carbon model directory from a `--model` argument that may point at
/// the directory itself or at the `model.safetensors`/`config.json` inside it.
pub fn model_dir_of(path: &Path) -> Result<std::path::PathBuf> {
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    path.parent()
        .map(|p| p.to_path_buf())
        .with_context(|| format!("cannot resolve model directory for {}", path.display()))
}
