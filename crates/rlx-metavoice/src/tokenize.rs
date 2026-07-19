// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! MetaVoice custom BPE (tiktoken-style) from `tokenizer_metavoice.json`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use regex::Regex;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TokFile {
    pat_str: String,
    mergeable_ranks_b64: Vec<(String, u32)>,
    special_tokens: HashMap<String, u32>,
    offset: u32,
}

/// Trained BPE used by MetaVoice first-stage (text tokens + `offset` for audio).
pub struct MetaTokenizer {
    pat: Regex,
    ranks: HashMap<Vec<u8>, u32>,
    special: HashMap<String, u32>,
    /// Audio token ids = EnCodec code + offset (HF dump uses 2049).
    pub offset: u32,
}

impl MetaTokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read tokenizer {}", path.display()))?;
        let f: TokFile = serde_json::from_str(&raw).context("parse tokenizer_metavoice.json")?;
        let pat = Regex::new(&f.pat_str).map_err(|e| anyhow!("compile tokenizer regex: {e}"))?;
        let mut ranks = HashMap::with_capacity(f.mergeable_ranks_b64.len());
        for (b64, id) in f.mergeable_ranks_b64 {
            let bytes = B64
                .decode(b64.as_bytes())
                .with_context(|| format!("b64 rank key {b64}"))?;
            ranks.insert(bytes, id);
        }
        Ok(Self {
            pat,
            ranks,
            special: f.special_tokens,
            offset: f.offset,
        })
    }

    pub fn eot_id(&self) -> u32 {
        *self.special.get("<|endoftext|>").unwrap_or(&512)
    }

    /// Encode text with the MetaVoice BPE (byte-level ranks, same pattern as HF).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for m in self.pat.find_iter(text) {
            let m = m.map_err(|e| anyhow!("regex match: {e}"))?;
            let piece = m.as_str().as_bytes();
            out.extend(self.bpe(piece)?);
        }
        Ok(out)
    }

    fn bpe(&self, piece: &[u8]) -> Result<Vec<u32>> {
        if piece.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(&id) = self.ranks.get(piece) {
            return Ok(vec![id]);
        }
        // Start as individual bytes, merge greedily by lowest rank.
        let mut parts: Vec<Vec<u8>> = piece.iter().map(|&b| vec![b]).collect();
        loop {
            if parts.len() < 2 {
                break;
            }
            let mut best: Option<(usize, u32)> = None;
            for i in 0..parts.len() - 1 {
                let mut merged = parts[i].clone();
                merged.extend_from_slice(&parts[i + 1]);
                if let Some(&rank) = self.ranks.get(&merged) {
                    if best.map(|(_, r)| rank < r).unwrap_or(true) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let mut merged = parts[i].clone();
            merged.extend_from_slice(&parts[i + 1]);
            parts[i] = merged;
            parts.remove(i + 1);
        }
        let mut ids = Vec::with_capacity(parts.len());
        for p in parts {
            let id = self
                .ranks
                .get(&p)
                .copied()
                .ok_or_else(|| anyhow!("unknown BPE piece {p:?}"))?;
            ids.push(id);
        }
        Ok(ids)
    }
}
