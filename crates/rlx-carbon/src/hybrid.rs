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

//! Carbon's hybrid coding over any Qwen3 BPE: text through the BPE, the
//! nucleotides inside `<dna>…</dna>` through the 6-mer table
//! ([`crate::dna_config`]). `HybridDnaTokenizer` (feature `tokenizer`) runs it over
//! `tokenizer.json`; the exported package's program over the package's
//! tokenizer.

use anyhow::Result;

use crate::dna_config::{DnaConfig, parse_dna_region, split_by_dna_tags};

/// The base BPE: text to ids and back, no special tokens added.
pub trait Bpe {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>;
}

/// Encode text/DNA; with `auto_dna_tags` an input without a `<dna>` tag is one
/// DNA region. No BOS/EOS is added (matches Carbon/Qwen3).
pub fn encode(
    base: &dyn Bpe,
    dna: &DnaConfig,
    text: &str,
    auto_dna_tags: bool,
) -> Result<Vec<u32>> {
    let use_auto = auto_dna_tags;
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
                ids.push(dna.begin_id());
            }
            if !region.content.is_empty() {
                ids.extend(dna.process_dna_sequence_ids(&region.content));
            }
            if region.has_end {
                ids.push(dna.end_id());
            }
        } else {
            ids.extend(base.encode(content.as_str())?);
        }
    }
    Ok(ids)
}

/// Decode token ids to text, dispatching DNA ids to the 6-mer table and
/// base ids to the Qwen3 BPE. Mirrors `HybridDNATokenizer.decode`.
pub fn decode(
    base: &dyn Bpe,
    dna: &DnaConfig,
    ids: &[u32],
    skip_special_tokens: bool,
    eos_id: u32,
    pad_id: u32,
) -> Result<String> {
    // Python drops eos/pad up front when skipping specials.
    let filtered: Vec<u32>;
    let ids: &[u32] = if skip_special_tokens {
        filtered = ids
            .iter()
            .copied()
            .filter(|&t| t != eos_id && t != pad_id)
            .collect();
        &filtered
    } else {
        ids
    };

    let begin = dna.begin_id();
    let end = dna.end_id();
    let oov = dna.oov_id();

    let mut out = String::new();
    let mut i = 0;
    while i < ids.len() {
        let tid = ids[i];
        if tid == begin {
            i += 1;
            let mut dna_seq = String::new();
            while i < ids.len() && ids[i] != end {
                if let Some(tok) = dna.id_to_dna_token(ids[i]) {
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
        } else if dna.is_dna_id(tid) {
            let is_special = tid == begin || tid == end || tid == oov;
            if !(skip_special_tokens && is_special)
                && let Some(tok) = dna.id_to_dna_token(tid)
            {
                out.push_str(&tok);
            }
            i += 1;
        } else {
            let start = i;
            while i < ids.len() && !dna.is_dna_id(ids[i]) {
                i += 1;
            }
            let text_ids = &ids[start..i];
            if !text_ids.is_empty() {
                let s = base.decode(text_ids, skip_special_tokens)?;
                out.push_str(&s);
            }
        }
    }
    Ok(out)
}
