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

//! NLLB SentencePiece tokenizer (`tokenizer.json`) with FLORES-200 lang codes.

use anyhow::{Result, anyhow, bail};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct NllbTokenizer {
    tk: Tokenizer,
    eos_id: u32,
}

impl NllbTokenizer {
    /// Load HF `tokenizer.json`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let tk = Tokenizer::from_file(path).map_err(|e| anyhow!("load tokenizer: {e}"))?;
        let eos_id = tk.token_to_id("</s>").unwrap_or(2);
        Ok(Self { tk, eos_id })
    }

    /// Resolve a FLORES-200 language token id. Probes `eng_Latn` and
    /// `__eng_Latn__` forms (HF checkpoints vary).
    pub fn lang_token_id(&self, flores: &str) -> Result<u32> {
        let code = flores.trim();
        let candidates = [code.to_string(), format!("__{code}__"), format!("<{code}>")];
        for c in &candidates {
            if let Some(id) = self.tk.token_to_id(c) {
                return Ok(id);
            }
        }
        bail!(
            "nllb tokenizer: language token for `{code}` not found \
             (tried {candidates:?})"
        )
    }

    /// Encode source text the NLLB way: `[src_lang] + tokens + [eos]`.
    pub fn encode(&self, text: &str, src_lang_flores: &str) -> Result<Vec<u32>> {
        let lang_id = self.lang_token_id(src_lang_flores)?;
        let enc = self
            .tk
            .encode(text, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let mut ids = Vec::with_capacity(enc.len() + 2);
        ids.push(lang_id);
        ids.extend_from_slice(enc.get_ids());
        if ids.last() != Some(&self.eos_id) {
            ids.push(self.eos_id);
        }
        Ok(ids)
    }

    /// Decode token ids to text, dropping special tokens.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tk
            .decode(ids, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Decode keeping special tokens.
    pub fn decode_keep_special(&self, ids: &[u32]) -> Result<String> {
        self.tk
            .decode(ids, false)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }
}
