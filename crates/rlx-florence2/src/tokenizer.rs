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

//! Florence-2 BART tokenizer with the 1024 task/location tokens the processor
//! adds at runtime (`<od>`…, `<loc_0>`…`<loc_999>`, `<cap>`…).

use anyhow::{Result, anyhow};
use std::path::Path;
use tokenizers::{AddedToken, Tokenizer};

/// Number of location tokens (`<loc_0>`..`<loc_999>`).
pub const NUM_LOC_BINS: usize = 1000;

pub struct Florence2Tokenizer {
    tk: Tokenizer,
    /// Token id of `<loc_0>` (subsequent bins are contiguous).
    loc_base: u32,
}

impl Florence2Tokenizer {
    /// Load `tokenizer.json` and append the Florence-2 task/location tokens in
    /// the exact order `Florence2Processor` uses (so their ids match the
    /// checkpoint embedding rows).
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut tk = Tokenizer::from_file(path).map_err(|e| anyhow!("load tokenizer: {e}"))?;
        let base = tk.get_vocab_size(true) as u32;
        let mut added: Vec<AddedToken> = Vec::with_capacity(1024);
        for s in ["<od>", "</od>", "<ocr>", "</ocr>"] {
            added.push(AddedToken::from(s, true));
        }
        for x in 0..NUM_LOC_BINS {
            added.push(AddedToken::from(format!("<loc_{x}>"), true));
        }
        for s in EXTRA_TOKENS {
            added.push(AddedToken::from(*s, true));
        }
        tk.add_special_tokens(&added);
        // <od>(+0..3), then <loc_0> at base+4.
        let loc_base = base + 4;
        Ok(Self { tk, loc_base })
    }

    /// Encode the task prompt the way the processor does: expand the task
    /// token to its prompt string, then BART-encode with `<s>`/`</s>`.
    pub fn encode_prompt(&self, text: &str) -> Result<Vec<u32>> {
        let prompt = crate::config::construct_prompt(text);
        let enc = self
            .tk
            .encode(prompt, true)
            .map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token ids to text, keeping special tokens (so `<loc_N>` survive
    /// for the post-processor).
    pub fn decode_keep_special(&self, ids: &[u32]) -> Result<String> {
        self.tk
            .decode(ids, false)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Decode to clean text, dropping special tokens (for captions).
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tk
            .decode(ids, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// If `id` is a `<loc_N>` token, return `N`.
    pub fn loc_index(&self, id: u32) -> Option<usize> {
        if id >= self.loc_base && (id as usize) < self.loc_base as usize + NUM_LOC_BINS {
            Some((id - self.loc_base) as usize)
        } else {
            None
        }
    }
}

const EXTRA_TOKENS: &[&str] = &[
    "<cap>",
    "</cap>",
    "<ncap>",
    "</ncap>",
    "<dcap>",
    "</dcap>",
    "<grounding>",
    "</grounding>",
    "<seg>",
    "</seg>",
    "<sep>",
    "<region_cap>",
    "</region_cap>",
    "<region_to_desciption>",
    "</region_to_desciption>",
    "<proposal>",
    "</proposal>",
    "<poly>",
    "</poly>",
    "<and>",
];
