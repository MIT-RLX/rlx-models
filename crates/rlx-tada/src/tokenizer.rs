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

//! Llama-3 tokenizer plumbing.
//!
//! TADA reuses Llama-3.2's vocabulary unchanged, including its special tokens —
//! which matters more than usual here, because `_generate` decides what to keep
//! versus mask in the prompt by *identity* of the header and turn-end tokens.
//! Those ids are resolved by name rather than hard-coded, so a checkpoint that
//! ships a differently-numbered vocabulary fails loudly instead of silently
//! masking the wrong positions.
//!
//! Upstream loads this from `meta-llama/Llama-3.2-1B`, which is gated; any
//! ungated mirror of the same tokenizer works (`unsloth/Llama-3.2-1B` is what
//! the `just fetch-tada` recipe uses).

#[cfg(feature = "tokenizer")]
use anyhow::Context;
use anyhow::Result;
#[cfg(not(feature = "tokenizer"))]
use anyhow::bail;
use std::path::Path;

/// Special token ids `_generate` branches on.
#[derive(Debug, Clone, Copy)]
pub struct SpecialIds {
    pub bos: u32,
    pub eos: u32,
    pub eot: u32,
    pub start_header: u32,
    pub end_header: u32,
    pub pad: u32,
}

impl SpecialIds {
    /// Whether a token survives prompt masking on its own merits (headers and
    /// turn boundaries stay; content is replaced).
    pub fn is_structural(&self, token: u32) -> bool {
        token == self.start_header || token == self.end_header || token == self.eot
    }
}

/// Text → token ids, plus the special ids above.
pub struct TadaTokenizer {
    #[cfg(feature = "tokenizer")]
    inner: tokenizers::Tokenizer,
    pub special: SpecialIds,
}

impl TadaTokenizer {
    /// Load `tokenizer.json` from a file or from a directory containing one.
    #[cfg(feature = "tokenizer")]
    pub fn load(path: &Path) -> Result<Self> {
        let file = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };
        let inner = tokenizers::Tokenizer::from_file(&file)
            .map_err(|e| anyhow::anyhow!("load {}: {e}", file.display()))?;
        let id = |name: &str| -> Result<u32> {
            inner
                .token_to_id(name)
                .with_context(|| format!("{} has no `{name}` token", file.display()))
        };
        let special = SpecialIds {
            bos: id("<|begin_of_text|>")?,
            eos: id("<|end_of_text|>")?,
            eot: id("<|eot_id|>")?,
            start_header: id("<|start_header_id|>")?,
            end_header: id("<|end_header_id|>")?,
            pad: id("<|finetune_right_pad_id|>")?,
        };
        Ok(Self { inner, special })
    }

    #[cfg(not(feature = "tokenizer"))]
    pub fn load(_path: &Path) -> Result<Self> {
        bail!("rlx-tada was built without the `tokenizer` feature")
    }

    /// Encode without adding special tokens — every special token TADA needs is
    /// placed explicitly by the prompt builder.
    #[cfg(feature = "tokenizer")]
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    #[cfg(not(feature = "tokenizer"))]
    pub fn encode(&self, _text: &str) -> Result<Vec<u32>> {
        bail!("rlx-tada was built without the `tokenizer` feature")
    }

    #[cfg(feature = "tokenizer")]
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("detokenize: {e}"))
    }

    #[cfg(not(feature = "tokenizer"))]
    pub fn decode(&self, _ids: &[u32]) -> Result<String> {
        bail!("rlx-tada was built without the `tokenizer` feature")
    }
}

/// The chat scaffolding `generate()` wraps every utterance in. The system turn
/// is deliberately empty: TADA uses the structure, not its content.
pub const PREFIX_TEMPLATE: &str = "<|start_header_id|>system<|end_header_id|><|eot_id|>\
<|start_header_id|>assistant<|end_header_id|>";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structural_tokens_are_exactly_the_header_and_turn_markers() {
        let s = SpecialIds {
            bos: 1,
            eos: 2,
            eot: 3,
            start_header: 4,
            end_header: 5,
            pad: 6,
        };
        assert!(s.is_structural(3));
        assert!(s.is_structural(4));
        assert!(s.is_structural(5));
        // bos / eos are kept by a separate rule in the prompt masker, not here.
        assert!(!s.is_structural(1));
        assert!(!s.is_structural(2));
        assert!(!s.is_structural(99));
    }
}
