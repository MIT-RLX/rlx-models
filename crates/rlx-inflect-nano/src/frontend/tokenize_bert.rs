//! bert-base-uncased WordPiece tokenizer (via the `tokenizers` crate) used only
//! to segment text into word groups, mirroring `tokenizer.tokenize(text)`.

use std::path::Path;

use anyhow::{Result, anyhow};
use tokenizers::Tokenizer;

pub struct BertTokenizer {
    tok: Tokenizer,
}

impl BertTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let tok = Tokenizer::from_file(path).map_err(|e| anyhow!("load bert tokenizer: {e}"))?;
        Ok(Self { tok })
    }

    /// Equivalent to HF `tokenizer.tokenize(text)`: wordpiece tokens, no specials.
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| anyhow!("bert encode: {e}"))?;
        Ok(enc.get_tokens().to_vec())
    }
}
