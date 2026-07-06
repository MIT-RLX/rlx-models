//! SentencePiece text tokenizer for Kyutai TTS (`tokenizer_spm_8k_en_fr_audio.model`).

use anyhow::{Context, Result};
use sentencepiece::SentencePieceProcessor;
use std::path::Path;

/// 8k en/fr + audio-control SPM tokenizer.
pub struct KyutaiTokenizer {
    sp: SentencePieceProcessor,
    pub pad_id: u32,
    pub new_word_id: u32,
}

impl KyutaiTokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let sp = SentencePieceProcessor::open(path)
            .with_context(|| format!("open SPM tokenizer {}", path.display()))?;
        Ok(Self {
            pad_id: 3,
            new_word_id: 0,
            sp,
        })
    }

    pub fn encode_word(&self, word: &str) -> Result<Vec<u32>> {
        let ids = self
            .sp
            .encode(word)
            .with_context(|| format!("encode {word:?}"))?;
        Ok(ids.into_iter().map(|id| id.id).collect())
    }

    pub fn encode_prompt_words(&self, prompt: &str) -> Result<Vec<Vec<u32>>> {
        let mut out = Vec::new();
        for word in prompt.split_whitespace() {
            if word.is_empty() {
                continue;
            }
            out.push(self.encode_word(word)?);
        }
        Ok(out)
    }
}
