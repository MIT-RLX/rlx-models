//! SentencePiece text tokenizer for Kyutai TTS (`tokenizer_spm_8k_en_fr_audio.model`).

use anyhow::Result;
use rlx_core::sentencepiece::SentencePiece;
use std::path::Path;

/// 8k en/fr + audio-control SPM tokenizer.
pub struct KyutaiTokenizer {
    sp: SentencePiece,
    pub pad_id: u32,
    pub new_word_id: u32,
}

impl KyutaiTokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let sp = SentencePiece::open(path)?;
        Ok(Self {
            pad_id: 3,
            new_word_id: 0,
            sp,
        })
    }

    pub fn encode_word(&self, word: &str) -> Result<Vec<u32>> {
        Ok(self.sp.encode(word))
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
