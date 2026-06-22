use anyhow::{Context, Result};
use sentencepiece::SentencePieceProcessor;
use std::path::Path;

/// Moshi text tokenizer (SentencePiece, 32k vocab).
pub struct MoshiTokenizer {
    inner: SentencePieceProcessor,
    pub text_start_token: u32,
    pub text_pad_token: u32,
}

impl MoshiTokenizer {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let inner = SentencePieceProcessor::open(path.as_ref())
            .with_context(|| format!("open tokenizer {}", path.as_ref().display()))?;
        Ok(Self {
            inner,
            text_start_token: 32_000,
            text_pad_token: 3,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let ids = self
            .inner
            .encode(text)
            .with_context(|| format!("encode {text:?}"))?;
        Ok(ids.into_iter().map(|p| p.id).collect())
    }

    pub fn decode_piece(&self, id: u32) -> Result<String> {
        Ok(self.inner.decode_piece_ids(&[id])?)
    }

    /// Build per-frame text tokens for one-way generation: pad + inner-monologue prefix.
    pub fn prompt_frame_tokens(&self, prompt: &str, num_frames: usize) -> Result<Vec<u32>> {
        let mut ids = self.encode(prompt)?;
        if ids.is_empty() {
            ids.push(self.text_pad_token);
        }
        let mut frames = Vec::with_capacity(num_frames);
        for fi in 0..num_frames {
            let tok = if fi == 0 {
                self.text_start_token
            } else if fi - 1 < ids.len() {
                ids[fi - 1]
            } else {
                self.text_pad_token
            };
            frames.push(tok);
        }
        Ok(frames)
    }
}
