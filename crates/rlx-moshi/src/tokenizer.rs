use anyhow::Result;
use rlx_core::sentencepiece::SentencePiece;
use std::path::Path;

/// Moshi text tokenizer (SentencePiece, 32k vocab).
pub struct MoshiTokenizer {
    inner: SentencePiece,
    pub text_start_token: u32,
    pub text_pad_token: u32,
}

impl MoshiTokenizer {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let inner = SentencePiece::open(path)?;
        Ok(Self {
            inner,
            text_start_token: 32_000,
            text_pad_token: 3,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.inner.encode(text))
    }

    pub fn decode_piece(&self, id: u32) -> Result<String> {
        self.inner.decode(&[id])
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
