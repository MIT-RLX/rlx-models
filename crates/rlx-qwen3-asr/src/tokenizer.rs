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

//! Qwen2 byte-level BPE tokenizer built from `vocab.json` + `merges.txt`, plus
//! the Qwen3-ASR chat-template prompt builder.
//!
//! Special tokens are spliced by id (text chunks never contain them), so the
//! base BPE need not register the `<|…|>` markers — the model only emits plain
//! text tokens during transcription.

use crate::config::Qwen3AsrConfig;
use anyhow::{Context, Result, anyhow};
use std::path::Path;
use tokenizers::Tokenizer;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

/// `<|im_start|>` / `<|im_end|>` chat markers (fixed Qwen2/3 ids).
const IM_START: u32 = 151644;
const IM_END: u32 = 151645;

pub struct AsrTokenizer {
    inner: Tokenizer,
}

impl AsrTokenizer {
    /// Load `vocab.json` + `merges.txt` from a model directory.
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let vocab = dir.join("vocab.json");
        let merges = dir.join("merges.txt");
        anyhow::ensure!(vocab.is_file(), "missing vocab.json under {dir:?}");
        anyhow::ensure!(merges.is_file(), "missing merges.txt under {dir:?}");

        let bpe = BPE::from_file(
            vocab.to_str().context("vocab path utf8")?,
            merges.to_str().context("merges path utf8")?,
        )
        .build()
        .map_err(|e| anyhow!("build BPE: {e}"))?;

        let mut inner = Tokenizer::new(bpe);
        // Qwen: byte-level, add_prefix_space=false, use_regex=true.
        inner.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
        inner.with_decoder(Some(ByteLevelDecoder::default()));
        Ok(Self { inner })
    }

    /// Byte-level BPE encode of a plain text chunk (no special tokens).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode generated text token ids to a string.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Build the Qwen3-ASR prompt:
    /// `<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n`
    /// `<|audio_start|>{<|audio_pad|>×n}<|audio_end|><|im_end|>\n<|im_start|>assistant\n`
    pub fn build_prompt(
        &self,
        cfg: &Qwen3AsrConfig,
        system_text: &str,
        n_audio: usize,
    ) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(n_audio + 32);

        ids.push(IM_START);
        ids.extend(self.encode(&format!("system\n{system_text}"))?);
        ids.push(IM_END);
        ids.extend(self.encode("\n")?);

        ids.push(IM_START);
        ids.extend(self.encode("user\n")?);
        ids.push(cfg.audio_start_token_id);
        ids.extend(std::iter::repeat_n(cfg.audio_token_id, n_audio));
        ids.push(cfg.audio_end_token_id);
        ids.push(IM_END);
        ids.extend(self.encode("\n")?);

        ids.push(IM_START);
        ids.extend(self.encode("assistant\n")?);
        Ok(ids)
    }
}
