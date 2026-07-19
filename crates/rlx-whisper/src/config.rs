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

//! Whisper configuration — HuggingFace `config.json` / OpenAI dims.

use serde::Deserialize;
use std::path::Path;

/// Whisper model dimensions (HF field names; see OpenAI `model.py` comments in candle).
#[derive(Debug, Clone, Deserialize)]
pub struct WhisperConfig {
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub d_model: usize,
    pub encoder_attention_heads: usize,
    pub encoder_layers: usize,
    pub vocab_size: usize,
    pub max_target_positions: usize,
    pub decoder_attention_heads: usize,
    pub decoder_layers: usize,
    #[serde(default)]
    pub suppress_tokens: Vec<u32>,
    /// Suppressed on the first generated token only (HF `begin_suppress_tokens`).
    #[serde(default)]
    pub begin_suppress_tokens: Vec<u32>,
}

impl WhisperConfig {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    pub fn decoder_head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }

    /// True for multilingual Whisper checkpoints (`vocab_size == 51865`).
    ///
    /// English-only models (`*.en`, vocab 51864) must not insert `<|en|>` /
    /// `<|transcribe|>` into the decoder prompt — those tokens exist in the
    /// tokenizer but collapse greedy decode to a single word.
    pub fn is_multilingual(&self) -> bool {
        self.vocab_size >= 51865
    }

    /// Encoder sequence length after the two conv layers (stride-2 on the second).
    pub fn encoder_seq_len(&self, mel_frames: usize) -> usize {
        let after_conv1 = mel_frames;
        let pad = 1usize;
        let k = 3usize;
        let stride2 = 2usize;
        (after_conv1 + 2 * pad - k) / stride2 + 1
    }

    /// Tiny config for compile tests (no real weights required).
    pub fn tiny_synthetic() -> Self {
        Self {
            num_mel_bins: 4,
            max_source_positions: 16,
            d_model: 8,
            encoder_attention_heads: 2,
            encoder_layers: 1,
            vocab_size: 32,
            max_target_positions: 16,
            decoder_attention_heads: 2,
            decoder_layers: 1,
            suppress_tokens: vec![],
            begin_suppress_tokens: vec![],
        }
    }

    /// `openai/whisper-tiny` dimensions.
    pub fn tiny() -> Self {
        Self {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 384,
            encoder_attention_heads: 6,
            encoder_layers: 4,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 6,
            decoder_layers: 4,
            suppress_tokens: vec![],
            begin_suppress_tokens: vec![220, 50257],
        }
    }
}
