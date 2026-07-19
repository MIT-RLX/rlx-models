// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

use serde::{Deserialize, Serialize};

/// Hugging Face model id for Parler-TTS Mini v1.
pub const DEFAULT_HF_REPO: &str = "parler-tts/parler-tts-mini-v1";
/// Local ONNX + tokenizer layout (`onnx/{text_encoder,decoder}.onnx`, `tokenizer.json`).
pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/parlertts";
/// Descript DAC 44.1 kHz / 9-codebook decoder used by Parler.
pub const DEFAULT_HF_DAC_REPO: &str = "parler-tts/dac_44khz";
pub const DEFAULT_DAC_DIR: &str = "weights/tts/parler-dac";
/// DAC sample rate (Hz).
pub const SAMPLE_RATE: u32 = 44_100;

/// Parler-TTS configuration (lightweight; heavy knobs live on [`crate::InferOpts`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParlerTTSConfig {
    /// Model directory (`weights/tts/parlertts`).
    pub model_dir: String,
    /// DAC codec directory (`weights/tts/parler-dac`).
    pub dac_dir: String,
    /// Language tag (documentation / future multilingual checkpoints).
    pub language: String,
}

impl Default for ParlerTTSConfig {
    fn default() -> Self {
        Self {
            model_dir: DEFAULT_LOCAL_DIR.to_string(),
            dac_dir: DEFAULT_DAC_DIR.to_string(),
            language: "en".to_string(),
        }
    }
}
