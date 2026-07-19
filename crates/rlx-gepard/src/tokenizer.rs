// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! HF `tokenizer.json` + Gepard text-region layout (`TextRepeater`).
//!
//! Layout (nineninesix MODEL_GUIDE §9.6 / `text_repetition.py`):
//! ```text
//! R == 1 → [SOT, *text, EOT, SOS]
//! R  > 1 → [SOT, *text, EOT] × (R-1) + [SOT, *text, EOT, SOS]
//! ```
//! Only the final copy carries SOS (audio start).

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use tokenizers::Tokenizer;

use crate::config::{GepardConfig, SpecialTokens, TextLayoutConfig};

/// Loaded HF tokenizer + Gepard special-token framing.
pub struct GepardTokenizer {
    tok: Tokenizer,
    special: SpecialTokens,
    layout: TextLayoutConfig,
}

impl GepardTokenizer {
    pub fn load(dir: impl AsRef<Path>, cfg: &GepardConfig) -> Result<Self> {
        let path = dir.as_ref().join("tokenizer.json");
        let tok =
            Tokenizer::from_file(&path).map_err(|e| anyhow!("load {}: {e}", path.display()))?;
        Ok(Self {
            tok,
            special: cfg.special_tokens.clone(),
            layout: cfg.text_layout.clone(),
        })
    }

    /// Encode raw transcript text (no specials) with the HF tokenizer.
    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Full prefill text region: optional TextRepeater + SOS.
    pub fn build_prompt_ids(&self, text: &str) -> Result<Vec<u32>> {
        let body = self.encode_text(text)?;
        Ok(build_input_ids(
            &body,
            target_r(body.len(), &self.layout),
            &self.special,
        ))
    }

    pub fn special(&self) -> &SpecialTokens {
        &self.special
    }
}

/// Deterministic inference `R` (no mixed-keep coin flip).
pub fn target_r(n_text_tokens: usize, cfg: &TextLayoutConfig) -> usize {
    if !cfg.enabled {
        return 1;
    }
    if n_text_tokens == 0 || n_text_tokens >= cfg.apply_below {
        return 1;
    }
    if n_text_tokens >= cfg.target_text_tokens {
        return 1;
    }
    let r = (cfg.target_text_tokens as f64 / n_text_tokens as f64).ceil() as usize;
    r.clamp(1, cfg.max_repeats)
}

pub fn build_input_ids(text_token_ids: &[u32], r: usize, special: &SpecialTokens) -> Vec<u32> {
    let r = r.max(1);
    let mut block = Vec::with_capacity(text_token_ids.len() + 2);
    block.push(special.start_of_text);
    block.extend_from_slice(text_token_ids);
    block.push(special.end_of_text);
    let mut out = Vec::with_capacity(block.len() * r + 1);
    for _ in 0..(r.saturating_sub(1)) {
        out.extend_from_slice(&block);
    }
    out.extend_from_slice(&block);
    out.push(special.start_of_speech);
    out
}

/// Load tokenizer from a bundle dir using that dir's `gepard_config.json`.
pub fn load_bundle_tokenizer(dir: impl AsRef<Path>) -> Result<GepardTokenizer> {
    let dir = dir.as_ref();
    let cfg =
        GepardConfig::from_path(dir).with_context(|| format!("config in {}", dir.display()))?;
    GepardTokenizer::load(dir, &cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_r1() {
        let sp = SpecialTokens {
            start_of_text: 1,
            end_of_text: 2,
            start_of_speech: 3,
            end_of_speech: 4,
            tts_pad: 5,
        };
        assert_eq!(build_input_ids(&[10, 11], 1, &sp), vec![1, 10, 11, 2, 3]);
    }

    #[test]
    fn layout_r2() {
        let sp = SpecialTokens {
            start_of_text: 1,
            end_of_text: 2,
            start_of_speech: 3,
            end_of_speech: 4,
            tts_pad: 5,
        };
        assert_eq!(build_input_ids(&[10], 2, &sp), vec![1, 10, 2, 1, 10, 2, 3]);
    }

    #[test]
    fn target_r_short() {
        let cfg = TextLayoutConfig {
            enabled: true,
            target_text_tokens: 16,
            apply_below: 13,
            max_repeats: 8,
        };
        assert_eq!(target_r(4, &cfg), 4); // ceil(16/4)=4
        assert_eq!(target_r(13, &cfg), 1);
        assert_eq!(target_r(1, &cfg), 8); // capped
    }

    #[test]
    fn loads_bundle_tokenizer_when_present() {
        let dir =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/gepard");
        if !dir.join("tokenizer.json").is_file() {
            return;
        }
        let tok = load_bundle_tokenizer(&dir).expect("load");
        let ids = tok.build_prompt_ids("Hello from Gepard.").expect("prompt");
        assert!(ids.len() >= 4);
        assert_eq!(*ids.last().unwrap(), tok.special().start_of_speech);
        assert_eq!(ids[0], tok.special().start_of_text);
    }
}
