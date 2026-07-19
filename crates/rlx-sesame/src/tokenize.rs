//! Llama-3.2 tokenizer + Sesame frame packing (33 = 32 codebooks + text).

use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;

use crate::config::SesameConfig;

pub struct SesameTokenizer {
    tok: Tokenizer,
    pub bos_id: u32,
    pub eos_id: u32,
}

impl SesameTokenizer {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("tokenizer.json");
        let tok = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", path.display()))?;
        // Llama-3 specials.
        let bos_id = tok
            .token_to_id("<|begin_of_text|>")
            .or_else(|| tok.token_to_id("<|begin_of_text|>"))
            .unwrap_or(128_000);
        let eos_id = tok.token_to_id("<|end_of_text|>").unwrap_or(128_001);
        Ok(Self {
            tok,
            bos_id,
            eos_id,
        })
    }

    /// Encode text without adding special tokens (caller wraps bos/eos).
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }
}

/// One sequence position: `num_codebooks` audio slots + 1 text slot.
#[derive(Debug, Clone)]
pub struct Frame {
    pub tokens: Vec<u32>,
    pub mask: Vec<bool>,
}

impl Frame {
    pub fn text(cfg: &SesameConfig, text_token: u32) -> Self {
        let k = cfg.num_codebooks;
        let mut tokens = vec![0u32; k + 1];
        let mut mask = vec![false; k + 1];
        tokens[k] = text_token;
        mask[k] = true;
        Self { tokens, mask }
    }

    pub fn audio(cfg: &SesameConfig, codes: &[u32]) -> Self {
        let k = cfg.num_codebooks;
        debug_assert_eq!(codes.len(), k);
        let mut tokens = vec![0u32; k + 1];
        let mut mask = vec![false; k + 1];
        tokens[..k].copy_from_slice(codes);
        for m in mask.iter_mut().take(k) {
            *m = true;
        }
        Self { tokens, mask }
    }

    /// Audio frame used during AR feedback (text column masked off).
    pub fn audio_with_empty_text(cfg: &SesameConfig, codes: &[u32]) -> Self {
        Self::audio(cfg, codes)
    }
}

/// Build prompt frames: `<bos>[speaker]text<eos>` as text-only frames.
pub fn tokenize_text_prompt(
    tok: &SesameTokenizer,
    cfg: &SesameConfig,
    text: &str,
    speaker: u32,
) -> Result<Vec<Frame>> {
    let body = format!("[{speaker}]{text}");
    let ids = tok.encode_raw(&body)?;
    let mut frames = Vec::with_capacity(ids.len() + 2);
    frames.push(Frame::text(cfg, tok.bos_id));
    for &id in &ids {
        frames.push(Frame::text(cfg, id));
    }
    frames.push(Frame::text(cfg, tok.eos_id));
    Ok(frames)
}

/// Append Mimi-encoded context as audio frames (+ trailing EOS frame of zeros).
pub fn frames_from_audio_codes(cfg: &SesameConfig, frames_codes: &[Vec<u32>]) -> Vec<Frame> {
    let k = cfg.num_codebooks;
    let mut out = Vec::with_capacity(frames_codes.len() + 1);
    for codes in frames_codes {
        let mut padded = codes.clone();
        if padded.len() < k {
            padded.resize(k, 0);
        } else if padded.len() > k {
            padded.truncate(k);
        }
        out.push(Frame::audio(cfg, &padded));
    }
    // EOS audio frame (all zeros) — matches Sesame generator.
    out.push(Frame::audio(cfg, &vec![0u32; k]));
    out
}

pub fn default_model_dir() -> std::path::PathBuf {
    std::env::var_os("RLX_SESAME_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("weights/tts/sesame"))
}

pub fn default_mimi_dir() -> std::path::PathBuf {
    rlx_mimi::default_mimi_dir()
}

pub fn ensure_model_dir(dir: &Path) -> Result<()> {
    let cfg = dir.join("config.json");
    let weights = dir.join("model.safetensors");
    let tok = dir.join("tokenizer.json");
    if !cfg.is_file() {
        anyhow::bail!(
            "missing {} — run `just fetch-sesame` (or download unsloth/csm-1b)",
            cfg.display()
        );
    }
    if !weights.is_file() {
        anyhow::bail!("missing {} — run `just fetch-sesame`", weights.display());
    }
    if !tok.is_file() {
        anyhow::bail!("missing {} — run `just fetch-sesame`", tok.display());
    }
    Ok(())
}
