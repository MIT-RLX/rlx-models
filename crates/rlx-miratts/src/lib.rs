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

//! MiraTTS — 48 kHz autoregressive LM + neural-codec TTS (voice cloning).
//! `YatharthS/MiraTTS`, CC-BY-NC-SA-4.0.
//!
//! **Architecture** (verified from the real HF/github repos, 2026-07):
//! - **LM**: a **Qwen2-0.5B** decoder (`Qwen2ForCausalLM`; hidden 896, 24 layers,
//!   14 heads, 2 KV heads/GQA, intermediate 4864, rope_theta 1e6, tied lm_head)
//!   with an **extended vocab of 166 000** — the ~151.9 k base text tokens plus
//!   ~14 k **audio codec tokens** (the Orpheus pattern: the LM emits audio tokens
//!   inline). It generates a discrete speech-token stream from text, conditioned
//!   on a reference clip.
//! - **Codec = LinaCodec** (`YatharthS/LinaCodec`, same author): the encoder is a
//!   **WavLM** SSL model + quantizer (`encode(wav) → (speech_tokens,
//!   global_embedding)`, 12.5 tokens/s); the decoder is a **Dual-Path Vocos**
//!   (`decode(tokens, global_embedding) → 48 kHz`). Voice cloning is carried by
//!   the per-utterance `global_embedding` from the reference audio.
//!
//! **Pipeline** (github `mira/model.py`, via the `ncodec` package's `TTSCodec`):
//! `ctx = codec.encode(ref_wav)` → `prompt = codec.format_prompt(text, ctx)` →
//! Qwen2 AR generation → `wav = codec.decode(generated, ctx)`.
//!
//! ## Port status — LM + codec decoder validated; full pipeline wired
//!
//! - ✅ **Step 1** — [`tokens`]: token layout + prompt builder, reverse-engineered
//!   from FastBiCodec `ncodec/codec.py` + `added_tokens.json` (unit-tested).
//! - ✅ **Step 2** — [`lm`]: the Qwen2-0.5B LM (via `rlx-qwen3`, `qk_norm=false /
//!   attention_bias=true`) + AR loop. **Greedy decode validated token-for-token
//!   (16/16) against HF transformers** (`tests/lm_parity.rs`).
//! - ✅ **Step 3** — [`codec`]: the FastBiCodec `detokenizer.onnx` (acoustic +
//!   context codes → 16 kHz wav), imported **natively** via rlx-tiny-tts and
//!   **bit-exact vs onnxruntime (cos 0.99999)** (`tests/codec_parity.rs`). Runs on
//!   any RLX backend — no ort at runtime. (Required three general
//!   rlx-onnx-import/rlx-cpu fixes: `broadcast_dims` leading-dim collapse, i32
//!   Gather indices read as f32, and a missing `CastI32ToI64`.)
//! - ✅ **Step 4** — speaker encoder (`s_encoder.onnx`): mel → 32 global tokens,
//!   native via rlx-tiny-tts. MiraTTS `encode_audio` uses this path only (no
//!   WavLM / `q_encoder`). [`MiraTts::synthesize_with_ref`] encodes a raw
//!   reference clip; [`synthesize`] still accepts precomputed codes.
//!
//! See the `miratts_port_scope` memory note.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_runtime::Device;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

/// STEP 1 (done): the codec token layout + prompt builder, reverse-engineered
/// from FastBiCodec `ncodec/codec.py` + the model's `added_tokens.json`.
pub mod tokens;

/// STEP 2 (done): the Qwen2-0.5B LM + AR loop, via `rlx-qwen3`.
pub mod lm;

/// STEP 3 (done): the FastBiCodec `detokenizer.onnx` (codes → 16 kHz wav),
/// imported natively via rlx-tiny-tts, bit-exact vs onnxruntime.
pub mod codec;

/// STEP 4 (done): `s_encoder.onnx` (mel → 32 global tokens) for voice cloning.
pub mod encoder;

/// Qwen2 LM configuration (subset), parsed from the model's `config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiraConfig {
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_kv")]
    pub num_key_value_heads: usize,
    #[serde(default = "d_inter")]
    pub intermediate_size: usize,
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_rope")]
    pub rope_theta: f32,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "d_true")]
    pub tie_word_embeddings: bool,
    #[serde(default = "d_bos")]
    pub bos_token_id: u32,
    #[serde(default = "d_eos")]
    pub eos_token_id: u32,
}

fn d_hidden() -> usize {
    896
}
fn d_layers() -> usize {
    24
}
fn d_heads() -> usize {
    14
}
fn d_kv() -> usize {
    2
}
fn d_inter() -> usize {
    4864
}
fn d_vocab() -> usize {
    166000
}
fn d_rope() -> f32 {
    1_000_000.0
}
fn d_eps() -> f32 {
    1e-6
}
fn d_true() -> bool {
    true
}
fn d_bos() -> u32 {
    151643
}
fn d_eos() -> u32 {
    151645
}

impl Default for MiraConfig {
    fn default() -> Self {
        Self {
            hidden_size: d_hidden(),
            num_hidden_layers: d_layers(),
            num_attention_heads: d_heads(),
            num_key_value_heads: d_kv(),
            intermediate_size: d_inter(),
            vocab_size: d_vocab(),
            rope_theta: d_rope(),
            rms_norm_eps: d_eps(),
            tie_word_embeddings: d_true(),
            bos_token_id: d_bos(),
            eos_token_id: d_eos(),
        }
    }
}

impl MiraConfig {
    /// Load `config.json` from a MiraTTS model directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        let s = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
        serde_json::from_str(&s).with_context(|| format!("parse {}", p.display()))
    }
}

/// Output sample rate of the codec decoder (`detokenizer.onnx`, pre-upsampler).
pub const SAMPLE_RATE: u32 = codec::SAMPLE_RATE;

/// Max acoustic frames the LM emits before the codec's fixed 249-frame decode
/// (`detokenizer.onnx` has a static `speech_tokens [1, 249]` input; extra codes
/// are truncated, short streams zero-padded — one decode = ~5 s @ 16 kHz).
pub const MAX_SPEECH_FRAMES: usize = codec::SPEECH_LEN;

/// A loaded MiraTTS model directory: Qwen2-0.5B LM + native FastBiCodec
/// decoder + speaker encoder.
///
/// Voice cloning from a raw reference clip goes through [`synthesize_with_ref`]
/// (`s_encoder` mel → 32 global tokens). Precomputed codes still work via
/// [`synthesize`].
pub struct MiraTts {
    dir: PathBuf,
    config: MiraConfig,
    device: Device,
    tokenizer: Tokenizer,
    lm: Option<lm::MiraLm>,
    codec: Option<codec::MiraCodec>,
    speaker: Option<encoder::MiraSpeakerEncoder>,
}

impl MiraTts {
    /// Resolve a model directory, parse its config and load the BPE tokenizer
    /// (the heavy LM + codec weights load lazily on first [`synthesize`]).
    ///
    /// [`synthesize`]: Self::synthesize
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        let config = MiraConfig::load(dir).unwrap_or_default();
        let tok_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tok_path.display()))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            config,
            device,
            tokenizer,
            lm: None,
            codec: None,
            speaker: None,
        })
    }

    pub fn config(&self) -> &MiraConfig {
        &self.config
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Encode `text` to Qwen2 BPE ids (no special tokens — the prompt builder
    /// supplies the structural ids itself).
    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Lazily load the LM + codec decoder + speaker encoder.
    fn ensure_loaded(&mut self) -> Result<()> {
        let dec = self.dir.join("decoders");
        if self.lm.is_none() {
            self.lm = Some(lm::MiraLm::load(&self.dir, &self.config, self.device)?);
        }
        if self.codec.is_none() {
            self.codec = Some(codec::MiraCodec::load(&dec, self.device)?);
        }
        if self.speaker.is_none() {
            self.speaker = Some(encoder::MiraSpeakerEncoder::load(&dec, self.device)?);
        }
        Ok(())
    }

    /// Encode a mono 16 kHz reference clip → 32 global tokens (`s_encoder`).
    pub fn encode_ref(&mut self, pcm_16k: &[f32]) -> Result<Vec<u32>> {
        self.ensure_loaded()?;
        self.speaker
            .as_ref()
            .context("speaker encoder loaded")?
            .encode_pcm(pcm_16k)
    }

    /// Synthesize `text` conditioned on a raw reference waveform (16 kHz mono).
    ///
    /// FastBiCodec `encode_audio` only runs `s_encoder` (mel → globals); the same
    /// 32 tokens condition the LM prompt and the detokenizer.
    pub fn synthesize_with_ref(
        &mut self,
        text: &str,
        reference_pcm_16k: &[f32],
        seed: u64,
    ) -> Result<Vec<f32>> {
        let globals = self.encode_ref(reference_pcm_16k)?;
        self.synthesize(text, &globals, &globals, seed)
    }

    /// Synthesize `text` conditioned on precomputed reference context codes.
    ///
    /// `semantic_context` / `global_tokens` are normally the same 32 globals from
    /// [`encode_ref`] / `s_encoder` (MiraTTS `encode_audio`). Returns 16 kHz mono.
    pub fn synthesize(
        &mut self,
        text: &str,
        semantic_context: &[u32],
        global_tokens: &[u32],
        seed: u64,
    ) -> Result<Vec<f32>> {
        let text_ids = self.encode_text(text)?;
        let prompt = tokens::build_tts_prompt(&text_ids, semantic_context);
        self.ensure_loaded()?;
        let lm = self.lm.as_mut().context("lm loaded")?;
        let speech_codes = lm.generate_speech_codes(&prompt, MAX_SPEECH_FRAMES, seed)?;
        anyhow::ensure!(
            !speech_codes.is_empty(),
            "MiraTTS LM produced no acoustic codes for {text:?}"
        );
        let codec = self.codec.as_ref().context("codec loaded")?;
        codec.decode(&speech_codes, global_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_qwen2_0_5b() {
        let c = MiraConfig::default();
        assert_eq!(c.hidden_size, 896);
        assert_eq!(c.num_hidden_layers, 24);
        assert_eq!(c.num_key_value_heads, 2);
        assert_eq!(c.vocab_size, 166000);
        assert!(c.tie_word_embeddings);
    }

    #[test]
    fn sample_rate_is_codec_rate() {
        assert_eq!(SAMPLE_RATE, 16_000);
        assert_eq!(MAX_SPEECH_FRAMES, 249);
    }
}
