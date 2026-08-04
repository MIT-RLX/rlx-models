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

//! # rlx-outetts
//!
//! **OuteTTS** (OuteAI) — a multilingual TTS / voice-cloning model. The 1.0 line
//! this crate targets is a **Llama-3 backbone** that autoregressively emits an
//! interleaved stream of text and audio tokens, where the audio tokens are two
//! **DAC** codebooks (`<|c1_N|>` / `<|c2_N|>`, `N ∈ [0, 1024)`) decoded to a
//! 24 kHz waveform.
//!
//! It is therefore pure composition of existing RLX crates:
//!
//! - **LM backbone** → [`rlx-llama32`](https://docs.rs/rlx-llama32) (Llama-3 with
//!   the Llama-3 RoPE scaling in [`RopeScaling`]).
//! - **Codec** → [`rlx-dac`](https://docs.rs/rlx-dac) (2-codebook DAC decoder,
//!   already bit-exact on CPU/Metal/MLX/wgpu).
//!
//! Both backbones already run on every RLX backend, so OuteTTS inherits full
//! backend coverage once wired.
//!
//! This crate implements the checkpoint-free, unit-testable core: the config
//! ([`OuteTtsConfig`], [`GenerationConfig`]), the prompt format
//! ([`build_prompt_string`]), and the audio-token ↔ DAC-codebook mapping
//! ([`AudioCodeMap`], [`collect_codebooks`]) — a faithful port of audio.cpp's
//! `community_models/outetts` tokenizer. **Next step** (needs a checkpoint): drive
//! `rlx_llama32` with these prompts + a repetition-penalised sampler, route
//! generated tokens through [`AudioCodeMap`], and decode with `rlx_dac`.

use std::collections::HashMap;

use anyhow::{Result, ensure};

/// Llama-3 RoPE scaling parameters (ported from audio.cpp `OuteTTSLlama3RopeConfig`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeScaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
}

impl Default for RopeScaling {
    fn default() -> Self {
        Self {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        }
    }
}

/// OuteTTS model config (Llama-3 backbone dims + DAC codec params). Dimensional
/// fields defaulted to `0` are read from the checkpoint; codec + rope fields carry
/// the fixed OuteTTS 1.0 topology.
#[derive(Debug, Clone, PartialEq)]
pub struct OuteTtsConfig {
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub pad_token_id: i64,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: RopeScaling,
    // DAC codec.
    pub sample_rate: usize,
    pub hop_length: usize,
    pub codebook_size: usize,
    pub codebooks: usize,
    pub dac_latent_dim: usize,
    pub dac_decoder_dim: usize,
}

impl Default for OuteTtsConfig {
    fn default() -> Self {
        Self {
            bos_token_id: 0,
            eos_token_id: 0,
            pad_token_id: 0,
            hidden_size: 0,
            intermediate_size: 0,
            max_position_embeddings: 0,
            num_attention_heads: 0,
            num_hidden_layers: 0,
            num_key_value_heads: 0,
            head_dim: 0,
            vocab_size: 0,
            rms_norm_eps: 1.0e-5,
            rope_theta: 500_000.0,
            rope_scaling: RopeScaling::default(),
            sample_rate: 24_000,
            hop_length: 320,
            codebook_size: 1024,
            codebooks: 2,
            dac_latent_dim: 1024,
            dac_decoder_dim: 1536,
        }
    }
}

impl OuteTtsConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.codebooks == 2,
            "OuteTTS 1.0 uses exactly 2 DAC codebooks"
        );
        ensure!(self.codebook_size > 0, "codebook_size must be > 0");
        ensure!(
            self.sample_rate > 0 && self.hop_length > 0,
            "invalid codec rate"
        );
        Ok(())
    }
}

/// Sampling defaults (ported from audio.cpp `OuteTTSGenerationConfig`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub max_length: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.4,
            repetition_penalty: 1.1,
            repetition_window: 64,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            max_length: 8192,
        }
    }
}

/// OuteTTS control tokens (string forms; ids come from the BPE tokenizer).
pub mod special {
    pub const IM_START: &str = "<|im_start|>";
    pub const IM_END: &str = "<|im_end|>"; // == EOS
    pub const TEXT_START: &str = "<|text_start|>";
    pub const TEXT_END: &str = "<|text_end|>";
    pub const AUDIO_START: &str = "<|audio_start|>";
    pub const AUDIO_END: &str = "<|audio_end|>";
    pub const WORD_START: &str = "<|word_start|>";
    pub const WORD_END: &str = "<|word_end|>";
}

/// Build the (un-tokenised) OuteTTS generation prompt for `text`, matching
/// audio.cpp's `build_prompt`:
/// `<|im_start|>\n<|text_start|>{text}<|text_end|>\n<|audio_start|>\n`.
/// The caller encodes this with the Llama-3 BPE tokenizer.
pub fn build_prompt_string(text: &str) -> Result<String> {
    let t = text.trim();
    ensure!(!t.is_empty(), "OuteTTS requires non-empty text");
    Ok(format!(
        "{}\n{}{}{}\n{}\n",
        special::IM_START,
        special::TEXT_START,
        t,
        special::TEXT_END,
        special::AUDIO_START,
    ))
}

/// Which DAC codebook an audio token belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codebook {
    One,
    Two,
}

/// Maps generated LM token ids ↔ DAC codebook codes. In OuteTTS the `<|c1_N|>`
/// and `<|c2_N|>` special tokens are assigned arbitrary ids by the BPE tokenizer,
/// so this holds explicit `token → code` maps (as audio.cpp does). Use
/// [`AudioCodeMap::contiguous`] for the common case where the ids are contiguous.
#[derive(Debug, Clone, Default)]
pub struct AudioCodeMap {
    c1: HashMap<i32, i32>,
    c2: HashMap<i32, i32>,
    codebook_size: i32,
}

impl AudioCodeMap {
    /// Build from explicit `token → code` maps (the fully general form).
    pub fn from_maps(c1: HashMap<i32, i32>, c2: HashMap<i32, i32>, codebook_size: i32) -> Self {
        Self {
            c1,
            c2,
            codebook_size,
        }
    }

    /// Build assuming `<|c1_N|>` and `<|c2_N|>` occupy contiguous id ranges
    /// starting at `c1_base` / `c2_base` for `N ∈ [0, codebook_size]` (inclusive,
    /// mirroring audio.cpp which registers `0..=codebook_size`).
    pub fn contiguous(c1_base: i32, c2_base: i32, codebook_size: i32) -> Self {
        let mut c1 = HashMap::new();
        let mut c2 = HashMap::new();
        for code in 0..=codebook_size {
            c1.insert(c1_base + code, code);
            c2.insert(c2_base + code, code);
        }
        Self {
            c1,
            c2,
            codebook_size,
        }
    }

    /// Is `token` any audio-code token?
    pub fn is_audio_code(&self, token: i32) -> bool {
        self.c1.contains_key(&token) || self.c2.contains_key(&token)
    }

    /// Resolve a token to its `(codebook, code)`, if it is an audio-code token.
    pub fn code_for(&self, token: i32) -> Option<(Codebook, i32)> {
        if let Some(&c) = self.c1.get(&token) {
            return Some((Codebook::One, c));
        }
        if let Some(&c) = self.c2.get(&token) {
            return Some((Codebook::Two, c));
        }
        None
    }

    /// Route a generated token into the running codebooks. Faithful to audio.cpp:
    /// returns `true` if `token` is an audio-code token (of either codebook), and
    /// pushes the code only when it is below `codebook_size` (the top value is a
    /// frame marker, not a real code).
    pub fn append_audio_code(
        &self,
        token: i32,
        codebook1: &mut Vec<i32>,
        codebook2: &mut Vec<i32>,
    ) -> bool {
        if let Some(&code) = self.c1.get(&token) {
            if code < self.codebook_size {
                codebook1.push(code);
            }
            return true;
        }
        if let Some(&code) = self.c2.get(&token) {
            if code < self.codebook_size {
                codebook2.push(code);
            }
            return true;
        }
        false
    }
}

/// Collect a generated token stream into `(codebook1, codebook2)` DAC codes,
/// stopping at the first non-audio token (e.g. `<|audio_end|>` / EOS).
pub fn collect_codebooks(tokens: &[i32], map: &AudioCodeMap) -> (Vec<i32>, Vec<i32>) {
    let mut c1 = Vec::new();
    let mut c2 = Vec::new();
    for &tok in tokens {
        if !map.append_audio_code(tok, &mut c1, &mut c2) {
            break;
        }
    }
    (c1, c2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_audio_cpp() {
        let c = OuteTtsConfig::default();
        assert_eq!(c.sample_rate, 24_000);
        assert_eq!(c.hop_length, 320);
        assert_eq!(c.codebook_size, 1024);
        assert_eq!(c.codebooks, 2);
        assert_eq!(c.dac_latent_dim, 1024);
        assert_eq!(c.dac_decoder_dim, 1536);
        assert_eq!(c.rope_theta, 500_000.0);
        assert_eq!(c.rope_scaling.factor, 32.0);
        assert_eq!(c.rope_scaling.original_max_position_embeddings, 8192);
        c.validate().unwrap();
    }

    #[test]
    fn generation_defaults() {
        let g = GenerationConfig::default();
        assert_eq!(g.temperature, 0.4);
        assert_eq!(g.top_k, 40);
        assert_eq!(g.top_p, 0.9);
        assert_eq!(g.min_p, 0.05);
        assert_eq!(g.repetition_penalty, 1.1);
        assert_eq!(g.max_length, 8192);
    }

    #[test]
    fn prompt_string_wraps_text_in_control_tokens() {
        let p = build_prompt_string("hello world").unwrap();
        assert!(p.contains(special::IM_START));
        assert!(p.contains("<|text_start|>hello world<|text_end|>"));
        assert!(p.contains(special::AUDIO_START));
        assert!(build_prompt_string("   ").is_err());
    }

    #[test]
    fn code_map_roundtrips_and_ranges_are_disjoint() {
        let map = AudioCodeMap::contiguous(1000, 3000, 1024);
        // token → code → token
        assert_eq!(map.code_for(1000 + 5), Some((Codebook::One, 5)));
        assert_eq!(map.code_for(3000 + 7), Some((Codebook::Two, 7)));
        assert_eq!(map.code_for(42), None);
        assert!(map.is_audio_code(1000));
        assert!(!map.is_audio_code(50));
    }

    #[test]
    fn append_routes_by_codebook_and_guards_top_code() {
        let map = AudioCodeMap::contiguous(1000, 3000, 1024);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        assert!(map.append_audio_code(1000 + 3, &mut a, &mut b)); // c1 code 3
        assert!(map.append_audio_code(3000 + 9, &mut a, &mut b)); // c2 code 9
        // code == codebook_size (1024) is recognised but NOT pushed.
        assert!(map.append_audio_code(1000 + 1024, &mut a, &mut b));
        // a non-audio token routes nowhere.
        assert!(!map.append_audio_code(7, &mut a, &mut b));
        assert_eq!(a, vec![3]);
        assert_eq!(b, vec![9]);
    }

    #[test]
    fn collect_codebooks_deinterleaves_until_stop() {
        let map = AudioCodeMap::contiguous(1000, 3000, 1024);
        let stream = [
            1000 + 3,  // c1=3
            3000 + 7,  // c2=7
            1000 + 10, // c1=10
            3000 + 2,  // c2=2
            55,        // non-audio → stop
            1000 + 99, // ignored
        ];
        let (c1, c2) = collect_codebooks(&stream, &map);
        assert_eq!(c1, vec![3, 10]);
        assert_eq!(c2, vec![7, 2]);
    }
}
