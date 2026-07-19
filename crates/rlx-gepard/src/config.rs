// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Gepard model configuration — parsed from `gepard_config.json`.
//!
//! The checkpoint is *self-describing*: `gepard_config.json` embeds the
//! full architecture (backbone config nested inside it) so the runner
//! can reconstruct the model without any training YAML.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

// std does not have IndexMap; use a Vec of (name, size) to preserve order.
/// Ordered audio-head specification — one entry per FSQ channel.
/// Order matches the 32 codebook heads (`level_audio_0..level_audio_31`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioHeadSpec {
    /// Channel name, e.g. `"level_audio_0"`.
    pub name: String,
    /// Vocabulary size for this channel (from the FSQ levels: 8, 7, 6, or 6).
    pub vocab_size: u32,
}

/// HF `gepard_config.json` stores `audio_heads` as a map `{name: vocab}`;
/// legacy configs use a list of [`AudioHeadSpec`].
fn deserialize_audio_heads<'de, D>(deserializer: D) -> Result<Vec<AudioHeadSpec>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, MapAccess, SeqAccess, Visitor};
    use std::fmt;

    struct AudioHeadsVisitor;
    impl<'de> Visitor<'de> for AudioHeadsVisitor {
        type Value = Vec<AudioHeadSpec>;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("audio_heads map or sequence")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut out = Vec::new();
            while let Some(v) = seq.next_element::<AudioHeadSpec>()? {
                out.push(v);
            }
            Ok(out)
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut out = Vec::new();
            while let Some((name, vocab_size)) = map.next_entry::<String, u32>()? {
                out.push(AudioHeadSpec { name, vocab_size });
            }
            // Stable order: level_audio_0 .. level_audio_31 when present.
            out.sort_by(|a, b| a.name.cmp(&b.name));
            Ok(out)
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(Vec::new())
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(Vec::new())
        }
    }

    deserializer.deserialize_any(AudioHeadsVisitor)
}

/// FSQ codec geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecConfig {
    /// Number of independent quantization layers (groups), e.g. 8.
    pub num_layers: usize,
    /// FSQ levels per group dimension, e.g. `[8, 7, 6, 6]`.
    pub fsq_levels: Vec<u32>,
    /// Whether codes have been unfolded to 32 per-dimension channels.
    #[serde(default = "default_true")]
    pub do_unfold: bool,
    /// Codec frame rate in Hz, e.g. 21.5.
    pub frame_rate_hz: f32,
    /// Hugging Face model id of the codec checkpoint.
    #[serde(default)]
    pub codec_id: String,
    /// Target waveform sample rate, e.g. 22050.
    pub sample_rate: u32,
}

fn default_true() -> bool {
    true
}

impl CodecConfig {
    /// NanoCodec defaults used by gepard-1.0.
    pub fn nanocodec_defaults() -> Self {
        Self {
            num_layers: 8,
            fsq_levels: vec![8, 7, 6, 6],
            do_unfold: true,
            frame_rate_hz: 21.5,
            codec_id: "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps".into(),
            sample_rate: 22050,
        }
    }

    /// Total unfolded channels per frame = num_layers × len(fsq_levels).
    pub fn num_channels(&self) -> usize {
        self.num_layers * self.fsq_levels.len()
    }

    /// Vocab sizes for the 32 channels in channel order.
    /// Repeats `fsq_levels` `num_layers` times.
    pub fn channel_vocabs(&self) -> Vec<u32> {
        self.fsq_levels
            .iter()
            .cloned()
            .cycle()
            .take(self.num_channels())
            .collect()
    }
}

/// Voice-cloning Q-Former compressor config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressorConfig {
    /// Number of output speaker-prefix query tokens (K=8).
    #[serde(default = "default_num_queries")]
    pub num_queries: usize,
    /// Number of Q-Former transformer blocks (L=2).
    #[serde(default = "default_num_layers_qformer")]
    pub num_layers: usize,
    /// Attention heads in Q-Former (default 8).
    #[serde(default = "default_num_heads_qformer")]
    pub num_heads: usize,
    /// Q-Former hidden dim — must equal backbone hidden_size (1024).
    pub d_model: Option<usize>,
}

fn default_num_queries() -> usize {
    8
}
fn default_num_layers_qformer() -> usize {
    2
}
fn default_num_heads_qformer() -> usize {
    8
}

impl CompressorConfig {
    pub fn d_model_or(&self, backbone_hidden: usize) -> usize {
        self.d_model.unwrap_or(backbone_hidden)
    }
}

/// Voice-cloning subsystem config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCloningConfig {
    /// Whether the compressor/null_prefix are present in the checkpoint.
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub compressor: Option<CompressorConfig>,
}

/// Text repetition policy for short prompts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextLayoutConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Target text-token budget (default 16).
    #[serde(default = "default_target_text_tokens")]
    pub target_text_tokens: usize,
    /// Apply repetition when token count is below this (default 13).
    #[serde(default = "default_apply_below")]
    pub apply_below: usize,
    /// Hard cap on repeat count (default 8).
    #[serde(default = "default_max_repeats")]
    pub max_repeats: usize,
}

fn default_target_text_tokens() -> usize {
    16
}
fn default_apply_below() -> usize {
    13
}
fn default_max_repeats() -> usize {
    8
}

impl Default for TextLayoutConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target_text_tokens: 16,
            apply_below: 13,
            max_repeats: 8,
        }
    }
}

/// Special token IDs (Qwen3.5 tokenizer + Gepard additions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    /// Start-of-text token ID (SOT).
    #[serde(default = "default_sot")]
    pub start_of_text: u32,
    /// End-of-text token ID (EOT).
    #[serde(default = "default_eot")]
    pub end_of_text: u32,
    /// Start-of-speech token ID (SOS) — triggers audio generation.
    #[serde(default = "default_sos")]
    pub start_of_speech: u32,
    /// End-of-speech token ID (EOS).
    #[serde(default = "default_eos")]
    pub end_of_speech: u32,
    /// Pad token used for audio sequences.
    #[serde(default = "default_tts_pad")]
    pub tts_pad: u32,
}

fn default_sot() -> u32 {
    248_073
}
fn default_eot() -> u32 {
    248_074
}
fn default_sos() -> u32 {
    248_070
}
fn default_eos() -> u32 {
    248_071
}
fn default_tts_pad() -> u32 {
    248_076
}

impl Default for SpecialTokens {
    /// Defaults from `nineninesix/gepard-1.0` `gepard_config.json`.
    fn default() -> Self {
        Self {
            start_of_text: default_sot(),
            end_of_text: default_eot(),
            start_of_speech: default_sos(),
            end_of_speech: default_eos(),
            tts_pad: default_tts_pad(),
        }
    }
}

/// RoPE / MRoPE parameters from HF `backbone_config.rope_parameters`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RopeParameters {
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

/// Backbone Qwen3.5 architecture config (subset needed for inference).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackboneConfig {
    /// Transformer hidden dimension (1024 for gepard-1.0).
    pub hidden_size: usize,
    /// Number of layers (14 for gepard-1.0).
    pub num_hidden_layers: usize,
    /// Number of query attention heads (8).
    pub num_attention_heads: usize,
    /// Number of KV heads for GQA (2).
    #[serde(default = "default_kv_heads")]
    pub num_key_value_heads: usize,
    /// Intermediate FFN size (3584 for gepard-1.0).
    #[serde(default = "default_intermediate")]
    pub intermediate_size: usize,
    /// Text vocabulary size (248320 for gepard-1.0 with extended vocab).
    pub vocab_size: usize,
    /// RMS norm epsilon.
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    /// RoPE base frequency (10M for gepard-1.0).
    #[serde(default = "default_rope_base")]
    pub rope_theta: f64,
    /// Maximum context length.
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    /// Per-head dimension (256 for gepard-1.0 extended attention).
    /// If absent, computed as hidden_size / num_attention_heads.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Qwen3.5 attention output gate (multiplicative gating of attn output).
    #[serde(default)]
    pub attn_output_gate: bool,
    /// MRoPE section layout (`rope_parameters.mrope_section`).
    #[serde(default)]
    pub rope_parameters: RopeParameters,
}

fn default_kv_heads() -> usize {
    2
}
fn default_intermediate() -> usize {
    3584
}
fn default_rms_eps() -> f64 {
    1e-6
}
fn default_rope_base() -> f64 {
    10_000_000.0
}
fn default_max_pos() -> usize {
    32768
}

impl BackboneConfig {
    /// Effective per-head dimension (may be larger than hidden/heads in Qwen3.5).
    pub fn effective_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Four-section MRoPE layout (text modality uses `[p,p,p,0]`).
    pub fn mrope_sections4(&self) -> [usize; 4] {
        let raw = &self.rope_parameters.mrope_section;
        if raw.is_empty() {
            [11, 11, 10, 0]
        } else {
            rlx_flow::rope::mrope_sections4(raw)
        }
    }

    /// Rotary dimension count. HF uses `sum(mrope_section)` pair slots × 2.
    pub fn rope_dim_count(&self) -> usize {
        let s = self.mrope_sections4();
        let pairs = s[0] + s[1] + s[2] + s[3];
        if pairs == 0 {
            self.effective_head_dim()
        } else {
            pairs * 2
        }
    }
}

/// Top-level Gepard model configuration (from `gepard_config.json`).
///
/// Supports both the legacy format (`backbone` key) and the HF repo format
/// (`backbone_config` key from `nineninesix/gepard-1.0`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepardConfig {
    /// Backbone transformer architecture.
    #[serde(alias = "backbone_config")]
    pub backbone: BackboneConfig,
    /// Per-channel FSQ audio head specs (32 entries in channel order).
    #[serde(default, deserialize_with = "deserialize_audio_heads")]
    pub audio_heads: Vec<AudioHeadSpec>,
    /// Embedding projection dim per FSQ channel (default 32).
    #[serde(default = "default_audio_embed_dim")]
    pub audio_embed_dim: usize,
    /// Codec geometry.
    #[serde(default = "CodecConfig::nanocodec_defaults")]
    pub codec: CodecConfig,
    /// Special token IDs.
    #[serde(default)]
    pub special_tokens: SpecialTokens,
    /// Voice-cloning subsystem.
    #[serde(default = "default_voice_cloning")]
    pub voice_cloning: VoiceCloningConfig,
    /// Text repetition policy (stamped from training / HF as `text_repetition`).
    #[serde(default, alias = "text_repetition")]
    pub text_layout: TextLayoutConfig,
}

fn default_audio_embed_dim() -> usize {
    32
}
fn default_voice_cloning() -> VoiceCloningConfig {
    VoiceCloningConfig {
        enabled: true,
        compressor: Some(CompressorConfig {
            num_queries: 8,
            num_layers: 2,
            num_heads: 8,
            d_model: None,
        }),
    }
}

impl GepardConfig {
    /// Load from a `gepard_config.json` path or a checkpoint directory.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let p = path.as_ref();
        let json_path = if p.is_dir() {
            p.join("gepard_config.json")
        } else {
            p.to_path_buf()
        };
        let text = std::fs::read_to_string(&json_path)
            .with_context(|| format!("read gepard_config.json at {}", json_path.display()))?;
        let cfg: Self = serde_json::from_str(&text)
            .with_context(|| format!("parse gepard_config.json at {}", json_path.display()))?;
        Ok(cfg)
    }

    /// Number of codebook heads (= total unfolded FSQ channels per frame).
    pub fn num_audio_heads(&self) -> usize {
        if !self.audio_heads.is_empty() {
            self.audio_heads.len()
        } else {
            self.codec.num_channels()
        }
    }

    /// Backbone hidden size.
    pub fn hidden_size(&self) -> usize {
        self.backbone.hidden_size
    }

    /// Number of speaker prefix tokens (K) from Q-Former (0 when VC disabled).
    pub fn num_prefix_tokens(&self) -> usize {
        if !self.voice_cloning.enabled {
            return 0;
        }
        self.voice_cloning
            .compressor
            .as_ref()
            .map(|c| c.num_queries)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_channel_vocabs_count() {
        let c = CodecConfig::nanocodec_defaults();
        let vocabs = c.channel_vocabs();
        assert_eq!(vocabs.len(), 32, "expected 32 channels");
        // Channels 0,4,8,… → level 8; channels 1,5,9,… → level 7; etc.
        assert_eq!(vocabs[0], 8);
        assert_eq!(vocabs[1], 7);
        assert_eq!(vocabs[2], 6);
        assert_eq!(vocabs[3], 6);
        assert_eq!(vocabs[4], 8); // second group
    }

    #[test]
    fn num_channels() {
        let c = CodecConfig::nanocodec_defaults();
        assert_eq!(c.num_channels(), 32);
    }
}
