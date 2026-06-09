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

//! HuggingFace `config.json` for Qwen3-TTS checkpoints.

use anyhow::{Context, Result};
use rlx_qwen3::Qwen3Config;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const HF_MODEL_ID_06B_CUSTOM: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice";
pub const HF_MODEL_ID_06B_BASE: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";
pub const HF_TOKENIZER_12HZ: &str = "Qwen/Qwen3-TTS-Tokenizer-12Hz";
pub const WEIGHTS_FILE: &str = "model.safetensors";

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub interleaved: bool,
    #[serde(default = "default_mrope_section")]
    pub mrope_section: [usize; 3],
}

fn default_mrope_section() -> [usize; 3] {
    [24, 20, 20]
}

#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub num_code_groups: usize,
    pub vocab_size: usize,
    pub text_hidden_size: usize,
    pub text_vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_true")]
    pub qk_norm: bool,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_pad_id: u32,
    #[serde(default = "default_codec_think_id")]
    pub codec_think_id: u32,
    #[serde(default = "default_codec_think_bos_id")]
    pub codec_think_bos_id: u32,
    #[serde(default = "default_codec_think_eos_id")]
    pub codec_think_eos_id: u32,
    #[serde(default = "default_codec_nothink_id")]
    pub codec_nothink_id: u32,
    #[serde(default = "default_position_id_per_seconds")]
    pub position_id_per_seconds: usize,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub spk_id: HashMap<String, u32>,
    #[serde(default)]
    pub codec_language_id: HashMap<String, u32>,
    #[serde(default)]
    pub spk_is_dialect: HashMap<String, serde_json::Value>,
}

impl CodePredictorConfig {
    pub fn to_talker_config(&self) -> TalkerConfig {
        TalkerConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            max_position_embeddings: self.max_position_embeddings,
            num_code_groups: self.num_code_groups,
            vocab_size: self.vocab_size,
            text_hidden_size: 2048,
            text_vocab_size: 151936,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            hidden_act: self.hidden_act.clone(),
            attention_bias: self.attention_bias,
            qk_norm: self.qk_norm,
            codec_bos_id: 0,
            codec_eos_token_id: 0,
            codec_pad_id: 0,
            codec_think_id: 0,
            codec_think_bos_id: 0,
            codec_think_eos_id: 0,
            codec_nothink_id: 0,
            position_id_per_seconds: 13,
            rope_scaling: None,
            spk_id: Default::default(),
            codec_language_id: Default::default(),
            spk_is_dialect: Default::default(),
        }
    }

    pub fn to_qwen3_config(&self) -> Qwen3Config {
        Qwen3Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            hidden_act: self.hidden_act.clone(),
            tie_word_embeddings: false,
            attention_bias: self.attention_bias,
            qk_norm: self.qk_norm,
            sliding_window: None,
            max_window_layers: self.num_hidden_layers,
            use_sliding_window: false,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub num_code_groups: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_true")]
    pub qk_norm: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsConfig {
    pub tts_model_type: String,
    pub tts_model_size: String,
    pub tokenizer_type: String,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    pub talker_config: TalkerConfigNested,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfigNested {
    #[serde(flatten)]
    pub talker: TalkerConfig,
    pub code_predictor_config: CodePredictorConfig,
}

impl TalkerConfig {
    pub fn rope_sections(&self) -> [usize; 4] {
        let s = self
            .rope_scaling
            .as_ref()
            .map(|r| r.mrope_section)
            .unwrap_or([24, 20, 20]);
        [s[0], s[1], s[2], 0]
    }

    pub fn rope_dim_count(&self) -> usize {
        self.head_dim
    }

    pub fn to_qwen3_config(&self) -> Qwen3Config {
        Qwen3Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            hidden_act: self.hidden_act.clone(),
            tie_word_embeddings: false,
            attention_bias: self.attention_bias,
            qk_norm: self.qk_norm,
            sliding_window: None,
            max_window_layers: self.num_hidden_layers,
            use_sliding_window: false,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }
}

impl Qwen3TtsConfig {
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&text).context("parse config.json")
    }

    pub fn talker(&self) -> &TalkerConfig {
        &self.talker_config.talker
    }

    pub fn code_predictor(&self) -> &CodePredictorConfig {
        &self.talker_config.code_predictor_config
    }
}

pub fn resolve_model_dir(dir: &Path) -> Result<PathBuf> {
    if dir.join("config.json").is_file() {
        return Ok(dir.to_path_buf());
    }
    let nested = dir.join("Qwen3-TTS-12Hz-0.6B-CustomVoice");
    if nested.join("config.json").is_file() {
        return Ok(nested);
    }
    anyhow::bail!(
        "no config.json under {} — set RLX_QWEN3_TTS_DIR or run `just fetch-qwen3-tts`",
        dir.display()
    );
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_hidden_act() -> String {
    "silu".into()
}
fn default_true() -> bool {
    true
}
fn default_position_id_per_seconds() -> usize {
    13
}
fn default_codec_think_id() -> u32 {
    2154
}
fn default_codec_think_bos_id() -> u32 {
    2156
}
fn default_codec_think_eos_id() -> u32 {
    2157
}
fn default_codec_nothink_id() -> u32 {
    2155
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    #[serde(default = "default_true")]
    pub do_sample: bool,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default = "default_true")]
    pub subtalker_dosample: bool,
    #[serde(default = "default_top_k")]
    pub subtalker_top_k: u32,
    #[serde(default = "default_top_p")]
    pub subtalker_top_p: f32,
    #[serde(default = "default_temperature")]
    pub subtalker_temperature: f32,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
    #[serde(default = "default_min_new_tokens")]
    pub min_new_tokens: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            do_sample: true,
            top_k: 50,
            top_p: 1.0,
            temperature: 0.9,
            repetition_penalty: 1.05,
            subtalker_dosample: true,
            subtalker_top_k: 50,
            subtalker_top_p: 1.0,
            subtalker_temperature: 0.9,
            max_new_tokens: 2048,
            min_new_tokens: 2,
        }
    }
}

impl GenerationConfig {
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("generation_config.json");
        if !path.is_file() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)?;
        serde_json::from_str(&text).context("parse generation_config.json")
    }

    pub fn greedy() -> Self {
        Self {
            do_sample: false,
            top_k: 1,
            top_p: 1.0,
            temperature: 0.0,
            repetition_penalty: 1.0,
            subtalker_dosample: false,
            subtalker_top_k: 1,
            subtalker_top_p: 1.0,
            subtalker_temperature: 0.0,
            max_new_tokens: 128,
            min_new_tokens: 2,
        }
    }

    /// Greedy sampling with `repetition_penalty` / limits from `generation_config.json` when present
    /// (matches HF `_merge_generate_kwargs(do_sample=false, …)`).
    pub fn greedy_for_model_dir(dir: &Path) -> Result<Self> {
        let mut cfg = Self::from_model_dir(dir)?;
        cfg.do_sample = false;
        cfg.top_k = 1;
        cfg.top_p = 1.0;
        cfg.temperature = 0.0;
        cfg.subtalker_dosample = false;
        cfg.subtalker_top_k = 1;
        cfg.subtalker_top_p = 1.0;
        cfg.subtalker_temperature = 0.0;
        if cfg.max_new_tokens == Self::default().max_new_tokens {
            cfg.max_new_tokens = 128;
        }
        cfg.min_new_tokens = cfg.min_new_tokens.max(2);
        Ok(cfg)
    }
}

fn default_top_k() -> u32 {
    50
}
fn default_top_p() -> f32 {
    1.0
}
fn default_temperature() -> f32 {
    0.9
}
fn default_repetition_penalty() -> f32 {
    1.05
}
fn default_max_new_tokens() -> usize {
    2048
}
fn default_min_new_tokens() -> usize {
    2
}
