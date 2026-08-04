//! MioTTS-0.6B LM (Qwen3) via `rlx-qwen3`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_qwen3::{
    Qwen3Config, Qwen3Generator, Qwen3RunnerBuilder, SampleOpts, apply_repetition_penalty,
    sample_token_at,
};
use rlx_runtime::Device;
use serde::Deserialize;

use crate::tokens::{self, SPEECH_BASE, SPEECH_CODEBOOK};

#[derive(Debug, Clone, Deserialize)]
pub struct MioLmConfig {
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_inter")]
    pub intermediate_size: usize,
    #[serde(default = "d_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_kv")]
    pub num_key_value_heads: usize,
    #[serde(default = "d_head")]
    pub head_dim: usize,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "d_rope")]
    pub rope_theta: f64,
    #[serde(default = "d_tie")]
    pub tie_word_embeddings: bool,
    #[serde(default = "d_bias")]
    pub attention_bias: bool,
    #[serde(default = "d_max")]
    pub max_position_embeddings: usize,
}

fn d_vocab() -> usize {
    164_480
}
fn d_hidden() -> usize {
    1024
}
fn d_inter() -> usize {
    3072
}
fn d_layers() -> usize {
    28
}
fn d_heads() -> usize {
    16
}
fn d_kv() -> usize {
    8
}
fn d_head() -> usize {
    128
}
fn d_eps() -> f64 {
    1e-6
}
fn d_rope() -> f64 {
    1_000_000.0
}
fn d_tie() -> bool {
    true
}
fn d_bias() -> bool {
    false
}
fn d_max() -> usize {
    32_768
}

impl Default for MioLmConfig {
    fn default() -> Self {
        Self {
            vocab_size: d_vocab(),
            hidden_size: d_hidden(),
            intermediate_size: d_inter(),
            num_hidden_layers: d_layers(),
            num_attention_heads: d_heads(),
            num_key_value_heads: d_kv(),
            head_dim: d_head(),
            rms_norm_eps: d_eps(),
            rope_theta: d_rope(),
            tie_word_embeddings: d_tie(),
            attention_bias: d_bias(),
            max_position_embeddings: d_max(),
        }
    }
}

impl MioLmConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        let s = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
        serde_json::from_str(&s).context("parse MioTTS config.json")
    }

    pub fn to_qwen3(&self) -> Qwen3Config {
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
            hidden_act: "silu".to_string(),
            tie_word_embeddings: self.tie_word_embeddings,
            attention_bias: self.attention_bias,
            qk_norm: true, // Qwen3
            sliding_window: None,
            max_window_layers: usize::MAX,
            use_sliding_window: false,
            num_experts: 0,
            num_experts_used: 0,
            expert_ffn_size: 0,
            shared_expert_ffn_size: 0,
            expert_weights_scale: 1.0,
        }
    }
}

pub struct MioLm {
    generator: Qwen3Generator,
}

impl MioLm {
    pub fn load(dir: &Path, cfg: &MioLmConfig, device: Device) -> Result<Self> {
        // MioTTS generates speech codes with the streaming decode loop
        // (`prefill_get_last_logits` + `decode_get_logits`), which only exists on
        // the **F32** generator — the packed-weights path builds no generator. The
        // model dir may ship both a packed `*.gguf` (which the builder would
        // auto-select once it is ≥256 MB) and an f32 `model.safetensors`, so prefer
        // the safetensors; otherwise force the F32 path off the GGUF.
        let mut builder = Qwen3RunnerBuilder::default();
        let safetensors = dir.join("model.safetensors");
        builder = if safetensors.is_file() {
            builder.weights(safetensors)
        } else {
            builder.weights(dir).packed_weights(false)
        };
        let runner = builder
            .config_value(cfg.to_qwen3())
            .device(device)
            .build()
            .context("build MioTTS Qwen3 runner")?;
        let generator = runner
            .into_generator()
            .context("MioTTS: f32 generator unavailable")?;
        Ok(Self { generator })
    }

    /// Sample speech codes (`0..12800`) from a chat-formatted prompt.
    pub fn generate_speech_codes(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
        seed: u64,
    ) -> Result<Vec<u32>> {
        let opts = SampleOpts::temperature(0.8, seed);
        let mut logits = self.generator.prefill_get_last_logits(prompt_ids)?;
        let mut counts: HashMap<u32, u32> = HashMap::new();
        let mut codes = Vec::new();
        for step in 0..max_new {
            apply_repetition_penalty(&mut logits, &counts, 1.0);
            let next = sample_token_at(&logits, opts, step as u64) as u32;
            if next == tokens::EOS {
                break;
            }
            if (SPEECH_BASE..SPEECH_BASE + SPEECH_CODEBOOK).contains(&next) {
                codes.push(next - SPEECH_BASE);
            }
            *counts.entry(next).or_insert(0) += 1;
            logits = self.generator.decode_get_logits(next)?;
        }
        Ok(codes)
    }
}
