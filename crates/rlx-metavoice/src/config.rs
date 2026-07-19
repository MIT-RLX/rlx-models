// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

use serde::{Deserialize, Serialize};

pub const DEFAULT_HF_REPO: &str = "metavoiceio/metavoice-1B-v0.1";
pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/metavoice";
pub const SAMPLE_RATE: u32 = 24_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaVoiceConfig {
    pub model_dir: String,
    pub language: String,
}

impl Default for MetaVoiceConfig {
    fn default() -> Self {
        Self {
            model_dir: DEFAULT_LOCAL_DIR.to_string(),
            language: "en".to_string(),
        }
    }
}

/// First-stage GPT hyperparams (from `first_stage_args.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstStageArgs {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub block_size: usize,
    #[serde(default)]
    pub bias: bool,
    pub vocab_sizes: Vec<usize>,
    #[serde(default)]
    pub dropout: f32,
    #[serde(default = "default_true")]
    pub causal: bool,
    #[serde(default = "default_rmsnorm")]
    pub norm_type: String,
    #[serde(default = "default_eps")]
    pub rmsnorm_eps: f32,
    #[serde(default = "default_swiglu")]
    pub nonlinearity_type: String,
    #[serde(default = "default_true")]
    pub spk_emb_on_text: bool,
    #[serde(default)]
    pub attn_kernel_type: Option<String>,
    #[serde(default)]
    pub swiglu_multiple_of: Option<usize>,
    #[serde(default)]
    pub spkemb_dropout: Option<f32>,
    #[serde(default = "default_spk_emb")]
    pub speaker_emb_size: usize,
}

fn default_true() -> bool {
    true
}
fn default_eps() -> f32 {
    1e-5
}
fn default_rmsnorm() -> String {
    "rmsnorm".into()
}
fn default_swiglu() -> String {
    "swiglu".into()
}

fn default_spk_emb() -> usize {
    256
}

impl Default for FirstStageArgs {
    fn default() -> Self {
        Self {
            n_layer: 24,
            n_head: 16,
            n_embd: 2048,
            block_size: 2048,
            bias: false,
            vocab_sizes: vec![2562],
            dropout: 0.0,
            causal: true,
            norm_type: "rmsnorm".into(),
            rmsnorm_eps: 1e-5,
            nonlinearity_type: "swiglu".into(),
            spk_emb_on_text: true,
            attn_kernel_type: Some("torch_attn".into()),
            swiglu_multiple_of: Some(256),
            spkemb_dropout: Some(0.1),
            speaker_emb_size: 256,
        }
    }
}

/// Second-stage hyperparams (from `second_stage_args.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecondStageArgs {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub block_size: usize,
    #[serde(default)]
    pub bias: bool,
    pub vocab_sizes: Vec<usize>,
    #[serde(default)]
    pub target_vocab_sizes: Vec<usize>,
    #[serde(default)]
    pub dropout: f32,
    #[serde(default)]
    pub causal: bool,
}

impl Default for SecondStageArgs {
    fn default() -> Self {
        Self {
            n_layer: 6,
            n_head: 6,
            n_embd: 384,
            block_size: 1024,
            bias: false,
            vocab_sizes: vec![1538, 1025],
            target_vocab_sizes: vec![1025; 6],
            dropout: 0.0,
            causal: false,
        }
    }
}
