// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Zonos-v0.1 transformer config (`Zyphra/Zonos-v0.1-transformer`).

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

pub const DEFAULT_LOCAL_DIR: &str = "weights/tts/zonos";
pub const DEFAULT_DAC_DIR: &str = "weights/tts/parler-dac";
pub const DEFAULT_HF_REPO: &str = "Zyphra/Zonos-v0.1-transformer";
pub const SAMPLE_RATE: u32 = 44_100;

pub const EOS_TOKEN_ID: i64 = 1024;
pub const MASKED_TOKEN_ID: i64 = 1025;
pub const CODEBOOK_SIZE: usize = 1024; // valid DAC codes [0, 1023]
pub const N_CODEBOOKS: usize = 9; // Descript DAC 44 kHz

#[derive(Debug, Clone, Deserialize)]
pub struct ZonosFileConfig {
    pub backbone: BackboneConfig,
    #[serde(default)]
    pub eos_token_id: i64,
    #[serde(default)]
    pub masked_token_id: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BackboneConfig {
    pub d_model: usize,
    pub n_layer: usize,
    pub attn_mlp_d_intermediate: usize,
    pub attn_cfg: AttnConfig,
    #[serde(default = "default_eps")]
    pub norm_epsilon: f32,
}

fn default_eps() -> f32 {
    1e-5
}

#[derive(Debug, Clone, Deserialize)]
pub struct AttnConfig {
    pub num_heads: usize,
    pub num_heads_kv: usize,
    pub rotary_emb_dim: usize,
    #[serde(default)]
    pub rotary_emb_interleaved: bool,
}

impl ZonosFileConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let s = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read {}", path.as_ref().display()))?;
        let mut c: Self = serde_json::from_str(&s).context("parse Zonos config.json")?;
        if c.eos_token_id == 0 {
            c.eos_token_id = EOS_TOKEN_ID;
        }
        if c.masked_token_id == 0 {
            c.masked_token_id = MASKED_TOKEN_ID;
        }
        Ok(c)
    }

    pub fn validate(&self) -> Result<()> {
        let h = self.backbone.attn_cfg.num_heads;
        let d = self.backbone.d_model;
        if d % h != 0 {
            bail!("d_model {d} not divisible by num_heads {h}");
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.backbone.d_model / self.backbone.attn_cfg.num_heads
    }
}
