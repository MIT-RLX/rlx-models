// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! SNAC 24 kHz decoder configuration (`snac_24khz_decoder_config.json`).

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// SNAC model hyper-parameters (hubertsiuzdak/snac_24khz).
#[derive(Debug, Clone, Deserialize)]
pub struct SnacConfig {
    pub sampling_rate: u32,
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub attn_window_size: Option<usize>,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub vq_strides: Vec<usize>,
    pub noise: bool,
    pub depthwise: bool,
    #[serde(default)]
    pub latent_dim: Option<usize>,
}

impl SnacConfig {
    /// Load JSON config from disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read SNAC config {}", path.display()))?;
        Self::from_json(&text).with_context(|| format!("parse SNAC config {}", path.display()))
    }

    /// Parse JSON config text.
    pub fn from_json(text: &str) -> Result<Self> {
        Ok(serde_json::from_str(text)?)
    }

    /// Latent channel width — explicit in export JSON or derived from encoder stack.
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
            .unwrap_or_else(|| self.encoder_dim * 2_usize.pow(self.encoder_rates.len() as u32))
    }

    pub fn n_codebooks(&self) -> usize {
        self.vq_strides.len()
    }
}
