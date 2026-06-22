// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SNAC config (hubertsiuzdak/snac_24khz et al).

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

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
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read SNAC config {}", path.display()))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        Ok(serde_json::from_str(text)?)
    }

    /// Latent channel width (explicit or derived from the encoder stack).
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
            .unwrap_or_else(|| self.encoder_dim * 2usize.pow(self.encoder_rates.len() as u32))
    }

    pub fn n_codebooks(&self) -> usize {
        self.vq_strides.len()
    }

    /// SNAC 24 kHz reference config.
    pub fn snac_24khz() -> Self {
        Self {
            sampling_rate: 24_000,
            encoder_dim: 48,
            encoder_rates: vec![2, 4, 8, 8],
            decoder_dim: 1024,
            decoder_rates: vec![8, 8, 4, 2],
            attn_window_size: None,
            codebook_size: 4096,
            codebook_dim: 8,
            vq_strides: vec![4, 2, 1],
            noise: true,
            depthwise: true,
            latent_dim: None,
        }
    }
}
