use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DacConfig {
    pub model_type: String,
    pub sample_rate: u32,
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub n_codebooks: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub hop_length: usize,
}

impl DacConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let cfg: Self = serde_json::from_str(&text).context("parse dac config.json")?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.encoder_dim > 0, "encoder_dim must be > 0");
        ensure!(!self.encoder_rates.is_empty(), "encoder_rates empty");
        ensure!(!self.decoder_rates.is_empty(), "decoder_rates empty");
        ensure!(self.latent_dim > 0, "latent_dim must be > 0");
        ensure!(self.n_codebooks > 0, "n_codebooks must be > 0");
        ensure!(self.codebook_size > 0, "codebook_size must be > 0");
        ensure!(self.codebook_dim > 0, "codebook_dim must be > 0");
        ensure!(self.hop_length > 0, "hop_length must be > 0");
        Ok(())
    }

    pub fn config_24khz() -> Self {
        Self {
            model_type: "24khz".into(),
            sample_rate: 24_000,
            encoder_dim: 64,
            encoder_rates: vec![2, 4, 5, 8],
            decoder_dim: 1536,
            decoder_rates: vec![8, 5, 4, 2],
            latent_dim: 1024,
            n_codebooks: 32,
            codebook_size: 1024,
            codebook_dim: 8,
            hop_length: 320,
        }
    }

    pub fn config_44khz() -> Self {
        Self {
            model_type: "44khz".into(),
            sample_rate: 44_100,
            encoder_dim: 64,
            encoder_rates: vec![2, 4, 8, 8],
            decoder_dim: 1536,
            decoder_rates: vec![8, 8, 4, 2],
            latent_dim: 1024,
            n_codebooks: 9,
            codebook_size: 1024,
            codebook_dim: 8,
            hop_length: 512,
        }
    }

    pub fn config_16khz() -> Self {
        Self {
            model_type: "16khz".into(),
            sample_rate: 16_000,
            encoder_dim: 64,
            encoder_rates: vec![2, 4, 5, 8],
            decoder_dim: 1536,
            decoder_rates: vec![8, 5, 4, 2],
            latent_dim: 512,
            n_codebooks: 32,
            codebook_size: 1024,
            codebook_dim: 8,
            hop_length: 320,
        }
    }
}
