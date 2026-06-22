//! TinyTTS bundle configuration (`config.json`).

use std::collections::BTreeMap;

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct BundleConfig {
    pub model: String,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_true")]
    pub add_blank: bool,
    #[serde(default = "default_language")]
    pub language: String,
    #[serde(default)]
    pub speakers: BTreeMap<String, i64>,
    #[serde(default)]
    pub default_speaker: Option<String>,
    #[serde(default = "default_noise_scale")]
    pub noise_scale: f32,
    #[serde(default = "default_noise_scale_w")]
    pub noise_scale_w: f32,
    #[serde(default = "default_length_scale")]
    pub length_scale: f32,
    /// Latent channel width of the VITS prior / flow / decoder input (ONNX `z`).
    #[serde(default = "default_inter_channels")]
    pub inter_channels: usize,
    /// Speaker-conditioning channel width (ONNX `g`).
    #[serde(default = "default_gin_channels")]
    pub gin_channels: usize,
}

fn default_sample_rate() -> u32 {
    44100
}
fn default_true() -> bool {
    true
}
fn default_language() -> String {
    "EN".to_string()
}
fn default_noise_scale() -> f32 {
    0.667
}
fn default_noise_scale_w() -> f32 {
    0.8
}
fn default_length_scale() -> f32 {
    1.0
}
fn default_inter_channels() -> usize {
    80
}
fn default_gin_channels() -> usize {
    80
}

impl BundleConfig {
    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    /// Default speaker id (`default_speaker` name → id, else first, else 0).
    pub fn default_speaker(&self) -> i64 {
        if let Some(name) = &self.default_speaker {
            if let Some(id) = self.speakers.get(name) {
                return *id;
            }
        }
        self.speakers.values().next().copied().unwrap_or(0)
    }

    /// Resolve a speaker name (case-insensitive) to its id, falling back to default.
    pub fn speaker_id(&self, name: &str) -> i64 {
        self.speakers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| *v)
            .unwrap_or_else(|| self.default_speaker())
    }
}
