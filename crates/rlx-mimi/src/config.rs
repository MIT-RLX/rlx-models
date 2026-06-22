use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Flat HF `config.json` for [`kyutai/mimi`](https://huggingface.co/kyutai/mimi).
#[derive(Debug, Clone, Deserialize)]
pub struct MimiConfig {
    pub audio_channels: usize,
    pub num_filters: usize,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub hidden_size: usize,
    pub upsampling_ratios: Vec<usize>,
    pub num_residual_layers: usize,
    pub residual_kernel_size: usize,
    pub dilation_growth_rate: usize,
    pub compress: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    #[serde(default = "default_num_quantizers")]
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub vector_quantization_hidden_dimension: usize,
    pub upsample_groups: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub sliding_window: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_layer_scale")]
    pub layer_scale_initial_scale: f32,
    pub sampling_rate: u32,
    #[serde(default = "default_frame_rate")]
    pub frame_rate: f32,
    #[serde(default = "default_trim_right_ratio")]
    pub trim_right_ratio: f32,
}

fn default_num_quantizers() -> usize {
    32
}
fn default_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_layer_scale() -> f32 {
    0.01
}
fn default_frame_rate() -> f32 {
    12.5
}
fn default_trim_right_ratio() -> f32 {
    1.0
}

impl MimiConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
    }

    /// SEANet hop length before the stride-2 frame-rate downsample.
    pub fn seanet_hop(&self) -> usize {
        self.upsampling_ratios.iter().product()
    }

    /// PCM samples per one codec frame @ [`Self::frame_rate`] Hz.
    pub fn samples_per_codec_frame(&self) -> usize {
        self.seanet_hop() * 2
    }

    /// Approximate encoded frame count for `pcm_len` samples (matches HF `get_encoded_length`).
    pub fn encoded_frame_count(&self, pcm_len: usize) -> usize {
        if pcm_len == 0 {
            return 0;
        }
        pcm_len.div_ceil(self.samples_per_codec_frame())
    }

    pub fn num_acoustic_quantizers(&self) -> usize {
        self.num_quantizers
            .saturating_sub(self.num_semantic_quantizers)
    }

    /// Nominal bitrate in bits per second (codebooks × log2(codebook_size) × frame_rate).
    pub fn bitrate_bps(&self) -> f32 {
        let bits_per_frame = self.num_quantizers as f32 * (self.codebook_size as f32).log2();
        bits_per_frame * self.frame_rate
    }

    /// Kernel size for the frame-rate `downsample` / `upsample` convs.
    pub fn frame_rate_kernel(&self) -> usize {
        let encodec_frame_rate = self.sampling_rate as f32 / self.seanet_hop() as f32;
        let ratio = (encodec_frame_rate / self.frame_rate) as usize;
        2 * ratio.max(1)
    }
}
