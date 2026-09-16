//! FireRedAudio config — defaults match
//! [FireRedTeam/FireRedAudio](https://huggingface.co/FireRedTeam/FireRedAudio)
//! `FireRedAudio/config.json`.

use anyhow::{Result, ensure};

/// Whisper-style understanding encoder (16 kHz mel → continuous embeds).
#[derive(Debug, Clone, PartialEq)]
pub struct AudioEncoderConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    /// Projected width into the backbone hidden size.
    pub output_dim: usize,
    pub max_source_positions: usize,
    pub n_window: usize,
}

impl Default for AudioEncoderConfig {
    fn default() -> Self {
        Self {
            d_model: 1280,
            encoder_layers: 32,
            encoder_attention_heads: 20,
            encoder_ffn_dim: 5120,
            num_mel_bins: 128,
            output_dim: 4096,
            max_source_positions: 1500,
            n_window: 1500,
        }
    }
}

impl AudioEncoderConfig {
    /// Per-head attention dimension (`d_model / heads`).
    pub fn head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    /// Raw mel frames per pre-CNN chunk (`2 * n_window`).
    pub fn chunk_frames(&self) -> usize {
        self.n_window * 2
    }
}

/// Qwen3.5 text backbone (hybrid linear / full attention).
#[derive(Debug, Clone, PartialEq)]
pub struct BackboneConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub full_attention_interval: usize,
    pub linear_conv_kernel_dim: usize,
    pub partial_rotary_factor: f32,
}

impl Default for BackboneConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 12_288,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            vocab_size: 248_320,
            max_position_embeddings: 262_144,
            rms_norm_eps: 1e-6,
            full_attention_interval: 4,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 0.25,
        }
    }
}

impl BackboneConfig {
    /// Released checkpoint uses full attention every 4th layer (8 of 32).
    pub fn num_full_attention_layers(&self) -> usize {
        self.num_hidden_layers / self.full_attention_interval
    }
}

/// RedAE continuous speech VAE (24 kHz → 25 Hz × 64-d latents).
#[derive(Debug, Clone, PartialEq)]
pub struct RedVaeConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub out_dim: usize,
    pub audio_sample_rate: usize,
    /// Waveform samples per 50 Hz patch before the extra ×2 downsample to 25 Hz.
    pub audio_patch_size: usize,
    pub extra_downsample_rate: usize,
    pub sliding_window: usize,
}

impl Default for RedVaeConfig {
    fn default() -> Self {
        Self {
            hidden_size: 896,
            intermediate_size: 3584,
            num_hidden_layers: 18,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            out_dim: 64,
            audio_sample_rate: 24_000,
            audio_patch_size: 480,
            extra_downsample_rate: 2,
            sliding_window: 64,
        }
    }
}

/// Groups 25 Hz VAE frames into backbone patch tokens (`patch_size` frames each).
#[derive(Debug, Clone, PartialEq)]
pub struct PatchEncoderConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub mlp_ratio: usize,
    pub out_dim: usize,
    pub patch_size: usize,
    pub vae_dim: usize,
}

impl Default for PatchEncoderConfig {
    fn default() -> Self {
        Self {
            depth: 8,
            hidden_size: 1024,
            num_heads: 16,
            mlp_ratio: 4,
            out_dim: 4096,
            patch_size: 4,
            vae_dim: 64,
        }
    }
}

/// Flow-matching DiT that predicts RedAE latents conditioned on the backbone.
#[derive(Debug, Clone, PartialEq)]
pub struct DitConfig {
    pub backbone_hidden_size: usize,
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub mlp_ratio: f32,
    pub patch_size: usize,
    pub history_patches: usize,
    pub vae_channels: usize,
    pub train_cfg_rate: f32,
}

impl Default for DitConfig {
    fn default() -> Self {
        Self {
            backbone_hidden_size: 4096,
            depth: 11,
            hidden_size: 1024,
            num_heads: 16,
            mlp_ratio: 4.0,
            patch_size: 4,
            history_patches: 2,
            vae_channels: 64,
            train_cfg_rate: 0.1,
        }
    }
}

/// Special-token ids and strings from the released tokenizer.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecialTokens {
    /// `<|sosp|>` — start of speech / audio segment.
    pub sosp_idx: u32,
    /// `<|eosp|>` — end of speech / audio segment.
    pub eosp_idx: u32,
    /// Understanding-side placeholder string.
    pub audio_special_token: &'static str,
    pub audio_special_token_id: u32,
    /// Generation-side placeholder string (RedAE / patch path).
    pub audio_special_token_no_latent: &'static str,
    pub audio_special_no_latent_id: u32,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            sosp_idx: 248_077,
            eosp_idx: 248_078,
            audio_special_token: "<|AUDIO|>",
            audio_special_token_id: 248_091,
            audio_special_token_no_latent: "<|AUDIO_NO_LATENT|>",
            audio_special_no_latent_id: 248_092,
        }
    }
}

/// Top-level FireRedAudio config (released checkpoint defaults).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FireRedAudioConfig {
    pub backbone: BackboneConfig,
    pub audio_encoder: AudioEncoderConfig,
    pub red_vae: RedVaeConfig,
    pub patch_encoder: PatchEncoderConfig,
    pub dit: DitConfig,
    pub tokens: SpecialTokens,
}

impl FireRedAudioConfig {
    pub fn validate(&self) -> Result<()> {
        let b = &self.backbone;
        ensure!(b.hidden_size > 0, "backbone.hidden_size must be > 0");
        ensure!(
            b.num_hidden_layers > 0,
            "backbone.num_hidden_layers must be > 0"
        );
        ensure!(
            b.num_attention_heads.is_multiple_of(b.num_key_value_heads),
            "backbone num_attention_heads must be divisible by num_key_value_heads"
        );
        ensure!(
            b.full_attention_interval > 0
                && b.num_hidden_layers
                    .is_multiple_of(b.full_attention_interval),
            "full_attention_interval must divide num_hidden_layers"
        );
        ensure!(
            self.audio_encoder.output_dim == b.hidden_size,
            "audio_encoder.output_dim must match backbone.hidden_size"
        );
        ensure!(
            self.patch_encoder.out_dim == b.hidden_size,
            "patch_encoder.out_dim must match backbone.hidden_size"
        );
        ensure!(
            self.dit.backbone_hidden_size == b.hidden_size,
            "dit.backbone_hidden_size must match backbone.hidden_size"
        );
        ensure!(
            self.red_vae.out_dim == self.dit.vae_channels,
            "red_vae.out_dim must match dit.vae_channels"
        );
        ensure!(
            self.patch_encoder.patch_size == self.dit.patch_size,
            "patch_encoder.patch_size must match dit.patch_size"
        );
        ensure!(
            self.red_vae.audio_patch_size > 0 && self.red_vae.extra_downsample_rate > 0,
            "invalid RedAE downsample settings"
        );
        Ok(())
    }

    /// 25 Hz RedAE latent rate (`sample_rate / (patch_size * extra_downsample)`).
    pub fn vae_frame_rate_hz(&self) -> f32 {
        let hop = self.red_vae.audio_patch_size * self.red_vae.extra_downsample_rate;
        self.red_vae.audio_sample_rate as f32 / hop as f32
    }

    /// Backbone patch-token rate after grouping `patch_size` VAE frames (6.25 Hz).
    pub fn patch_token_rate_hz(&self) -> f32 {
        self.vae_frame_rate_hz() / self.patch_encoder.patch_size as f32
    }
}
