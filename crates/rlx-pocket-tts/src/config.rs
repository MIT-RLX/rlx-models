// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Pocket TTS configuration. Defaults match the english config in
//! [`kyutai-labs/pocket-tts`](https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/config/english.yaml).

use serde::{Deserialize, Serialize};

/// Top-level config — wired to match the english model in the ungated mirror.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketTtsConfig {
    pub flow_lm: FlowLmConfig,
    pub mimi: MimiConfig,

    /// SentencePiece vocab size (= `flow_lm.conditioner.embed.num_embeddings - 1`).
    #[serde(default = "default_n_bins")]
    pub n_bins: usize,

    /// EOS logit threshold: `out_eos(...) > eos_threshold` ⇒ stop.
    #[serde(default = "default_eos_threshold")]
    pub eos_threshold: f32,

    /// Number of Euler steps for the flow head (1 in the released model).
    #[serde(default = "default_lsd_steps")]
    pub lsd_decode_steps: usize,

    /// Temperature for the flow head's noise init (`std = sqrt(temp)`).
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Extra latent frames decoded after EOS triggers (smoothes utterance tail).
    #[serde(default = "default_frames_after_eos")]
    pub frames_after_eos: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowLmConfig {
    pub transformer: TransformerConfig,

    /// Continuous latent dimension consumed by the FlowLM (32).
    #[serde(default = "default_latent_dim")]
    pub latent_dim: usize,

    /// Layernorm epsilon for the FlowLM transformer + flow head LNs.
    #[serde(default = "default_eps")]
    pub norm_eps: f32,

    /// Hidden dim of the flow head (`SimpleMLPAdaLN`).
    #[serde(default = "default_flow_dim")]
    pub flow_dim: usize,

    /// Number of res blocks in the flow head.
    #[serde(default = "default_flow_blocks")]
    pub flow_blocks: usize,

    /// Sinusoid frequency count for `time_embed` (half of input dim before MLP).
    #[serde(default = "default_time_half")]
    pub time_half: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    #[serde(default = "default_lm_d_model")]
    pub d_model: usize,
    #[serde(default = "default_lm_heads")]
    pub num_heads: usize,
    #[serde(default = "default_lm_layers")]
    pub num_layers: usize,
    #[serde(default = "default_lm_ffn")]
    pub dim_feedforward: usize,
    #[serde(default = "default_max_period")]
    pub max_period: f32,
    /// Sliding-window context. `None` ⇒ full causal mask.
    #[serde(default)]
    pub context: Option<usize>,
    /// LayerScale init. `None` ⇒ no layer scale (FlowLM transformer).
    #[serde(default)]
    pub layer_scale: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MimiConfig {
    /// Internal channel count of the codec (512).
    #[serde(default = "default_outer_dim")]
    pub outer_dim: usize,
    /// Latent channel count consumed from the FlowLM (32).
    #[serde(default = "default_latent_dim")]
    pub inner_dim: usize,
    /// SEANet base filter count (64) — channels at the audio side of the stack.
    #[serde(default = "default_n_filters")]
    pub n_filters: usize,
    /// SEANet kernel size for entry/exit convs (7).
    #[serde(default = "default_seanet_kernel")]
    pub kernel_size: usize,
    /// SEANet kernel size for the very last conv (3).
    #[serde(default = "default_last_kernel")]
    pub last_kernel_size: usize,
    /// SEANet residual block kernel (3).
    #[serde(default = "default_residual_kernel")]
    pub residual_kernel_size: usize,
    /// SEANet residual block dilation base (2).
    #[serde(default = "default_dilation_base")]
    pub dilation_base: usize,
    /// SEANet residual block channel compression (2).
    #[serde(default = "default_compress")]
    pub compress: usize,
    /// SEANet stride ratios (decoder applies in order; encoder reversed).
    #[serde(default = "default_ratios")]
    pub ratios: Vec<usize>,
    /// Frame-rate downsample stride (encoder_frame_rate / frame_rate = 200/12.5 = 16).
    #[serde(default = "default_downsample_stride")]
    pub downsample_stride: usize,
    /// Mimi decoder transformer config.
    pub decoder_transformer: TransformerConfig,
}

impl Default for PocketTtsConfig {
    fn default() -> Self {
        Self::english()
    }
}

impl PocketTtsConfig {
    /// English default config (matches the ungated mirror's
    /// `tts_b6369a24.safetensors`).
    pub fn english() -> Self {
        Self {
            flow_lm: FlowLmConfig {
                transformer: TransformerConfig {
                    d_model: 1024,
                    num_heads: 16,
                    num_layers: 6,
                    dim_feedforward: 4096,
                    max_period: 10_000.0,
                    context: None,
                    layer_scale: None,
                },
                latent_dim: 32,
                norm_eps: 1e-5,
                flow_dim: 512,
                flow_blocks: 6,
                time_half: 128,
            },
            mimi: MimiConfig {
                outer_dim: 512,
                inner_dim: 32,
                n_filters: 64,
                kernel_size: 7,
                last_kernel_size: 3,
                residual_kernel_size: 3,
                dilation_base: 2,
                compress: 2,
                ratios: vec![6, 5, 4],
                downsample_stride: 16,
                decoder_transformer: TransformerConfig {
                    d_model: 512,
                    num_heads: 8,
                    num_layers: 2,
                    dim_feedforward: 2048,
                    max_period: 10_000.0,
                    context: Some(250),
                    layer_scale: Some(0.01),
                },
            },
            n_bins: 4000,
            eos_threshold: -4.0,
            lsd_decode_steps: 1,
            temperature: 0.7,
            frames_after_eos: 1,
        }
    }

    /// Head dim of the FlowLM attention (1024 / 16 = 64).
    pub fn lm_head_dim(&self) -> usize {
        self.flow_lm.transformer.d_model / self.flow_lm.transformer.num_heads
    }

    /// Head dim of the Mimi decoder transformer (512 / 8 = 64).
    pub fn mimi_head_dim(&self) -> usize {
        self.mimi.decoder_transformer.d_model / self.mimi.decoder_transformer.num_heads
    }
}

fn default_n_bins() -> usize {
    4000
}
fn default_eos_threshold() -> f32 {
    -4.0
}
fn default_lsd_steps() -> usize {
    1
}
fn default_temperature() -> f32 {
    0.7
}
fn default_frames_after_eos() -> usize {
    1
}
fn default_latent_dim() -> usize {
    32
}
fn default_eps() -> f32 {
    1e-5
}
fn default_flow_dim() -> usize {
    512
}
fn default_flow_blocks() -> usize {
    6
}
fn default_time_half() -> usize {
    128
}
fn default_lm_d_model() -> usize {
    1024
}
fn default_lm_heads() -> usize {
    16
}
fn default_lm_layers() -> usize {
    6
}
fn default_lm_ffn() -> usize {
    4096
}
fn default_max_period() -> f32 {
    10_000.0
}
fn default_outer_dim() -> usize {
    512
}
fn default_n_filters() -> usize {
    64
}
fn default_seanet_kernel() -> usize {
    7
}
fn default_last_kernel() -> usize {
    3
}
fn default_residual_kernel() -> usize {
    3
}
fn default_dilation_base() -> usize {
    2
}
fn default_compress() -> usize {
    2
}
fn default_ratios() -> Vec<usize> {
    vec![6, 5, 4]
}
fn default_downsample_stride() -> usize {
    16
}
