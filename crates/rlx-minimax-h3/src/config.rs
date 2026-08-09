// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! MiniMax-H3 configuration parsing.
//!
//! The checkpoint is a `diffusers` modular pipeline (`MiniMaxH3ModularPipeline`)
//! whose `model_index.json` names one subfolder per component. Each subfolder
//! carries its own `config.json`; this module mirrors the fields the port reads.
//!
//! Layout on disk:
//!
//! ```text
//! MiniMax-H3/
//!   model_index.json
//!   transformer/config.json        MiniMaxH3Transformer3DModel  (video+audio DiT)
//!   transformer_ref/config.json    same class, Ref2VA weights
//!   vae/config.json                AutoencoderKLMiniMaxH3       (video)
//!   audio_vae/config.json          AutoencoderKLMiniMaxH3Audio
//!   scheduler/scheduler_config.json        shift = 12.0 (video)
//!   audio_scheduler/scheduler_config.json  shift =  3.0 (audio)
//!   text_encoder/config.json       Qwen3VLForConditionalGeneration
//! ```

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Number of modality tags MiniMax-H3 keeps AdaLN parameters for:
/// `0 = video`, `1 = text`, `2 = audio`.
pub const MODALITY_NUM: usize = 3;

/// Modality tag of a row in the packed sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Modality {
    Video = 0,
    Text = 1,
    Audio = 2,
}

impl Modality {
    #[must_use]
    pub fn tag(self) -> u32 {
        self as u32
    }
}

/// `transformer/config.json` — [`crate::transformer`].
#[derive(Debug, Clone, Deserialize)]
pub struct H3TransformerConfig {
    #[serde(default = "d_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_attention_head_dim")]
    pub attention_head_dim: usize,
    #[serde(default = "d_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "d_num_layers")]
    pub num_layers: usize,
    #[serde(default = "d_num_refiner_layers")]
    pub num_refiner_layers: usize,
    #[serde(default = "d_ffn_dim")]
    pub ffn_dim: usize,
    #[serde(default = "d_in_channels")]
    pub in_channels: usize,
    #[serde(default = "d_audio_in_channels")]
    pub audio_in_channels: usize,
    #[serde(default = "d_patch_size")]
    pub patch_size: [usize; 3],
    #[serde(default = "d_text_dim")]
    pub text_dim: usize,
    #[serde(default = "d_freq_dim")]
    pub freq_dim: usize,
    #[serde(default = "d_time_embed_hidden_dim")]
    pub time_embed_hidden_dim: usize,
    #[serde(default = "d_time_embed_dim")]
    pub time_embed_dim: usize,
    #[serde(default = "d_rope_freq_dim")]
    pub rope_freq_dim: usize,
    #[serde(default = "d_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "d_eps")]
    pub norm_eps: f32,
    #[serde(default = "d_eps")]
    pub qk_norm_eps: f32,
    #[serde(default = "d_eps")]
    pub final_norm_eps: f32,
}

fn d_num_attention_heads() -> usize {
    56
}
fn d_attention_head_dim() -> usize {
    128
}
fn d_hidden_size() -> usize {
    5376
}
fn d_num_layers() -> usize {
    50
}
fn d_num_refiner_layers() -> usize {
    2
}
fn d_ffn_dim() -> usize {
    14336
}
fn d_in_channels() -> usize {
    24
}
fn d_audio_in_channels() -> usize {
    32
}
fn d_patch_size() -> [usize; 3] {
    [1, 2, 2]
}
fn d_text_dim() -> usize {
    5120
}
fn d_freq_dim() -> usize {
    256
}
fn d_time_embed_hidden_dim() -> usize {
    5376
}
fn d_time_embed_dim() -> usize {
    2688
}
fn d_rope_freq_dim() -> usize {
    16
}
fn d_rope_theta() -> f32 {
    10_000.0
}
fn d_eps() -> f32 {
    1e-5
}

impl Default for H3TransformerConfig {
    fn default() -> Self {
        Self {
            num_attention_heads: d_num_attention_heads(),
            attention_head_dim: d_attention_head_dim(),
            hidden_size: d_hidden_size(),
            num_layers: d_num_layers(),
            num_refiner_layers: d_num_refiner_layers(),
            ffn_dim: d_ffn_dim(),
            in_channels: d_in_channels(),
            audio_in_channels: d_audio_in_channels(),
            patch_size: d_patch_size(),
            text_dim: d_text_dim(),
            freq_dim: d_freq_dim(),
            time_embed_hidden_dim: d_time_embed_hidden_dim(),
            time_embed_dim: d_time_embed_dim(),
            rope_freq_dim: d_rope_freq_dim(),
            rope_theta: d_rope_theta(),
            norm_eps: d_eps(),
            qk_norm_eps: d_eps(),
            final_norm_eps: d_eps(),
        }
    }
}

impl H3TransformerConfig {
    /// Parse `<dir>/config.json`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        read_json(&dir.join("config.json"))
    }

    /// `num_attention_heads * attention_head_dim`. Note this is **larger** than
    /// [`Self::hidden_size`] in MiniMax-H3 (7168 vs 5376).
    #[must_use]
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Channels of one patchified video row: `in_channels * prod(patch_size)`.
    #[must_use]
    pub fn video_patch_dim(&self) -> usize {
        self.in_channels * self.patch_size.iter().product::<usize>()
    }

    /// Rotated channels per head: `2 * 3 * rope_freq_dim` (96 of 128).
    #[must_use]
    pub fn rope_rotary_dim(&self) -> usize {
        2 * 3 * self.rope_freq_dim
    }

    /// Rows of one block's AdaLN modulation table per timestep.
    #[must_use]
    pub fn adaln_rows_per_timestep(&self) -> usize {
        MODALITY_NUM
    }

    /// Output width of a block's `adaln_proj.linear`.
    #[must_use]
    pub fn adaln_proj_out(&self) -> usize {
        6 * self.hidden_size * MODALITY_NUM
    }

    pub fn validate(&self) -> Result<()> {
        if self.rope_rotary_dim() > self.attention_head_dim {
            bail!(
                "rope rotary dim {} exceeds attention_head_dim {}",
                self.rope_rotary_dim(),
                self.attention_head_dim
            );
        }
        if self.patch_size.contains(&0) {
            bail!(
                "patch_size entries must be non-zero, got {:?}",
                self.patch_size
            );
        }
        // `num_layers == 0` is allowed: it reduces the graph to the input
        // projections plus the output heads, which is how the stack is bisected
        // when a change makes the block body misbehave.
        if self.hidden_size == 0 {
            bail!("hidden_size must be non-zero");
        }
        if self.attention_head_dim == 0 || self.num_attention_heads == 0 {
            bail!("attention_head_dim and num_attention_heads must be non-zero");
        }
        Ok(())
    }
}

/// `vae/config.json` — [`crate::vae_video`].
#[derive(Debug, Clone, Deserialize)]
pub struct H3VideoVaeConfig {
    #[serde(default = "d_three")]
    pub in_channels: usize,
    #[serde(default = "d_three")]
    pub out_channels: usize,
    #[serde(default = "d_in_channels")]
    pub latent_channels: usize,
    #[serde(default = "d_vae_block_out_channels")]
    pub block_out_channels: Vec<usize>,
    #[serde(default = "d_two")]
    pub layers_per_block: usize,
    #[serde(default = "d_vae_spatial_ds")]
    pub spatial_downsample_factors: Vec<usize>,
    #[serde(default = "d_vae_temporal_ds")]
    pub temporal_downsample_factors: Vec<usize>,
    #[serde(default = "d_thirty_two")]
    pub norm_num_groups: usize,
    #[serde(default = "d_eps_1e6")]
    pub norm_eps: f32,
    #[serde(default = "d_reflect")]
    pub spatial_padding_mode: String,
    #[serde(default = "d_dec_layers")]
    pub decoder_num_layers: usize,
    #[serde(default = "d_thirty_two")]
    pub decoder_num_attention_heads: usize,
    #[serde(default = "d_sixty_four")]
    pub decoder_attention_head_dim: usize,
    #[serde(default = "d_four")]
    pub decoder_num_register_tokens: usize,
    #[serde(default = "d_four")]
    pub decoder_ffn_mult: usize,
    #[serde(default = "d_dec_rope_theta")]
    pub decoder_rope_theta: f32,
    #[serde(default = "d_dec_rope_ratio")]
    pub decoder_rope_dim_ratio: f32,
    #[serde(default = "d_eps")]
    pub decoder_norm_eps: f32,
    #[serde(default = "d_clip_length")]
    pub clip_length: usize,
    #[serde(default = "d_token_drop")]
    pub token_drop: usize,
    #[serde(default)]
    pub latents_mean: Vec<f32>,
    #[serde(default)]
    pub latents_std: Vec<f32>,
}

fn d_two() -> usize {
    2
}
fn d_three() -> usize {
    3
}
fn d_four() -> usize {
    4
}
fn d_thirty_two() -> usize {
    32
}
fn d_sixty_four() -> usize {
    64
}
fn d_dec_layers() -> usize {
    36
}
fn d_clip_length() -> usize {
    17
}
fn d_token_drop() -> usize {
    3
}
fn d_eps_1e6() -> f32 {
    1e-6
}
fn d_dec_rope_theta() -> f32 {
    100.0
}
fn d_dec_rope_ratio() -> f32 {
    0.75
}
fn d_reflect() -> String {
    "reflect".to_string()
}
fn d_vae_block_out_channels() -> Vec<usize> {
    vec![128, 256, 256, 512, 512, 1024]
}
fn d_vae_spatial_ds() -> Vec<usize> {
    vec![2, 2, 2, 2, 1, 1]
}
fn d_vae_temporal_ds() -> Vec<usize> {
    vec![1, 2, 2, 1, 1, 1]
}

impl H3VideoVaeConfig {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        read_json(&dir.join("config.json"))
    }

    /// Total spatial downsample factor of the CNN encoder.
    #[must_use]
    pub fn spatial_compression(&self) -> usize {
        self.spatial_downsample_factors.iter().product()
    }

    /// Total temporal downsample factor of the CNN encoder.
    #[must_use]
    pub fn temporal_compression(&self) -> usize {
        self.temporal_downsample_factors.iter().product()
    }

    /// Width of the decoder transformer stack.
    #[must_use]
    pub fn decoder_hidden_size(&self) -> usize {
        self.decoder_num_attention_heads * self.decoder_attention_head_dim
    }

    pub fn validate(&self) -> Result<()> {
        let n = self.block_out_channels.len();
        if self.spatial_downsample_factors.len() != n || self.temporal_downsample_factors.len() != n
        {
            bail!(
                "vae: block_out_channels ({n}), spatial ({}) and temporal ({}) downsample factors must have equal length",
                self.spatial_downsample_factors.len(),
                self.temporal_downsample_factors.len()
            );
        }
        if !self.latents_mean.is_empty() && self.latents_mean.len() != self.latent_channels {
            bail!(
                "vae: latents_mean has {} entries, expected latent_channels = {}",
                self.latents_mean.len(),
                self.latent_channels
            );
        }
        if !self.latents_std.is_empty() && self.latents_std.len() != self.latent_channels {
            bail!(
                "vae: latents_std has {} entries, expected latent_channels = {}",
                self.latents_std.len(),
                self.latent_channels
            );
        }
        Ok(())
    }
}

/// `audio_vae/config.json` — [`crate::vae_audio`].
#[derive(Debug, Clone, Deserialize)]
pub struct H3AudioVaeConfig {
    #[serde(default = "d_sixty_four")]
    pub encoder_dim: usize,
    #[serde(default = "d_enc_rates")]
    pub encoder_rates: Vec<usize>,
    #[serde(default = "d_latent_dim")]
    pub latent_dim: usize,
    #[serde(default = "d_audio_in_channels")]
    pub latent_channels: usize,
    #[serde(default = "d_decoder_dim")]
    pub decoder_dim: usize,
    #[serde(default = "d_dec_rates")]
    pub decoder_rates: Vec<usize>,
    #[serde(default = "d_dec_kernels")]
    pub decoder_kernel_sizes: Vec<usize>,
    #[serde(default = "d_eight")]
    pub num_attention_heads: usize,
    #[serde(default = "d_resblock_kernels")]
    pub resblock_kernel_sizes: Vec<usize>,
    #[serde(default = "d_resblock_dilations")]
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    #[serde(default = "d_sampling_rate")]
    pub sampling_rate: usize,
    #[serde(default)]
    pub latents_mean: Vec<f32>,
    #[serde(default)]
    pub latents_std: Vec<f32>,
}

fn d_eight() -> usize {
    8
}
fn d_latent_dim() -> usize {
    2048
}
fn d_decoder_dim() -> usize {
    1024
}
fn d_sampling_rate() -> usize {
    32_000
}
fn d_enc_rates() -> Vec<usize> {
    vec![2, 4, 4, 5, 5]
}
fn d_dec_rates() -> Vec<usize> {
    vec![5, 5, 2, 2, 2, 2, 2]
}
fn d_dec_kernels() -> Vec<usize> {
    vec![9, 9, 4, 4, 4, 4, 4]
}
fn d_resblock_kernels() -> Vec<usize> {
    vec![3, 7, 11]
}
fn d_resblock_dilations() -> Vec<Vec<usize>> {
    vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]
}

impl H3AudioVaeConfig {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        read_json(&dir.join("config.json"))
    }

    /// Encoder hop: how many waveform samples one latent frame covers.
    #[must_use]
    pub fn encoder_hop(&self) -> usize {
        self.encoder_rates.iter().product()
    }

    /// Decoder upsample: how many waveform samples one latent frame expands to.
    #[must_use]
    pub fn decoder_hop(&self) -> usize {
        self.decoder_rates.iter().product()
    }

    pub fn validate(&self) -> Result<()> {
        if self.decoder_rates.len() != self.decoder_kernel_sizes.len() {
            bail!(
                "audio vae: decoder_rates ({}) and decoder_kernel_sizes ({}) must have equal length",
                self.decoder_rates.len(),
                self.decoder_kernel_sizes.len()
            );
        }
        if self.resblock_kernel_sizes.len() != self.resblock_dilation_sizes.len() {
            bail!("audio vae: resblock kernel/dilation lists must have equal length");
        }
        if !self.latents_mean.is_empty() && self.latents_mean.len() != self.latent_channels {
            bail!(
                "audio vae: latents_mean has {} entries, expected {}",
                self.latents_mean.len(),
                self.latent_channels
            );
        }
        Ok(())
    }
}

/// `scheduler/scheduler_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct H3SchedulerConfig {
    #[serde(default = "d_video_shift")]
    pub shift: f32,
}

fn d_video_shift() -> f32 {
    12.0
}

impl H3SchedulerConfig {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        read_json(&dir.join("scheduler_config.json"))
    }
}

/// The Qwen3-VL text encoder's `text_config` block, plus the tap layer.
///
/// MiniMax-H3 reads the **unnormalized** hidden state after decoder layer
/// [`H3TextEncoderConfig::TAP_LAYER`] rather than the final layer.
#[derive(Debug, Clone, Deserialize)]
pub struct H3TextEncoderConfig {
    #[serde(default = "d_text_dim")]
    pub hidden_size: usize,
    #[serde(default = "d_te_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_te_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_te_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "d_attention_head_dim")]
    pub head_dim: usize,
    #[serde(default = "d_te_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "d_eps_1e6")]
    pub rms_norm_eps: f32,
    #[serde(default = "d_te_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "d_te_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_mrope_section")]
    pub mrope_section: [usize; 3],
    #[serde(default = "d_true")]
    pub mrope_interleaved: bool,
}

fn d_te_layers() -> usize {
    64
}
fn d_te_heads() -> usize {
    64
}
fn d_te_kv_heads() -> usize {
    8
}
fn d_te_intermediate() -> usize {
    25_600
}
fn d_te_rope_theta() -> f32 {
    5_000_000.0
}
fn d_te_vocab() -> usize {
    151_936
}
fn d_mrope_section() -> [usize; 3] {
    [24, 20, 20]
}
fn d_true() -> bool {
    true
}

impl H3TextEncoderConfig {
    /// MiniMax-H3 conditions on the hidden state after this decoder layer.
    pub const TAP_LAYER: usize = 50;

    /// The released embedding-table width.
    ///
    /// This is **larger** than the tokenizer's vocabulary (151669): the table is
    /// padded up, so a valid token id is always in range but not every row is
    /// reachable.
    #[must_use]
    pub fn default_vocab_size() -> usize {
        d_te_vocab()
    }

    /// Parse the nested `text_config` (and `rope_scaling`) out of the
    /// `Qwen3VLForConditionalGeneration` config.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read text encoder config {}", path.display()))?;
        let root: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse text encoder config {}", path.display()))?;
        let text = root
            .get("text_config")
            .cloned()
            .unwrap_or_else(|| root.clone());
        let mut cfg: Self = serde_json::from_value(text.clone())
            .with_context(|| format!("parse text_config in {}", path.display()))?;
        if let Some(scaling) = text.get("rope_scaling") {
            if let Some(sec) = scaling.get("mrope_section").and_then(|v| v.as_array()) {
                for (i, v) in sec.iter().take(3).enumerate() {
                    if let Some(n) = v.as_u64() {
                        cfg.mrope_section[i] = n as usize;
                    }
                }
            }
            if let Some(b) = scaling.get("mrope_interleaved").and_then(|v| v.as_bool()) {
                cfg.mrope_interleaved = b;
            }
        }
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        if Self::TAP_LAYER > self.num_hidden_layers {
            bail!(
                "text encoder has {} layers, cannot tap layer {}",
                self.num_hidden_layers,
                Self::TAP_LAYER
            );
        }
        if self.mrope_section.iter().sum::<usize>() * 2 != self.head_dim {
            bail!(
                "mrope_section {:?} sums to {}, expected head_dim/2 = {}",
                self.mrope_section,
                self.mrope_section.iter().sum::<usize>(),
                self.head_dim / 2
            );
        }
        Ok(())
    }
}

/// Every component config of one MiniMax-H3 checkpoint directory.
#[derive(Debug, Clone)]
pub struct H3Config {
    pub root: PathBuf,
    pub transformer: H3TransformerConfig,
    /// Present when the checkpoint ships the Ref2VA weights.
    pub transformer_ref: Option<H3TransformerConfig>,
    pub vae: H3VideoVaeConfig,
    pub audio_vae: H3AudioVaeConfig,
    pub scheduler: H3SchedulerConfig,
    pub audio_scheduler: H3SchedulerConfig,
    pub text_encoder: H3TextEncoderConfig,
}

impl H3Config {
    /// Load every component config from a checkpoint root.
    pub fn from_root(root: &Path) -> Result<Self> {
        let transformer = H3TransformerConfig::from_dir(&root.join("transformer"))
            .context("MiniMax-H3: transformer/config.json")?;
        let transformer_ref = {
            let dir = root.join("transformer_ref");
            if dir.join("config.json").is_file() {
                Some(
                    H3TransformerConfig::from_dir(&dir)
                        .context("MiniMax-H3: transformer_ref/config.json")?,
                )
            } else {
                None
            }
        };
        let vae =
            H3VideoVaeConfig::from_dir(&root.join("vae")).context("MiniMax-H3: vae/config.json")?;
        let audio_vae = H3AudioVaeConfig::from_dir(&root.join("audio_vae"))
            .context("MiniMax-H3: audio_vae/config.json")?;
        let scheduler = H3SchedulerConfig::from_dir(&root.join("scheduler"))
            .context("MiniMax-H3: scheduler/scheduler_config.json")?;
        let audio_scheduler = H3SchedulerConfig::from_dir(&root.join("audio_scheduler"))
            .context("MiniMax-H3: audio_scheduler/scheduler_config.json")?;
        let text_encoder = H3TextEncoderConfig::from_dir(&root.join("text_encoder"))
            .context("MiniMax-H3: text_encoder/config.json")?;

        let cfg = Self {
            root: root.to_path_buf(),
            transformer,
            transformer_ref,
            vae,
            audio_vae,
            scheduler,
            audio_scheduler,
            text_encoder,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        self.transformer.validate()?;
        if let Some(t) = &self.transformer_ref {
            t.validate()?;
        }
        self.vae.validate()?;
        self.audio_vae.validate()?;
        self.text_encoder.validate()?;
        if self.transformer.text_dim != self.text_encoder.hidden_size {
            bail!(
                "transformer.text_dim {} does not match text encoder hidden_size {}",
                self.transformer.text_dim,
                self.text_encoder.hidden_size
            );
        }
        if self.transformer.in_channels != self.vae.latent_channels {
            bail!(
                "transformer.in_channels {} does not match vae.latent_channels {}",
                self.transformer.in_channels,
                self.vae.latent_channels
            );
        }
        if self.transformer.audio_in_channels != self.audio_vae.latent_channels {
            bail!(
                "transformer.audio_in_channels {} does not match audio_vae.latent_channels {}",
                self.transformer.audio_in_channels,
                self.audio_vae.latent_channels
            );
        }
        Ok(())
    }

    #[must_use]
    pub fn subdir(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("read config {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parse config {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformer_defaults_match_released_checkpoint() {
        let c = H3TransformerConfig::default();
        assert_eq!(c.inner_dim(), 7168);
        assert_eq!(c.video_patch_dim(), 24 * 2 * 2);
        assert_eq!(c.rope_rotary_dim(), 96);
        assert!(c.rope_rotary_dim() < c.attention_head_dim);
        assert_eq!(c.adaln_proj_out(), 6 * 5376 * 3);
        c.validate().unwrap();
    }

    #[test]
    fn transformer_config_parses() {
        let json = r#"{
          "num_attention_heads": 56, "attention_head_dim": 128, "hidden_size": 5376,
          "num_layers": 50, "num_refiner_layers": 2, "ffn_dim": 14336,
          "in_channels": 24, "audio_in_channels": 32, "patch_size": [1, 2, 2],
          "text_dim": 5120, "freq_dim": 256, "time_embed_hidden_dim": 5376,
          "time_embed_dim": 2688, "rope_freq_dim": 16, "rope_theta": 10000.0,
          "norm_eps": 1e-05, "qk_norm_eps": 1e-05, "final_norm_eps": 1e-05
        }"#;
        let c: H3TransformerConfig = serde_json::from_str(json).unwrap();
        assert_eq!(c.num_layers, 50);
        assert_eq!(c.patch_size, [1, 2, 2]);
        c.validate().unwrap();
    }

    #[test]
    fn video_vae_compression_factors() {
        let json = r#"{
          "in_channels": 3, "out_channels": 3, "latent_channels": 24,
          "block_out_channels": [128, 256, 256, 512, 512, 1024],
          "layers_per_block": 2,
          "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
          "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
          "norm_num_groups": 32, "norm_eps": 1e-06,
          "spatial_padding_mode": "reflect",
          "decoder_num_layers": 36, "decoder_num_attention_heads": 32,
          "decoder_attention_head_dim": 64, "decoder_num_register_tokens": 4,
          "decoder_ffn_mult": 4, "decoder_rope_theta": 100.0,
          "decoder_rope_dim_ratio": 0.75, "decoder_norm_eps": 1e-05,
          "clip_length": 17, "token_drop": 3
        }"#;
        let c: H3VideoVaeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(c.spatial_compression(), 16);
        assert_eq!(c.temporal_compression(), 4);
        assert_eq!(c.decoder_hidden_size(), 2048);
        c.validate().unwrap();
    }

    #[test]
    fn audio_vae_hop_sizes() {
        let c = H3AudioVaeConfig {
            encoder_dim: 64,
            encoder_rates: d_enc_rates(),
            latent_dim: 2048,
            latent_channels: 32,
            decoder_dim: 1024,
            decoder_rates: d_dec_rates(),
            decoder_kernel_sizes: d_dec_kernels(),
            num_attention_heads: 8,
            resblock_kernel_sizes: d_resblock_kernels(),
            resblock_dilation_sizes: d_resblock_dilations(),
            sampling_rate: 32_000,
            latents_mean: vec![],
            latents_std: vec![],
        };
        // 2*4*4*5*5 = 800 and 5*5*2*2*2*2*2 = 800 — encoder and decoder agree.
        assert_eq!(c.encoder_hop(), 800);
        assert_eq!(c.decoder_hop(), 800);
        assert_eq!(c.sampling_rate / c.encoder_hop(), 40); // 40 latent frames / s
        c.validate().unwrap();
    }

    #[test]
    fn text_encoder_tap_layer_is_within_stack() {
        let c = H3TextEncoderConfig {
            hidden_size: 5120,
            num_hidden_layers: 64,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 25_600,
            rms_norm_eps: 1e-6,
            rope_theta: 5e6,
            vocab_size: 151_936,
            mrope_section: [24, 20, 20],
            mrope_interleaved: true,
        };
        assert_eq!(H3TextEncoderConfig::TAP_LAYER, 50);
        assert!(H3TextEncoderConfig::TAP_LAYER < c.num_hidden_layers);
        c.validate().unwrap();
    }

    #[test]
    fn modality_tags_match_reference() {
        assert_eq!(Modality::Video.tag(), 0);
        assert_eq!(Modality::Text.tag(), 1);
        assert_eq!(Modality::Audio.tag(), 2);
        assert_eq!(MODALITY_NUM, 3);
    }
}
