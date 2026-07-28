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

//! TRELLIS.2-4B configuration.
//!
//! Mirrors the JSON checkpoints published under `microsoft/TRELLIS.2-4B`:
//!   - `pipeline.json` — the [`PipelineConfig`]: which sub-models to load,
//!     sampler hyper-parameters, latent normalization stats, and the image
//!     conditioner.
//!   - `ckpts/*.json` — per-model architecture configs. The generative DiTs
//!     (`SparseStructureFlowModel`, `SLatFlowModel`) share [`DitConfig`]; the
//!     sparse VAEs use [`SparseVaeConfig`]; the (reused) dense sparse-structure
//!     decoder uses [`SparseStructureVaeConfig`].
//!
//! All fields keep the upstream names so a checkpoint's JSON deserializes
//! verbatim.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

/// Generic `{ "name": ..., "args": {...} }` checkpoint wrapper.
#[derive(Debug, Clone, Deserialize)]
struct NamedArgs<T> {
    name: String,
    args: T,
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading config {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
}

// ---------------------------------------------------------------------------
// Generative DiTs: SparseStructureFlowModel + SLatFlowModel
// ---------------------------------------------------------------------------

/// Which flow-matching DiT variant a checkpoint is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitKind {
    /// Dense DiT over a fixed `resolution³` occupancy-latent grid.
    SparseStructureFlow,
    /// Sparse DiT over active voxels (variable token count).
    SLatFlow,
}

/// Architecture config shared by both generative DiTs.
///
/// Deserialized from the `args` object of `ckpts/{ss_flow,slat_flow}_*.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DitArgs {
    /// Spatial side length. `16` for the dense structure DiT; the SLat DiTs
    /// carry `32` (coords are quantized into a `resolution³` grid for RoPE).
    pub resolution: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    /// Conditioner (DINOv3) feature width.
    pub cond_channels: usize,
    pub num_blocks: usize,
    #[serde(default)]
    pub num_heads: Option<usize>,
    #[serde(default = "default_head_channels")]
    pub num_head_channels: usize,
    pub mlp_ratio: f32,
    /// `"ape"` (absolute) or `"rope"` (rotary). All 4B checkpoints use `rope`.
    #[serde(default = "default_pe_mode")]
    pub pe_mode: String,
    #[serde(default = "default_rope_freq")]
    pub rope_freq: (f32, f32),
    #[serde(default)]
    pub share_mod: bool,
    #[serde(default)]
    pub initialization: Option<String>,
    #[serde(default)]
    pub qk_rms_norm: bool,
    #[serde(default)]
    pub qk_rms_norm_cross: bool,
    #[serde(default = "default_dtype")]
    pub dtype: String,
}

fn default_head_channels() -> usize {
    64
}
fn default_pe_mode() -> String {
    "ape".into()
}
fn default_rope_freq() -> (f32, f32) {
    (1.0, 10000.0)
}
fn default_dtype() -> String {
    "float32".into()
}

/// A generative DiT config bound to its concrete [`DitKind`].
#[derive(Debug, Clone)]
pub struct DitConfig {
    pub kind: DitKind,
    pub args: DitArgs,
}

impl DitConfig {
    /// Load and tag a DiT config JSON (`ss_flow_*` or `slat_flow_*`).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let wrapped: NamedArgs<DitArgs> = read_json(path)?;
        let kind = match wrapped.name.as_str() {
            "SparseStructureFlowModel" => DitKind::SparseStructureFlow,
            "SLatFlowModel" => DitKind::SLatFlow,
            other => bail!("unexpected DiT model name {other:?} in {}", path.display()),
        };
        Ok(Self {
            kind,
            args: wrapped.args,
        })
    }

    /// Number of attention heads (`num_heads` or `model_channels / num_head_channels`).
    pub fn num_heads(&self) -> usize {
        self.args
            .num_heads
            .unwrap_or(self.args.model_channels / self.args.num_head_channels)
    }

    /// Per-head channel width.
    pub fn head_dim(&self) -> usize {
        self.args.model_channels / self.num_heads()
    }

    /// Hidden width of the FFN (`int(model_channels * mlp_ratio)`, matching
    /// PyTorch truncation).
    pub fn mlp_hidden(&self) -> usize {
        (self.args.model_channels as f32 * self.args.mlp_ratio) as usize
    }

    pub fn uses_rope(&self) -> bool {
        self.args.pe_mode == "rope"
    }
}

// ---------------------------------------------------------------------------
// Sparse VAEs (shape / texture decoders + encoders)
// ---------------------------------------------------------------------------

/// Config for the sparse-3D-conv VAEs:
/// `FlexiDualGridVae{Encoder,Decoder}` (shape) and
/// `SparseUnetVae{Encoder,Decoder}` (texture).
#[derive(Debug, Clone, Deserialize)]
pub struct SparseVaeArgs {
    /// Present on decoders only (shape decoder: implied 7; texture: 6).
    #[serde(default)]
    pub out_channels: Option<usize>,
    /// Present on encoders only (shape enc: implied 6; texture: 6).
    #[serde(default)]
    pub in_channels: Option<usize>,
    /// Per-stage widths (coarse→fine for decoders, fine→coarse for encoders).
    pub model_channels: Vec<usize>,
    pub latent_channels: usize,
    /// Number of `block_type` blocks per stage.
    pub num_blocks: Vec<usize>,
    pub block_type: Vec<String>,
    /// Upsample (decoder) block per stage boundary.
    #[serde(default)]
    pub up_block_type: Vec<String>,
    /// Downsample (encoder) block per stage boundary.
    #[serde(default)]
    pub down_block_type: Vec<String>,
    #[serde(default)]
    pub block_args: Vec<serde_json::Value>,
    /// Texture decoder only: whether to predict subdivision.
    #[serde(default)]
    pub pred_subdiv: bool,
    #[serde(default)]
    pub use_fp16: bool,
}

/// Which sparse VAE a checkpoint is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseVaeKind {
    FlexiDualGridEncoder,
    FlexiDualGridDecoder,
    SparseUnetEncoder,
    SparseUnetDecoder,
}

/// A sparse VAE config bound to its concrete [`SparseVaeKind`].
#[derive(Debug, Clone)]
pub struct SparseVaeConfig {
    pub kind: SparseVaeKind,
    pub args: SparseVaeArgs,
}

impl SparseVaeConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let wrapped: NamedArgs<SparseVaeArgs> = read_json(path)?;
        Self::from_named(wrapped, &path.display().to_string())
    }

    /// Parse from a JSON string (`{ "name": ..., "args": {...} }`).
    pub fn from_json_str(json: &str) -> Result<Self> {
        let wrapped: NamedArgs<SparseVaeArgs> = serde_json::from_str(json)?;
        Self::from_named(wrapped, "<json>")
    }

    fn from_named(wrapped: NamedArgs<SparseVaeArgs>, src: &str) -> Result<Self> {
        let kind = match wrapped.name.as_str() {
            "FlexiDualGridVaeEncoder" => SparseVaeKind::FlexiDualGridEncoder,
            "FlexiDualGridVaeDecoder" => SparseVaeKind::FlexiDualGridDecoder,
            "SparseUnetVaeEncoder" => SparseVaeKind::SparseUnetEncoder,
            "SparseUnetVaeDecoder" => SparseVaeKind::SparseUnetDecoder,
            other => bail!("unexpected sparse VAE name {other:?} in {src}"),
        };
        Ok(Self {
            kind,
            args: wrapped.args,
        })
    }

    /// FlexiDualGrid decoder emits 7 channels (vertex offset 3, intersected 3,
    /// and quad-lerp 1); the texture decoder emits `out_channels` (6). Encoders
    /// have no meaningful output-channel here.
    pub fn decoder_out_channels(&self) -> usize {
        match self.kind {
            SparseVaeKind::FlexiDualGridDecoder => 7,
            SparseVaeKind::SparseUnetDecoder => self.args.out_channels.unwrap_or(6),
            _ => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Dense sparse-structure VAE decoder (reused from TRELLIS-image-large)
// ---------------------------------------------------------------------------

/// Config for the dense conv3d `SparseStructureDecoder`
/// (`ss_dec_conv3d_16l8_fp16`, borrowed from `microsoft/TRELLIS-image-large`).
#[derive(Debug, Clone, Deserialize)]
pub struct SparseStructureVaeArgs {
    pub out_channels: usize,
    pub latent_channels: usize,
    pub num_res_blocks: usize,
    pub channels: Vec<usize>,
    #[serde(default = "default_res_blocks_middle")]
    pub num_res_blocks_middle: usize,
    #[serde(default = "default_norm_type")]
    pub norm_type: String,
    #[serde(default)]
    pub use_fp16: bool,
}

fn default_res_blocks_middle() -> usize {
    2
}
fn default_norm_type() -> String {
    "layer".into()
}

impl SparseStructureVaeArgs {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let wrapped: NamedArgs<SparseStructureVaeArgs> = read_json(path.as_ref())?;
        Ok(wrapped.args)
    }
}

// ---------------------------------------------------------------------------
// Pipeline config (pipeline.json)
// ---------------------------------------------------------------------------

/// Flow-Euler + CFG + guidance-interval sampler hyper-parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct SamplerParams {
    pub steps: usize,
    pub guidance_strength: f32,
    #[serde(default)]
    pub guidance_rescale: f32,
    #[serde(default = "default_guidance_interval")]
    pub guidance_interval: [f32; 2],
    #[serde(default = "default_rescale_t")]
    pub rescale_t: f32,
}

fn default_guidance_interval() -> [f32; 2] {
    [0.0, 1.0]
}
fn default_rescale_t() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
struct SamplerArgs {
    #[serde(default = "default_sigma_min")]
    sigma_min: f32,
}
fn default_sigma_min() -> f32 {
    1e-5
}

/// A full sampler spec (`name` + `args.sigma_min` + `params`).
#[derive(Debug, Clone, Deserialize)]
pub struct SamplerConfig {
    pub name: String,
    #[serde(default)]
    args: SamplerArgs,
    pub params: SamplerParams,
}

impl Default for SamplerArgs {
    fn default() -> Self {
        Self {
            sigma_min: default_sigma_min(),
        }
    }
}

impl SamplerConfig {
    pub fn sigma_min(&self) -> f32 {
        self.args.sigma_min
    }
}

/// Per-channel latent normalization (`(x - mean) / std` encode, inverse decode).
#[derive(Debug, Clone, Deserialize)]
pub struct Normalization {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

/// The image conditioner spec (`DinoV3FeatureExtractor` + HF model name).
#[derive(Debug, Clone, Deserialize)]
pub struct ImageCondModel {
    pub name: String,
    pub args: ImageCondArgs,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageCondArgs {
    pub model_name: String,
}

/// Relative checkpoint paths for every sub-model, from `pipeline.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct PipelineModels {
    pub sparse_structure_decoder: String,
    pub sparse_structure_flow_model: String,
    pub shape_slat_decoder: String,
    pub shape_slat_flow_model_512: String,
    pub shape_slat_flow_model_1024: String,
    pub tex_slat_decoder: String,
    pub tex_slat_flow_model_512: String,
    pub tex_slat_flow_model_1024: String,
}

/// The top-level `pipeline.json` args block.
#[derive(Debug, Clone, Deserialize)]
pub struct PipelineArgs {
    pub models: PipelineModels,
    pub sparse_structure_sampler: SamplerConfig,
    pub shape_slat_sampler: SamplerConfig,
    pub tex_slat_sampler: SamplerConfig,
    pub shape_slat_normalization: Normalization,
    pub tex_slat_normalization: Normalization,
    pub image_cond_model: ImageCondModel,
    #[serde(default)]
    pub rembg_model: Option<serde_json::Value>,
    #[serde(default = "default_pipeline_type")]
    pub default_pipeline_type: String,
}

fn default_pipeline_type() -> String {
    "1024_cascade".into()
}

/// The parsed `pipeline.json`.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub args: PipelineArgs,
}

impl PipelineConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let wrapped: NamedArgs<PipelineArgs> = read_json(path)?;
        if wrapped.name != "Trellis2ImageTo3DPipeline" {
            bail!(
                "unexpected pipeline name {:?} in {}",
                wrapped.name,
                path.display()
            );
        }
        Ok(Self { args: wrapped.args })
    }
}

/// The supported end-to-end pipeline resolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineType {
    /// Single-pass 512³.
    Res512,
    /// Single-pass 1024³.
    Res1024,
    /// 512→1024 shape cascade (default).
    Cascade1024,
    /// 512→1536 shape cascade.
    Cascade1536,
}

impl PipelineType {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "512" => Self::Res512,
            "1024" => Self::Res1024,
            "1024_cascade" => Self::Cascade1024,
            "1536_cascade" => Self::Cascade1536,
            other => bail!("invalid pipeline type {other:?}"),
        })
    }

    /// Sparse-structure decode resolution used to seed active voxels.
    pub fn sparse_structure_res(&self) -> usize {
        match self {
            Self::Res512 => 32,
            Self::Res1024 => 64,
            Self::Cascade1024 | Self::Cascade1536 => 32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_dit_args() {
        let json = r#"{"name":"SparseStructureFlowModel","args":{
            "resolution":16,"in_channels":8,"out_channels":8,"model_channels":1536,
            "cond_channels":1024,"num_blocks":30,"num_heads":12,"mlp_ratio":5.3334,
            "pe_mode":"rope","share_mod":true,"initialization":"scaled",
            "qk_rms_norm":true,"qk_rms_norm_cross":true,"dtype":"bfloat16"}}"#;
        let wrapped: NamedArgs<DitArgs> = serde_json::from_str(json).unwrap();
        let cfg = DitConfig {
            kind: DitKind::SparseStructureFlow,
            args: wrapped.args,
        };
        assert_eq!(cfg.num_heads(), 12);
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.mlp_hidden(), (1536.0 * 5.3334) as usize);
        assert!(cfg.uses_rope());
    }
}
