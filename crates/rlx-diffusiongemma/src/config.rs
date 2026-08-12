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

//! `config.json` and `generation_config.json` for
//! `model_type = "diffusion_gemma"`.
//!
//! (The checkpoint also ships a diffusers `scheduler/scheduler_config.json`;
//! this crate follows the transformers generation path, whose schedule lives in
//! `generation_config.json`, so that file is not read.)
//!
//! The text tower is Gemma 4's MoE stack, so most fields mirror
//! `Gemma4TextConfig`. The parts that actually vary per layer — head_dim,
//! KV-head count, V-aliased-to-K, and the RoPE schedule — are resolved through
//! the `layer_*` helpers rather than being flattened at parse time.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Attention flavour of one layer, from `text_config.layer_types`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Local attention, window `sliding_window`, base `head_dim`/`num_key_value_heads`.
    Sliding,
    /// Global attention: `global_head_dim`, `num_global_key_value_heads`, V aliased to K.
    Full,
}

/// `rope_parameters.<layer_type>.rope_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeKind {
    #[default]
    Default,
    /// Rotates the leading `partial_rotary_factor · head_dim / 2` angle slots and
    /// leaves the rest un-rotated, while still emitting a full `head_dim`-wide
    /// table (HF `_compute_proportional_rope_parameters`).
    Proportional,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    #[serde(default)]
    pub rope_type: RopeKind,
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    #[serde(default)]
    pub factor: Option<f64>,
}

impl RopeParams {
    fn default_sliding() -> Self {
        Self {
            rope_type: RopeKind::Default,
            rope_theta: 10_000.0,
            partial_rotary_factor: None,
            factor: None,
        }
    }

    fn default_full() -> Self {
        Self {
            rope_type: RopeKind::Proportional,
            rope_theta: 1_000_000.0,
            partial_rotary_factor: Some(0.25),
            factor: None,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RopeParamsMap {
    #[serde(default)]
    pub sliding_attention: Option<RopeParams>,
    #[serde(default)]
    pub full_attention: Option<RopeParams>,
}

fn de_layer_types<'de, D>(d: D) -> std::result::Result<Vec<LayerType>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = Option::<Vec<String>>::deserialize(d)?.unwrap_or_default();
    Ok(raw
        .into_iter()
        .map(|s| {
            if s == "full_attention" {
                LayerType::Full
            } else {
                LayerType::Sliding
            }
        })
        .collect())
}

/// `text_config` — the Gemma 4 MoE language stack shared by the encoder and the
/// diffusion decoder.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    /// Shared-expert (`mlp`) FFN width. Also the `self_conditioning` width.
    #[serde(default = "d_inter")]
    pub intermediate_size: usize,
    #[serde(default = "d_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    #[serde(default = "d_head_dim")]
    pub head_dim: usize,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    #[serde(default, deserialize_with = "de_layer_types")]
    pub layer_types: Vec<LayerType>,
    #[serde(default = "d_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub rope_parameters: RopeParamsMap,
    #[serde(default = "d_softcap")]
    pub final_logit_softcapping: f32,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub top_k_experts: usize,
    /// Per-expert FFN width (`704`), distinct from `intermediate_size` (`2112`).
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default = "d_act")]
    pub hidden_activation: String,
    #[serde(default = "d_tie")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub pad_token_id: u32,
    #[serde(default = "d_max_pos")]
    pub max_position_embeddings: usize,
}

fn d_vocab() -> usize {
    262_144
}
fn d_hidden() -> usize {
    2816
}
fn d_inter() -> usize {
    2112
}
fn d_layers() -> usize {
    30
}
fn d_heads() -> usize {
    16
}
fn d_kv_heads() -> usize {
    8
}
fn d_head_dim() -> usize {
    256
}
fn d_sliding_window() -> usize {
    1024
}
fn d_eps() -> f32 {
    1e-6
}
fn d_softcap() -> f32 {
    30.0
}
fn d_act() -> String {
    "gelu_pytorch_tanh".to_string()
}
fn d_tie() -> bool {
    true
}
fn d_max_pos() -> usize {
    262_144
}

impl TextConfig {
    /// `layer_types` with the HF default 5:1 sliding:full pattern filled in when
    /// the checkpoint omits it (and the last layer forced to full attention).
    pub fn resolved_layer_types(&self) -> Vec<LayerType> {
        let mut lt = if self.layer_types.len() == self.num_hidden_layers {
            self.layer_types.clone()
        } else {
            (0..self.num_hidden_layers)
                .map(|i| {
                    if (i + 1).is_multiple_of(6) {
                        LayerType::Full
                    } else {
                        LayerType::Sliding
                    }
                })
                .collect()
        };
        if let Some(last) = lt.last_mut() {
            *last = LayerType::Full;
        }
        lt
    }

    pub fn layer_type(&self, layer: usize) -> LayerType {
        self.resolved_layer_types()[layer]
    }

    pub fn is_full(&self, layer: usize) -> bool {
        self.layer_type(layer) == LayerType::Full
    }

    /// Full-attention layers use `global_head_dim` (512); sliding layers the base
    /// `head_dim` (256).
    pub fn layer_head_dim(&self, layer: usize) -> usize {
        if self.is_full(layer) {
            self.global_head_dim.unwrap_or(self.head_dim)
        } else {
            self.head_dim
        }
    }

    /// Full-attention layers use `num_global_key_value_heads` (2).
    pub fn layer_kv_heads(&self, layer: usize) -> usize {
        if self.is_full(layer) {
            self.num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }

    /// Full-attention layers ship no `v_proj`: V is the (pre-`k_norm`) K
    /// projection. Mirrors HF `use_alternative_attention`.
    pub fn layer_k_eq_v(&self, layer: usize) -> bool {
        self.is_full(layer)
    }

    fn layer_rope(&self, layer: usize) -> RopeParams {
        if self.is_full(layer) {
            self.rope_parameters
                .full_attention
                .clone()
                .unwrap_or_else(RopeParams::default_full)
        } else {
            self.rope_parameters
                .sliding_attention
                .clone()
                .unwrap_or_else(RopeParams::default_sliding)
        }
    }

    /// Inverse frequencies for layer `layer`, always `head_dim/2` entries.
    ///
    /// `Default` fills every slot; `Proportional` fills the leading
    /// `partial_rotary_factor · head_dim / 2` and zeroes the tail, which makes
    /// those dimensions NoPE (cos 1 / sin 0) while keeping the table full width.
    pub fn layer_inv_freq(&self, layer: usize) -> Vec<f64> {
        let hd = self.layer_head_dim(layer);
        let half = hd / 2;
        let p = self.layer_rope(layer);
        match p.rope_type {
            RopeKind::Default => (0..half)
                .map(|i| 1.0 / p.rope_theta.powf((2 * i) as f64 / hd as f64))
                .collect(),
            RopeKind::Proportional => {
                let prop = p.partial_rotary_factor.unwrap_or(1.0);
                // HF: `int(rope_proportion * head_dim // 2)` — the floor divide
                // binds after the multiply.
                let angles = ((prop * hd as f64).floor() / 2.0).floor() as usize;
                let factor = p.factor.unwrap_or(1.0);
                (0..half)
                    .map(|i| {
                        if i < angles {
                            (1.0 / p.rope_theta.powf((2 * i) as f64 / hd as f64)) / factor
                        } else {
                            0.0
                        }
                    })
                    .collect()
            }
        }
    }

    /// Half-width NeoX RoPE tables `[len, head_dim/2]` for positions
    /// `offset .. offset+len`, flattened row-major.
    pub fn rope_tables(&self, layer: usize, offset: usize, len: usize) -> (Vec<f32>, Vec<f32>) {
        let inv = self.layer_inv_freq(layer);
        let half = inv.len();
        let mut cos = vec![0f32; len * half];
        let mut sin = vec![0f32; len * half];
        for t in 0..len {
            let pos = (offset + t) as f64;
            for (j, f) in inv.iter().enumerate() {
                let a = pos * f;
                cos[t * half + j] = a.cos() as f32;
                sin[t * half + j] = a.sin() as f32;
            }
        }
        (cos, sin)
    }

    /// Distinct RoPE table names — one per layer *type*, since every layer of a
    /// type shares `inv_freq`.
    pub fn rope_input_names(&self, layer: usize) -> (&'static str, &'static str) {
        if self.is_full(layer) {
            ("rope_cos_full", "rope_sin_full")
        } else {
            ("rope_cos_sliding", "rope_sin_sliding")
        }
    }

    /// `hidden_size^-0.5`, the router's `scalar_root_size`.
    pub fn router_root_scale(&self) -> f32 {
        (self.hidden_size as f32).powf(-0.5)
    }

    /// `sqrt(hidden_size)`, applied to token embeddings.
    pub fn embed_scale(&self) -> f32 {
        (self.hidden_size as f32).sqrt()
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts > 0 && self.top_k_experts > 0
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.num_attention_heads
                .is_multiple_of(self.num_key_value_heads.max(1)),
            "num_attention_heads {} not divisible by num_key_value_heads {}",
            self.num_attention_heads,
            self.num_key_value_heads
        );
        if let Some(g) = self.num_global_key_value_heads {
            anyhow::ensure!(
                g > 0 && self.num_attention_heads.is_multiple_of(g),
                "num_attention_heads {} not divisible by num_global_key_value_heads {g}",
                self.num_attention_heads
            );
        }
        anyhow::ensure!(
            self.layer_head_dim(0).is_multiple_of(2),
            "head_dim must be even for RoPE"
        );
        anyhow::ensure!(
            self.is_moe(),
            "DiffusionGemma is MoE-only; got num_experts={} top_k={}",
            self.num_experts,
            self.top_k_experts
        );
        anyhow::ensure!(
            self.moe_intermediate_size > 0,
            "moe_intermediate_size must be set"
        );
        Ok(())
    }
}

/// `vision_config` (`model_type = "gemma4_vision"`).
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "dv_hidden")]
    pub hidden_size: usize,
    #[serde(default = "dv_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "dv_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "dv_head_dim")]
    pub head_dim: usize,
    #[serde(default = "dv_inter")]
    pub intermediate_size: usize,
    #[serde(default = "dv_patch")]
    pub patch_size: usize,
    #[serde(default = "dv_pool")]
    pub pooling_kernel_size: usize,
    #[serde(default = "dv_pos")]
    pub position_embedding_size: usize,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub rope_parameters: Option<RopeParams>,
    #[serde(default = "dv_standardize")]
    pub standardize: bool,
    #[serde(default)]
    pub use_clipped_linears: bool,
    #[serde(default = "dv_out_len")]
    pub default_output_length: usize,
}

fn dv_hidden() -> usize {
    1152
}
fn dv_layers() -> usize {
    27
}
fn dv_heads() -> usize {
    16
}
fn dv_head_dim() -> usize {
    72
}
fn dv_inter() -> usize {
    4304
}
fn dv_patch() -> usize {
    16
}
fn dv_pool() -> usize {
    3
}
fn dv_pos() -> usize {
    10240
}
fn dv_standardize() -> bool {
    true
}
fn dv_out_len() -> usize {
    280
}

impl VisionConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map_or(100.0, |p| p.rope_theta)
    }
}

/// Top-level `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionGemmaConfig {
    #[serde(default)]
    pub model_type: String,
    pub text_config: TextConfig,
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
    /// Block length of the diffusion canvas (256).
    #[serde(default = "d_canvas")]
    pub canvas_length: usize,
    #[serde(default = "d_boi")]
    pub boi_token_id: u32,
    #[serde(default = "d_eoi")]
    pub eoi_token_id: u32,
    #[serde(default = "d_img")]
    pub image_token_id: u32,
    #[serde(default = "d_tie")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub vision_soft_tokens_per_image: Option<usize>,
    #[serde(default)]
    pub eos_token_id: Vec<u32>,
}

fn d_canvas() -> usize {
    256
}
fn d_boi() -> u32 {
    255_999
}
fn d_eoi() -> u32 {
    258_882
}
fn d_img() -> u32 {
    258_880
}

impl DiffusionGemmaConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_json(&raw)
    }

    pub fn from_json(raw: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(raw).context("parsing diffusion_gemma config.json")?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        if !self.model_type.is_empty() {
            anyhow::ensure!(
                self.model_type == super::MODEL_TYPE,
                "expected model_type={}, got {}",
                super::MODEL_TYPE,
                self.model_type
            );
        }
        anyhow::ensure!(self.canvas_length > 0, "canvas_length must be > 0");
        self.text_config.validate()
    }
}

/// `generation_config.json` — the block-diffusion sampler schedule.
#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionGenerationConfig {
    #[serde(default = "dg_steps")]
    pub max_denoising_steps: usize,
    #[serde(default = "dg_tmax")]
    pub t_max: f32,
    #[serde(default = "dg_tmin")]
    pub t_min: f32,
    #[serde(default = "dg_max_new")]
    pub max_new_tokens: usize,
    /// `StableAndConfidentStoppingCriteria.stability_threshold`.
    #[serde(default = "dg_stability")]
    pub stability_threshold: usize,
    /// `StableAndConfidentStoppingCriteria.confidence_threshold`.
    #[serde(default = "dg_confidence")]
    pub confidence_threshold: f32,
    #[serde(default)]
    pub sampler_config: SamplerConfig,
    #[serde(default = "dg_eos")]
    pub eos_token_id: Vec<u32>,
    #[serde(default)]
    pub pad_token_id: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SamplerConfig {
    #[serde(default = "dg_entropy_bound")]
    pub entropy_bound: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            entropy_bound: dg_entropy_bound(),
        }
    }
}

fn dg_steps() -> usize {
    48
}
fn dg_tmax() -> f32 {
    0.8
}
fn dg_tmin() -> f32 {
    0.4
}
fn dg_max_new() -> usize {
    256
}
fn dg_stability() -> usize {
    1
}
fn dg_confidence() -> f32 {
    0.005
}
fn dg_entropy_bound() -> f32 {
    0.1
}
fn dg_eos() -> Vec<u32> {
    vec![1, 106, 50]
}

impl Default for DiffusionGenerationConfig {
    fn default() -> Self {
        Self {
            max_denoising_steps: dg_steps(),
            t_max: dg_tmax(),
            t_min: dg_tmin(),
            max_new_tokens: dg_max_new(),
            stability_threshold: dg_stability(),
            confidence_threshold: dg_confidence(),
            sampler_config: SamplerConfig::default(),
            eos_token_id: dg_eos(),
            pad_token_id: 0,
        }
    }
}

impl DiffusionGenerationConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        // HF writes `confidence_threshold` at the top level under that name in
        // some revisions and as `stability_threshold`/`confidence_threshold`
        // pairs in others; serde defaults cover either spelling.
        let mut cfg: Self =
            serde_json::from_str(&raw).context("parsing diffusion_gemma generation_config.json")?;
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw)
            && let Some(t) = v.get("confidence_threshold").and_then(|t| t.as_f64())
        {
            cfg.confidence_threshold = t as f32;
        }
        Ok(cfg)
    }

    /// `t = t_min + (t_max - t_min)·(step / max_denoising_steps)`.
    ///
    /// `step` counts *down* from `max_denoising_steps` to 1, so the schedule
    /// anneals from `t_max` toward `t_min`.
    pub fn temperature(&self, step: usize) -> f32 {
        self.t_min + (self.t_max - self.t_min) * (step as f32 / self.max_denoising_steps as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped `google/diffusiongemma-26B-A4B-it` config, trimmed to the
    /// fields this crate reads.
    pub(crate) const REAL_CONFIG: &str = r#"{
      "architectures": ["DiffusionGemmaForBlockDiffusion"],
      "boi_token_id": 255999, "canvas_length": 256, "eoi_token_id": 258882,
      "eos_token_id": [1, 106], "image_token_id": 258880,
      "model_type": "diffusion_gemma",
      "text_config": {
        "final_logit_softcapping": 30.0, "global_head_dim": 512, "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 2816,
        "intermediate_size": 2112,
        "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
        "max_position_embeddings": 262144, "model_type": "diffusion_gemma_text",
        "moe_intermediate_size": 704, "num_attention_heads": 16, "num_experts": 128,
        "num_global_key_value_heads": 2, "num_hidden_layers": 30,
        "num_key_value_heads": 8, "pad_token_id": 0, "rms_norm_eps": 1e-06,
        "rope_parameters": {
          "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
          "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "sliding_window": 1024, "tie_word_embeddings": true, "top_k_experts": 8,
        "use_bidirectional_attention": "vision", "vocab_size": 262144
      },
      "tie_word_embeddings": true,
      "vision_config": {
        "head_dim": 72, "hidden_size": 1152, "intermediate_size": 4304,
        "model_type": "gemma4_vision", "num_attention_heads": 16,
        "num_hidden_layers": 27, "patch_size": 16, "pooling_kernel_size": 3,
        "position_embedding_size": 10240, "rms_norm_eps": 1e-06,
        "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
        "standardize": true, "use_clipped_linears": false,
        "default_output_length": 280
      },
      "vision_soft_tokens_per_image": 280
    }"#;

    fn cfg() -> DiffusionGemmaConfig {
        DiffusionGemmaConfig::from_json(REAL_CONFIG).unwrap()
    }

    #[test]
    fn parses_the_shipped_config() {
        let c = cfg();
        let t = &c.text_config;
        assert_eq!(c.canvas_length, 256);
        assert_eq!(t.num_hidden_layers, 30);
        assert_eq!(t.hidden_size, 2816);
        assert_eq!(t.num_experts, 128);
        assert_eq!(t.top_k_experts, 8);
        assert_eq!(t.moe_intermediate_size, 704);
        assert_eq!(t.intermediate_size, 2112);
        assert_eq!(t.final_logit_softcapping, 30.0);
        let v = c.vision_config.as_ref().unwrap();
        assert_eq!((v.hidden_size, v.num_hidden_layers), (1152, 27));
    }

    #[test]
    fn layer_geometry_matches_the_checkpoint_shapes() {
        let c = cfg();
        let t = &c.text_config;
        // Sliding layer 0: q [16*256, 2816], k/v [8*256, 2816].
        assert!(!t.is_full(0));
        assert_eq!(t.layer_head_dim(0), 256);
        assert_eq!(t.layer_kv_heads(0), 8);
        assert!(!t.layer_k_eq_v(0));
        // Full layers 5/11/17/23/29: q [16*512, ...], k [2*512, ...], no v_proj.
        for l in [5, 11, 17, 23, 29] {
            assert!(t.is_full(l), "layer {l} should be full attention");
            assert_eq!(t.layer_head_dim(l), 512);
            assert_eq!(t.layer_kv_heads(l), 2);
            assert!(t.layer_k_eq_v(l));
        }
        assert_eq!(t.resolved_layer_types().len(), 30);
    }

    #[test]
    fn proportional_rope_zeroes_the_nope_tail() {
        let c = cfg();
        let t = &c.text_config;
        // Full layer: head_dim 512 → 256 angle slots, of which
        // int(0.25 * 512 // 2) = 64 are rotated.
        let inv = t.layer_inv_freq(5);
        assert_eq!(inv.len(), 256);
        assert!(inv[..64].iter().all(|&f| f > 0.0));
        assert!(inv[64..].iter().all(|&f| f == 0.0));
        // Exponent divides by head_dim (512), not by 2·rope_angles.
        assert!((inv[0] - 1.0).abs() < 1e-12);
        let want = 1.0 / 1_000_000f64.powf(2.0 / 512.0);
        assert!((inv[1] - want).abs() < 1e-12, "got {} want {want}", inv[1]);

        // Sliding layer: every slot rotated, theta 1e4 over head_dim 256.
        let inv_s = t.layer_inv_freq(0);
        assert_eq!(inv_s.len(), 128);
        assert!(inv_s.iter().all(|&f| f > 0.0));
        let want_s = 1.0 / 10_000f64.powf(2.0 / 256.0);
        assert!((inv_s[1] - want_s).abs() < 1e-12);
    }

    #[test]
    fn nope_slots_are_identity_in_the_tables() {
        let c = cfg();
        let (cos, sin) = c.text_config.rope_tables(5, 7, 3);
        assert_eq!(cos.len(), 3 * 256);
        // Zero inv_freq ⇒ angle 0 ⇒ cos 1 / sin 0 ⇒ those dims pass through.
        for t in 0..3 {
            for j in 64..256 {
                assert_eq!(cos[t * 256 + j], 1.0);
                assert_eq!(sin[t * 256 + j], 0.0);
            }
        }
        // Non-zero slots track the position offset.
        let inv = c.text_config.layer_inv_freq(5);
        let want = ((8.0_f64) * inv[3]).cos() as f32;
        assert!((cos[256 + 3] - want).abs() < 1e-6);
    }

    #[test]
    fn embed_and_router_scales() {
        let c = cfg();
        let t = &c.text_config;
        assert!((t.embed_scale() - (2816f32).sqrt()).abs() < 1e-4);
        assert!((t.router_root_scale() - 1.0 / (2816f32).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn generation_config_temperature_schedule() {
        let g = DiffusionGenerationConfig::default();
        assert_eq!(g.max_denoising_steps, 48);
        // Step counts down: first step is t_max, last (step 1) approaches t_min.
        assert!((g.temperature(48) - 0.8).abs() < 1e-6);
        assert!((g.temperature(0) - 0.4).abs() < 1e-6);
        assert!(g.temperature(24) > g.temperature(1));
    }

    #[test]
    fn missing_layer_types_fall_back_to_5_to_1() {
        let raw = REAL_CONFIG.replace("\"layer_types\"", "\"layer_types_unused\"");
        let c = DiffusionGemmaConfig::from_json(&raw).unwrap();
        let lt = c.text_config.resolved_layer_types();
        assert_eq!(lt.len(), 30);
        for l in [5, 11, 17, 23, 29] {
            assert_eq!(lt[l], LayerType::Full);
        }
        assert_eq!(lt[0], LayerType::Sliding);
    }

    #[test]
    fn rejects_a_non_moe_config() {
        let raw = REAL_CONFIG.replace("\"num_experts\": 128", "\"num_experts\": 0");
        assert!(DiffusionGemmaConfig::from_json(&raw).is_err());
    }
}
