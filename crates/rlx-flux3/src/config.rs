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

//! Model and package configuration.
//!
//! [`Flux3Config`] is the DiT's shape, derived from the checkpoint's tensors;
//! [`PackageConfig`] is the package's own `config.json` (text-encoder layers,
//! padding, sampler defaults).

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::path::Path;

/// Sinusoidal timestep embedding width (`time_in.in_layer` columns).
pub const TIME_EMBED_DIM: usize = 256;
/// LayerNorm / QK-RMSNorm epsilon.
pub const NORM_EPS: f32 = 1e-6;

/// The three token kinds of a sample, in the order they sit in the joint
/// sequence (BFL `MODES`, sorted: image, image_cond, text).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    /// The latents being generated.
    Image,
    /// Reference-image latents (editing).
    Cond,
    /// Text-encoder states.
    Text,
}

impl Mode {
    pub const ALL: [Mode; 3] = [Mode::Image, Mode::Cond, Mode::Text];

    /// Checkpoint key of the per-mode modulation (`early_stream_modulations.{key}`).
    pub fn mod_key(self) -> &'static str {
        match self {
            Mode::Image => "image",
            Mode::Cond => "image_cond",
            Mode::Text => "txt",
        }
    }

    /// Prefix of the mode's early blocks.
    pub fn block_prefix(self) -> &'static str {
        match self {
            Mode::Image => "content_mode_blocks.image",
            Mode::Cond => "content_mode_blocks.image_cond",
            Mode::Text => "txt_mode_blocks",
        }
    }

    /// The mode's input projection.
    pub fn embed_key(self) -> &'static str {
        match self {
            Mode::Image => "emb_in.image",
            Mode::Cond => "emb_in.image_cond",
            Mode::Text => "txt_in",
        }
    }
}

/// The DiT's shape.
#[derive(Debug, Clone, PartialEq)]
pub struct Flux3Config {
    pub hidden: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// MLP inner width (half of `mlp_in` rows: gate and up are stacked).
    pub mlp_hidden: usize,
    /// Per-mode blocks.
    pub depth: usize,
    /// Shared single-stream blocks.
    pub depth_single: usize,
    /// Latent token width (`emb_in.image` columns, 32 channels · 2 · 2).
    pub in_channels: usize,
    /// Text token width (`txt_in` columns: 8 encoder layers × 2560).
    pub context_dim: usize,
    /// RoPE axes `(t, h, w, l)`.
    pub axes_dim: [usize; 4],
    pub theta: f64,
    /// Image row/column ids are scaled by this before RoPE.
    pub image_rope_scale: f32,
    /// The distilled checkpoint is a flow map: it also embeds the time each
    /// step jumps to (`ref_time_in`) and runs without CFG.
    pub flow_map: bool,
}

impl Flux3Config {
    /// FLUX 3 Image Klein (5.5B).
    pub fn klein() -> Self {
        Self {
            hidden: 3072,
            num_heads: 24,
            head_dim: 128,
            mlp_hidden: 9216,
            depth: 5,
            depth_single: 28,
            in_channels: 128,
            context_dim: 20480,
            axes_dim: [32, 32, 32, 32],
            theta: 10_000.0,
            image_rope_scale: 0.5,
            flow_map: false,
        }
    }

    /// A small configuration with the real layout, for synthetic tests.
    pub fn tiny() -> Self {
        Self {
            hidden: 64,
            num_heads: 2,
            head_dim: 32,
            mlp_hidden: 192,
            depth: 2,
            depth_single: 2,
            in_channels: 16,
            context_dim: 40,
            axes_dim: [8, 8, 8, 8],
            theta: 10_000.0,
            image_rope_scale: 0.5,
            flow_map: false,
        }
    }

    /// Derive every dimension from the checkpoint's tensor shapes.
    pub fn from_shapes(shape_of: impl Fn(&str) -> Option<Vec<usize>>) -> Result<Self> {
        let s = |k: &str| shape_of(k).with_context(|| format!("checkpoint has no `{k}`"));
        let emb = s("emb_in.image.weight")?;
        let txt = s("txt_in.weight")?;
        let mlp = s("single_blocks.0.mlp_in.weight")?;
        let head_dim = s("single_blocks.0.norm.query_norm.scale")?[0];
        ensure!(
            emb.len() == 2 && txt.len() == 2 && mlp.len() == 2,
            "bad DiT shapes"
        );
        let hidden = emb[0];
        ensure!(
            hidden.is_multiple_of(head_dim) && head_dim.is_multiple_of(8),
            "hidden {hidden} / head_dim {head_dim}"
        );
        let count = |prefix: &str| {
            (0..)
                .take_while(|i| shape_of(&format!("{prefix}.{i}.q_proj.weight")).is_some())
                .count()
        };
        let depth = count("txt_mode_blocks");
        let depth_single = count("single_blocks");
        if depth == 0 || depth_single == 0 {
            bail!("no per-mode or single-stream blocks in the checkpoint");
        }
        let q = head_dim / 4;
        Ok(Self {
            hidden,
            num_heads: hidden / head_dim,
            head_dim,
            mlp_hidden: mlp[0] / 2,
            depth,
            depth_single,
            in_channels: emb[1],
            context_dim: txt[1],
            axes_dim: [q, q, q, q],
            theta: 10_000.0,
            image_rope_scale: 0.5,
            flow_map: shape_of("ref_time_in.in_layer.weight").is_some(),
        })
    }

    /// Take `axes_dim`, `theta` and `image_rope_scale` from the package config.
    pub fn with_package(mut self, pkg: &PackageConfig) -> Result<Self> {
        if let Some(d) = &pkg.dit {
            if let Some(a) = &d.axes_dim {
                ensure!(a.len() == 4, "axes_dim {a:?}: expected 4 axes");
                self.axes_dim = [a[0], a[1], a[2], a[3]];
            }
            if let Some(t) = d.theta {
                self.theta = t;
            }
            if let Some(r) = d.image_rope_scale {
                self.image_rope_scale = r;
            }
        }
        ensure!(
            self.axes_dim.iter().sum::<usize>() == self.head_dim,
            "axes_dim {:?} does not sum to head_dim {}",
            self.axes_dim,
            self.head_dim
        );
        Ok(self)
    }
}

/// The package's `config.json` (only the fields used here).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PackageConfig {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub dit: Option<PackageDit>,
    #[serde(default)]
    pub text_encoder: Option<PackageTextEncoder>,
    #[serde(default)]
    pub sampler: Option<PackageSampler>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PackageDit {
    pub axes_dim: Option<Vec<usize>>,
    pub theta: Option<f64>,
    pub image_rope_scale: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PackageTextEncoder {
    pub layers: Option<Vec<usize>>,
    pub pad_multiple: Option<usize>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PackageSampler {
    pub steps: Option<usize>,
    pub guidance: Option<f32>,
    pub alpha: Option<f32>,
    pub size: Option<[usize; 2]>,
}

impl PackageConfig {
    /// Read `config.json` from a package directory; a missing file gives the defaults.
    pub fn load(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        if !p.is_file() {
            return Ok(Self::default());
        }
        let s = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        serde_json::from_str(&s).with_context(|| format!("parsing {}", p.display()))
    }

    /// Hidden-state indices stacked into a text token.
    pub fn text_layers(&self) -> Vec<usize> {
        self.text_encoder
            .as_ref()
            .and_then(|t| t.layers.clone())
            .unwrap_or_else(|| vec![4, 8, 12, 16, 20, 24, 28, 32])
    }

    /// Text sequences are padded to a multiple of this.
    pub fn pad_multiple(&self) -> usize {
        self.text_encoder
            .as_ref()
            .and_then(|t| t.pad_multiple)
            .unwrap_or(80)
    }

    /// Prompt truncation length.
    pub fn max_tokens(&self) -> usize {
        self.text_encoder
            .as_ref()
            .and_then(|t| t.max_tokens)
            .unwrap_or(4096)
    }

    /// Default sampler: the package's, else the reference code's
    /// (`flux3/pipeline.py`: 50 steps, CFG 4, alpha 16 for base; 16 steps
    /// without CFG for the flow map).
    pub fn sampler(&self, flow_map: bool) -> SamplerDefaults {
        let s = self.sampler.clone().unwrap_or_default();
        let (steps, guidance) = if flow_map { (16, 1.0) } else { (50, 4.0) };
        SamplerDefaults {
            steps: s.steps.unwrap_or(steps),
            guidance: if flow_map {
                1.0
            } else {
                s.guidance.unwrap_or(guidance)
            },
            alpha: s.alpha.unwrap_or(16.0),
            size: s.size.unwrap_or([1024, 1024]),
        }
    }
}

/// Sampler settings a package ships.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerDefaults {
    pub steps: usize,
    /// CFG scale (1 = no CFG).
    pub guidance: f32,
    /// Time-shift of the schedule.
    pub alpha: f32,
    /// `[width, height]` for text-to-image.
    pub size: [usize; 2],
}

/// The Qwen3-VL text tower's shape (`text_encoder/config.json` → `text_config`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TextEncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
}

impl TextEncoderConfig {
    /// Parse `config.json` of a `Qwen3VLForConditionalGeneration` (its
    /// `text_config`) or of a plain Qwen3 text model.
    pub fn load(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        let v: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?,
        )?;
        let t = v.get("text_config").unwrap_or(&v);
        let mut t = t.clone();
        // Qwen3-VL keeps rope_theta at the top of text_config; newer
        // configs move it into `rope_parameters`.
        if t.get("rope_theta").is_none()
            && let Some(r) = t.pointer("/rope_parameters/rope_theta").cloned()
        {
            t["rope_theta"] = r;
        }
        serde_json::from_value(t).with_context(|| format!("text config in {}", p.display()))
    }

    /// A small text tower for synthetic tests.
    pub fn tiny() -> Self {
        Self {
            hidden_size: 16,
            intermediate_size: 24,
            num_hidden_layers: 8,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            vocab_size: 32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_defaults_follow_the_package() {
        let pkg: PackageConfig = serde_json::from_str(
            r#"{"sampler": {"steps": 50, "guidance": 3.5, "alpha": 6.93, "size": [1024, 1024]}}"#,
        )
        .unwrap();
        let s = pkg.sampler(false);
        assert_eq!((s.steps, s.guidance, s.alpha), (50, 3.5, 6.93));
        // A flow map never takes CFG, whatever the package says.
        assert_eq!(pkg.sampler(true).guidance, 1.0);
        let none = PackageConfig::default().sampler(false);
        assert_eq!((none.steps, none.guidance, none.alpha), (50, 4.0, 16.0));
    }
}
