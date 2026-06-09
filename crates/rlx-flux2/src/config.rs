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

//! FLUX.2 transformer configuration (matches HuggingFace / diffusers / BFL).

use anyhow::{Context, Result, bail};
use rlx_core::gguf_support::gguf_architecture_str;
use rlx_gguf::{GgufFile, MetaValue};
use serde::Deserialize;
use std::path::Path;

/// FLUX.2 rectified-flow transformer (denoiser) configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Flux2Config {
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    pub out_channels: Option<usize>,
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    #[serde(default = "default_num_single_layers")]
    pub num_single_layers: usize,
    #[serde(default = "default_attention_head_dim")]
    pub attention_head_dim: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_joint_attention_dim")]
    pub joint_attention_dim: usize,
    #[serde(default = "default_timestep_guidance_channels")]
    pub timestep_guidance_channels: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
    #[serde(default = "default_axes_dims_rope")]
    pub axes_dims_rope: Vec<usize>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: usize,
    #[serde(default = "default_eps")]
    pub eps: f64,
    #[serde(default = "default_guidance_embeds")]
    pub guidance_embeds: bool,
}

fn default_patch_size() -> usize {
    1
}
fn default_in_channels() -> usize {
    128
}
fn default_num_layers() -> usize {
    8
}
fn default_num_single_layers() -> usize {
    48
}
fn default_attention_head_dim() -> usize {
    128
}
fn default_num_attention_heads() -> usize {
    48
}
fn default_joint_attention_dim() -> usize {
    15360
}
fn default_timestep_guidance_channels() -> usize {
    256
}
fn default_mlp_ratio() -> f64 {
    3.0
}
fn default_axes_dims_rope() -> Vec<usize> {
    vec![32, 32, 32, 32]
}
fn default_rope_theta() -> usize {
    2000
}
fn default_eps() -> f64 {
    1e-6
}
fn default_guidance_embeds() -> bool {
    true
}

impl Flux2Config {
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).with_context(|| format!("reading {path:?}"))?;
        Ok(serde_json::from_str(&data)?)
    }

    /// Hidden width (`num_attention_heads * attention_head_dim`).
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn ff_inner_dim(&self) -> usize {
        (self.inner_dim() as f64 * self.mlp_ratio) as usize
    }

    pub fn out_ch(&self) -> usize {
        self.out_channels.unwrap_or(self.in_channels)
    }

    pub fn proj_out_dim(&self) -> usize {
        self.patch_size * self.patch_size * self.out_ch()
    }

    /// FLUX.2 [dev] defaults (32B-class; not runnable on commodity RAM at F32).
    pub fn flux2_dev() -> Self {
        Self {
            patch_size: 1,
            in_channels: 128,
            out_channels: None,
            num_layers: 8,
            num_single_layers: 48,
            attention_head_dim: 128,
            num_attention_heads: 48,
            joint_attention_dim: 15360,
            timestep_guidance_channels: 256,
            mlp_ratio: 3.0,
            axes_dims_rope: vec![32, 32, 32, 32],
            rope_theta: 2000,
            eps: 1e-6,
            guidance_embeds: true,
        }
    }

    /// FLUX.2 [klein] 4B-style defaults (guidance embedder optional).
    pub fn flux2_klein_4b() -> Self {
        Self {
            num_layers: 4,
            num_single_layers: 16,
            num_attention_heads: 24,
            attention_head_dim: 128,
            joint_attention_dim: 7680,
            guidance_embeds: false,
            ..Self::flux2_dev()
        }
    }

    /// FLUX.2 [klein] 9B defaults (BFL `Klein9BParams`: 8 double + 24 single, 32 heads).
    pub fn flux2_klein_9b() -> Self {
        Self {
            num_layers: 8,
            num_single_layers: 24,
            num_attention_heads: 32,
            attention_head_dim: 128,
            joint_attention_dim: 12288,
            guidance_embeds: false,
            ..Self::flux2_dev()
        }
    }

    /// Infer variant from checkpoint tensor names (BFL `double_blocks.*` or diffusers `transformer_blocks.*`).
    pub fn infer_from_weight_keys<'a>(keys: impl IntoIterator<Item = &'a str>) -> Self {
        let keys: Vec<&str> = keys.into_iter().collect();
        let double = max_block_layers(&keys, &["double_blocks.", "transformer_blocks."]);
        let single = max_block_layers(&keys, &["single_blocks.", "single_transformer_blocks."]);
        let guidance = keys.iter().any(|k| {
            k.contains("guidance_in.") || k.contains("time_guidance_embed.guidance_embedder.")
        });
        match (double, single) {
            (8, 24) => Self::flux2_klein_9b(),
            (4, 16) => Self::flux2_klein_4b(),
            (8, 48) => Self::flux2_dev(),
            (d, s) if d > 0 && s > 0 => {
                let mut cfg = Self::flux2_klein_9b();
                cfg.num_layers = d;
                cfg.num_single_layers = s;
                cfg.guidance_embeds = guidance;
                cfg
            }
            _ => Self::flux2_klein_9b(),
        }
    }

    /// Read `flux.*` metadata when present; otherwise infer from `general.basename`.
    pub fn from_gguf(raw: &GgufFile) -> Result<Self> {
        let arch = gguf_architecture_str(raw).unwrap_or("flux");
        if arch != "flux" {
            bail!("Flux2Config::from_gguf expected architecture `flux`, got {arch}");
        }
        if let Some(name) = raw
            .metadata
            .get("general.basename")
            .and_then(MetaValue::as_str)
        {
            let lower = name.to_lowercase();
            if lower.contains("klein") && (lower.contains("9b") || lower.contains("9-b")) {
                return Ok(Self::flux2_klein_9b());
            }
            if lower.contains("klein") {
                return Ok(Self::flux2_klein_4b());
            }
            if lower.contains("dev") {
                return Ok(Self::flux2_dev());
            }
        }
        Ok(Self::infer_from_weight_keys(
            raw.tensors.keys().map(|s| s.as_str()),
        ))
    }

    /// Tiny config for unit tests and graph minimal builds.
    pub fn tiny() -> Self {
        Self {
            patch_size: 1,
            in_channels: 8,
            out_channels: None,
            num_layers: 1,
            num_single_layers: 1,
            attention_head_dim: 16,
            num_attention_heads: 2,
            joint_attention_dim: 16,
            timestep_guidance_channels: 32,
            mlp_ratio: 2.0,
            axes_dims_rope: vec![4, 4, 4, 4],
            rope_theta: 2000,
            eps: 1e-6,
            guidance_embeds: true,
        }
    }
}

fn max_block_layers(keys: &[&str], prefixes: &[&str]) -> usize {
    let mut max_idx = 0usize;
    for key in keys {
        for pfx in prefixes {
            if let Some(rest) = key.strip_prefix(pfx) {
                if let Ok(i) = rest.split('.').next().unwrap_or("").parse::<usize>() {
                    max_idx = max_idx.max(i + 1);
                }
            }
        }
    }
    max_idx
}

#[cfg(test)]
mod gguf_config_tests {
    use super::*;

    #[test]
    fn infer_klein_9b_from_bfl_keys() {
        let keys = [
            "double_blocks.0.img_attn.qkv.weight",
            "double_blocks.7.img_attn.qkv.weight",
            "single_blocks.23.linear1.weight",
        ];
        let cfg = Flux2Config::infer_from_weight_keys(keys);
        assert_eq!(cfg.num_layers, 8);
        assert_eq!(cfg.num_single_layers, 24);
        assert!(!cfg.guidance_embeds);
    }
}
