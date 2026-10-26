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

//! FLUX.1 configuration: the MMDiT denoiser, read from the checkpoint's own
//! tensor shapes (BFL single-file layout), plus the two text encoders.

use anyhow::{Context, Result, bail};
use std::collections::HashSet;

/// FLUX.1 rectified-flow transformer dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct Flux1Config {
    /// Residual width (`img_in.weight` rows).
    pub hidden: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// MLP inner width (`double_blocks.0.img_mlp.0.weight` rows).
    pub mlp_hidden: usize,
    /// Dual-stream ("double") blocks.
    pub depth: usize,
    /// Single-stream blocks.
    pub depth_single: usize,
    /// Packed latent channels (`16 · 2 · 2`).
    pub in_channels: usize,
    /// T5 hidden width (`txt_in.weight` columns).
    pub context_dim: usize,
    /// Pooled CLIP width (`vector_in.in_layer.weight` columns).
    pub vec_dim: usize,
    /// RoPE axes (`[16, 56, 56]`: unused text axis, row, column).
    pub axes_dim: [usize; 3],
    pub theta: f64,
    /// `[dev]` distills CFG into a guidance embedding; `[schnell]` has none.
    pub guidance_embed: bool,
}

/// Channels of one sinusoidal timestep / guidance embedding.
pub const TIME_EMBED_DIM: usize = 256;
/// LayerNorm / RMSNorm epsilon used throughout the denoiser.
pub const NORM_EPS: f32 = 1e-6;

impl Flux1Config {
    /// FLUX.1 `[dev]` (12B).
    pub fn dev() -> Self {
        Self {
            hidden: 3072,
            num_heads: 24,
            head_dim: 128,
            mlp_hidden: 12288,
            depth: 19,
            depth_single: 38,
            in_channels: 64,
            context_dim: 4096,
            vec_dim: 768,
            axes_dim: [16, 56, 56],
            theta: 10_000.0,
            guidance_embed: true,
        }
    }

    /// FLUX.1 `[schnell]`: `[dev]` without the guidance embedding.
    pub fn schnell() -> Self {
        Self {
            guidance_embed: false,
            ..Self::dev()
        }
    }

    /// A small configuration with the real layout, for synthetic tests.
    pub fn tiny() -> Self {
        Self {
            hidden: 64,
            num_heads: 2,
            head_dim: 32,
            mlp_hidden: 256,
            depth: 2,
            depth_single: 2,
            in_channels: 64,
            context_dim: 48,
            vec_dim: 24,
            axes_dim: [4, 14, 14],
            theta: 10_000.0,
            guidance_embed: true,
        }
    }

    /// Derive every dimension from a BFL-layout checkpoint's tensor shapes.
    pub fn from_shapes(shape_of: impl Fn(&str) -> Option<Vec<usize>>) -> Result<Self> {
        let get = |k: &str| shape_of(k).with_context(|| format!("FLUX.1 checkpoint lacks `{k}`"));
        let img_in = get("img_in.weight")?;
        let (hidden, in_channels) = (img_in[0], img_in[1]);
        let qk = get("double_blocks.0.img_attn.norm.query_norm.scale")?;
        let head_dim = qk[0];
        if hidden % head_dim != 0 {
            bail!("hidden {hidden} is not a multiple of head_dim {head_dim}");
        }
        let mlp_hidden = get("double_blocks.0.img_mlp.0.weight")?[0];
        let context_dim = get("txt_in.weight")?[1];
        let vec_dim = get("vector_in.in_layer.weight")?[1];
        let count = |prefix: &str| {
            (0..)
                .take_while(|i| {
                    shape_of(&format!("{prefix}.{i}.norm.key_norm.scale")).is_some()
                        || shape_of(&format!("{prefix}.{i}.img_attn.norm.key_norm.scale")).is_some()
                })
                .count()
        };
        let depth = count("double_blocks");
        let depth_single = count("single_blocks");
        if depth == 0 || depth_single == 0 {
            bail!("FLUX.1 checkpoint has {depth} double / {depth_single} single blocks");
        }
        // The axes split is fixed by BFL's `axes_dim = [16, 56, 56]` for
        // head_dim 128; scale it for other head widths so the three still sum.
        let axes_dim = if head_dim == 128 {
            [16, 56, 56]
        } else {
            let a = head_dim / 8;
            let rest = (head_dim - a) / 2;
            [a, rest, head_dim - a - rest]
        };
        Ok(Self {
            hidden,
            num_heads: hidden / head_dim,
            head_dim,
            mlp_hidden,
            depth,
            depth_single,
            in_channels,
            context_dim,
            vec_dim,
            axes_dim,
            theta: 10_000.0,
            guidance_embed: shape_of("guidance_in.in_layer.weight").is_some(),
        })
    }

    /// Names every tensor a BFL checkpoint of this shape must contain.
    pub fn expected_tensors(&self) -> HashSet<String> {
        let mut out = HashSet::new();
        let mut lin = |p: String| {
            out.insert(format!("{p}.weight"));
            out.insert(format!("{p}.bias"));
        };
        for p in [
            "img_in",
            "txt_in",
            "final_layer.linear",
            "final_layer.adaLN_modulation.1",
        ] {
            lin(p.to_string());
        }
        let mut mlps = vec!["time_in", "vector_in"];
        if self.guidance_embed {
            mlps.push("guidance_in");
        }
        for m in mlps {
            lin(format!("{m}.in_layer"));
            lin(format!("{m}.out_layer"));
        }
        for i in 0..self.depth {
            for s in ["img", "txt"] {
                let p = format!("double_blocks.{i}.{s}");
                lin(format!("{p}_mod.lin"));
                lin(format!("{p}_attn.qkv"));
                lin(format!("{p}_attn.proj"));
                lin(format!("{p}_mlp.0"));
                lin(format!("{p}_mlp.2"));
            }
        }
        for i in 0..self.depth_single {
            let p = format!("single_blocks.{i}");
            lin(format!("{p}.linear1"));
            lin(format!("{p}.linear2"));
            lin(format!("{p}.modulation.lin"));
        }
        for i in 0..self.depth {
            for s in ["img", "txt"] {
                for n in ["query_norm", "key_norm"] {
                    out.insert(format!("double_blocks.{i}.{s}_attn.norm.{n}.scale"));
                }
            }
        }
        for i in 0..self.depth_single {
            for n in ["query_norm", "key_norm"] {
                out.insert(format!("single_blocks.{i}.norm.{n}.scale"));
            }
        }
        out
    }
}

/// T5 v1.1 encoder dimensions (`text_encoder_2/config.json`).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct T5Config {
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    #[serde(default = "default_t5_buckets")]
    pub relative_attention_num_buckets: usize,
    #[serde(default = "default_t5_max_distance")]
    pub relative_attention_max_distance: usize,
    #[serde(default = "default_eps")]
    pub layer_norm_epsilon: f64,
    pub vocab_size: usize,
}

fn default_t5_buckets() -> usize {
    32
}
fn default_t5_max_distance() -> usize {
    128
}
fn default_eps() -> f64 {
    1e-6
}

impl T5Config {
    /// `google/t5-v1_1-xxl`.
    pub fn xxl() -> Self {
        Self {
            d_model: 4096,
            d_kv: 64,
            d_ff: 10240,
            num_heads: 64,
            num_layers: 24,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_epsilon: 1e-6,
            vocab_size: 32128,
        }
    }

    pub fn tiny() -> Self {
        Self {
            d_model: 32,
            d_kv: 8,
            d_ff: 48,
            num_heads: 4,
            num_layers: 3,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_epsilon: 1e-6,
            vocab_size: 100,
        }
    }

    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.d_kv
    }
}

/// CLIP text tower dimensions (`text_encoder/config.json`, or the
/// `text_config` of a full CLIP checkpoint).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct ClipTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_clip_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_clip_act")]
    pub hidden_act: String,
    pub vocab_size: usize,
    #[serde(default = "default_clip_eos")]
    pub eos_token_id: u32,
}

fn default_clip_eps() -> f64 {
    1e-5
}
fn default_clip_act() -> String {
    "quick_gelu".into()
}
fn default_clip_eos() -> u32 {
    49407
}

impl ClipTextConfig {
    /// `openai/clip-vit-large-patch14` text tower.
    pub fn vit_l_14() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            hidden_act: "quick_gelu".into(),
            vocab_size: 49408,
            eos_token_id: 49407,
        }
    }

    pub fn tiny() -> Self {
        Self {
            hidden_size: 24,
            intermediate_size: 48,
            num_attention_heads: 2,
            num_hidden_layers: 2,
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            hidden_act: "quick_gelu".into(),
            vocab_size: 300,
            eos_token_id: 299,
        }
    }

    /// Read `config.json`, accepting both a bare `CLIPTextConfig` and a full
    /// `CLIPConfig` (whose text half sits under `text_config`).
    pub fn from_json(raw: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(raw)?;
        let text = v.get("text_config").cloned().unwrap_or(v);
        // `eos_token_id: 2` is the legacy placeholder in openai's config; the
        // real end-of-text id is the last vocabulary entry.
        let mut cfg: Self = serde_json::from_value(text)?;
        if cfg.eos_token_id < 3 {
            cfg.eos_token_id = (cfg.vocab_size - 1) as u32;
        }
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_shapes_recovers_dev_and_schnell() {
        for cfg in [Flux1Config::dev(), Flux1Config::schnell()] {
            let names = cfg.expected_tensors();
            let shape = |k: &str| -> Option<Vec<usize>> {
                if !names.contains(k) {
                    return None;
                }
                Some(match k {
                    "img_in.weight" => vec![cfg.hidden, cfg.in_channels],
                    "txt_in.weight" => vec![cfg.hidden, cfg.context_dim],
                    "vector_in.in_layer.weight" => vec![cfg.hidden, cfg.vec_dim],
                    "double_blocks.0.img_mlp.0.weight" => vec![cfg.mlp_hidden, cfg.hidden],
                    k if k.ends_with("_norm.scale") => vec![cfg.head_dim],
                    _ => vec![1],
                })
            };
            assert_eq!(Flux1Config::from_shapes(shape).unwrap(), cfg);
        }
    }
}
