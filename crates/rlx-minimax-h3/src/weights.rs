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

//! Loading MiniMax-H3's sharded `safetensors` components.
//!
//! The released checkpoint is **mixed precision**: the two input patch
//! projections, the timestep MLP and the two output heads are `float32` while
//! everything else — including the AdaLN projections, which hold most of the
//! parameters — is `bfloat16`. [`rlx_core::weight_map::WeightMap`] widens
//! everything to `f32` on load, so nothing here has to branch on dtype; the
//! distinction only matters if a future packed path wants to keep the block
//! stack narrow.
//!
//! Component directories and their expected weight prefixes:
//!
//! | Directory | Shards | Notes |
//! |---|---|---|
//! | `transformer/` | 14 | `t2va` / `i2va` / `fl2va` |
//! | `transformer_ref/` | 14 | `ref2va` — same architecture, separate weights |
//! | `vae/` | 3 | video VAE |
//! | `audio_vae/` | 1 | audio VAE |
//! | `text_encoder/` | 14 | Qwen3-VL |

use crate::config::H3TransformerConfig;
use anyhow::{Context, Result, bail};
use rlx_core::weight_map::WeightMap;
use std::path::Path;

/// Which DiT partition a task loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitPartition {
    /// `transformer/` — `t2va`, `i2va`, `fl2va`.
    Base,
    /// `transformer_ref/` — `ref2va`.
    Reference,
}

impl DitPartition {
    #[must_use]
    pub fn subdir(self) -> &'static str {
        match self {
            Self::Base => "transformer",
            Self::Reference => "transformer_ref",
        }
    }
}

/// Load one DiT partition's weights.
pub fn load_dit(root: &Path, partition: DitPartition) -> Result<WeightMap> {
    let dir = root.join(partition.subdir());
    if !dir.is_dir() {
        bail!(
            "MiniMax-H3: {} is missing; the {:?} partition is not present in this checkpoint",
            dir.display(),
            partition
        );
    }
    WeightMap::from_safetensors_dir(&dir)
        .with_context(|| format!("MiniMax-H3: load DiT weights from {}", dir.display()))
}

/// Load the video VAE weights.
pub fn load_video_vae(root: &Path) -> Result<WeightMap> {
    let dir = root.join("vae");
    WeightMap::from_safetensors_dir(&dir)
        .with_context(|| format!("MiniMax-H3: load video VAE from {}", dir.display()))
}

/// Load only the video VAE's ViT decoder.
///
/// The `vae/` directory is ~10 GB and the encoder half is dead weight when a
/// run only turns latents into pixels, so this selects the decoder keys instead
/// of widening the whole thing to `f32`.
pub fn load_video_vae_decoder(
    root: &Path,
    cfg: &crate::config::H3VideoVaeConfig,
) -> Result<WeightMap> {
    let dir = root.join("vae");
    let want: std::collections::HashSet<String> = crate::vae_video::decoder_parameter_keys(cfg)
        .into_iter()
        .collect();
    let mut wm = WeightMap::from_safetensors_dir_selected(&dir, &want)
        .with_context(|| format!("MiniMax-H3: load video VAE decoder from {}", dir.display()))?;
    squeeze_pointwise_conv(&mut wm, "post_quant_conv.weight")?;
    Ok(wm)
}

/// Flatten a `1x1x1` `Conv3d` kernel to the `[out, in]` matrix it really is.
///
/// `post_quant_conv` is stored as `[out, in, 1, 1, 1]`, but a pointwise
/// convolution is a linear map on the channel axis and the decoder graph
/// consumes it as a matmul.
pub fn squeeze_pointwise_conv(wm: &mut WeightMap, key: &str) -> Result<()> {
    let Some((data, shape)) = wm.get(key) else {
        return Ok(());
    };
    if shape.len() <= 2 {
        return Ok(());
    }
    if shape[2..].iter().any(|&d| d != 1) {
        bail!("{key}: expected a 1x1x1 kernel, got {shape:?}");
    }
    let squeezed = vec![shape[0], shape[1]];
    let data = data.to_vec();
    let mut tensors: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)> =
        std::collections::HashMap::new();
    for k in wm.keys() {
        let (d, s) = wm.get(k).expect("key came from keys()");
        tensors.insert(k.to_string(), (d.to_vec(), s.to_vec()));
    }
    tensors.insert(key.to_string(), (data, squeezed));
    *wm = WeightMap::from_tensors(tensors);
    Ok(())
}

/// Load the audio VAE weights.
pub fn load_audio_vae(root: &Path) -> Result<WeightMap> {
    let dir = root.join("audio_vae");
    WeightMap::from_safetensors_dir(&dir)
        .with_context(|| format!("MiniMax-H3: load audio VAE from {}", dir.display()))
}

/// Load the Qwen3-VL text encoder weights.
pub fn load_text_encoder(root: &Path) -> Result<WeightMap> {
    let dir = root.join("text_encoder");
    WeightMap::from_safetensors_dir(&dir)
        .with_context(|| format!("MiniMax-H3: load text encoder from {}", dir.display()))
}

/// Every parameter key one DiT partition must provide, in load order.
///
/// Used to fail a run before compiling a 33B graph rather than midway through
/// it, and to build synthetic weights in tests.
#[must_use]
pub fn dit_parameter_keys(cfg: &H3TransformerConfig) -> Vec<String> {
    let mut keys = vec![
        "proj_in.weight".to_string(),
        "proj_in.bias".to_string(),
        "audio_proj_in.weight".to_string(),
        "audio_proj_in.bias".to_string(),
        "context_embedder.weight".to_string(),
        "context_embedder.bias".to_string(),
        "time_embedder.linear_1.weight".to_string(),
        "time_embedder.linear_1.bias".to_string(),
        "time_embedder.linear_2.weight".to_string(),
        "time_embedder.linear_2.bias".to_string(),
        "token_refiner.final_norm.weight".to_string(),
        "norm_out.norm.weight".to_string(),
        "norm_out.linear.weight".to_string(),
        "norm_out.linear.bias".to_string(),
        "proj_out.weight".to_string(),
        "proj_out.bias".to_string(),
        "audio_proj_out.weight".to_string(),
        "audio_proj_out.bias".to_string(),
    ];
    for b in 0..cfg.num_refiner_layers {
        let p = format!("token_refiner.refiner_blocks.{b}");
        keys.extend(attention_block_keys(&p, false));
    }
    for b in 0..cfg.num_layers {
        let p = format!("transformer_blocks.{b}");
        keys.extend(attention_block_keys(&p, true));
    }
    keys
}

fn attention_block_keys(prefix: &str, adaln: bool) -> Vec<String> {
    let mut keys = vec![
        format!("{prefix}.norm1.weight"),
        format!("{prefix}.attn.to_q.weight"),
        format!("{prefix}.attn.to_k.weight"),
        format!("{prefix}.attn.to_v.weight"),
        format!("{prefix}.attn.norm_q.weight"),
        format!("{prefix}.attn.norm_k.weight"),
        format!("{prefix}.attn.to_out.0.weight"),
        format!("{prefix}.norm2.weight"),
        format!("{prefix}.ff.net.0.proj.weight"),
        format!("{prefix}.ff.net.2.weight"),
    ];
    if adaln {
        keys.push(format!("{prefix}.adaln_proj.linear.weight"));
        keys.push(format!("{prefix}.adaln_proj.linear.bias"));
    }
    keys
}

/// The shape every DiT parameter must have, so a mismatch is reported by key
/// rather than as a downstream matmul error.
#[must_use]
pub fn dit_parameter_shape(cfg: &H3TransformerConfig, key: &str) -> Option<Vec<usize>> {
    let hidden = cfg.hidden_size;
    let inner = cfg.inner_dim();
    let ted = cfg.time_embed_dim;
    let vpd = cfg.video_patch_dim();

    let shape = match key {
        "proj_in.weight" => vec![hidden, vpd],
        "proj_in.bias" => vec![hidden],
        "audio_proj_in.weight" => vec![hidden, cfg.audio_in_channels],
        "audio_proj_in.bias" => vec![hidden],
        "context_embedder.weight" => vec![hidden, cfg.text_dim],
        "context_embedder.bias" => vec![hidden],
        "time_embedder.linear_1.weight" => vec![cfg.time_embed_hidden_dim, cfg.freq_dim],
        "time_embedder.linear_1.bias" => vec![cfg.time_embed_hidden_dim],
        "time_embedder.linear_2.weight" => vec![ted, cfg.time_embed_hidden_dim],
        "time_embedder.linear_2.bias" => vec![ted],
        "token_refiner.final_norm.weight" | "norm_out.norm.weight" => vec![hidden],
        "norm_out.linear.weight" => vec![2 * hidden, ted],
        "norm_out.linear.bias" => vec![2 * hidden],
        "proj_out.weight" => vec![vpd, hidden],
        "proj_out.bias" => vec![vpd],
        "audio_proj_out.weight" => vec![cfg.audio_in_channels, hidden],
        "audio_proj_out.bias" => vec![cfg.audio_in_channels],
        // Per-block keys, matched on their suffix so the same table serves the
        // refiner blocks and the main stack.
        k if k.ends_with(".norm1.weight") || k.ends_with(".norm2.weight") => vec![hidden],
        k if k.ends_with(".attn.to_q.weight")
            || k.ends_with(".attn.to_k.weight")
            || k.ends_with(".attn.to_v.weight") =>
        {
            vec![inner, hidden]
        }
        k if k.ends_with(".attn.norm_q.weight") || k.ends_with(".attn.norm_k.weight") => {
            vec![cfg.attention_head_dim]
        }
        k if k.ends_with(".attn.to_out.0.weight") => vec![hidden, inner],
        k if k.ends_with(".ff.net.0.proj.weight") => vec![2 * cfg.ffn_dim, hidden],
        k if k.ends_with(".ff.net.2.weight") => vec![hidden, cfg.ffn_dim],
        k if k.ends_with(".adaln_proj.linear.weight") => vec![cfg.adaln_proj_out(), ted],
        k if k.ends_with(".adaln_proj.linear.bias") => vec![cfg.adaln_proj_out()],
        _ => return None,
    };
    Some(shape)
}

/// Fail early if the loaded weights do not cover the configured architecture.
pub fn validate_dit_weights(cfg: &H3TransformerConfig, weights: &WeightMap) -> Result<()> {
    let mut missing = Vec::new();
    let mut wrong = Vec::new();
    for key in dit_parameter_keys(cfg) {
        match weights.get(&key) {
            None => missing.push(key),
            Some((_, shape)) => {
                if let Some(want) = dit_parameter_shape(cfg, &key) {
                    if shape != want.as_slice() {
                        wrong.push(format!("{key}: got {shape:?}, expected {want:?}"));
                    }
                }
            }
        }
    }
    if !missing.is_empty() {
        bail!(
            "MiniMax-H3: {} DiT parameter(s) missing, first few: {:?}",
            missing.len(),
            &missing[..missing.len().min(5)]
        );
    }
    if !wrong.is_empty() {
        bail!(
            "MiniMax-H3: {} DiT parameter shape mismatch(es), first few: {:?}",
            wrong.len(),
            &wrong[..wrong.len().min(5)]
        );
    }
    Ok(())
}

/// Build a deterministic synthetic weight set for the configured architecture.
///
/// Values come from a small counter-based hash so tests are reproducible
/// without a random-number dependency, and are scaled by `1/sqrt(fan_in)` so a
/// stack of 50 blocks does not blow up.
#[must_use]
pub fn synthetic_dit_weights(cfg: &H3TransformerConfig, seed: u64) -> WeightMap {
    use std::collections::HashMap;
    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for (i, key) in dit_parameter_keys(cfg).into_iter().enumerate() {
        let shape = dit_parameter_shape(cfg, &key).unwrap_or_else(|| vec![cfg.hidden_size]);
        let n: usize = shape.iter().product();
        let fan_in = *shape.last().unwrap_or(&1);
        let is_norm = key.ends_with("norm1.weight")
            || key.ends_with("norm2.weight")
            || key.ends_with("norm_q.weight")
            || key.ends_with("norm_k.weight")
            || key == "norm_out.norm.weight"
            || key == "token_refiner.final_norm.weight";
        let scale = if is_norm {
            0.0
        } else {
            1.0 / (fan_in as f32).sqrt()
        };
        let data = (0..n)
            .map(|j| {
                let base = if is_norm { 1.0 } else { 0.0 };
                base + scale * unit_hash(seed, i as u64, j as u64)
            })
            .collect();
        tensors.insert(key, (data, shape));
    }
    WeightMap::from_tensors(tensors)
}

/// A deterministic value in `[-1, 1)` from three counters.
fn unit_hash(seed: u64, a: u64, b: u64) -> f32 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(a.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(b.wrapping_mul(0x94D0_49BB_1331_11EB));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    // Keep 12 bits so the result really lands in [-1, 1): a wider shift leaves
    // values in the thousands, which a 50-block residual stack turns into NaN.
    ((x >> 52) as f32 / 2048.0) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> H3TransformerConfig {
        H3TransformerConfig {
            num_attention_heads: 2,
            attention_head_dim: 16,
            hidden_size: 24,
            num_layers: 2,
            num_refiner_layers: 1,
            ffn_dim: 32,
            in_channels: 4,
            audio_in_channels: 6,
            patch_size: [1, 2, 2],
            text_dim: 8,
            freq_dim: 16,
            time_embed_hidden_dim: 24,
            time_embed_dim: 12,
            rope_freq_dim: 2,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }

    #[test]
    fn released_key_count_matches_the_checkpoint() {
        // The released `transformer/` index holds 638 tensors.
        let cfg = H3TransformerConfig::default();
        assert_eq!(dit_parameter_keys(&cfg).len(), 638);
    }

    #[test]
    fn every_key_has_a_declared_shape() {
        let cfg = H3TransformerConfig::default();
        for key in dit_parameter_keys(&cfg) {
            assert!(
                dit_parameter_shape(&cfg, &key).is_some(),
                "no shape declared for {key}"
            );
        }
    }

    #[test]
    fn adaln_projection_dominates_the_parameter_count() {
        let cfg = H3TransformerConfig::default();
        let mut adaln = 0usize;
        let mut total = 0usize;
        for key in dit_parameter_keys(&cfg) {
            let n: usize = dit_parameter_shape(&cfg, &key)
                .expect("shape")
                .iter()
                .product();
            total += n;
            if key.contains("adaln_proj") {
                adaln += n;
            }
        }
        // The model card puts ~13B of ~33B in the AdaLN branches.
        assert!(
            (12.0e9..14.0e9).contains(&(adaln as f64)),
            "AdaLN parameters = {adaln}"
        );
        assert!(
            (30.0e9..36.0e9).contains(&(total as f64)),
            "total parameters = {total}"
        );
    }

    #[test]
    fn synthetic_weights_pass_validation() {
        let cfg = tiny();
        let w = synthetic_dit_weights(&cfg, 7);
        validate_dit_weights(&cfg, &w).unwrap();
    }

    #[test]
    fn synthetic_weights_are_deterministic_and_finite() {
        let cfg = tiny();
        let a = synthetic_dit_weights(&cfg, 7);
        let b = synthetic_dit_weights(&cfg, 7);
        let c = synthetic_dit_weights(&cfg, 8);
        let k = "transformer_blocks.0.attn.to_q.weight";
        assert_eq!(a.get(k).unwrap().0, b.get(k).unwrap().0);
        assert_ne!(a.get(k).unwrap().0, c.get(k).unwrap().0);
        assert!(a.get(k).unwrap().0.iter().all(|v| v.is_finite()));
        // Norm gammas initialize to exactly one.
        let g = a.get("transformer_blocks.0.norm1.weight").unwrap().0;
        assert!(g.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn validation_reports_missing_and_mismatched_keys() {
        let cfg = tiny();
        let mut tensors = std::collections::HashMap::new();
        tensors.insert("proj_in.weight".to_string(), (vec![0.0; 4], vec![2, 2]));
        let w = WeightMap::from_tensors(tensors);
        let err = validate_dit_weights(&cfg, &w).unwrap_err().to_string();
        assert!(err.contains("missing"), "unexpected error: {err}");
    }

    #[test]
    fn partition_subdirs() {
        assert_eq!(DitPartition::Base.subdir(), "transformer");
        assert_eq!(DitPartition::Reference.subdir(), "transformer_ref");
    }
}
