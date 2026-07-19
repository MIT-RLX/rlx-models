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

//! Safetensors weight loading for HOCT (`general_v0` Torch key names).
//!
//! Export with `scripts/export_jit_safetensors.py` (clone shared RoPE buffers so
//! each key is unique on disk).

use anyhow::{Context, Result, bail};
use rlx_core::weight_map::WeightMap;
use std::path::Path;

/// Gated attention projections + 3D RoPE buffers for one block.
#[derive(Debug, Clone)]
pub struct AttnWeights {
    pub q_proj_weight: Vec<f32>,
    pub kv_proj_weight: Vec<f32>,
    pub proj_weight: Vec<f32>,
    pub proj_bias: Vec<f32>,
    /// Learnable `log_freq` table `(1, H, 1, 12, 1)`.
    pub log_freq: Vec<f32>,
    /// Householder reflection vectors `(H, Hd)`.
    pub reflect_vec: Vec<f32>,
    /// Identity `(Hd, Hd)` used with the reflection.
    pub eye: Vec<f32>,
}

/// Two-layer GELU MLP weights.
#[derive(Debug, Clone)]
pub struct MlpWeights {
    pub fc1_weight: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc2_weight: Vec<f32>,
    pub fc2_bias: Vec<f32>,
}

/// One transformer block (node or edge): dual RMSNorm, gated attn, MLP, optional dist bias.
#[derive(Debug, Clone)]
pub struct BlockWeights {
    pub norm1_x_weight: Vec<f32>,
    pub norm1_y_weight: Vec<f32>,
    pub norm2_weight: Vec<f32>,
    pub attn: AttnWeights,
    pub mlp: MlpWeights,
    /// Edge-only: softplus scale per head.
    pub dist_scaling: Vec<f32>,
    /// Edge-only: per-head direction multiplier for line-to-line distance.
    pub dist_scaling_head_direction: Vec<f32>,
}

/// All tensors for `general_v0` (input projs, 4+4 blocks, score head).
#[derive(Debug, Clone)]
pub struct HoctWeights {
    pub input_proj_weight: Vec<f32>,
    pub input_proj_bias: Vec<f32>,
    pub edge_input_proj_weight: Vec<f32>,
    pub edge_input_proj_bias: Vec<f32>,
    pub edge_gatherer: MlpWeights,
    pub node_blocks: Vec<BlockWeights>,
    pub edge_blocks: Vec<BlockWeights>,
    pub head_norm_weight: Vec<f32>,
    pub head_norm_bias: Vec<f32>,
    pub head_weight: Vec<f32>,
    pub head_bias: Vec<f32>,
}

fn take_vec(wm: &mut WeightMap, key: &str) -> Result<Vec<f32>> {
    let (data, _shape) = wm
        .take(key)
        .with_context(|| format!("missing weight `{key}`"))?;
    Ok(data)
}

fn load_attn(wm: &mut WeightMap, prefix: &str) -> Result<AttnWeights> {
    Ok(AttnWeights {
        q_proj_weight: take_vec(wm, &format!("{prefix}.attn.q_proj.weight"))?,
        kv_proj_weight: take_vec(wm, &format!("{prefix}.attn.kv_proj.weight"))?,
        proj_weight: take_vec(wm, &format!("{prefix}.attn.proj.weight"))?,
        proj_bias: take_vec(wm, &format!("{prefix}.attn.proj.bias"))?,
        log_freq: take_vec(wm, &format!("{prefix}.attn.pos_enc.log_freq"))?,
        reflect_vec: take_vec(wm, &format!("{prefix}.attn.pos_enc.reflect_vec"))?,
        eye: take_vec(wm, &format!("{prefix}.attn.pos_enc.eye"))?,
    })
}

fn load_mlp(wm: &mut WeightMap, prefix: &str) -> Result<MlpWeights> {
    Ok(MlpWeights {
        fc1_weight: take_vec(wm, &format!("{prefix}.fc1.weight"))?,
        fc1_bias: take_vec(wm, &format!("{prefix}.fc1.bias"))?,
        fc2_weight: take_vec(wm, &format!("{prefix}.fc2.weight"))?,
        fc2_bias: take_vec(wm, &format!("{prefix}.fc2.bias"))?,
    })
}

fn load_block(wm: &mut WeightMap, prefix: &str, edge: bool) -> Result<BlockWeights> {
    let dist_scaling = if edge {
        take_vec(wm, &format!("{prefix}.dist_scaling"))?
    } else {
        Vec::new()
    };
    let dist_scaling_head_direction = if edge {
        take_vec(wm, &format!("{prefix}.dist_scaling_head_direction"))?
    } else {
        Vec::new()
    };
    // Module-level pos_enc duplicates attn.pos_enc; drop unused keys.
    let _ = wm.take(&format!("{prefix}.pos_enc.log_freq"));
    let _ = wm.take(&format!("{prefix}.pos_enc.reflect_vec"));
    let _ = wm.take(&format!("{prefix}.pos_enc.eye"));
    Ok(BlockWeights {
        norm1_x_weight: take_vec(wm, &format!("{prefix}.norm1_x.weight"))?,
        norm1_y_weight: take_vec(wm, &format!("{prefix}.norm1_y.weight"))?,
        norm2_weight: take_vec(wm, &format!("{prefix}.norm2.weight"))?,
        attn: load_attn(wm, prefix)?,
        mlp: load_mlp(wm, &format!("{prefix}.mlp"))?,
        dist_scaling,
        dist_scaling_head_direction,
    })
}

/// Load and type-check all `general_v0` tensors from a safetensors file.
///
/// Fails if required keys are missing or unexpected leftovers remain.
pub fn load_hoct_weights(path: impl AsRef<Path>) -> Result<HoctWeights> {
    let path = path.as_ref();
    let snapshot = WeightMap::snapshot_from_path(path.to_str().unwrap())
        .with_context(|| format!("read weights {}", path.display()))?;
    let mut wm = WeightMap::from_tensors(snapshot);

    let mut node_blocks = Vec::with_capacity(4);
    for i in 0..4 {
        node_blocks.push(load_block(&mut wm, &format!("node_blocks.{i}"), false)?);
    }
    let mut edge_blocks = Vec::with_capacity(4);
    for i in 0..4 {
        edge_blocks.push(load_block(&mut wm, &format!("edge_blocks.{i}"), true)?);
    }

    let weights = HoctWeights {
        input_proj_weight: take_vec(&mut wm, "input_proj.weight")?,
        input_proj_bias: take_vec(&mut wm, "input_proj.bias")?,
        edge_input_proj_weight: take_vec(&mut wm, "edge_input_proj.weight")?,
        edge_input_proj_bias: take_vec(&mut wm, "edge_input_proj.bias")?,
        edge_gatherer: load_mlp(&mut wm, "edge_gatherer.mlp")?,
        node_blocks,
        edge_blocks,
        head_norm_weight: take_vec(&mut wm, "head_norm.weight")?,
        head_norm_bias: take_vec(&mut wm, "head_norm.bias")?,
        head_weight: take_vec(&mut wm, "head.weight")?,
        head_bias: take_vec(&mut wm, "head.bias")?,
    };

    if !wm.is_empty() {
        let remaining: Vec<_> = wm.keys().take(8).map(|s| s.to_string()).collect();
        bail!("unexpected weight keys after load (showing up to 8): {remaining:?}");
    }
    Ok(weights)
}
