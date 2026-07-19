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

//! Dense sparse-structure decoder (`SparseStructureDecoder`,
//! `trellis2/models/sparse_structure_vae.py`), reused from
//! `microsoft/TRELLIS-image-large` (`ss_dec_conv3d_16l8`).
//!
//! Decodes the `[8, 16³]` structure latent to a `[1, 64³]` occupancy logit
//! volume, then thresholds + max-pools it down to the requested resolution to
//! seed the active-voxel coordinates for the shape-SLat stage.
//!
//! ```text
//!   input_layer conv3d(8→512)  → middle 2×ResBlock(512)
//!   → 2×ResBlock(512) → up(512→128) → 2×ResBlock(128) → up(128→32) → 2×ResBlock(32)
//!   → ChannelLayerNorm → SiLU → conv3d(32→1)
//! ```

use crate::config::SparseStructureVaeArgs;
use crate::conv3d::{Vol, channel_layer_norm, conv3d_same, pixel_shuffle_3d, silu};
use anyhow::{Context, Result};
use rlx_core::weight_map::WeightMap;

fn get<'a>(wm: &'a WeightMap, key: &str) -> Result<&'a [f32]> {
    wm.get(key)
        .map(|(d, _)| d)
        .with_context(|| format!("missing weight {key}"))
}

/// `ResBlock3d`: `x + conv2(silu(norm2(conv1(silu(norm1(x))))))` (skip is
/// identity when in/out channels match, which is always true here).
fn res_block(wm: &WeightMap, prefix: &str, x: &Vol, ch: usize) -> Result<Vol> {
    let n1w = get(wm, &format!("{prefix}.norm1.weight"))?;
    let n1b = get(wm, &format!("{prefix}.norm1.bias"))?;
    let n2w = get(wm, &format!("{prefix}.norm2.weight"))?;
    let n2b = get(wm, &format!("{prefix}.norm2.bias"))?;
    let c1w = get(wm, &format!("{prefix}.conv1.weight"))?;
    let c1b = get(wm, &format!("{prefix}.conv1.bias"))?;
    let c2w = get(wm, &format!("{prefix}.conv2.weight"))?;
    let c2b = get(wm, &format!("{prefix}.conv2.bias"))?;

    let mut h = channel_layer_norm(x, n1w, n1b);
    silu(&mut h);
    let h = conv3d_same(&h, c1w, c1b, ch);
    let mut h = channel_layer_norm(&h, n2w, n2b);
    silu(&mut h);
    let h = conv3d_same(&h, c2w, c2b, ch);
    let mut out = h;
    for i in 0..out.data.len() {
        out.data[i] += x.data[i];
    }
    Ok(out)
}

/// Decode `[8, 16³]` latent → `[1, 64³]` occupancy logits.
pub fn decode_occupancy(cfg: &SparseStructureVaeArgs, wm: &WeightMap, latent: &Vol) -> Result<Vol> {
    let channels = &cfg.channels; // [512, 128, 32]
    // input_layer conv3d(latent_channels -> channels[0])
    let mut h = conv3d_same(
        latent,
        get(wm, "input_layer.weight")?,
        get(wm, "input_layer.bias")?,
        channels[0],
    );
    // middle blocks
    for i in 0..cfg.num_res_blocks_middle {
        h = res_block(wm, &format!("middle_block.{i}"), &h, channels[0])?;
    }
    // decoder stages
    let mut blk = 0usize;
    for (i, &ch) in channels.iter().enumerate() {
        for _ in 0..cfg.num_res_blocks {
            h = res_block(wm, &format!("blocks.{blk}"), &h, ch)?;
            blk += 1;
        }
        if i < channels.len() - 1 {
            // UpsampleBlock3d: conv(ch -> next*8) then pixel_shuffle_3d(2)
            let next = channels[i + 1];
            let cw = get(wm, &format!("blocks.{blk}.conv.weight"))?;
            let cb = get(wm, &format!("blocks.{blk}.conv.bias"))?;
            let up = conv3d_same(&h, cw, cb, next * 8);
            h = pixel_shuffle_3d(&up, 2);
            blk += 1;
        }
    }
    // out_layer: norm -> silu -> conv(ch_last -> out_channels)
    let mut o = channel_layer_norm(
        &h,
        get(wm, "out_layer.0.weight")?,
        get(wm, "out_layer.0.bias")?,
    );
    silu(&mut o);
    let out = conv3d_same(
        &o,
        get(wm, "out_layer.2.weight")?,
        get(wm, "out_layer.2.bias")?,
        cfg.out_channels,
    );
    Ok(out)
}

/// Threshold occupancy logits (`>0`) and, if the decode resolution differs from
/// `target_res`, OR-pool down by the integer ratio (matching
/// `max_pool3d(occ, ratio) > 0.5`). Returns active voxel coords `[batch, x, y, z]`
/// (batch is always 0 for a single sample), row-major over `(x,y,z)`.
pub fn occupancy_to_coords(occ: &Vol, target_res: usize) -> Vec<[i32; 4]> {
    let full = occ.d; // 64 (assumes cubic)
    let ratio = (full / target_res).max(1);
    let mut coords = Vec::new();
    for x in 0..target_res {
        for y in 0..target_res {
            for z in 0..target_res {
                // OR over the ratio³ block
                let mut any = false;
                'blk: for dx in 0..ratio {
                    for dy in 0..ratio {
                        for dz in 0..ratio {
                            let (px, py, pz) = (x * ratio + dx, y * ratio + dy, z * ratio + dz);
                            if occ.data[occ.idx(0, px, py, pz)] > 0.0 {
                                any = true;
                                break 'blk;
                            }
                        }
                    }
                }
                if any {
                    coords.push([0, x as i32, y as i32, z as i32]);
                }
            }
        }
    }
    coords
}
