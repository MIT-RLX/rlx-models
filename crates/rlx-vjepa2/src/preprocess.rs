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

//! Host-side 3-D patch embedding and video normalization for V-JEPA2.

use super::config::{IMAGENET_MEAN, IMAGENET_STD, Vjepa2Config};
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;

#[derive(Clone)]
pub struct Vjepa2PatchEmbedWeights {
    /// Conv3d kernel `[embed_dim * in_chans * t * ph * pw]` in PyTorch order
    /// `[oc, ic, kt, kh, kw]` flattened.
    pub proj_w: Vec<f32>,
    pub proj_b: Vec<f32>,
    pub embed_dim: usize,
    pub in_chans: usize,
    pub tubelet_size: usize,
    pub patch_size: usize,
}

pub fn extract_patch_embed_weights(
    weights: &mut WeightMap,
    cfg: &Vjepa2Config,
) -> Result<Vjepa2PatchEmbedWeights> {
    let e = cfg.hidden_size;
    let c = cfg.in_chans;
    let ts = cfg.tubelet_size;
    let ps = cfg.patch_size;
    let expected = vec![e, c, ts, ps, ps];

    let w_keys = [
        "encoder.embeddings.patch_embeddings.proj.weight",
        "patch_embed.proj.weight",
    ];
    let b_keys = [
        "encoder.embeddings.patch_embeddings.proj.bias",
        "patch_embed.proj.bias",
    ];

    let (proj_w, shape) = take_first(weights, &w_keys)?;
    ensure!(
        shape == expected,
        "patch embed weight expected {expected:?}, got {shape:?}"
    );
    let (proj_b, bshape) = take_first(weights, &b_keys)?;
    ensure!(bshape == vec![e], "patch embed bias expected [{e}]");

    Ok(Vjepa2PatchEmbedWeights {
        proj_w,
        proj_b,
        embed_dim: e,
        in_chans: c,
        tubelet_size: ts,
        patch_size: ps,
    })
}

/// Normalize RGB u8 frames to NCTHW f32 in `[0,1]` then ImageNet stats.
/// `frames` is `[num_frames, crop, crop, 3]` HWC u8 row-major.
pub fn normalize_video_hwc(frames: &[u8], num_frames: usize, crop: usize) -> Vec<f32> {
    let plane = crop * crop;
    let mut out = vec![0f32; 3 * num_frames * plane];
    for t in 0..num_frames {
        for y in 0..crop {
            for x in 0..crop {
                let src = (t * plane + y * crop + x) * 3;
                for c in 0..3 {
                    let v = frames[src + c] as f32 / 255.0;
                    let norm = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                    out[c * num_frames * plane + t * plane + y * crop + x] = norm;
                }
            }
        }
    }
    out
}

/// 3-D conv patch embedding: input `[C, T, H, W]` → tokens `[seq, embed_dim]`.
pub fn conv3d_patch_embed(
    patch: &Vjepa2PatchEmbedWeights,
    video_ncthw: &[f32],
    frames: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>> {
    let c = patch.in_chans;
    let ts = patch.tubelet_size;
    let ps = patch.patch_size;
    let e = patch.embed_dim;
    ensure!(
        video_ncthw.len() == c * frames * height * width,
        "video tensor size mismatch"
    );
    ensure!(frames.is_multiple_of(ts) && height.is_multiple_of(ps) && width.is_multiple_of(ps));

    let t_out = frames / ts;
    let h_out = height / ps;
    let w_out = width / ps;
    let seq = t_out * h_out * w_out;
    let mut tokens = vec![0f32; seq * e];

    let plane = height * width;
    let vol = frames * plane;

    for ot in 0..t_out {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let tok = (ot * h_out + oh) * w_out + ow;
                for oc in 0..e {
                    let mut acc = patch.proj_b[oc];
                    for ic in 0..c {
                        for kt in 0..ts {
                            for kh in 0..ps {
                                for kw in 0..ps {
                                    let it = ot * ts + kt;
                                    let ih = oh * ps + kh;
                                    let iw = ow * ps + kw;
                                    let in_idx = ic * vol + it * plane + ih * width + iw;
                                    let w_idx = oc * (c * ts * ps * ps)
                                        + ic * (ts * ps * ps)
                                        + kt * (ps * ps)
                                        + kh * ps
                                        + kw;
                                    acc += patch.proj_w[w_idx] * video_ncthw[in_idx];
                                }
                            }
                        }
                    }
                    tokens[tok * e + oc] = acc;
                }
            }
        }
    }
    Ok(tokens)
}

fn take_first(weights: &mut WeightMap, keys: &[&str]) -> Result<(Vec<f32>, Vec<usize>)> {
    for key in keys {
        if weights.has(key) {
            return weights.take(key);
        }
    }
    anyhow::bail!("none of the patch-embed keys found: {keys:?}")
}
