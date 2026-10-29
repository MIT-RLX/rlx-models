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

//! Native CPU forward for the V-JEPA2 video encoder.

use super::config::Vjepa2Config;
use super::layers::block_forward;
use super::preprocess::conv3d_patch_embed;
use super::weights::Vjepa2EncoderWeights;
use anyhow::Result;
use rlx_core::host_kernels::layer_norm;

pub struct Vjepa2EncoderOutput {
    pub tokens: Vec<f32>,
    pub seq: usize,
    pub hidden: usize,
}

/// Encode a pre-normalized video tensor `[C, T, H, W]`.
pub fn encode_video_native(
    weights: &Vjepa2EncoderWeights,
    cfg: &Vjepa2Config,
    video_ncthw: &[f32],
    batch: usize,
) -> Result<Vjepa2EncoderOutput> {
    encode_video_native_ext(weights, cfg, video_ncthw, batch, None)
}

/// Like [`encode_video_native`], but stop after transformer block `stop_after_block`
/// (inclusive). Skips final layer norm unless stopping at the last block.
pub fn encode_video_native_ext(
    weights: &Vjepa2EncoderWeights,
    cfg: &Vjepa2Config,
    video_ncthw: &[f32],
    batch: usize,
    stop_after_block: Option<usize>,
) -> Result<Vjepa2EncoderOutput> {
    let e = cfg.hidden_size;
    let frames = cfg.frames_per_clip;
    let crop = cfg.crop_size;
    let seq = cfg.num_patches();
    let head_dim = cfg.head_dim();
    let nh = cfg.num_attention_heads;
    let (d_dim, h_dim, w_dim) = cfg.rope_segment_dims();
    let grid_t = cfg.grid_temporal();
    let grid_h = cfg.grid_spatial();
    let grid_w = cfg.grid_spatial();
    let eps = cfg.layer_norm_eps as f32;

    let mut x = conv3d_patch_embed(&weights.patch, video_ncthw, frames, crop, crop)?;
    if batch > 1 {
        let per = x.len();
        let mut batched = Vec::with_capacity(per * batch);
        for _ in 0..batch {
            batched.extend_from_slice(&x);
        }
        x = batched;
    }

    let last_block = weights.blocks.len().saturating_sub(1);
    for (i, block) in weights.blocks.iter().enumerate() {
        block_forward(
            &mut x, block, batch, seq, e, nh, head_dim, d_dim, h_dim, w_dim, grid_t, grid_h,
            grid_w, eps, None,
        )?;
        if stop_after_block == Some(i) {
            break;
        }
    }

    if stop_after_block.is_none() || stop_after_block == Some(last_block) {
        x = layer_norm(&x, &weights.norm_w, &weights.norm_b, e, eps)?;
    }

    Ok(Vjepa2EncoderOutput {
        tokens: x,
        seq,
        hidden: e,
    })
}

/// One forward pass returning the hidden state after each block in `taps`, in
/// `taps` order. Each entry equals `encode_video_native_ext(.., Some(tap))`,
/// so a tap at the last block includes the final layer norm. Runs only up to
/// the deepest tap instead of once per tap.
pub fn encode_video_native_taps(
    weights: &Vjepa2EncoderWeights,
    cfg: &Vjepa2Config,
    video_ncthw: &[f32],
    batch: usize,
    taps: &[usize],
) -> Result<Vec<Vjepa2EncoderOutput>> {
    let n_blocks = weights.blocks.len();
    if let Some(&bad) = taps.iter().find(|&&t| t >= n_blocks) {
        anyhow::bail!("tap {bad} out of range (encoder has {n_blocks} blocks)");
    }
    // Block `i`'s output is hidden state `i + 1`; the last block also gets the
    // final layer norm, like `encode_video_native_ext`.
    let wanted: Vec<usize> = taps.iter().map(|t| t + 1).collect();
    let (seq, hidden) = (cfg.num_patches(), cfg.hidden_size);
    let eps = cfg.layer_norm_eps as f32;
    let mut captured: Vec<Option<Vec<f32>>> = vec![None; taps.len()];
    let mut failed = None;
    encode_video_native_hidden_states(weights, cfg, video_ncthw, batch, &wanted, |hs, tokens| {
        let state = if hs == n_blocks {
            match layer_norm(tokens, &weights.norm_w, &weights.norm_b, hidden, eps) {
                Ok(t) => t,
                Err(e) => {
                    failed = Some(e);
                    return;
                }
            }
        } else {
            tokens.to_vec()
        };
        for (slot, _) in captured.iter_mut().zip(&wanted).filter(|(_, w)| **w == hs) {
            *slot = Some(state.clone());
        }
    })?;
    if let Some(e) = failed {
        return Err(e);
    }
    Ok(captured
        .into_iter()
        .map(|tokens| Vjepa2EncoderOutput {
            tokens: tokens.expect("every tap is visited"),
            seq,
            hidden,
        })
        .collect())
}

/// Visit `VJEPA2Model(output_hidden_states=True).hidden_states` in one
/// forward pass, numbered as transformers 5.x records them: state `0` is the
/// patch embedding and state `k` (`1..=n_blocks`) the raw output of block
/// `k - 1`. The last state is NOT layer-normed: 5.x records layer outputs with
/// hooks and `tie_last_hidden_states=False`, whereas 4.x replaced it with the
/// final layer norm (cosine ~0.81 apart on a real clip). `f(k, tokens)` is
/// called once per distinct requested `k`, in increasing `k`, with
/// `[batch * seq, hidden]` tokens; blocks past the deepest request are not run.
pub fn encode_video_native_hidden_states(
    weights: &Vjepa2EncoderWeights,
    cfg: &Vjepa2Config,
    video_ncthw: &[f32],
    batch: usize,
    wanted: &[usize],
    mut f: impl FnMut(usize, &[f32]),
) -> Result<()> {
    let e = cfg.hidden_size;
    let frames = cfg.frames_per_clip;
    let crop = cfg.crop_size;
    let seq = cfg.num_patches();
    let head_dim = cfg.head_dim();
    let nh = cfg.num_attention_heads;
    let (d_dim, h_dim, w_dim) = cfg.rope_segment_dims();
    let grid_t = cfg.grid_temporal();
    let grid_h = cfg.grid_spatial();
    let grid_w = cfg.grid_spatial();
    let eps = cfg.layer_norm_eps as f32;

    let n_blocks = weights.blocks.len();
    if let Some(&bad) = wanted.iter().find(|&&k| k > n_blocks) {
        anyhow::bail!("hidden state {bad} out of range (0..={n_blocks})");
    }
    let Some(&deepest) = wanted.iter().max() else {
        return Ok(());
    };

    let mut x = conv3d_patch_embed(&weights.patch, video_ncthw, frames, crop, crop)?;
    if batch > 1 {
        let per = x.len();
        let mut batched = Vec::with_capacity(per * batch);
        for _ in 0..batch {
            batched.extend_from_slice(&x);
        }
        x = batched;
    }
    if wanted.contains(&0) {
        f(0, &x);
    }
    for (i, block) in weights.blocks.iter().enumerate().take(deepest) {
        block_forward(
            &mut x, block, batch, seq, e, nh, head_dim, d_dim, h_dim, w_dim, grid_t, grid_h,
            grid_w, eps, None,
        )?;
        let k = i + 1;
        if wanted.contains(&k) {
            f(k, &x);
        }
    }
    Ok(())
}
