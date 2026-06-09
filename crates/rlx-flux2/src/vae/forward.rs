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

use super::config::Flux2VaeConfig;
use super::layers::{
    conv2d_1x1, conv2d_3x3_pad1, group_norm, silu, spatial_attention, upsample_nearest_2x,
};
use super::weights::{Conv2dWeight, Flux2VaeWeights, ResnetBlockWeights, UpDecoderBlockWeights};
use crate::latent_ops::{denorm_patchified_latents, unpack_latents_with_ids, unpatchify_latents};
use anyhow::Result;

pub(crate) fn resnet_forward(
    x: &[f32],
    b: &ResnetBlockWeights,
    batch: usize,
    in_c: usize,
    h: usize,
    w: usize,
    eps: f32,
    groups: usize,
) -> Result<Vec<f32>> {
    let mut residual = x.to_vec();
    let mut h1 = group_norm(
        x,
        batch,
        in_c,
        h,
        w,
        groups,
        &b.norm1.gamma,
        &b.norm1.beta,
        eps,
    );
    let mut tmp = vec![0.0f32; h1.len()];
    silu(&h1, &mut tmp);
    h1 = conv2d_3x3_pad1(
        &tmp,
        in_c,
        b.conv1.out_c,
        h,
        w,
        &b.conv1.weight,
        &b.conv1.bias,
    );
    h1 = group_norm(
        &h1,
        batch,
        b.conv1.out_c,
        h,
        w,
        groups,
        &b.norm2.gamma,
        &b.norm2.beta,
        eps,
    );
    tmp.resize(h1.len(), 0.0);
    silu(&h1, &mut tmp);
    h1 = conv2d_3x3_pad1(
        &tmp,
        b.conv1.out_c,
        b.conv2.out_c,
        h,
        w,
        &b.conv2.weight,
        &b.conv2.bias,
    );
    if let Some(sc) = &b.shortcut {
        residual = conv2d_1x1(x, in_c, sc.out_c, h, w, &sc.weight, &sc.bias);
    }
    for i in 0..h1.len() {
        h1[i] += residual[i];
    }
    Ok(h1)
}

fn up_block_forward(
    x: &[f32],
    block: &UpDecoderBlockWeights,
    batch: usize,
    mut in_c: usize,
    h: usize,
    w: usize,
    eps: f32,
    groups: usize,
) -> Result<(Vec<f32>, usize, usize, usize)> {
    let mut cur = x.to_vec();
    for resnet in &block.resnets {
        let out_c = resnet.conv2.out_c;
        cur = resnet_forward(&cur, resnet, batch, in_c, h, w, eps, groups)?;
        in_c = out_c;
    }
    let mut out_h = h;
    let mut out_w = w;
    if let Some(up) = &block.upsample {
        let (uped, h2, w2) = upsample_nearest_2x(&cur, in_c, h, w);
        cur = conv2d_3x3_pad1(&uped, in_c, up.out_c, h2, w2, &up.weight, &up.bias);
        out_h = h2;
        out_w = w2;
        in_c = up.out_c;
    }
    Ok((cur, in_c, out_h, out_w))
}

/// Decode VAE latents `[batch, latent_channels, H, W]` → RGB `[batch, 3, H', W']`.
pub fn flux2_vae_decode(
    weights: &Flux2VaeWeights,
    cfg: &Flux2VaeConfig,
    latents: &[f32],
    batch: usize,
    mut h: usize,
    mut w: usize,
) -> Result<Vec<f32>> {
    let mut x = latents.to_vec();
    let lc = cfg.latent_channels;
    ensure_latent_shape(&x, batch, lc, h, w)?;

    x = apply_conv(&x, weights.post_quant_conv.as_ref(), batch, lc, h, w)?;
    let mut channels = lc;
    x = apply_conv(&x, Some(&weights.conv_in), batch, channels, h, w)?;
    channels = weights.conv_in.out_c;

    let eps = 1e-6f32;
    let groups = cfg.norm_num_groups;
    for resnet in &weights.mid_resnets {
        x = resnet_forward(&x, resnet, batch, channels, h, w, eps, groups)?;
        channels = resnet.conv2.out_c;
    }
    if let Some(attn) = &weights.mid_attn {
        x = spatial_attention(
            &x,
            batch,
            channels,
            h,
            w,
            &attn.to_q.weight,
            &attn.to_q.bias,
            &attn.to_k.weight,
            &attn.to_k.bias,
            &attn.to_v.weight,
            &attn.to_v.bias,
            &attn.to_out.weight,
            &attn.to_out.bias,
            &attn.norm.gamma,
            &attn.norm.beta,
            groups,
            eps,
        );
    }

    for block in &weights.up_blocks {
        let (cur, c, hh, ww) = up_block_forward(&x, block, batch, channels, h, w, eps, groups)?;
        x = cur;
        channels = c;
        h = hh;
        w = ww;
    }

    x = group_norm(
        &x,
        batch,
        channels,
        h,
        w,
        groups,
        &weights.conv_norm_out.gamma,
        &weights.conv_norm_out.beta,
        eps,
    );
    let mut tmp = vec![0.0f32; x.len()];
    silu(&x, &mut tmp);
    x = conv2d_3x3_pad1(
        &tmp,
        channels,
        weights.conv_out.out_c,
        h,
        w,
        &weights.conv_out.weight,
        &weights.conv_out.bias,
    );
    Ok(x)
}

fn apply_conv(
    x: &[f32],
    conv: Option<&Conv2dWeight>,
    batch: usize,
    in_c: usize,
    h: usize,
    w: usize,
) -> Result<Vec<f32>> {
    let Some(c) = conv else {
        return Ok(x.to_vec());
    };
    let mut out = Vec::new();
    for b in 0..batch {
        let plane = &x[b * in_c * h * w..(b + 1) * in_c * h * w];
        let decoded = if c.weight.len() == c.out_c * c.in_c {
            conv2d_1x1(plane, c.in_c, c.out_c, h, w, &c.weight, &c.bias)
        } else {
            conv2d_3x3_pad1(plane, c.in_c, c.out_c, h, w, &c.weight, &c.bias)
        };
        out.extend_from_slice(&decoded);
    }
    Ok(out)
}

fn ensure_latent_shape(x: &[f32], batch: usize, c: usize, h: usize, w: usize) -> Result<()> {
    anyhow::ensure!(x.len() == batch * c * h * w, "latent shape mismatch");
    Ok(())
}

/// Full post-denoise decode: packed transformer latents → 8-bit RGB planar `[batch, 3, H, W]`.
pub fn flux2_decode_packed_latents(
    vae_weights: &Flux2VaeWeights,
    vae_cfg: &Flux2VaeConfig,
    packed: &[f32],
    img_ids: &[f32],
    batch: usize,
    img_seq: usize,
    packed_channels: usize,
    latent_h: usize,
    latent_w: usize,
) -> Result<Vec<f32>> {
    let spatial = unpack_latents_with_ids(
        packed,
        img_ids,
        batch,
        img_seq,
        packed_channels,
        latent_h,
        latent_w,
    )?;
    let denorm = denorm_patchified_latents(
        &spatial,
        &vae_weights.bn_running_mean,
        &vae_weights.bn_running_var,
        vae_cfg.batch_norm_eps,
    );
    let unpatch = unpatchify_latents(&denorm, batch, packed_channels, latent_h, latent_w);
    let h2 = latent_h * 2;
    let w2 = latent_w * 2;
    let mut latents = unpatch;
    if vae_cfg.scaling_factor != 1.0 || vae_cfg.shift_factor != 0.0 {
        for v in &mut latents {
            *v = *v / vae_cfg.scaling_factor + vae_cfg.shift_factor;
        }
    }
    flux2_vae_decode(vae_weights, vae_cfg, &latents, batch, h2, w2)
}

/// Planar RGB `[-1,1]` → interleaved `u8` HWC for PNG.
pub fn flux2_rgb_to_u8(rgb: &[f32], batch: usize, channels: usize, h: usize, w: usize) -> Vec<u8> {
    let mut out = vec![0u8; batch * h * w * channels];
    for b in 0..batch {
        for y in 0..h {
            for x in 0..w {
                for c in 0..channels.min(3) {
                    let v = rgb[b * channels * h * w + c * h * w + y * w + x];
                    let byte = ((v * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
                    out[(b * h * w + y * w + x) * channels + c] = byte;
                }
            }
        }
    }
    out
}
