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
use super::forward::resnet_forward;
use super::layers::{
    conv2d_1x1, conv2d_3x3_pad1, downsample_conv2d, group_norm, silu, spatial_attention,
};
use super::weights::{DownEncoderBlockWeights, Flux2VaeWeights};
use anyhow::Result;

fn down_block_forward(
    x: &[f32],
    block: &DownEncoderBlockWeights,
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
    if let Some(down) = &block.downsample {
        let (downed, h2, w2) = downsample_conv2d(&cur, in_c, h, w, &down.weight, &down.bias);
        cur = downed;
        out_h = h2;
        out_w = w2;
    }
    Ok((cur, in_c, out_h, out_w))
}

/// VAE encode RGB planar `[-1,1]` `[batch, 3, H, W]` → latent `[batch, latent_channels, h, w]`.
pub fn flux2_vae_encode(
    weights: &Flux2VaeWeights,
    cfg: &Flux2VaeConfig,
    rgb: &[f32],
    batch: usize,
    h: usize,
    w: usize,
) -> Result<Vec<f32>> {
    let mut x = rgb.to_vec();
    let eps = 1e-6f32;
    let groups = cfg.norm_num_groups;

    x = conv2d_3x3_pad1(
        &x,
        cfg.in_channels,
        weights.encoder_conv_in.out_c,
        h,
        w,
        &weights.encoder_conv_in.weight,
        &weights.encoder_conv_in.bias,
    );
    let mut channels = weights.encoder_conv_in.out_c;
    let mut cur_h = h;
    let mut cur_w = w;

    for block in &weights.encoder_down_blocks {
        let (cur, c, hh, ww) =
            down_block_forward(&x, block, batch, channels, cur_h, cur_w, eps, groups)?;
        x = cur;
        channels = c;
        cur_h = hh;
        cur_w = ww;
    }

    for resnet in &weights.encoder_mid_resnets {
        x = resnet_forward(&x, resnet, batch, channels, cur_h, cur_w, eps, groups)?;
        channels = resnet.conv2.out_c;
    }
    if let Some(attn) = &weights.encoder_mid_attn {
        x = spatial_attention(
            &x,
            batch,
            channels,
            cur_h,
            cur_w,
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

    x = group_norm(
        &x,
        batch,
        channels,
        cur_h,
        cur_w,
        groups,
        &weights.encoder_conv_norm_out.gamma,
        &weights.encoder_conv_norm_out.beta,
        eps,
    );
    let mut tmp = vec![0.0f32; x.len()];
    silu(&x, &mut tmp);
    x = conv2d_3x3_pad1(
        &tmp,
        channels,
        weights.encoder_conv_out.out_c,
        cur_h,
        cur_w,
        &weights.encoder_conv_out.weight,
        &weights.encoder_conv_out.bias,
    );

    // quant_conv → take mean half
    let qc = &weights.quant_conv;
    let q = conv2d_1x1(&x, qc.in_c, qc.out_c, cur_h, cur_w, &qc.weight, &qc.bias);
    let mean_c = qc.out_c / 2;
    let spatial = cur_h * cur_w;
    let mut latent = vec![0.0f32; batch * mean_c * spatial];
    for b in 0..batch {
        for c in 0..mean_c {
            for i in 0..spatial {
                latent[b * mean_c * spatial + c * spatial + i] =
                    q[b * qc.out_c * spatial + c * spatial + i];
            }
        }
    }
    if cfg.scaling_factor != 1.0 || cfg.shift_factor != 0.0 {
        for v in &mut latent {
            *v = (*v - cfg.shift_factor) * cfg.scaling_factor;
        }
    }
    Ok(latent)
}
