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

//! Seed encoder weights from the shipped decoder (public checkpoints omit encoder).

use crate::codec::layout::{
    decoder_conv_block_for_encoder_downsample, decoder_transformer_block_for_encoder_stage,
    encoder_execution_plan,
};
use crate::config::CodecArgs;
use crate::load::PREFIX_CODEC;
use anyhow::{Context, Result, bail};
use std::collections::HashMap;

/// Copy decoder tensors into encoder slots for training bootstrap / reference encode.
pub fn seed_encoder_from_decoder(
    codec: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    cfg: &CodecArgs,
) -> Result<()> {
    let prefix = PREFIX_CODEC;
    let plan = encoder_execution_plan(cfg)?;
    let n_stages = cfg.encoder_transformer_lengths().len();
    let _ = &plan;

    for stage in 0..n_stages {
        let dec_t = decoder_transformer_block_for_encoder_stage(stage);
        copy_transformer_block(codec, prefix, stage, dec_t)?;
    }

    for stage in 0..n_stages {
        let enc_conv_idx = stage * 2 + 1;
        let dec_conv_idx = decoder_conv_block_for_encoder_downsample(stage);
        if stage + 1 == n_stages {
            copy_latent_proj_conv(codec, prefix, enc_conv_idx, 0)?;
        } else {
            copy_conv_block(codec, prefix, enc_conv_idx, dec_conv_idx)?;
        }
    }

    copy_input_proj_from_output_proj(codec, prefix, cfg)?;

    Ok(())
}

fn copy_transformer_block(
    codec: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    enc_stage: usize,
    dec_block: usize,
) -> Result<()> {
    let enc_block = enc_stage * 2;
    let enc_pfx = format!("{prefix}encoder_blocks.{enc_block}.layers.");
    let dec_pfx = format!("{prefix}decoder_blocks.{dec_block}.layers.");
    let keys: Vec<String> = codec
        .keys()
        .filter(|k| k.starts_with(&dec_pfx))
        .cloned()
        .collect();
    if keys.is_empty() {
        bail!("decoder transformer {dec_pfx}* not found for encoder stage {enc_stage}");
    }
    for dk in keys {
        let suffix = dk.strip_prefix(&dec_pfx).unwrap();
        let ek = format!("{enc_pfx}{suffix}");
        let entry = codec.get(&dk).context("decoder tensor")?.clone();
        codec.insert(ek, entry);
    }
    Ok(())
}

fn copy_conv_block(
    codec: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    enc_block: usize,
    dec_block: usize,
) -> Result<()> {
    let enc_pfx = format!("{prefix}encoder_blocks.{enc_block}.conv");
    let dec_pfx = format!("{prefix}decoder_blocks.{dec_block}.conv");
    for suffix in [
        ".weight",
        ".parametrizations.weight.original0",
        ".parametrizations.weight.original1",
    ] {
        let dk = format!("{dec_pfx}{suffix}");
        if let Some(v) = codec.get(&dk).cloned() {
            codec.insert(format!("{enc_pfx}{suffix}"), v);
        }
    }
    Ok(())
}

/// Decoder block-0 projects latent→dim; encoder last block projects dim→latent (flip in/out).
fn copy_latent_proj_conv(
    codec: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    enc_block: usize,
    dec_block: usize,
) -> Result<()> {
    let enc_pfx = format!("{prefix}encoder_blocks.{enc_block}.conv");
    let dec_pfx = format!("{prefix}decoder_blocks.{dec_block}.conv");
    if let Some((data, shape)) = codec.get(&format!("{dec_pfx}.weight")).cloned() {
        let flipped = flip_conv_in_out(&data, &shape)?;
        codec.insert(
            format!("{enc_pfx}.weight"),
            (flipped, vec![shape[1], shape[0], shape[2]]),
        );
        return Ok(());
    }
    let g_key = format!("{dec_pfx}.parametrizations.weight.original0");
    let v_key = format!("{dec_pfx}.parametrizations.weight.original1");
    let (g, gs) = codec.get(&g_key).context("decoder latent conv g")?.clone();
    let (v, vs) = codec.get(&v_key).context("decoder latent conv v")?.clone();
    ensure_conv_rank3(&vs)?;
    let w = reconstruct_param_conv(&g, &v, &vs)?;
    let flipped = flip_conv_in_out(&w, &vs)?;
    codec.insert(
        format!("{enc_pfx}.weight"),
        (flipped, vec![vs[1], vs[0], vs[2]]),
    );
    let _ = gs;
    Ok(())
}

fn copy_input_proj_from_output_proj(
    codec: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    cfg: &CodecArgs,
) -> Result<()> {
    let out_pfx = format!("{prefix}output_proj.conv");
    let in_key = format!("{prefix}input_proj.conv.weight");
    if let Some((data, shape)) = codec.get(&format!("{out_pfx}.weight")).cloned() {
        let flipped = flip_conv_in_out(&data, &shape)?;
        codec.insert(
            in_key,
            (
                flipped,
                vec![
                    cfg.dim,
                    cfg.pretransform_patch_size,
                    cfg.patch_proj_kernel_size,
                ],
            ),
        );
        return Ok(());
    }
    let g_key = format!("{out_pfx}.parametrizations.weight.original0");
    let v_key = format!("{out_pfx}.parametrizations.weight.original1");
    let (g, gs) = codec.get(&g_key).context("output_proj conv g")?.clone();
    let (v, vs) = codec.get(&v_key).context("output_proj conv v")?.clone();
    ensure_conv_rank3(&vs)?;
    let w = reconstruct_param_conv(&g, &v, &vs)?;
    let flipped = flip_conv_in_out(&w, &vs)?;
    codec.insert(
        in_key,
        (
            flipped,
            vec![
                cfg.dim,
                cfg.pretransform_patch_size,
                cfg.patch_proj_kernel_size,
            ],
        ),
    );
    let _ = gs;
    Ok(())
}

fn reconstruct_param_conv(g: &[f32], v: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    ensure_conv_rank3(shape)?;
    let (out_ch, fan_in, _) = (shape[0], shape[1] * shape[2], shape[2]);
    let mut w = vec![0f32; v.len()];
    for oc in 0..out_ch {
        let mut norm_sq = 0f32;
        for i in 0..fan_in {
            let idx = oc * fan_in + i;
            norm_sq += v[idx] * v[idx];
        }
        let scale = g[oc] / (norm_sq.sqrt() + 1e-12);
        for i in 0..fan_in {
            let idx = oc * fan_in + i;
            w[idx] = v[idx] * scale;
        }
    }
    Ok(w)
}

fn flip_conv_in_out(data: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    ensure_conv_rank3(shape)?;
    let (oc, ic, k) = (shape[0], shape[1], shape[2]);
    let mut out = vec![0f32; data.len()];
    for o in 0..oc {
        for i in 0..ic {
            for ki in 0..k {
                out[i * oc * k + o * k + ki] = data[o * ic * k + i * k + ki];
            }
        }
    }
    Ok(out)
}

fn ensure_conv_rank3(shape: &[usize]) -> Result<()> {
    if shape.len() != 3 {
        bail!("expected rank-3 conv shape, got {shape:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_conv_swaps_in_out() {
        let shape = vec![4, 2, 3];
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let out = flip_conv_in_out(&data, &shape).unwrap();
        assert_eq!(out.len(), 24);
        assert_eq!(out[0], data[0]);
        assert_eq!(out[3], data[6]);
    }
}
