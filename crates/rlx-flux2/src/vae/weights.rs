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
use anyhow::{Result, ensure};
use rlx_core::weight_map::WeightMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Conv2dWeight {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_c: usize,
    pub out_c: usize,
}

#[derive(Debug, Clone)]
pub struct GroupNormWeight {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ResnetBlockWeights {
    pub norm1: GroupNormWeight,
    pub conv1: Conv2dWeight,
    pub norm2: GroupNormWeight,
    pub conv2: Conv2dWeight,
    pub shortcut: Option<Conv2dWeight>,
}

#[derive(Debug, Clone)]
pub struct AttnBlockWeights {
    pub norm: GroupNormWeight,
    pub to_q: Conv2dWeight,
    pub to_k: Conv2dWeight,
    pub to_v: Conv2dWeight,
    pub to_out: Conv2dWeight,
}

#[derive(Debug, Clone)]
pub struct UpDecoderBlockWeights {
    pub resnets: Vec<ResnetBlockWeights>,
    pub upsample: Option<Conv2dWeight>,
}

#[derive(Debug, Clone)]
pub struct DownEncoderBlockWeights {
    pub resnets: Vec<ResnetBlockWeights>,
    pub downsample: Option<Conv2dWeight>,
}

#[derive(Debug, Clone)]
pub struct Flux2VaeWeights {
    pub encoder_conv_in: Conv2dWeight,
    pub encoder_down_blocks: Vec<DownEncoderBlockWeights>,
    pub encoder_mid_resnets: Vec<ResnetBlockWeights>,
    pub encoder_mid_attn: Option<AttnBlockWeights>,
    pub encoder_conv_norm_out: GroupNormWeight,
    pub encoder_conv_out: Conv2dWeight,
    pub quant_conv: Conv2dWeight,
    pub post_quant_conv: Option<Conv2dWeight>,
    pub conv_in: Conv2dWeight,
    pub mid_resnets: Vec<ResnetBlockWeights>,
    pub mid_attn: Option<AttnBlockWeights>,
    pub up_blocks: Vec<UpDecoderBlockWeights>,
    pub conv_norm_out: GroupNormWeight,
    pub conv_out: Conv2dWeight,
    pub bn_running_mean: Vec<f32>,
    pub bn_running_var: Vec<f32>,
}

/// Resolve `vae/` next to a transformer weights file or model root.
pub fn resolve_vae_dir(model_path: &Path) -> Option<PathBuf> {
    crate::paths::find_component_dir(model_path, "vae")
}

pub fn load_flux2_vae_weights(path: &Path, cfg: &Flux2VaeConfig) -> Result<Flux2VaeWeights> {
    let wm = if path.is_dir() {
        WeightMap::from_safetensors_dir(path)?
    } else {
        WeightMap::from_file(
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("non-utf8 path"))?,
        )?
    };
    extract_flux2_vae_weights(wm, cfg)
}

pub fn extract_flux2_vae_weights(
    mut wm: WeightMap,
    cfg: &Flux2VaeConfig,
) -> Result<Flux2VaeWeights> {
    let encoder_conv_in = load_conv(&mut wm, "encoder.conv_in.weight", "encoder.conv_in.bias")?;

    let mut encoder_down_blocks = Vec::new();
    let channels: Vec<usize> = cfg.block_out_channels.clone();
    for (i, &out_ch) in channels.iter().enumerate() {
        let in_ch = if i == 0 { channels[0] } else { channels[i - 1] };
        let num_layers = cfg.layers_per_block;
        let mut resnets = Vec::with_capacity(num_layers);
        for j in 0..num_layers {
            resnets.push(load_resnet(
                &mut wm,
                &format!("encoder.down_blocks.{i}.resnets.{j}"),
                cfg.norm_num_groups,
            )?);
            let _ = if j == 0 { in_ch } else { out_ch };
        }
        let downsample = if i + 1 < channels.len() {
            Some(load_conv(
                &mut wm,
                &format!("encoder.down_blocks.{i}.downsamplers.0.conv.weight"),
                &format!("encoder.down_blocks.{i}.downsamplers.0.conv.bias"),
            )?)
        } else {
            None
        };
        encoder_down_blocks.push(DownEncoderBlockWeights {
            resnets,
            downsample,
        });
    }

    let mut encoder_mid_resnets = Vec::new();
    for i in 0..2 {
        encoder_mid_resnets.push(load_resnet(
            &mut wm,
            &format!("encoder.mid_block.resnets.{i}"),
            cfg.norm_num_groups,
        )?);
    }
    let encoder_mid_attn = if cfg.mid_block_add_attention {
        let p = "encoder.mid_block.attentions.0";
        Some(AttnBlockWeights {
            norm: load_gn(&mut wm, &format!("{p}.group_norm"))?,
            to_q: load_conv(
                &mut wm,
                &format!("{p}.to_q.weight"),
                &format!("{p}.to_q.bias"),
            )?,
            to_k: load_conv(
                &mut wm,
                &format!("{p}.to_k.weight"),
                &format!("{p}.to_k.bias"),
            )?,
            to_v: load_conv(
                &mut wm,
                &format!("{p}.to_v.weight"),
                &format!("{p}.to_v.bias"),
            )?,
            to_out: load_conv(
                &mut wm,
                &format!("{p}.to_out.0.weight"),
                &format!("{p}.to_out.0.bias"),
            )?,
        })
    } else {
        None
    };
    let encoder_conv_norm_out = load_gn(&mut wm, "encoder.conv_norm_out")?;
    let encoder_conv_out = load_conv(&mut wm, "encoder.conv_out.weight", "encoder.conv_out.bias")?;
    let quant_conv = load_conv(&mut wm, "quant_conv.weight", "quant_conv.bias")?;

    let post_quant_conv = if cfg.use_post_quant_conv {
        Some(load_conv(
            &mut wm,
            "post_quant_conv.weight",
            "post_quant_conv.bias",
        )?)
    } else {
        None
    };
    let conv_in = load_conv(&mut wm, "decoder.conv_in.weight", "decoder.conv_in.bias")?;

    let mut mid_resnets = Vec::new();
    for i in 0..2 {
        mid_resnets.push(load_resnet(
            &mut wm,
            &format!("decoder.mid_block.resnets.{i}"),
            cfg.norm_num_groups,
        )?);
    }
    let mid_attn = if cfg.mid_block_add_attention {
        let p = "decoder.mid_block.attentions.0";
        Some(AttnBlockWeights {
            norm: load_gn(&mut wm, &format!("{p}.group_norm"))?,
            to_q: load_conv(
                &mut wm,
                &format!("{p}.to_q.weight"),
                &format!("{p}.to_q.bias"),
            )?,
            to_k: load_conv(
                &mut wm,
                &format!("{p}.to_k.weight"),
                &format!("{p}.to_k.bias"),
            )?,
            to_v: load_conv(
                &mut wm,
                &format!("{p}.to_v.weight"),
                &format!("{p}.to_v.bias"),
            )?,
            to_out: load_conv(
                &mut wm,
                &format!("{p}.to_out.0.weight"),
                &format!("{p}.to_out.0.bias"),
            )?,
        })
    } else {
        None
    };

    let channels: Vec<usize> = cfg.block_out_channels.clone();
    let mut up_blocks = Vec::new();
    let reversed: Vec<usize> = channels.iter().copied().rev().collect();
    for (i, &out_ch) in reversed.iter().enumerate() {
        let in_ch = if i == 0 {
            *channels.last().unwrap()
        } else {
            reversed[i - 1]
        };
        let num_layers = cfg.layers_per_block + 1;
        let mut resnets = Vec::with_capacity(num_layers);
        for j in 0..num_layers {
            let block_in = if j == 0 { in_ch } else { out_ch };
            resnets.push(load_resnet(
                &mut wm,
                &format!("decoder.up_blocks.{i}.resnets.{j}"),
                cfg.norm_num_groups,
            )?);
            let _ = block_in;
        }
        let upsample = if i + 1 < reversed.len() {
            Some(load_conv(
                &mut wm,
                &format!("decoder.up_blocks.{i}.upsamplers.0.conv.weight"),
                &format!("decoder.up_blocks.{i}.upsamplers.0.conv.bias"),
            )?)
        } else {
            None
        };
        up_blocks.push(UpDecoderBlockWeights { resnets, upsample });
    }

    let conv_norm_out = load_gn(&mut wm, "decoder.conv_norm_out")?;
    let conv_out = load_conv(&mut wm, "decoder.conv_out.weight", "decoder.conv_out.bias")?;
    let (bn_running_mean, _) = wm.take("bn.running_mean")?;
    let (bn_running_var, _) = wm.take("bn.running_var")?;
    ensure!(
        bn_running_mean.len() == cfg.bn_channels(),
        "bn.running_mean len {} != {}",
        bn_running_mean.len(),
        cfg.bn_channels()
    );

    Ok(Flux2VaeWeights {
        encoder_conv_in,
        encoder_down_blocks,
        encoder_mid_resnets,
        encoder_mid_attn,
        encoder_conv_norm_out,
        encoder_conv_out,
        quant_conv,
        post_quant_conv,
        conv_in,
        mid_resnets,
        mid_attn,
        up_blocks,
        conv_norm_out,
        conv_out,
        bn_running_mean,
        bn_running_var,
    })
}

fn load_conv(wm: &mut WeightMap, w_key: &str, b_key: &str) -> Result<Conv2dWeight> {
    let (data, shape) = wm.take(w_key)?;
    let (bias, _) = wm.take(b_key)?;
    let (out_c, in_c, kh, kw) = match shape.as_slice() {
        [o, i, 3, 3] => (*o, *i, 3, 3),
        [o, i, 1, 1] => (*o, *i, 1, 1),
        [o, i] => (*o, *i, 1, 1),
        _ => anyhow::bail!("conv weight shape {shape:?}"),
    };
    ensure!(kh == kw && (kh == 3 || kh == 1), "expected 1x1 or 3x3 conv");
    let weight = if kh == 3 {
        let mut w = vec![0.0f32; out_c * in_c * 9];
        for oc in 0..out_c {
            for ic in 0..in_c {
                for ky in 0..3 {
                    for kx in 0..3 {
                        w[(oc * in_c + ic) * 9 + ky * 3 + kx] =
                            data[((oc * in_c + ic) * 3 + ky) * 3 + kx];
                    }
                }
            }
        }
        w
    } else {
        data
    };
    Ok(Conv2dWeight {
        weight,
        bias,
        in_c,
        out_c,
    })
}

fn load_gn(wm: &mut WeightMap, prefix: &str) -> Result<GroupNormWeight> {
    let (gamma, _) = wm.take(&format!("{prefix}.weight"))?;
    let (beta, _) = wm.take(&format!("{prefix}.bias"))?;
    Ok(GroupNormWeight { gamma, beta })
}

fn zero_conv3(in_c: usize, out_c: usize) -> Conv2dWeight {
    Conv2dWeight {
        weight: vec![0.0; out_c * in_c * 9],
        bias: vec![0.0; out_c],
        in_c,
        out_c,
    }
}

fn zero_conv1(in_c: usize, out_c: usize) -> Conv2dWeight {
    Conv2dWeight {
        weight: vec![0.0; out_c * in_c],
        bias: vec![0.0; out_c],
        in_c,
        out_c,
    }
}

fn zero_gn(ch: usize) -> GroupNormWeight {
    GroupNormWeight {
        gamma: vec![1.0; ch],
        beta: vec![0.0; ch],
    }
}

fn zero_resnet(in_c: usize, out_c: usize) -> ResnetBlockWeights {
    ResnetBlockWeights {
        norm1: zero_gn(in_c),
        conv1: zero_conv3(in_c, out_c),
        norm2: zero_gn(out_c),
        conv2: zero_conv3(out_c, out_c),
        shortcut: if in_c != out_c {
            Some(zero_conv1(in_c, out_c))
        } else {
            None
        },
    }
}

/// Zero weights for [`Flux2VaeConfig::tiny`] basic tests.
pub fn synthetic_vae_weights(cfg: &Flux2VaeConfig) -> Flux2VaeWeights {
    let last = *cfg.block_out_channels.last().unwrap_or(&8);
    let channels: Vec<usize> = cfg.block_out_channels.clone();
    let reversed: Vec<usize> = channels.iter().copied().rev().collect();
    let mut up_blocks = Vec::new();
    for (i, &out_ch) in reversed.iter().enumerate() {
        let in_ch = if i == 0 { last } else { reversed[i - 1] };
        let num_layers = cfg.layers_per_block + 1;
        let resnets = (0..num_layers)
            .map(|j| {
                let cin = if j == 0 { in_ch } else { out_ch };
                zero_resnet(cin, out_ch)
            })
            .collect();
        let upsample = if i + 1 < reversed.len() {
            Some(zero_conv3(out_ch, out_ch))
        } else {
            None
        };
        up_blocks.push(UpDecoderBlockWeights { resnets, upsample });
    }
    Flux2VaeWeights {
        encoder_conv_in: zero_conv3(cfg.in_channels, channels[0]),
        encoder_down_blocks: {
            let mut blocks = Vec::new();
            for (i, &out_ch) in channels.iter().enumerate() {
                let in_ch = if i == 0 { channels[0] } else { channels[i - 1] };
                let num_layers = cfg.layers_per_block;
                let resnets = (0..num_layers)
                    .map(|j| {
                        let cin = if j == 0 { in_ch } else { out_ch };
                        zero_resnet(cin, out_ch)
                    })
                    .collect();
                let downsample = if i + 1 < channels.len() {
                    Some(zero_conv3(out_ch, out_ch))
                } else {
                    None
                };
                blocks.push(DownEncoderBlockWeights {
                    resnets,
                    downsample,
                });
            }
            blocks
        },
        encoder_mid_resnets: vec![zero_resnet(last, last), zero_resnet(last, last)],
        encoder_mid_attn: None,
        encoder_conv_norm_out: zero_gn(last),
        encoder_conv_out: zero_conv3(last, cfg.latent_channels * 2),
        quant_conv: zero_conv1(cfg.latent_channels * 2, cfg.latent_channels * 2),
        post_quant_conv: cfg
            .use_post_quant_conv
            .then(|| zero_conv1(cfg.latent_channels, cfg.latent_channels)),
        conv_in: zero_conv3(cfg.latent_channels, last),
        mid_resnets: vec![zero_resnet(last, last), zero_resnet(last, last)],
        mid_attn: None,
        up_blocks,
        conv_norm_out: zero_gn(cfg.block_out_channels[0]),
        conv_out: zero_conv3(cfg.block_out_channels[0], cfg.out_channels),
        bn_running_mean: vec![0.0; cfg.bn_channels()],
        bn_running_var: vec![1.0; cfg.bn_channels()],
    }
}

fn load_resnet(wm: &mut WeightMap, prefix: &str, groups: usize) -> Result<ResnetBlockWeights> {
    let norm1 = load_gn(wm, &format!("{prefix}.norm1"))?;
    let conv1 = load_conv(
        wm,
        &format!("{prefix}.conv1.weight"),
        &format!("{prefix}.conv1.bias"),
    )?;
    let norm2 = load_gn(wm, &format!("{prefix}.norm2"))?;
    let conv2 = load_conv(
        wm,
        &format!("{prefix}.conv2.weight"),
        &format!("{prefix}.conv2.bias"),
    )?;
    let shortcut = if wm.has(&format!("{prefix}.conv_shortcut.weight")) {
        Some(load_conv(
            wm,
            &format!("{prefix}.conv_shortcut.weight"),
            &format!("{prefix}.conv_shortcut.bias"),
        )?)
    } else {
        None
    };
    let _ = groups;
    Ok(ResnetBlockWeights {
        norm1,
        conv1,
        norm2,
        conv2,
        shortcut,
    })
}
