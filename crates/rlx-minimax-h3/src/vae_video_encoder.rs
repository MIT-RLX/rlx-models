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

//! The video VAE's 3-D **causal CNN** encoder.
//!
//! This is the half that turns real pixels into conditioning anchors, so it is
//! what `i2va`, `fl2va` and `ref2va` need. The decoder is a ViT and lives in
//! [`crate::vae_video`]; the two halves share nothing but the latent statistics.
//!
//! Three conventions carry the whole thing:
//!
//! - **Causal time.** Every convolution prepends `kernel_t - 1` zero frames and
//!   appends none, so a latent frame never sees the future. Spatial padding is
//!   symmetric and *reflective*.
//! - **Frame-isolated GroupNorm.** The temporal axis is folded into the batch
//!   axis before normalizing, so statistics never mix across frames. Normalizing
//!   over the clip instead would make a single conditioning image encode
//!   differently from the same image inside a video.
//! - **Asymmetric downsampling.** A spatial stride of 2 is preceded by a
//!   bottom/right reflect pad of 1 and the convolution itself carries no spatial
//!   padding, so the output is exactly `ceil(size / 2)`.
//!
//! A **single frame skips the temporal path entirely** ([`H3VideoEncoder::encode`]).
//! Padding one image up to `clip_length` by repetition would run the temporal
//! convolutions over 17 copies of it and return several latent frames instead of
//! one — not the conditioning H3 was trained with.

use crate::config::H3VideoVaeConfig;
use anyhow::{Context, Result, bail, ensure};
use rlx_core::weight_map::WeightMap;

/// A `(C, T, H, W)` volume, contiguous in that order.
#[derive(Debug, Clone, PartialEq)]
pub struct Volume {
    pub channels: usize,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl Volume {
    pub fn new(channels: usize, frames: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            frames,
            height,
            width,
            data: vec![0.0; channels * frames * height * width],
        }
    }

    pub fn from_data(
        channels: usize,
        frames: usize,
        height: usize,
        width: usize,
        data: Vec<f32>,
    ) -> Result<Self> {
        ensure!(
            data.len() == channels * frames * height * width,
            "volume data of {} values does not match ({channels}, {frames}, {height}, {width})",
            data.len()
        );
        Ok(Self {
            channels,
            frames,
            height,
            width,
            data,
        })
    }

    #[must_use]
    pub fn voxels(&self) -> usize {
        self.frames * self.height * self.width
    }

    #[inline]
    fn at(&self, c: usize, t: usize, h: usize, w: usize) -> f32 {
        self.data[((c * self.frames + t) * self.height + h) * self.width + w]
    }

    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }
}

/// A 3-D convolution kernel, `[out, in, kt, kh, kw]`.
#[derive(Debug, Clone)]
pub struct Conv3dWeights {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub out_channels: usize,
    pub in_channels: usize,
    pub kt: usize,
    pub kh: usize,
    pub kw: usize,
}

impl Conv3dWeights {
    pub fn new(weight: Vec<f32>, shape: &[usize], bias: Option<Vec<f32>>) -> Result<Self> {
        ensure!(
            shape.len() == 5,
            "a Conv3d kernel must be rank 5, got {shape:?}"
        );
        ensure!(
            weight.len() == shape.iter().product::<usize>(),
            "kernel data ({}) does not match shape {shape:?}",
            weight.len()
        );
        if let Some(b) = &bias {
            ensure!(
                b.len() == shape[0],
                "bias of {} does not match {} output channels",
                b.len(),
                shape[0]
            );
        }
        Ok(Self {
            weight,
            bias,
            out_channels: shape[0],
            in_channels: shape[1],
            kt: shape[2],
            kh: shape[3],
            kw: shape[4],
        })
    }
}

/// How one convolution pads its input.
#[derive(Debug, Clone, Copy)]
pub struct CausalPad {
    /// Symmetric reflective padding on height and width.
    pub spatial: usize,
    /// Zero frames prepended on the time axis; nothing is appended.
    pub temporal: usize,
    pub stride_t: usize,
    pub stride_hw: usize,
}

impl CausalPad {
    #[must_use]
    pub fn same() -> Self {
        Self {
            spatial: 1,
            temporal: 2,
            stride_t: 1,
            stride_hw: 1,
        }
    }

    #[must_use]
    pub fn pointwise() -> Self {
        Self {
            spatial: 0,
            temporal: 0,
            stride_t: 1,
            stride_hw: 1,
        }
    }
}

/// Reflect-pad height and width, zero-pad the *front* of the time axis.
///
/// Reflection mirrors about the edge sample without repeating it, matching
/// `F.pad(..., mode="reflect")`.
pub fn pad_causal(x: &Volume, spatial: usize, temporal: usize) -> Result<Volume> {
    if spatial > 0 {
        ensure!(
            spatial < x.height && spatial < x.width,
            "reflect padding of {spatial} needs more than {}x{} samples",
            x.height,
            x.width
        );
    }
    let (h2, w2) = (x.height + 2 * spatial, x.width + 2 * spatial);
    let t2 = x.frames + temporal;
    let mut out = Volume::new(x.channels, t2, h2, w2);
    let reflect = |i: isize, n: isize| -> usize {
        let mut i = i;
        if i < 0 {
            i = -i;
        }
        if i >= n {
            i = 2 * (n - 1) - i;
        }
        i.clamp(0, n - 1) as usize
    };
    for c in 0..x.channels {
        for t in 0..t2 {
            // Prepended frames stay zero — causal padding is constant, not
            // reflective.
            if t < temporal {
                continue;
            }
            let ts = t - temporal;
            for h in 0..h2 {
                let hs = reflect(h as isize - spatial as isize, x.height as isize);
                for w in 0..w2 {
                    let ws = reflect(w as isize - spatial as isize, x.width as isize);
                    out.data[((c * t2 + t) * h2 + h) * w2 + w] = x.at(c, ts, hs, ws);
                }
            }
        }
    }
    Ok(out)
}

/// Reflect-pad the bottom and right edges by one sample.
///
/// The stride-2 downsampler's asymmetric pad, which is what makes the output
/// exactly `ceil(size / 2)`.
pub fn pad_bottom_right(x: &Volume) -> Volume {
    let (h2, w2) = (x.height + 1, x.width + 1);
    let mut out = Volume::new(x.channels, x.frames, h2, w2);
    for c in 0..x.channels {
        for t in 0..x.frames {
            for h in 0..h2 {
                // Reflection about the last row: index h - 2 for the extra row.
                let hs = if h < x.height {
                    h
                } else {
                    x.height.saturating_sub(2).min(x.height - 1)
                };
                for w in 0..w2 {
                    let ws = if w < x.width {
                        w
                    } else {
                        x.width.saturating_sub(2).min(x.width - 1)
                    };
                    out.data[((c * x.frames + t) * h2 + h) * w2 + w] = x.at(c, t, hs, ws);
                }
            }
        }
    }
    out
}

/// Convolve an **already padded** volume — no further padding is applied.
pub fn conv3d(x: &Volume, w: &Conv3dWeights, stride_t: usize, stride_hw: usize) -> Result<Volume> {
    ensure!(
        stride_t > 0 && stride_hw > 0,
        "conv3d strides must be positive"
    );
    ensure!(
        x.channels == w.in_channels,
        "conv3d: kernel expects {} input channels, volume has {}",
        w.in_channels,
        x.channels
    );
    if x.frames < w.kt || x.height < w.kh || x.width < w.kw {
        bail!(
            "conv3d: volume ({}, {}, {}) is smaller than the kernel ({}, {}, {})",
            x.frames,
            x.height,
            x.width,
            w.kt,
            w.kh,
            w.kw
        );
    }
    let ot = (x.frames - w.kt) / stride_t + 1;
    let oh = (x.height - w.kh) / stride_hw + 1;
    let ow = (x.width - w.kw) / stride_hw + 1;
    let mut out = Volume::new(w.out_channels, ot, oh, ow);

    for oc in 0..w.out_channels {
        let bias = w.bias.as_ref().map_or(0.0, |b| b[oc]);
        for t in 0..ot {
            for h in 0..oh {
                for wi in 0..ow {
                    let mut acc = bias;
                    for ic in 0..w.in_channels {
                        let kbase = ((oc * w.in_channels + ic) * w.kt) * w.kh * w.kw;
                        for dt in 0..w.kt {
                            let st = t * stride_t + dt;
                            for dh in 0..w.kh {
                                let sh = h * stride_hw + dh;
                                let xrow = ((ic * x.frames + st) * x.height + sh) * x.width
                                    + wi * stride_hw;
                                let krow = kbase + (dt * w.kh + dh) * w.kw;
                                for dw in 0..w.kw {
                                    acc += w.weight[krow + dw] * x.data[xrow + dw];
                                }
                            }
                        }
                    }
                    out.data[((oc * ot + t) * oh + h) * ow + wi] = acc;
                }
            }
        }
    }
    Ok(out)
}

/// Pad and convolve in one step.
pub fn causal_conv3d(x: &Volume, w: &Conv3dWeights, pad: CausalPad) -> Result<Volume> {
    let padded = if pad.spatial > 0 || pad.temporal > 0 {
        pad_causal(x, pad.spatial, pad.temporal)?
    } else {
        x.clone()
    };
    conv3d(&padded, w, pad.stride_t, pad.stride_hw)
}

/// GroupNorm applied to each frame in isolation.
///
/// The temporal axis is folded into the batch axis, so statistics are taken over
/// `(channels_in_group, height, width)` of a single frame.
pub fn group_norm_per_frame(
    x: &mut Volume,
    groups: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<()> {
    ensure!(groups > 0, "group count must be positive");
    ensure!(
        x.channels.is_multiple_of(groups),
        "{} channels do not split into {groups} groups",
        x.channels
    );
    ensure!(
        gamma.len() == x.channels && beta.len() == x.channels,
        "group norm affine covers {}/{} of {} channels",
        gamma.len(),
        beta.len(),
        x.channels
    );
    let per_group = x.channels / groups;
    let hw = x.height * x.width;
    let n = (per_group * hw) as f32;

    for t in 0..x.frames {
        for g in 0..groups {
            let (mut sum, mut sq) = (0.0f64, 0.0f64);
            for k in 0..per_group {
                let c = g * per_group + k;
                let base = ((c * x.frames + t) * x.height) * x.width;
                for v in &x.data[base..base + hw] {
                    sum += *v as f64;
                    sq += (*v as f64) * (*v as f64);
                }
            }
            let mean = sum / n as f64;
            let var = (sq / n as f64 - mean * mean).max(0.0);
            let inv = 1.0 / (var + eps as f64).sqrt();
            for k in 0..per_group {
                let c = g * per_group + k;
                let (gm, bt) = (gamma[c], beta[c]);
                let base = ((c * x.frames + t) * x.height) * x.width;
                for v in &mut x.data[base..base + hw] {
                    *v = (((*v as f64 - mean) * inv) as f32) * gm + bt;
                }
            }
        }
    }
    Ok(())
}

/// SiLU, in place.
pub fn silu(x: &mut Volume) {
    for v in x.data.iter_mut() {
        *v /= 1.0 + (-*v).exp();
    }
}

/// A mean / log-variance pair over the latent grid.
#[derive(Debug, Clone)]
pub struct DiagonalGaussian {
    pub mean: Volume,
    pub logvar: Volume,
}

impl DiagonalGaussian {
    /// Split a `2 * latent_channels` volume into its two halves.
    pub fn split(x: &Volume) -> Result<Self> {
        ensure!(
            x.channels.is_multiple_of(2),
            "a diagonal Gaussian needs an even channel count, got {}",
            x.channels
        );
        let c = x.channels / 2;
        let per = x.voxels();
        let mean = Volume::from_data(c, x.frames, x.height, x.width, x.data[..c * per].to_vec())?;
        let logvar = Volume::from_data(c, x.frames, x.height, x.width, x.data[c * per..].to_vec())?;
        Ok(Self { mean, logvar })
    }

    /// The distribution's mode, which is its mean.
    ///
    /// H3 encodes its conditioning anchors under a fixed seed rather than
    /// sampling freshly per request, so the mode is what most callers want.
    #[must_use]
    pub fn mode(&self) -> &Volume {
        &self.mean
    }

    /// Draw a sample, `mean + exp(0.5 * clamp(logvar)) * noise`.
    pub fn sample(&self, seed: u64) -> Result<Volume> {
        let noise = crate::pipeline::noise(self.mean.data.len(), seed);
        let mut out = self.mean.clone();
        for ((v, &lv), &n) in out.data.iter_mut().zip(&self.logvar.data).zip(&noise) {
            *v += (0.5 * lv.clamp(-30.0, 20.0)).exp() * n;
        }
        Ok(out)
    }
}

/// One residual block of the encoder.
#[derive(Debug, Clone)]
struct ResnetBlock3d {
    norm1_g: Vec<f32>,
    norm1_b: Vec<f32>,
    conv1: Conv3dWeights,
    norm2_g: Vec<f32>,
    norm2_b: Vec<f32>,
    conv2: Conv3dWeights,
    shortcut: Option<Conv3dWeights>,
}

impl ResnetBlock3d {
    fn forward(&self, x: &Volume, groups: usize, eps: f32) -> Result<Volume> {
        let mut h = x.clone();
        group_norm_per_frame(&mut h, groups, &self.norm1_g, &self.norm1_b, eps)?;
        silu(&mut h);
        let mut h = causal_conv3d(&h, &self.conv1, CausalPad::same())?;
        group_norm_per_frame(&mut h, groups, &self.norm2_g, &self.norm2_b, eps)?;
        silu(&mut h);
        let h = causal_conv3d(&h, &self.conv2, CausalPad::same())?;

        let residual = match &self.shortcut {
            Some(w) => causal_conv3d(x, w, CausalPad::pointwise())?,
            None => x.clone(),
        };
        ensure!(
            residual.data.len() == h.data.len(),
            "resnet shortcut produced {} values against {} from the block body",
            residual.data.len(),
            h.data.len()
        );
        let mut out = h;
        for (v, r) in out.data.iter_mut().zip(&residual.data) {
            *v += r;
        }
        Ok(out)
    }
}

/// One level of the encoder: `layers_per_block` resnets, then an optional
/// strided downsample.
#[derive(Debug, Clone)]
struct DownBlock3d {
    resnets: Vec<ResnetBlock3d>,
    downsampler: Option<(Conv3dWeights, usize, usize)>,
}

impl DownBlock3d {
    fn forward(&self, x: &Volume, groups: usize, eps: f32) -> Result<Volume> {
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h, groups, eps)?;
        }
        if let Some((w, stride_t, stride_hw)) = &self.downsampler {
            // A stride of 2 gets the asymmetric bottom/right pad first.
            let padded = if *stride_hw == 2 {
                pad_bottom_right(&h)
            } else {
                h
            };
            h = causal_conv3d(
                &padded,
                w,
                CausalPad {
                    spatial: 0,
                    temporal: 2,
                    stride_t: *stride_t,
                    stride_hw: *stride_hw,
                },
            )?;
        }
        Ok(h)
    }
}

/// The video VAE's causal 3-D CNN encoder.
pub struct H3VideoEncoder {
    cfg: H3VideoVaeConfig,
    conv_in: Conv3dWeights,
    down_blocks: Vec<DownBlock3d>,
    norm_out_g: Vec<f32>,
    norm_out_b: Vec<f32>,
    conv_out: Conv3dWeights,
    quant_conv: Conv3dWeights,
}

impl H3VideoEncoder {
    /// Load the encoder from a checkpoint weight map.
    pub fn load(cfg: &H3VideoVaeConfig, w: &WeightMap) -> Result<Self> {
        cfg.validate()?;
        let conv_in = conv(w, "encoder.conv_in")?;
        let block_out = &cfg.block_out_channels;
        let mut down_blocks = Vec::with_capacity(block_out.len());

        for i in 0..block_out.len() {
            let p = format!("encoder.down_blocks.{i}");
            let mut resnets = Vec::with_capacity(cfg.layers_per_block);
            for r in 0..cfg.layers_per_block {
                let rp = format!("{p}.resnets.{r}");
                let shortcut_key = format!("{rp}.conv_shortcut.weight");
                let shortcut = if w.has(&shortcut_key) {
                    Some(conv(w, &format!("{rp}.conv_shortcut"))?)
                } else {
                    None
                };
                resnets.push(ResnetBlock3d {
                    norm1_g: vec_of(w, &format!("{rp}.norm1.weight"))?,
                    norm1_b: vec_of(w, &format!("{rp}.norm1.bias"))?,
                    conv1: conv(w, &format!("{rp}.conv1"))?,
                    norm2_g: vec_of(w, &format!("{rp}.norm2.weight"))?,
                    norm2_b: vec_of(w, &format!("{rp}.norm2.bias"))?,
                    conv2: conv(w, &format!("{rp}.conv2"))?,
                    shortcut,
                });
            }
            let ts = cfg.temporal_downsample_factors[i];
            let ss = cfg.spatial_downsample_factors[i];
            let downsampler = if ts * ss > 1 {
                Some((conv(w, &format!("{p}.downsamplers.0.conv"))?, ts, ss))
            } else {
                None
            };
            down_blocks.push(DownBlock3d {
                resnets,
                downsampler,
            });
        }

        Ok(Self {
            cfg: cfg.clone(),
            conv_in,
            down_blocks,
            norm_out_g: vec_of(w, "encoder.norm_out.weight")?,
            norm_out_b: vec_of(w, "encoder.norm_out.bias")?,
            conv_out: conv(w, "encoder.conv_out")?,
            quant_conv: conv(w, "quant_conv")?,
        })
    }

    #[must_use]
    pub fn config(&self) -> &H3VideoVaeConfig {
        &self.cfg
    }

    /// Encode pixels to a latent posterior.
    ///
    /// `video` is `(3, T, H, W)` in ImageNet-normalized range — see
    /// [`normalize_pixels`]. A single frame runs the spatial path alone.
    pub fn encode(&self, video: &Volume) -> Result<DiagonalGaussian> {
        ensure!(
            video.channels == self.cfg.in_channels,
            "encoder expects {} input channels, got {}",
            self.cfg.in_channels,
            video.channels
        );
        let ratio = self.cfg.spatial_compression();
        ensure!(
            video.height.is_multiple_of(ratio) && video.width.is_multiple_of(ratio),
            "pixel size {}x{} is not a multiple of the {ratio}x spatial compression",
            video.height,
            video.width
        );

        let groups = self.cfg.norm_num_groups;
        let eps = self.cfg.norm_eps;

        let mut h = causal_conv3d(video, &self.conv_in, CausalPad::same())?;
        for block in &self.down_blocks {
            h = block.forward(&h, groups, eps)?;
        }
        group_norm_per_frame(&mut h, groups, &self.norm_out_g, &self.norm_out_b, eps)?;
        silu(&mut h);
        let h = causal_conv3d(&h, &self.conv_out, CausalPad::same())?;
        let h = causal_conv3d(&h, &self.quant_conv, CausalPad::pointwise())?;
        DiagonalGaussian::split(&h)
    }

    /// Latent frames this encoder produces for a pixel frame count.
    ///
    /// A single frame stays a single latent frame; longer clips compress by
    /// [`H3VideoVaeConfig::temporal_compression`].
    #[must_use]
    pub fn latent_frames(&self, frames: usize) -> usize {
        if frames <= 1 {
            1
        } else {
            frames.div_ceil(self.cfg.temporal_compression())
        }
    }
}

/// ImageNet-normalize pixels in `[0, 1]` into the encoder's input range.
pub fn normalize_pixels(pixels: &mut Volume) -> Result<()> {
    ensure!(
        pixels.channels == 3,
        "pixel normalization expects 3 channels, got {}",
        pixels.channels
    );
    let per = pixels.voxels();
    for c in 0..3 {
        let (m, s) = (crate::layout::PIXEL_MEAN[c], crate::layout::PIXEL_STD[c]);
        let inv = 1.0 / s;
        for v in &mut pixels.data[c * per..(c + 1) * per] {
            *v = (*v - m) * inv;
        }
    }
    Ok(())
}

fn get<'a>(w: &'a WeightMap, key: &str) -> Result<(&'a [f32], &'a [usize])> {
    w.get(key)
        .ok_or_else(|| anyhow::anyhow!("MiniMax-H3 video VAE: missing weight `{key}`"))
}

fn vec_of(w: &WeightMap, key: &str) -> Result<Vec<f32>> {
    Ok(get(w, key)?.0.to_vec())
}

fn conv(w: &WeightMap, prefix: &str) -> Result<Conv3dWeights> {
    let (weight, shape) = get(w, &format!("{prefix}.weight"))?;
    let bias_key = format!("{prefix}.bias");
    let bias = if w.has(&bias_key) {
        Some(vec_of(w, &bias_key)?)
    } else {
        None
    };
    Conv3dWeights::new(weight.to_vec(), shape, bias)
        .with_context(|| format!("MiniMax-H3 video VAE: build conv `{prefix}`"))
}

/// Every parameter key the CNN encoder reads.
#[must_use]
pub fn encoder_parameter_keys(cfg: &H3VideoVaeConfig) -> Vec<String> {
    let mut keys = vec![
        "encoder.conv_in.weight".to_string(),
        "encoder.conv_in.bias".to_string(),
        "encoder.norm_out.weight".to_string(),
        "encoder.norm_out.bias".to_string(),
        "encoder.conv_out.weight".to_string(),
        "encoder.conv_out.bias".to_string(),
        "quant_conv.weight".to_string(),
        "quant_conv.bias".to_string(),
    ];
    let block_in: Vec<usize> = std::iter::once(cfg.block_out_channels[0])
        .chain(
            cfg.block_out_channels[..cfg.block_out_channels.len() - 1]
                .iter()
                .copied(),
        )
        .collect();
    for i in 0..cfg.block_out_channels.len() {
        let p = format!("encoder.down_blocks.{i}");
        for r in 0..cfg.layers_per_block {
            let rp = format!("{p}.resnets.{r}");
            for s in [
                "norm1.weight",
                "norm1.bias",
                "conv1.weight",
                "conv1.bias",
                "norm2.weight",
                "norm2.bias",
                "conv2.weight",
                "conv2.bias",
            ] {
                keys.push(format!("{rp}.{s}"));
            }
            // Only the first resnet of a level can change the channel count.
            let c_in = if r == 0 {
                block_in[i]
            } else {
                cfg.block_out_channels[i]
            };
            if c_in != cfg.block_out_channels[i] {
                keys.push(format!("{rp}.conv_shortcut.weight"));
                keys.push(format!("{rp}.conv_shortcut.bias"));
            }
        }
        if cfg.temporal_downsample_factors[i] * cfg.spatial_downsample_factors[i] > 1 {
            keys.push(format!("{p}.downsamplers.0.conv.weight"));
            keys.push(format!("{p}.downsamplers.0.conv.bias"));
        }
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> H3VideoVaeConfig {
        serde_json::from_str(
            r#"{"in_channels":3,"out_channels":3,"latent_channels":24,
                "block_out_channels":[128,256,256,512,512,1024],"layers_per_block":2,
                "spatial_downsample_factors":[2,2,2,2,1,1],
                "temporal_downsample_factors":[1,2,2,1,1,1],
                "norm_num_groups":32,"norm_eps":1e-06,"spatial_padding_mode":"reflect",
                "decoder_num_layers":36,"decoder_num_attention_heads":32,
                "decoder_attention_head_dim":64,"decoder_num_register_tokens":4,
                "decoder_ffn_mult":4,"decoder_rope_theta":100.0,
                "decoder_rope_dim_ratio":0.75,"decoder_norm_eps":1e-05,
                "clip_length":17,"token_drop":3}"#,
        )
        .unwrap()
    }

    #[test]
    fn causal_padding_prepends_zeros_only() {
        let x = Volume::from_data(1, 2, 2, 2, (0..8).map(|i| (i + 1) as f32).collect()).unwrap();
        let p = pad_causal(&x, 0, 2).unwrap();
        assert_eq!(p.frames, 4);
        // The two prepended frames are zero...
        assert!(p.data[..2 * 4].iter().all(|&v| v == 0.0));
        // ...and nothing is appended, so the tail is the original last frame.
        assert_eq!(&p.data[3 * 4..4 * 4], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn spatial_padding_reflects() {
        // A 3x3 block of 1..9. Row [1, 2, 3] reflects to [2, 1, 2, 3, 2].
        let x = Volume::from_data(1, 1, 3, 3, (1..=9).map(|i| i as f32).collect()).unwrap();
        let p = pad_causal(&x, 1, 0).unwrap();
        assert_eq!((p.height, p.width), (5, 5));
        // The row that was originally the first one, now at index 1.
        assert_eq!(p.data[5..10].to_vec(), vec![2.0, 1.0, 2.0, 3.0, 2.0]);
        // The padded top row mirrors the *second* source row, not the first.
        assert_eq!(p.data[0..5].to_vec(), vec![5.0, 4.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn reflect_padding_needs_room_to_mirror() {
        // PyTorch's reflect mode has the same requirement.
        let x = Volume::from_data(1, 1, 1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        assert!(pad_causal(&x, 1, 0).is_err());
    }

    #[test]
    fn bottom_right_pad_adds_exactly_one() {
        let x = Volume::from_data(1, 1, 2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let p = pad_bottom_right(&x);
        assert_eq!((p.height, p.width), (3, 3));
        // The original block is preserved in the top-left corner.
        assert_eq!(p.data[0], 1.0);
        assert_eq!(p.data[1], 2.0);
        assert_eq!(p.data[3], 3.0);
        assert_eq!(p.data[4], 4.0);
    }

    #[test]
    fn conv3d_identity_kernel_passes_through() {
        let x = Volume::from_data(1, 2, 2, 2, (0..8).map(|i| i as f32).collect()).unwrap();
        let w = Conv3dWeights::new(vec![1.0], &[1, 1, 1, 1, 1], None).unwrap();
        let y = conv3d(&x, &w, 1, 1).unwrap();
        assert_eq!(y.data, x.data);
    }

    #[test]
    fn causal_conv_preserves_the_frame_count() {
        // k=3 with temporal padding 2 and no appended frames keeps T.
        let x = Volume::from_data(1, 3, 4, 4, vec![0.5; 48]).unwrap();
        let w = Conv3dWeights::new(vec![0.0; 27], &[1, 1, 3, 3, 3], Some(vec![1.0])).unwrap();
        let y = causal_conv3d(&x, &w, CausalPad::same()).unwrap();
        assert_eq!((y.frames, y.height, y.width), (3, 4, 4));
        assert!(y.data.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn stride_two_downsample_halves_the_canvas() {
        let x = Volume::from_data(1, 2, 8, 8, vec![0.25; 128]).unwrap();
        let w = Conv3dWeights::new(vec![0.0; 27], &[1, 1, 3, 3, 3], Some(vec![0.0])).unwrap();
        let padded = pad_bottom_right(&x);
        let y = causal_conv3d(
            &padded,
            &w,
            CausalPad {
                spatial: 0,
                temporal: 2,
                stride_t: 1,
                stride_hw: 2,
            },
        )
        .unwrap();
        assert_eq!(
            (y.height, y.width),
            (4, 4),
            "stride 2 must halve the canvas"
        );
        assert_eq!(y.frames, 2, "a temporal stride of 1 keeps the frame count");
    }

    #[test]
    fn temporal_stride_two_halves_the_frames() {
        let x = Volume::from_data(1, 4, 4, 4, vec![0.25; 64]).unwrap();
        let w = Conv3dWeights::new(vec![0.0; 27], &[1, 1, 3, 3, 3], Some(vec![0.0])).unwrap();
        let padded = pad_bottom_right(&x);
        let y = causal_conv3d(
            &padded,
            &w,
            CausalPad {
                spatial: 0,
                temporal: 2,
                stride_t: 2,
                stride_hw: 2,
            },
        )
        .unwrap();
        assert_eq!(y.frames, 2);
    }

    #[test]
    fn group_norm_is_isolated_per_frame() {
        // Two frames with very different scales must both normalize to zero
        // mean; a clip-wide norm would leave one of them off-centre.
        let mut x = Volume::new(2, 2, 2, 2);
        for i in 0..8 {
            x.data[i] = 1.0 + i as f32; // channel 0
            x.data[8 + i] = 100.0 * (i as f32 + 1.0); // channel 1
        }
        group_norm_per_frame(&mut x, 2, &[1.0, 1.0], &[0.0, 0.0], 1e-6).unwrap();
        for t in 0..2 {
            for c in 0..2 {
                let base = ((c * 2 + t) * 2) * 2;
                let m: f32 = x.data[base..base + 4].iter().sum::<f32>() / 4.0;
                assert!(m.abs() < 1e-4, "frame {t} channel {c} mean = {m}");
            }
        }
    }

    #[test]
    fn group_norm_applies_its_affine() {
        let mut x = Volume::new(2, 1, 2, 2);
        x.data
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = i as f32);
        group_norm_per_frame(&mut x, 2, &[2.0, 3.0], &[5.0, -5.0], 1e-6).unwrap();
        // Beta shifts each channel's mean to exactly beta.
        let m0: f32 = x.data[0..4].iter().sum::<f32>() / 4.0;
        let m1: f32 = x.data[4..8].iter().sum::<f32>() / 4.0;
        assert!((m0 - 5.0).abs() < 1e-4, "{m0}");
        assert!((m1 + 5.0).abs() < 1e-4, "{m1}");
    }

    #[test]
    fn silu_matches_the_closed_form() {
        let mut x = Volume::from_data(1, 1, 1, 3, vec![-1.0, 0.0, 2.0]).unwrap();
        silu(&mut x);
        assert!((x.data[0] - (-1.0 / (1.0 + 1.0f32.exp()))).abs() < 1e-6);
        assert_eq!(x.data[1], 0.0);
        assert!((x.data[2] - (2.0 / (1.0 + (-2.0f32).exp()))).abs() < 1e-6);
    }

    #[test]
    fn diagonal_gaussian_splits_in_half() {
        let x = Volume::from_data(1, 1, 1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let x = Volume::from_data(4, 1, 1, 1, x.data).unwrap();
        let g = DiagonalGaussian::split(&x).unwrap();
        assert_eq!(g.mean.channels, 2);
        assert_eq!(g.mean.data, vec![1.0, 2.0]);
        assert_eq!(g.logvar.data, vec![3.0, 4.0]);
        assert_eq!(g.mode().data, vec![1.0, 2.0]);
    }

    #[test]
    fn gaussian_sampling_is_seeded_and_finite() {
        let x = Volume::from_data(4, 1, 1, 2, vec![0.0; 8]).unwrap();
        let g = DiagonalGaussian::split(&x).unwrap();
        let a = g.sample(42).unwrap();
        let b = g.sample(42).unwrap();
        let c = g.sample(43).unwrap();
        assert_eq!(a.data, b.data);
        assert_ne!(a.data, c.data);
        assert!(a.is_finite());
    }

    #[test]
    fn gaussian_sampling_survives_extreme_logvar() {
        let mut x = Volume::new(2, 1, 1, 1);
        x.data[1] = 1.0e6; // an absurd log-variance
        let g = DiagonalGaussian::split(&x).unwrap();
        assert!(g.sample(1).unwrap().is_finite(), "logvar must be clamped");
    }

    #[test]
    fn parameter_keys_match_the_released_encoder() {
        let c = cfg();
        let keys = encoder_parameter_keys(&c);
        // Shortcuts only where the channel count changes: blocks 1, 3 and 5.
        let shortcuts: Vec<&String> = keys
            .iter()
            .filter(|k| k.contains("conv_shortcut"))
            .collect();
        assert_eq!(shortcuts.len(), 6, "3 blocks x weight+bias");
        assert!(shortcuts.iter().all(|k| k.contains("resnets.0")));
        for b in [1usize, 3, 5] {
            assert!(
                keys.contains(&format!(
                    "encoder.down_blocks.{b}.resnets.0.conv_shortcut.weight"
                )),
                "block {b} should carry a shortcut"
            );
        }
        // Downsamplers only where a factor exceeds 1: blocks 0..3.
        let down: Vec<&String> = keys
            .iter()
            .filter(|k| k.contains("downsamplers") && k.ends_with(".weight"))
            .collect();
        assert_eq!(down.len(), 4);
    }

    #[test]
    fn latent_frame_count_treats_a_single_image_specially() {
        let c = cfg();
        let e = H3VideoEncoder {
            cfg: c.clone(),
            conv_in: Conv3dWeights::new(vec![0.0], &[1, 1, 1, 1, 1], None).unwrap(),
            down_blocks: Vec::new(),
            norm_out_g: Vec::new(),
            norm_out_b: Vec::new(),
            conv_out: Conv3dWeights::new(vec![0.0], &[1, 1, 1, 1, 1], None).unwrap(),
            quant_conv: Conv3dWeights::new(vec![0.0], &[1, 1, 1, 1, 1], None).unwrap(),
        };
        assert_eq!(e.latent_frames(1), 1, "one image stays one latent frame");
        assert_eq!(e.latent_frames(4), 1);
        assert_eq!(e.latent_frames(8), 2);
    }

    #[test]
    fn pixel_normalization_matches_imagenet() {
        let mut v = Volume::from_data(3, 1, 1, 1, vec![0.485, 0.456, 0.406]).unwrap();
        normalize_pixels(&mut v).unwrap();
        assert!(v.data.iter().all(|x| x.abs() < 1e-6), "{:?}", v.data);
    }

    #[test]
    fn normalization_round_trips_through_the_display_transform() {
        let original = vec![0.1f32, 0.9, 0.5];
        let mut v = Volume::from_data(3, 1, 1, 1, original.clone()).unwrap();
        normalize_pixels(&mut v).unwrap();
        crate::vae_video::to_display_range(&mut v.data, 3).unwrap();
        for (a, b) in v.data.iter().zip(&original) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn encoder_rejects_an_off_grid_canvas() {
        let c = cfg();
        let e = H3VideoEncoder {
            cfg: c,
            conv_in: Conv3dWeights::new(vec![0.0], &[1, 1, 1, 1, 1], None).unwrap(),
            down_blocks: Vec::new(),
            norm_out_g: Vec::new(),
            norm_out_b: Vec::new(),
            conv_out: Conv3dWeights::new(vec![0.0], &[1, 1, 1, 1, 1], None).unwrap(),
            quant_conv: Conv3dWeights::new(vec![0.0], &[1, 1, 1, 1, 1], None).unwrap(),
        };
        // 20 is not a multiple of the 16x spatial compression.
        let v = Volume::new(3, 1, 20, 32);
        assert!(e.encode(&v).is_err());
    }
}
