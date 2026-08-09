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

//! `AutoencoderKLMiniMaxH3Audio` — a DAC encoder paired with a BigVGAN decoder.
//!
//! The soundtrack runs at 32 kHz with an 800-sample hop, i.e. 40 latent frames
//! per second, which is exactly the rate the packed sequence's rotary clock
//! advances at. Stereo is carried as **two independent mono passes** through the
//! same weights — the encoder and decoder are mono, and the two channels are
//! recombined afterwards.
//!
//! ## What makes this decoder unusual
//!
//! Every activation is *alias-free*: the signal is upsampled 2x by a transposed
//! depthwise Kaiser-sinc convolution, the Snake non-linearity is applied at the
//! doubled rate, and the result is low-pass filtered back down. That triple is
//! [`alias_free_snake_beta`], and it is applied 6 times per AMP block across 21
//! blocks, so it dominates the decoder's cost.
//!
//! The Kaiser-sinc filters are **persistent buffers in the checkpoint** rather
//! than recomputed here, so the port reads them instead of reproducing
//! `alias-free-torch`'s window arithmetic. [`kaiser_sinc_filter1d`] exists to
//! check the shipped buffers, not to replace them.
//!
//! Weights are stored weight-normalized (`weight_g` / `weight_v`);
//! [`resolve_weight_norm`] folds them into a plain kernel at load time.

use crate::config::H3AudioVaeConfig;
use anyhow::{Context, Result, bail, ensure};
use rlx_core::weight_map::WeightMap;

/// The `+ 1e-9` guard both Snake variants apply before reciprocating.
const SNAKE_EPS: f32 = 1e-9;
/// Resampling ratio of the alias-free activation wrapper.
const ACT_RATIO: usize = 2;

/// A 1-D signal as `[channels][length]`, channel-major and contiguous.
#[derive(Debug, Clone, PartialEq)]
pub struct Signal {
    pub channels: usize,
    pub length: usize,
    pub data: Vec<f32>,
}

impl Signal {
    pub fn new(channels: usize, length: usize) -> Self {
        Self {
            channels,
            length,
            data: vec![0.0; channels * length],
        }
    }

    pub fn from_data(channels: usize, data: Vec<f32>) -> Result<Self> {
        ensure!(channels > 0, "a signal needs at least one channel");
        ensure!(
            data.len().is_multiple_of(channels),
            "signal data of {} values does not divide into {channels} channels",
            data.len()
        );
        let length = data.len() / channels;
        Ok(Self {
            channels,
            length,
            data,
        })
    }

    #[must_use]
    pub fn row(&self, c: usize) -> &[f32] {
        &self.data[c * self.length..(c + 1) * self.length]
    }

    fn row_mut(&mut self, c: usize) -> &mut [f32] {
        let l = self.length;
        &mut self.data[c * l..(c + 1) * l]
    }

    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }
}

/// A convolution kernel with its bias, already de-weight-normalized.
#[derive(Debug, Clone)]
pub struct Conv1dWeights {
    /// `[out_channels][in_channels_per_group][kernel]`, row-major.
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub out_channels: usize,
    pub in_channels_per_group: usize,
    pub kernel: usize,
}

impl Conv1dWeights {
    /// A forward convolution: kernel `[out, in/groups, k]`, bias sized by `out`.
    pub fn new(weight: Vec<f32>, shape: &[usize], bias: Option<Vec<f32>>) -> Result<Self> {
        Self::build(weight, shape, bias, false)
    }

    /// A transposed convolution: kernel `[in, out/groups, k]`, so the bias is
    /// sized by the **second** axis rather than the first.
    pub fn new_transposed(
        weight: Vec<f32>,
        shape: &[usize],
        bias: Option<Vec<f32>>,
    ) -> Result<Self> {
        Self::build(weight, shape, bias, true)
    }

    fn build(
        weight: Vec<f32>,
        shape: &[usize],
        bias: Option<Vec<f32>>,
        transposed: bool,
    ) -> Result<Self> {
        ensure!(
            shape.len() == 3,
            "a Conv1d kernel must be rank 3, got {shape:?}"
        );
        ensure!(
            weight.len() == shape.iter().product::<usize>(),
            "kernel data ({}) does not match shape {shape:?}",
            weight.len()
        );
        let bias_axis = if transposed { shape[1] } else { shape[0] };
        if let Some(b) = &bias {
            ensure!(
                b.len() == bias_axis,
                "bias of {} does not match {bias_axis} output channels for a {}convolution",
                b.len(),
                if transposed { "transposed " } else { "" }
            );
        }
        Ok(Self {
            weight,
            bias,
            out_channels: shape[0],
            in_channels_per_group: shape[1],
            kernel: shape[2],
        })
    }
}

/// Fold PyTorch weight normalization into a plain kernel.
///
/// `w = g * v / ||v||`, where the norm is taken over every axis except the
/// output-channel axis — the `dim = 0` convention `torch.nn.utils.weight_norm`
/// uses for convolutions.
pub fn resolve_weight_norm(g: &[f32], v: &[f32], v_shape: &[usize]) -> Result<Vec<f32>> {
    ensure!(
        v_shape.len() == 3,
        "weight_v must be rank 3, got {v_shape:?}"
    );
    let (out_c, inner) = (v_shape[0], v_shape[1] * v_shape[2]);
    ensure!(
        v.len() == out_c * inner,
        "weight_v has {} values, expected {}",
        v.len(),
        out_c * inner
    );
    ensure!(
        g.len() == out_c,
        "weight_g has {} values, expected {out_c}",
        g.len()
    );
    let mut w = vec![0.0f32; v.len()];
    for o in 0..out_c {
        let row = &v[o * inner..(o + 1) * inner];
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = if norm > 0.0 { g[o] / norm } else { 0.0 };
        for (d, &s) in w[o * inner..(o + 1) * inner].iter_mut().zip(row) {
            *d = s * scale;
        }
    }
    Ok(w)
}

/// Grouped, dilated, strided 1-D convolution with zero padding.
pub fn conv1d(
    x: &Signal,
    w: &Conv1dWeights,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Signal> {
    ensure!(
        stride > 0 && dilation > 0 && groups > 0,
        "conv1d: stride, dilation and groups must be positive"
    );
    ensure!(
        x.channels.is_multiple_of(groups) && w.out_channels.is_multiple_of(groups),
        "conv1d: {} in / {} out channels do not split into {groups} groups",
        x.channels,
        w.out_channels
    );
    ensure!(
        x.channels / groups == w.in_channels_per_group,
        "conv1d: kernel expects {} input channels per group, signal has {}",
        w.in_channels_per_group,
        x.channels / groups
    );
    let eff = dilation * (w.kernel - 1) + 1;
    let padded = x.length + 2 * padding;
    if padded < eff {
        bail!("conv1d: padded length {padded} is shorter than the {eff}-tap effective kernel");
    }
    let out_len = (padded - eff) / stride + 1;
    let mut out = Signal::new(w.out_channels, out_len);
    let in_per_group = x.channels / groups;
    let out_per_group = w.out_channels / groups;

    for g in 0..groups {
        for oc in 0..out_per_group {
            let o = g * out_per_group + oc;
            let bias = w.bias.as_ref().map_or(0.0, |b| b[o]);
            for t in 0..out_len {
                let mut acc = bias;
                let base = (t * stride) as isize - padding as isize;
                for ic in 0..in_per_group {
                    let xrow = x.row(g * in_per_group + ic);
                    let krow =
                        &w.weight[(o * w.in_channels_per_group + ic) * w.kernel..][..w.kernel];
                    for (k, &kv) in krow.iter().enumerate() {
                        let p = base + (k * dilation) as isize;
                        if p >= 0 && (p as usize) < x.length {
                            acc += kv * xrow[p as usize];
                        }
                    }
                }
                out.row_mut(o)[t] = acc;
            }
        }
    }
    Ok(out)
}

/// Grouped transposed 1-D convolution.
///
/// The kernel is `[in_channels][out_channels_per_group][kernel]`, PyTorch's
/// `ConvTranspose1d` layout.
pub fn conv_transpose1d(
    x: &Signal,
    w: &Conv1dWeights,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Result<Signal> {
    ensure!(
        stride > 0 && groups > 0,
        "conv_transpose1d: stride and groups must be positive"
    );
    // Here `out_channels` names the kernel's *first* axis, which for a
    // transposed convolution is the input channel count.
    ensure!(
        x.channels == w.out_channels,
        "conv_transpose1d: kernel expects {} input channels, signal has {}",
        w.out_channels,
        x.channels
    );
    let out_c = w.in_channels_per_group * groups;
    let full = (x.length - 1) * stride + w.kernel;
    ensure!(
        full > 2 * padding,
        "conv_transpose1d: padding {padding} removes the whole {full}-sample output"
    );
    let out_len = full - 2 * padding;
    let mut out = Signal::new(out_c, out_len);
    let in_per_group = x.channels / groups;
    let out_per_group = w.in_channels_per_group;

    for g in 0..groups {
        for ic in 0..in_per_group {
            let i = g * in_per_group + ic;
            let xrow_start = i * x.length;
            for oc in 0..out_per_group {
                let o = g * out_per_group + oc;
                let krow = &w.weight[(i * out_per_group + oc) * w.kernel..][..w.kernel];
                for t in 0..x.length {
                    let xv = x.data[xrow_start + t];
                    if xv == 0.0 {
                        continue;
                    }
                    for (k, &kv) in krow.iter().enumerate() {
                        let p = (t * stride + k) as isize - padding as isize;
                        if p >= 0 && (p as usize) < out_len {
                            out.data[o * out_len + p as usize] += kv * xv;
                        }
                    }
                }
            }
        }
    }
    if let Some(b) = &w.bias {
        for o in 0..out_c {
            let bv = b[o];
            for t in 0..out_len {
                out.data[o * out_len + t] += bv;
            }
        }
    }
    Ok(out)
}

/// Replicate-pad both ends of every channel.
pub fn pad_replicate(x: &Signal, left: usize, right: usize) -> Signal {
    let out_len = x.length + left + right;
    let mut out = Signal::new(x.channels, out_len);
    for c in 0..x.channels {
        let src = x.row(c);
        let first = *src.first().unwrap_or(&0.0);
        let last = *src.last().unwrap_or(&0.0);
        let dst = &mut out.data[c * out_len..(c + 1) * out_len];
        dst[..left].fill(first);
        dst[left..left + x.length].copy_from_slice(src);
        dst[left + x.length..].fill(last);
    }
    out
}

/// DAC's Snake: `x + (alpha + 1e-9)^-1 * sin(alpha * x)^2`, `alpha` per channel.
pub fn snake1d(x: &mut Signal, alpha: &[f32]) -> Result<()> {
    ensure!(
        alpha.len() == x.channels,
        "snake1d: alpha has {} entries for {} channels",
        alpha.len(),
        x.channels
    );
    for c in 0..x.channels {
        let a = alpha[c];
        let inv = 1.0 / (a + SNAKE_EPS);
        for v in x.row_mut(c) {
            let s = (a * *v).sin();
            *v += inv * s * s;
        }
    }
    Ok(())
}

/// BigVGAN's SnakeBeta: `x + (e^beta + 1e-9)^-1 * sin(e^alpha * x)^2`.
///
/// Both parameters are stored in log space as `[channels]` vectors.
pub fn snake_beta(x: &mut Signal, log_alpha: &[f32], log_beta: &[f32]) -> Result<()> {
    ensure!(
        log_alpha.len() == x.channels && log_beta.len() == x.channels,
        "snake_beta: alpha/beta have {}/{} entries for {} channels",
        log_alpha.len(),
        log_beta.len(),
        x.channels
    );
    for c in 0..x.channels {
        let a = log_alpha[c].exp();
        let inv = 1.0 / (log_beta[c].exp() + SNAKE_EPS);
        for v in x.row_mut(c) {
            let s = (a * *v).sin();
            *v += inv * s * s;
        }
    }
    Ok(())
}

/// The Kaiser-sinc resampling filters of one alias-free activation, as shipped
/// in the checkpoint.
#[derive(Debug, Clone)]
pub struct ResampleFilters {
    /// `[kernel]`, the 2x upsampling filter.
    pub up: Vec<f32>,
    /// `[kernel]`, the matching low-pass for the downsampling half.
    pub down: Vec<f32>,
}

/// Upsample 2x with a transposed depthwise Kaiser-sinc convolution.
pub fn alias_free_upsample(x: &Signal, filter: &[f32]) -> Result<Signal> {
    let kernel = filter.len();
    let ratio = ACT_RATIO;
    let stride = ratio;
    let pad = kernel / ratio - 1;
    let pad_left = pad * stride + (kernel - stride) / 2;
    let pad_right = pad * stride + (kernel - stride).div_ceil(2);

    let padded = pad_replicate(x, pad, pad);
    // Depthwise: one filter shared by every channel.
    let w = Conv1dWeights {
        weight: filter.to_vec().repeat(x.channels),
        bias: None,
        out_channels: x.channels,
        in_channels_per_group: 1,
        kernel,
    };
    let mut up = conv_transpose1d(&padded, &w, stride, 0, x.channels)?;
    for v in up.data.iter_mut() {
        *v *= ratio as f32;
    }
    ensure!(
        up.length > pad_left + pad_right,
        "alias-free upsample trimmed the whole {}-sample output",
        up.length
    );
    let out_len = up.length - pad_left - pad_right;
    let mut out = Signal::new(up.channels, out_len);
    for c in 0..up.channels {
        let src = &up.data[c * up.length + pad_left..c * up.length + pad_left + out_len];
        out.data[c * out_len..(c + 1) * out_len].copy_from_slice(src);
    }
    Ok(out)
}

/// Anti-aliased 2x downsample: replicate-pad, depthwise low-pass, stride 2.
pub fn alias_free_downsample(x: &Signal, filter: &[f32]) -> Result<Signal> {
    let kernel = filter.len();
    let even = kernel.is_multiple_of(2);
    let pad_left = kernel / 2 - usize::from(even);
    let pad_right = kernel / 2;
    let padded = pad_replicate(x, pad_left, pad_right);
    let w = Conv1dWeights {
        weight: filter.to_vec().repeat(x.channels),
        bias: None,
        out_channels: x.channels,
        in_channels_per_group: 1,
        kernel,
    };
    conv1d(&padded, &w, ACT_RATIO, 0, 1, x.channels)
}

/// The alias-free activation wrapper: upsample 2x, SnakeBeta, downsample 2x.
pub fn alias_free_snake_beta(
    x: &Signal,
    log_alpha: &[f32],
    log_beta: &[f32],
    filters: &ResampleFilters,
) -> Result<Signal> {
    let mut up = alias_free_upsample(x, &filters.up)?;
    snake_beta(&mut up, log_alpha, log_beta)?;
    alias_free_downsample(&up, &filters.down)
}

/// Kaiser-windowed sinc low-pass filter, for checking the shipped buffers.
///
/// Kept arithmetically identical to `alias-free-torch`, which is what the
/// checkpoint's persistent buffers were produced by.
#[must_use]
pub fn kaiser_sinc_filter1d(cutoff: f64, half_width: f64, kernel_size: usize) -> Vec<f32> {
    let half_size = kernel_size / 2;
    let attenuation =
        2.285 * (half_size as f64 - 1.0) * std::f64::consts::PI * (4.0 * half_width) + 7.95;
    let beta = if attenuation > 50.0 {
        0.1102 * (attenuation - 8.7)
    } else if attenuation >= 21.0 {
        0.5842 * (attenuation - 21.0).powf(0.4) + 0.07886 * (attenuation - 21.0)
    } else {
        0.0
    };
    let window = kaiser_window(kernel_size, beta);
    let time: Vec<f64> = if kernel_size.is_multiple_of(2) {
        (0..kernel_size)
            .map(|i| (i as f64 - half_size as f64) + 0.5)
            .collect()
    } else {
        (0..kernel_size)
            .map(|i| i as f64 - half_size as f64)
            .collect()
    };
    let mut f: Vec<f64> = time
        .iter()
        .zip(&window)
        .map(|(&t, &w)| 2.0 * cutoff * w * sinc(2.0 * cutoff * t))
        .collect();
    let sum: f64 = f.iter().sum();
    if sum != 0.0 {
        for v in f.iter_mut() {
            *v /= sum;
        }
    }
    f.into_iter().map(|v| v as f32).collect()
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let p = std::f64::consts::PI * x;
        p.sin() / p
    }
}

/// Symmetric (non-periodic) Kaiser window, matching `torch.kaiser_window`.
fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    let denom = bessel_i0(beta);
    (0..n)
        .map(|i| {
            let r = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
            bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denom
        })
        .collect()
}

/// Modified Bessel function of the first kind, order zero.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x / 2.0;
    for k in 1..64 {
        term *= (half / k as f64) * (half / k as f64);
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

/// Undo the per-channel latent normalization the DiT operates in.
pub fn denormalize_latents(latents: &mut Signal, cfg: &H3AudioVaeConfig) -> Result<()> {
    if cfg.latents_mean.is_empty() || cfg.latents_std.is_empty() {
        return Ok(());
    }
    ensure!(
        cfg.latents_mean.len() == latents.channels && cfg.latents_std.len() == latents.channels,
        "latent statistics cover {} channels, signal has {}",
        cfg.latents_mean.len(),
        latents.channels
    );
    for c in 0..latents.channels {
        let (m, s) = (cfg.latents_mean[c], cfg.latents_std[c]);
        for v in latents.row_mut(c) {
            *v = *v * s + m;
        }
    }
    Ok(())
}

/// Apply the per-channel latent normalization the DiT operates in.
pub fn normalize_latents(latents: &mut Signal, cfg: &H3AudioVaeConfig) -> Result<()> {
    if cfg.latents_mean.is_empty() || cfg.latents_std.is_empty() {
        return Ok(());
    }
    ensure!(
        cfg.latents_mean.len() == latents.channels,
        "latent statistics cover {} channels, signal has {}",
        cfg.latents_mean.len(),
        latents.channels
    );
    for c in 0..latents.channels {
        let (m, s) = (cfg.latents_mean[c], cfg.latents_std[c]);
        let inv = if s != 0.0 { 1.0 / s } else { 0.0 };
        for v in latents.row_mut(c) {
            *v = (*v - m) * inv;
        }
    }
    Ok(())
}

/// One AMP (anti-aliased multi-periodicity) block of the BigVGAN decoder.
#[derive(Debug, Clone)]
struct AmpBlock {
    convs1: Vec<Conv1dWeights>,
    convs2: Vec<Conv1dWeights>,
    /// `(log_alpha, log_beta, filters)` per activation, two per dilation.
    activations: Vec<(Vec<f32>, Vec<f32>, ResampleFilters)>,
    dilations: Vec<usize>,
    kernel: usize,
}

impl AmpBlock {
    fn forward(&self, x: &Signal) -> Result<Signal> {
        let mut h = x.clone();
        for (i, &d) in self.dilations.iter().enumerate() {
            let (a1, b1, f1) = &self.activations[2 * i];
            let (a2, b2, f2) = &self.activations[2 * i + 1];
            let t = alias_free_snake_beta(&h, a1, b1, f1)?;
            let pad = ((self.kernel - 1) * d) / 2;
            let t = conv1d(&t, &self.convs1[i], 1, pad, d, 1)?;
            let t = alias_free_snake_beta(&t, a2, b2, f2)?;
            let pad2 = (self.kernel - 1) / 2;
            let t = conv1d(&t, &self.convs2[i], 1, pad2, 1, 1)?;
            ensure!(
                t.length == h.length,
                "AMP block changed the time axis: {} -> {}",
                h.length,
                t.length
            );
            for (dst, src) in h.data.iter_mut().zip(&t.data) {
                *dst += src;
            }
        }
        Ok(h)
    }
}

/// The BigVGAN decoder: latent frames to a mono waveform.
pub struct H3AudioDecoder {
    cfg: H3AudioVaeConfig,
    dec_in_proj: Conv1dWeights,
    conv_pre: Conv1dWeights,
    ups: Vec<Conv1dWeights>,
    /// `resblock_kernel_sizes.len()` blocks per upsample stage.
    resblocks: Vec<AmpBlock>,
    post_alpha: Vec<f32>,
    post_beta: Vec<f32>,
    post_filters: ResampleFilters,
    conv_post: Conv1dWeights,
}

impl H3AudioDecoder {
    /// Load the decoder from a checkpoint weight map.
    pub fn load(cfg: &H3AudioVaeConfig, w: &WeightMap) -> Result<Self> {
        cfg.validate()?;
        let n_kernels = cfg.resblock_kernel_sizes.len();

        let dec_in_proj = plain_conv(w, "dec_in_proj", true)?;
        let conv_pre = wn_conv(w, "decoder.conv_pre", true, false)?;

        let mut ups = Vec::with_capacity(cfg.decoder_rates.len());
        for i in 0..cfg.decoder_rates.len() {
            ups.push(wn_conv(w, &format!("decoder.ups.{i}.0"), true, true)?);
        }

        let mut resblocks = Vec::with_capacity(cfg.decoder_rates.len() * n_kernels);
        for stage in 0..cfg.decoder_rates.len() {
            for k in 0..n_kernels {
                let idx = stage * n_kernels + k;
                let p = format!("decoder.resblocks.{idx}");
                let dilations = cfg.resblock_dilation_sizes[k].clone();
                let mut convs1 = Vec::with_capacity(dilations.len());
                let mut convs2 = Vec::with_capacity(dilations.len());
                for j in 0..dilations.len() {
                    convs1.push(wn_conv(w, &format!("{p}.convs1.{j}"), true, false)?);
                    convs2.push(wn_conv(w, &format!("{p}.convs2.{j}"), true, false)?);
                }
                let mut activations = Vec::with_capacity(2 * dilations.len());
                for j in 0..2 * dilations.len() {
                    activations.push(load_activation(w, &format!("{p}.activations.{j}"))?);
                }
                resblocks.push(AmpBlock {
                    convs1,
                    convs2,
                    activations,
                    dilations,
                    kernel: cfg.resblock_kernel_sizes[k],
                });
            }
        }

        let (post_alpha, post_beta, post_filters) = load_activation(w, "decoder.activation_post")?;
        let conv_post = wn_conv(w, "decoder.conv_post", false, false)?;

        Ok(Self {
            cfg: cfg.clone(),
            dec_in_proj,
            conv_pre,
            ups,
            resblocks,
            post_alpha,
            post_beta,
            post_filters,
            conv_post,
        })
    }

    /// Decode one **mono** latent stream to a waveform.
    ///
    /// `latents` is `[latent_channels][frames]`, already denormalized.
    pub fn decode_mono(&self, latents: &Signal) -> Result<Vec<f32>> {
        ensure!(
            latents.channels == self.cfg.latent_channels,
            "audio decoder expects {} latent channels, got {}",
            self.cfg.latent_channels,
            latents.channels
        );
        let n_kernels = self.cfg.resblock_kernel_sizes.len();

        let h = conv1d(latents, &self.dec_in_proj, 1, 0, 1, 1)?;
        let mut h = conv1d(&h, &self.conv_pre, 1, 3, 1, 1)?;

        for (stage, (&rate, &kernel)) in self
            .cfg
            .decoder_rates
            .iter()
            .zip(&self.cfg.decoder_kernel_sizes)
            .enumerate()
        {
            let (a, b, f) = &self.resblocks[stage * n_kernels].activations[0];
            let _ = (a, b, f);
            // BigVGAN upsamples with a transposed convolution whose padding is
            // `(kernel - rate) / 2`.
            let pad = (kernel - rate) / 2;
            h = conv_transpose1d(&h, &self.ups[stage], rate, pad, 1)?;

            // The `resblock_kernel_sizes` variants are averaged, not summed.
            let mut acc: Option<Signal> = None;
            for k in 0..n_kernels {
                let out = self.resblocks[stage * n_kernels + k].forward(&h)?;
                match &mut acc {
                    None => acc = Some(out),
                    Some(a) => {
                        for (d, s) in a.data.iter_mut().zip(&out.data) {
                            *d += s;
                        }
                    }
                }
            }
            let mut a = acc.context("a decoder stage has no residual blocks")?;
            let inv = 1.0 / n_kernels as f32;
            for v in a.data.iter_mut() {
                *v *= inv;
            }
            h = a;
        }

        let h = alias_free_snake_beta(&h, &self.post_alpha, &self.post_beta, &self.post_filters)?;
        let out = conv1d(&h, &self.conv_post, 1, 3, 1, 1)?;
        ensure!(
            out.channels == 1,
            "the audio decoder must end on a single channel, got {}",
            out.channels
        );
        Ok(out.data.iter().map(|v| v.tanh()).collect())
    }

    /// Decode a stereo pair.
    ///
    /// H3 packs stereo as two channel-major blocks of audio rows and the VAE
    /// itself is mono, so each channel runs through the same weights
    /// independently and the two are recombined here.
    pub fn decode_stereo(&self, left: &Signal, right: &Signal) -> Result<Vec<Vec<f32>>> {
        Ok(vec![self.decode_mono(left)?, self.decode_mono(right)?])
    }

    #[must_use]
    pub fn config(&self) -> &H3AudioVaeConfig {
        &self.cfg
    }

    /// Waveform samples one latent frame expands to.
    #[must_use]
    pub fn hop(&self) -> usize {
        self.cfg.decoder_hop()
    }
}

fn get<'a>(w: &'a WeightMap, key: &str) -> Result<(&'a [f32], &'a [usize])> {
    w.get(key)
        .ok_or_else(|| anyhow::anyhow!("MiniMax-H3 audio VAE: missing weight `{key}`"))
}

/// Load a weight-normalized convolution (`weight_g` / `weight_v`).
fn wn_conv(w: &WeightMap, prefix: &str, bias: bool, transposed: bool) -> Result<Conv1dWeights> {
    let (g, _) = get(w, &format!("{prefix}.weight_g"))?;
    let (v, vs) = get(w, &format!("{prefix}.weight_v"))?;
    let weight = resolve_weight_norm(g, v, vs)
        .with_context(|| format!("MiniMax-H3 audio VAE: resolve weight norm for `{prefix}`"))?;
    let b = if bias {
        Some(get(w, &format!("{prefix}.bias"))?.0.to_vec())
    } else {
        None
    };
    if transposed {
        Conv1dWeights::new_transposed(weight, vs, b)
    } else {
        Conv1dWeights::new(weight, vs, b)
    }
}

fn vec_of(w: &WeightMap, key: &str) -> Result<Vec<f32>> {
    Ok(get(w, key)?.0.to_vec())
}

/// Load a plain convolution.
fn plain_conv(w: &WeightMap, prefix: &str, bias: bool) -> Result<Conv1dWeights> {
    let (weight, shape) = get(w, &format!("{prefix}.weight"))?;
    let b = if bias {
        Some(get(w, &format!("{prefix}.bias"))?.0.to_vec())
    } else {
        None
    };
    Conv1dWeights::new(weight.to_vec(), shape, b)
}

/// Load one alias-free activation: its SnakeBeta parameters and the two
/// resampling filters shipped alongside them.
fn load_activation(w: &WeightMap, prefix: &str) -> Result<(Vec<f32>, Vec<f32>, ResampleFilters)> {
    let alpha = get(w, &format!("{prefix}.act.alpha"))?.0.to_vec();
    let beta = get(w, &format!("{prefix}.act.beta"))?.0.to_vec();
    let up = get(w, &format!("{prefix}.upsample.filter"))?.0.to_vec();
    let down = get(w, &format!("{prefix}.downsample.lowpass.filter"))?
        .0
        .to_vec();
    Ok((alpha, beta, ResampleFilters { up, down }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> H3AudioVaeConfig {
        serde_json::from_str(
            r#"{"encoder_dim":64,"encoder_rates":[2,4,4,5,5],"latent_dim":2048,
                "latent_channels":4,"decoder_dim":1024,"decoder_rates":[2],
                "decoder_kernel_sizes":[4],"num_attention_heads":8,
                "resblock_kernel_sizes":[3],"resblock_dilation_sizes":[[1]],
                "sampling_rate":32000,"latents_mean":[0.5,-0.5,1.0,0.0],
                "latents_std":[2.0,1.0,0.5,4.0]}"#,
        )
        .unwrap()
    }

    #[test]
    fn weight_norm_reproduces_the_original_kernel() {
        // If g is the row norm of v, folding is the identity.
        let v = vec![3.0f32, 4.0, 0.0, 1.0, 0.0, 0.0];
        let shape = [2usize, 1, 3];
        let g = vec![5.0f32, 1.0];
        let w = resolve_weight_norm(&g, &v, &shape).unwrap();
        for (a, b) in w.iter().zip(&v) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn weight_norm_scales_rows_to_g() {
        let v = vec![3.0f32, 4.0];
        let w = resolve_weight_norm(&[10.0], &v, &[1, 1, 2]).unwrap();
        let norm = (w[0] * w[0] + w[1] * w[1]).sqrt();
        assert!((norm - 10.0).abs() < 1e-5, "row norm = {norm}");
    }

    #[test]
    fn weight_norm_handles_a_zero_row() {
        let w = resolve_weight_norm(&[1.0], &[0.0, 0.0], &[1, 1, 2]).unwrap();
        assert_eq!(w, vec![0.0, 0.0]);
    }

    #[test]
    fn conv1d_identity_kernel_passes_through() {
        let x = Signal::from_data(1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let w = Conv1dWeights::new(vec![1.0], &[1, 1, 1], None).unwrap();
        let y = conv1d(&x, &w, 1, 0, 1, 1).unwrap();
        assert_eq!(y.data, x.data);
    }

    #[test]
    fn conv1d_same_padding_preserves_length() {
        let x = Signal::from_data(1, (0..16).map(|i| i as f32).collect()).unwrap();
        let w = Conv1dWeights::new(vec![0.25; 7], &[1, 1, 7], None).unwrap();
        let y = conv1d(&x, &w, 1, 3, 1, 1).unwrap();
        assert_eq!(y.length, 16);
    }

    #[test]
    fn conv1d_dilation_matches_hand_computation() {
        // kernel [1, 1] with dilation 2 over [1,2,3,4]: y[t] = x[t] + x[t+2]
        let x = Signal::from_data(1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let w = Conv1dWeights::new(vec![1.0, 1.0], &[1, 1, 2], None).unwrap();
        let y = conv1d(&x, &w, 1, 0, 2, 1).unwrap();
        assert_eq!(y.data, vec![4.0, 6.0]);
    }

    #[test]
    fn conv1d_groups_are_independent() {
        // Two channels, depthwise, different kernels.
        let x = Signal::from_data(2, vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let w = Conv1dWeights::new(vec![2.0, 3.0], &[2, 1, 1], None).unwrap();
        let y = conv1d(&x, &w, 1, 0, 1, 2).unwrap();
        assert_eq!(y.data, vec![2.0, 2.0, 6.0, 6.0]);
    }

    #[test]
    fn conv_transpose1d_upsamples_by_stride() {
        let x = Signal::from_data(1, vec![1.0, 2.0]).unwrap();
        let w = Conv1dWeights::new(vec![1.0, 1.0], &[1, 1, 2], None).unwrap();
        let y = conv_transpose1d(&x, &w, 2, 0, 1).unwrap();
        // (2-1)*2 + 2 = 4 samples
        assert_eq!(y.length, 4);
        assert_eq!(y.data, vec![1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn pad_replicate_repeats_the_edges() {
        let x = Signal::from_data(1, vec![5.0, 6.0, 7.0]).unwrap();
        let y = pad_replicate(&x, 2, 1);
        assert_eq!(y.data, vec![5.0, 5.0, 5.0, 6.0, 7.0, 7.0]);
    }

    #[test]
    fn snake_is_identity_at_zero_input() {
        let mut x = Signal::from_data(2, vec![0.0; 6]).unwrap();
        snake1d(&mut x, &[1.0, 2.0]).unwrap();
        assert!(x.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn snake_adds_a_non_negative_term() {
        // sin^2 >= 0 and the reciprocal is positive, so Snake never decreases x.
        let mut x = Signal::from_data(1, vec![-1.0, -0.5, 0.5, 1.0]).unwrap();
        let before = x.data.clone();
        snake1d(&mut x, &[1.5]).unwrap();
        for (a, b) in x.data.iter().zip(&before) {
            assert!(a >= b, "{a} < {b}");
        }
    }

    #[test]
    fn snake_beta_uses_log_space_parameters() {
        // alpha = beta = 0 in log space means e^0 = 1 for both.
        let mut x = Signal::from_data(1, vec![1.0]).unwrap();
        snake_beta(&mut x, &[0.0], &[0.0]).unwrap();
        let want = 1.0 + (1.0f32).sin().powi(2) / (1.0 + SNAKE_EPS);
        assert!((x.data[0] - want).abs() < 1e-6, "{} vs {want}", x.data[0]);
    }

    #[test]
    fn kaiser_filter_sums_to_one_and_is_symmetric() {
        let f = kaiser_sinc_filter1d(0.25, 0.3, 12);
        assert_eq!(f.len(), 12);
        let sum: f32 = f.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
        for i in 0..6 {
            assert!(
                (f[i] - f[11 - i]).abs() < 1e-6,
                "filter is not symmetric at {i}"
            );
        }
    }

    #[test]
    fn bessel_i0_matches_known_values() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
        assert!((bessel_i0(1.0) - 1.266_065_877_75).abs() < 1e-9);
        assert!((bessel_i0(2.0) - 2.279_585_302_336).abs() < 1e-9);
    }

    #[test]
    fn alias_free_activation_preserves_length() {
        let filters = ResampleFilters {
            up: kaiser_sinc_filter1d(0.25, 0.3, 12),
            down: kaiser_sinc_filter1d(0.25, 0.3, 12),
        };
        let x = Signal::from_data(2, (0..64).map(|i| (i as f32 / 8.0).sin()).collect()).unwrap();
        let y = alias_free_snake_beta(&x, &[0.0, 0.0], &[0.0, 0.0], &filters).unwrap();
        assert_eq!(y.length, x.length, "alias-free activation changed the rate");
        assert_eq!(y.channels, 2);
        assert!(y.is_finite());
    }

    #[test]
    fn upsample_then_downsample_roughly_reconstructs() {
        // With a matched Kaiser pair and no non-linearity in between, the round
        // trip should stay close to the input away from the edges.
        let filters = ResampleFilters {
            up: kaiser_sinc_filter1d(0.25, 0.3, 12),
            down: kaiser_sinc_filter1d(0.25, 0.3, 12),
        };
        let x = Signal::from_data(1, (0..64).map(|i| (i as f32 / 10.0).sin()).collect()).unwrap();
        let up = alias_free_upsample(&x, &filters.up).unwrap();
        assert_eq!(up.length, 2 * x.length);
        let back = alias_free_downsample(&up, &filters.down).unwrap();
        assert_eq!(back.length, x.length);
        let interior: f32 = (8..56)
            .map(|i| (back.data[i] - x.data[i]).abs())
            .fold(0.0, f32::max);
        assert!(interior < 0.05, "round-trip error {interior}");
    }

    #[test]
    fn latent_normalization_round_trips() {
        let c = cfg();
        let mut s = Signal::from_data(4, (0..12).map(|i| i as f32 * 0.25).collect()).unwrap();
        let original = s.data.clone();
        normalize_latents(&mut s, &c).unwrap();
        assert_ne!(s.data, original);
        denormalize_latents(&mut s, &c).unwrap();
        for (a, b) in s.data.iter().zip(&original) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn hop_sizes_match_forty_latents_per_second() {
        let c = H3AudioVaeConfig {
            encoder_dim: 64,
            encoder_rates: vec![2, 4, 4, 5, 5],
            latent_dim: 2048,
            latent_channels: 32,
            decoder_dim: 1024,
            decoder_rates: vec![5, 5, 2, 2, 2, 2, 2],
            decoder_kernel_sizes: vec![9, 9, 4, 4, 4, 4, 4],
            num_attention_heads: 8,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5]; 3],
            sampling_rate: 32_000,
            latents_mean: vec![],
            latents_std: vec![],
        };
        assert_eq!(c.encoder_hop(), 800);
        assert_eq!(c.decoder_hop(), 800);
        assert_eq!(c.sampling_rate / c.decoder_hop(), 40);
    }

    #[test]
    fn signal_rejects_ragged_data() {
        assert!(Signal::from_data(3, vec![0.0; 7]).is_err());
        assert!(Signal::from_data(0, vec![]).is_err());
    }
}

// ---------------------------------------------------------------------------
// Encoder — DAC trunk plus the causal-attention projection that produces the
// diffusion latents.
// ---------------------------------------------------------------------------

/// Posterior of the audio autoencoder, parameterized as `(mean, log_std)`.
///
/// The checkpoint keeps two separate `Conv1d` heads rather than one fused
/// moments projection, and the second predicts the **log standard deviation**,
/// not the log variance. H3 always consumes [`Self::mode`].
#[derive(Debug, Clone)]
pub struct AudioGaussian {
    pub mean: Signal,
    pub logs: Signal,
}

impl AudioGaussian {
    /// The mode, which is the mean — bit-for-bit what `mean_proj` produced.
    #[must_use]
    pub fn mode(&self) -> &Signal {
        &self.mean
    }

    /// `mean + exp(logs) * noise`.
    pub fn sample(&self, seed: u64) -> Signal {
        let noise = crate::pipeline::noise(self.mean.data.len(), seed);
        let mut out = self.mean.clone();
        for ((v, &ls), &n) in out.data.iter_mut().zip(&self.logs.data).zip(&noise) {
            *v += ls.clamp(-30.0, 20.0).exp() * n;
        }
        out
    }
}

/// DAC residual unit: `Snake -> dilated Conv1d(k=7) -> Snake -> Conv1d(k=1)`,
/// plus a shortcut that is centre-cropped when the body shortens the time axis.
#[derive(Debug, Clone)]
struct ResidualUnit {
    alpha1: Vec<f32>,
    conv1: Conv1dWeights,
    alpha2: Vec<f32>,
    conv2: Conv1dWeights,
    dilation: usize,
}

impl ResidualUnit {
    fn forward(&self, x: &Signal) -> Result<Signal> {
        let mut h = x.clone();
        snake1d(&mut h, &self.alpha1)?;
        let pad = ((self.conv1.kernel - 1) * self.dilation) / 2;
        let mut h = conv1d(&h, &self.conv1, 1, pad, self.dilation, 1)?;
        snake1d(&mut h, &self.alpha2)?;
        let h = conv1d(&h, &self.conv2, 1, 0, 1, 1)?;

        // The reference centre-crops the shortcut when the dilated convolution
        // shrinks the signal.
        let crop = x.length.saturating_sub(h.length) / 2;
        let mut out = h;
        ensure!(
            out.channels == x.channels,
            "residual unit changed the channel count: {} -> {}",
            x.channels,
            out.channels
        );
        for c in 0..out.channels {
            for t in 0..out.length {
                out.data[c * out.length + t] += x.data[c * x.length + crop + t];
            }
        }
        Ok(out)
    }
}

/// One encoder level: three residual units at dilations 1/3/9, then a strided
/// channel-doubling convolution.
#[derive(Debug, Clone)]
struct EncoderBlock {
    units: Vec<ResidualUnit>,
    alpha: Vec<f32>,
    down: Conv1dWeights,
    stride: usize,
}

impl EncoderBlock {
    fn forward(&self, x: &Signal) -> Result<Signal> {
        let mut h = x.clone();
        for u in &self.units {
            h = u.forward(&h)?;
        }
        snake1d(&mut h, &self.alpha)?;
        let pad = self.stride.div_ceil(2);
        conv1d(&h, &self.down, self.stride, pad, 1, 1)
    }
}

/// `pre_block` — a residual causal-attention + GeGLU block that rewires the
/// 2048-wide encoder trunk down to the 32-channel latent width.
///
/// The attention is unusual in two ways: the heads are **mean-pooled away**
/// rather than concatenated, and the surviving head dimension is adaptively
/// average-pooled down to `out_dim`.
#[derive(Debug, Clone)]
struct AttnProjection {
    norm1_w: Vec<f32>,
    norm1_b: Vec<f32>,
    norm3_w: Vec<f32>,
    norm3_b: Vec<f32>,
    norm2_w: Vec<f32>,
    norm2_b: Vec<f32>,
    qkv: Vec<f32>,
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
    v_bias: Vec<f32>,
    attn_proj_w: Vec<f32>,
    attn_proj_b: Vec<f32>,
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
    mlp_norm_w: Vec<f32>,
    mlp_norm_b: Vec<f32>,
    w0: (Vec<f32>, Vec<f32>),
    w1: (Vec<f32>, Vec<f32>),
    w2: (Vec<f32>, Vec<f32>),
    in_dim: usize,
    out_dim: usize,
    heads: usize,
}

impl AttnProjection {
    /// `rows` is `[seq_len * in_dim]`, row-major. Returns `[seq_len * out_dim]`.
    fn forward(&self, rows: &[f32], seq: usize) -> Result<Vec<f32>> {
        let (d, o, h) = (self.in_dim, self.out_dim, self.heads);
        ensure!(
            rows.len() == seq * d,
            "pre_block input holds {} values for {seq} rows of {d}",
            rows.len()
        );
        let head_dim = d / h;

        let n1 = layer_norm_rows(rows, seq, d, &self.norm1_w, &self.norm1_b);
        let n3 = layer_norm_rows(rows, seq, d, &self.norm3_w, &self.norm3_b);

        // Causal attention. qkv is a single bias-less linear; the key bias is a
        // frozen zero buffer.
        let mut q = vec![0.0f32; seq * d];
        let mut k = vec![0.0f32; seq * d];
        let mut v = vec![0.0f32; seq * d];
        for t in 0..seq {
            for j in 0..d {
                q[t * d + j] =
                    dot(&self.qkv[j * d..(j + 1) * d], &n1[t * d..(t + 1) * d]) + self.q_bias[j];
                let kr = (d + j) * d;
                k[t * d + j] = dot(&self.qkv[kr..kr + d], &n1[t * d..(t + 1) * d]) + self.k_bias[j];
                let vr = (2 * d + j) * d;
                v[t * d + j] = dot(&self.qkv[vr..vr + d], &n1[t * d..(t + 1) * d]) + self.v_bias[j];
            }
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        // Heads are averaged away, so accumulate straight into a [seq, head_dim]
        // buffer instead of concatenating.
        let mut pooled = vec![0.0f32; seq * head_dim];
        let mut scores = vec![0.0f32; seq];
        for hd in 0..h {
            let off = hd * head_dim;
            for t in 0..seq {
                let qs = &q[t * d + off..t * d + off + head_dim];
                let mut max = f32::NEG_INFINITY;
                for s in 0..=t {
                    let ks = &k[s * d + off..s * d + off + head_dim];
                    let sc = dot(qs, ks) * scale;
                    scores[s] = sc;
                    if sc > max {
                        max = sc;
                    }
                }
                let mut sum = 0.0f32;
                for s in 0..=t {
                    scores[s] = (scores[s] - max).exp();
                    sum += scores[s];
                }
                let inv = 1.0 / sum;
                for s in 0..=t {
                    let wgt = scores[s] * inv;
                    let vs = &v[s * d + off..s * d + off + head_dim];
                    for (dst, &src) in pooled[t * head_dim..(t + 1) * head_dim].iter_mut().zip(vs) {
                        *dst += wgt * src;
                    }
                }
            }
        }
        let inv_h = 1.0 / h as f32;
        for v in pooled.iter_mut() {
            *v *= inv_h;
        }

        // Adaptive average pool head_dim -> out_dim, then the attention's own
        // projection.
        let mut attn_out = vec![0.0f32; seq * o];
        for t in 0..seq {
            let src = &pooled[t * head_dim..(t + 1) * head_dim];
            let mut pool = vec![0.0f32; o];
            for (i, p) in pool.iter_mut().enumerate() {
                let start = i * head_dim / o;
                let end = ((i + 1) * head_dim).div_ceil(o);
                let end = end.max(start + 1).min(head_dim);
                *p = src[start..end].iter().sum::<f32>() / (end - start) as f32;
            }
            for j in 0..o {
                attn_out[t * o + j] =
                    dot(&self.attn_proj_w[j * o..(j + 1) * o], &pool) + self.attn_proj_b[j];
            }
        }

        // h = proj(norm3(x)) + attn(norm1(x))
        let mut hs = vec![0.0f32; seq * o];
        for t in 0..seq {
            for j in 0..o {
                hs[t * o + j] = dot(&self.proj_w[j * d..(j + 1) * d], &n3[t * d..(t + 1) * d])
                    + self.proj_b[j]
                    + attn_out[t * o + j];
            }
        }

        // h = h + mlp(norm2(h)); the GeGLU carries its own LayerNorm as well.
        let n2 = layer_norm_rows(&hs, seq, o, &self.norm2_w, &self.norm2_b);
        let n2 = layer_norm_rows(&n2, seq, o, &self.mlp_norm_w, &self.mlp_norm_b);
        let hidden = self.w0.1.len();
        for t in 0..seq {
            let row = &n2[t * o..(t + 1) * o];
            let mut gated = vec![0.0f32; hidden];
            for j in 0..hidden {
                let a = dot(&self.w0.0[j * o..(j + 1) * o], row) + self.w0.1[j];
                let b = dot(&self.w1.0[j * o..(j + 1) * o], row) + self.w1.1[j];
                gated[j] = gelu_tanh(a) * b;
            }
            for j in 0..o {
                hs[t * o + j] +=
                    dot(&self.w2.0[j * hidden..(j + 1) * hidden], &gated) + self.w2.1[j];
            }
        }
        Ok(hs)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Tanh-approximated GELU, matching `nn.GELU(approximate="tanh")`.
fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

/// Row-wise LayerNorm over `[seq, dim]`.
fn layer_norm_rows(x: &[f32], seq: usize, dim: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * dim];
    for t in 0..seq {
        let row = &x[t * dim..(t + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for j in 0..dim {
            out[t * dim + j] = (row[j] - mean) * inv * w[j] + b[j];
        }
    }
    out
}

/// The DAC waveform encoder plus the posterior heads.
pub struct H3AudioEncoder {
    cfg: H3AudioVaeConfig,
    conv_in: Conv1dWeights,
    blocks: Vec<EncoderBlock>,
    alpha_out: Vec<f32>,
    conv_out: Conv1dWeights,
    pre_block: AttnProjection,
    mean_proj: Conv1dWeights,
    logs_proj: Conv1dWeights,
}

impl H3AudioEncoder {
    /// Load the encoder from a checkpoint weight map.
    pub fn load(cfg: &H3AudioVaeConfig, w: &WeightMap) -> Result<Self> {
        cfg.validate()?;
        let conv_in = wn_conv(w, "encoder.block.0", true, false)?;

        let mut blocks = Vec::with_capacity(cfg.encoder_rates.len());
        for (i, &stride) in cfg.encoder_rates.iter().enumerate() {
            let p = format!("encoder.block.{}", i + 1);
            let mut units = Vec::with_capacity(3);
            for (u, dilation) in [1usize, 3, 9].into_iter().enumerate() {
                let up = format!("{p}.block.{u}.block");
                units.push(ResidualUnit {
                    alpha1: vec_of(w, &format!("{up}.0.alpha"))?,
                    conv1: wn_conv(w, &format!("{up}.1"), true, false)?,
                    alpha2: vec_of(w, &format!("{up}.2.alpha"))?,
                    conv2: wn_conv(w, &format!("{up}.3"), true, false)?,
                    dilation,
                });
            }
            blocks.push(EncoderBlock {
                units,
                alpha: vec_of(w, &format!("{p}.block.3.alpha"))?,
                down: wn_conv(w, &format!("{p}.block.4"), true, false)?,
                stride,
            });
        }

        let tail = cfg.encoder_rates.len() + 1;
        let alpha_out = vec_of(w, &format!("encoder.block.{tail}.alpha"))?;
        let conv_out = wn_conv(w, &format!("encoder.block.{}", tail + 1), true, false)?;

        let pre_block = AttnProjection {
            norm1_w: vec_of(w, "pre_block.norm1.weight")?,
            norm1_b: vec_of(w, "pre_block.norm1.bias")?,
            norm3_w: vec_of(w, "pre_block.norm3.weight")?,
            norm3_b: vec_of(w, "pre_block.norm3.bias")?,
            norm2_w: vec_of(w, "pre_block.norm2.weight")?,
            norm2_b: vec_of(w, "pre_block.norm2.bias")?,
            qkv: vec_of(w, "pre_block.attn.qkv.weight")?,
            q_bias: vec_of(w, "pre_block.attn.q_bias")?,
            k_bias: vec_of(w, "pre_block.attn.zero_k_bias")?,
            v_bias: vec_of(w, "pre_block.attn.v_bias")?,
            attn_proj_w: vec_of(w, "pre_block.attn.proj.weight")?,
            attn_proj_b: vec_of(w, "pre_block.attn.proj.bias")?,
            proj_w: vec_of(w, "pre_block.proj.weight")?,
            proj_b: vec_of(w, "pre_block.proj.bias")?,
            mlp_norm_w: vec_of(w, "pre_block.mlp.norm.weight")?,
            mlp_norm_b: vec_of(w, "pre_block.mlp.norm.bias")?,
            w0: (
                vec_of(w, "pre_block.mlp.w0.weight")?,
                vec_of(w, "pre_block.mlp.w0.bias")?,
            ),
            w1: (
                vec_of(w, "pre_block.mlp.w1.weight")?,
                vec_of(w, "pre_block.mlp.w1.bias")?,
            ),
            w2: (
                vec_of(w, "pre_block.mlp.w2.weight")?,
                vec_of(w, "pre_block.mlp.w2.bias")?,
            ),
            in_dim: cfg.latent_dim,
            out_dim: cfg.latent_channels,
            heads: cfg.num_attention_heads,
        };

        Ok(Self {
            cfg: cfg.clone(),
            conv_in,
            blocks,
            alpha_out,
            conv_out,
            pre_block,
            mean_proj: plain_conv(w, "mean_proj", true)?,
            logs_proj: plain_conv(w, "logs_proj", true)?,
        })
    }

    #[must_use]
    pub fn config(&self) -> &H3AudioVaeConfig {
        &self.cfg
    }

    /// Encode a **mono** waveform to a latent posterior.
    pub fn encode_mono(&self, wave: &[f32]) -> Result<AudioGaussian> {
        let hop = self.cfg.encoder_hop();
        ensure!(
            wave.len() >= hop,
            "the audio encoder needs at least one {hop}-sample hop, got {}",
            wave.len()
        );
        let x = Signal::from_data(1, wave.to_vec())?;
        let mut h = conv1d(&x, &self.conv_in, 1, 3, 1, 1)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        snake1d(&mut h, &self.alpha_out)?;
        let h = conv1d(&h, &self.conv_out, 1, 1, 1, 1)?;

        // The projection reads the trunk as a sequence, so transpose
        // `[channels][frames]` into `[frames][channels]` and back.
        let seq = h.length;
        let d = h.channels;
        let mut rows = vec![0.0f32; seq * d];
        for c in 0..d {
            for t in 0..seq {
                rows[t * d + c] = h.data[c * seq + t];
            }
        }
        let projected = self.pre_block.forward(&rows, seq)?;
        let o = self.cfg.latent_channels;
        let mut planar = Signal::new(o, seq);
        for t in 0..seq {
            for c in 0..o {
                planar.data[c * seq + t] = projected[t * o + c];
            }
        }

        Ok(AudioGaussian {
            mean: conv1d(&planar, &self.mean_proj, 1, 0, 1, 1)?,
            logs: conv1d(&planar, &self.logs_proj, 1, 0, 1, 1)?,
        })
    }

    /// Latent frames a waveform of `samples` produces.
    #[must_use]
    pub fn latent_frames(&self, samples: usize) -> usize {
        samples / self.cfg.encoder_hop()
    }
}

#[cfg(test)]
mod encoder_tests {
    use super::*;

    #[test]
    fn gelu_tanh_matches_known_values() {
        assert!((gelu_tanh(0.0)).abs() < 1e-7);
        assert!(
            (gelu_tanh(1.0) - 0.841_192).abs() < 1e-4,
            "{}",
            gelu_tanh(1.0)
        );
        assert!((gelu_tanh(-1.0) + 0.158_808).abs() < 1e-4);
        // Large positive inputs approach the identity.
        assert!((gelu_tanh(6.0) - 6.0).abs() < 1e-4);
    }

    #[test]
    fn layer_norm_centres_and_scales_each_row() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let out = layer_norm_rows(&x, 2, 4, &[1.0; 4], &[0.0; 4]);
        for t in 0..2 {
            let row = &out[t * 4..(t + 1) * 4];
            let m: f32 = row.iter().sum::<f32>() / 4.0;
            let v: f32 = row.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / 4.0;
            assert!(m.abs() < 1e-5, "row {t} mean {m}");
            assert!((v - 1.0).abs() < 1e-3, "row {t} variance {v}");
        }
    }

    #[test]
    fn layer_norm_applies_its_affine() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = layer_norm_rows(&x, 1, 4, &[2.0; 4], &[7.0; 4]);
        let m: f32 = out.iter().sum::<f32>() / 4.0;
        assert!((m - 7.0).abs() < 1e-5, "{m}");
    }

    #[test]
    fn audio_gaussian_uses_log_std_not_log_variance() {
        // sample = mean + exp(logs) * noise, so logs = 0 gives unit std.
        let mean = Signal::from_data(1, vec![0.0; 4096]).unwrap();
        let logs = Signal::from_data(1, vec![0.0; 4096]).unwrap();
        let g = AudioGaussian { mean, logs };
        let s = g.sample(3);
        let var = s.data.iter().map(|v| v * v).sum::<f32>() / s.data.len() as f32;
        assert!((var - 1.0).abs() < 0.15, "variance {var}");
        assert_eq!(g.mode().data, vec![0.0; 4096]);
    }

    #[test]
    fn audio_gaussian_sampling_is_seeded_and_clamped() {
        let mean = Signal::from_data(1, vec![0.0, 0.0]).unwrap();
        let logs = Signal::from_data(1, vec![1.0e6, -1.0e6]).unwrap();
        let g = AudioGaussian { mean, logs };
        let a = g.sample(9);
        assert_eq!(a.data, g.sample(9).data);
        assert!(a.data.iter().all(|v| v.is_finite()), "logs must be clamped");
    }

    #[test]
    fn encoder_hop_maps_samples_to_latent_frames() {
        let cfg = H3AudioVaeConfig {
            encoder_dim: 64,
            encoder_rates: vec![2, 4, 4, 5, 5],
            latent_dim: 2048,
            latent_channels: 32,
            decoder_dim: 1024,
            decoder_rates: vec![5, 5, 2, 2, 2, 2, 2],
            decoder_kernel_sizes: vec![9, 9, 4, 4, 4, 4, 4],
            num_attention_heads: 8,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5]; 3],
            sampling_rate: 32_000,
            latents_mean: vec![],
            latents_std: vec![],
        };
        // The strided-convolution chain has to land exactly on the 800-sample
        // hop, or the encoder and decoder disagree about the latent rate.
        let mut len = 1600usize;
        for &s in &cfg.encoder_rates {
            let pad = s.div_ceil(2);
            len = (len + 2 * pad - 2 * s) / s + 1;
        }
        assert_eq!(
            len, 2,
            "1600 samples must encode to exactly 2 latent frames"
        );
    }
}

/// Every tensor the audio VAE ships, split by which half reads it.
///
/// Returns `(encoder_side, decoder_side)`. The encoder side covers the DAC
/// trunk, the `pre_block` projection and the two posterior heads; the decoder
/// side covers `dec_in_proj` and BigVGAN.
///
/// This exists so a completeness check can assert the two halves account for
/// the whole checkpoint. A skipped module produces finite, plausible, wrong
/// audio — it is not visible in shapes.
#[must_use]
pub fn audio_parameter_keys(cfg: &H3AudioVaeConfig) -> (Vec<String>, Vec<String>) {
    let wn = |p: &str, bias: bool| -> Vec<String> {
        let mut v = vec![format!("{p}.weight_g"), format!("{p}.weight_v")];
        if bias {
            v.push(format!("{p}.bias"));
        }
        v
    };
    let act = |p: &str| -> Vec<String> {
        vec![
            format!("{p}.act.alpha"),
            format!("{p}.act.beta"),
            format!("{p}.upsample.filter"),
            format!("{p}.downsample.lowpass.filter"),
        ]
    };

    // ---- encoder side ----
    let mut enc: Vec<String> = wn("encoder.block.0", true);
    for i in 0..cfg.encoder_rates.len() {
        let p = format!("encoder.block.{}", i + 1);
        for u in 0..3 {
            let up = format!("{p}.block.{u}.block");
            enc.push(format!("{up}.0.alpha"));
            enc.extend(wn(&format!("{up}.1"), true));
            enc.push(format!("{up}.2.alpha"));
            enc.extend(wn(&format!("{up}.3"), true));
        }
        enc.push(format!("{p}.block.3.alpha"));
        enc.extend(wn(&format!("{p}.block.4"), true));
    }
    let tail = cfg.encoder_rates.len() + 1;
    enc.push(format!("encoder.block.{tail}.alpha"));
    enc.extend(wn(&format!("encoder.block.{}", tail + 1), true));

    for p in ["pre_block.norm1", "pre_block.norm2", "pre_block.norm3"] {
        enc.push(format!("{p}.weight"));
        enc.push(format!("{p}.bias"));
    }
    enc.extend([
        "pre_block.attn.qkv.weight".to_string(),
        "pre_block.attn.q_bias".to_string(),
        "pre_block.attn.v_bias".to_string(),
        "pre_block.attn.zero_k_bias".to_string(),
        "pre_block.attn.proj.weight".to_string(),
        "pre_block.attn.proj.bias".to_string(),
        "pre_block.proj.weight".to_string(),
        "pre_block.proj.bias".to_string(),
    ]);
    for p in [
        "pre_block.mlp.norm",
        "pre_block.mlp.w0",
        "pre_block.mlp.w1",
        "pre_block.mlp.w2",
    ] {
        enc.push(format!("{p}.weight"));
        enc.push(format!("{p}.bias"));
    }
    for p in ["mean_proj", "logs_proj"] {
        enc.push(format!("{p}.weight"));
        enc.push(format!("{p}.bias"));
    }

    // ---- decoder side ----
    let mut dec = vec![
        "dec_in_proj.weight".to_string(),
        "dec_in_proj.bias".to_string(),
    ];
    dec.extend(wn("decoder.conv_pre", true));
    for i in 0..cfg.decoder_rates.len() {
        dec.extend(wn(&format!("decoder.ups.{i}.0"), true));
    }
    let n_kernels = cfg.resblock_kernel_sizes.len();
    for stage in 0..cfg.decoder_rates.len() {
        for k in 0..n_kernels {
            let idx = stage * n_kernels + k;
            let p = format!("decoder.resblocks.{idx}");
            let dilations = cfg.resblock_dilation_sizes[k].len();
            for j in 0..dilations {
                dec.extend(wn(&format!("{p}.convs1.{j}"), true));
                dec.extend(wn(&format!("{p}.convs2.{j}"), true));
            }
            for j in 0..2 * dilations {
                dec.extend(act(&format!("{p}.activations.{j}")));
            }
        }
    }
    dec.extend(act("decoder.activation_post"));
    dec.extend(wn("decoder.conv_post", false));
    (enc, dec)
}
