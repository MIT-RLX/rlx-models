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

//! Native speech tokenizer **encode** — Mimi SEANet conv encoder stack.
//!
//! Layer-by-layer parity-tested against `transformers.MimiEncoder` from the HF
//! `qwen_tts` Base checkpoint. The transformer + RVQ stages are separate
//! follow-on PRs; this module only covers the conv stack:
//!
//!   conv k=7 (1→64)
//!   { ResnetBlock @ N → ELU → downsample (kernel=2*r, stride=r, N→2N) } x4
//!   ELU
//!   conv k=3 (1024→512)
//!
//! Output: `[hidden_size=512, T]` where `T = ceil(L / prod(downsample_strides))`.

use anyhow::{Context, Result, bail, ensure};
use ndarray::{Array1, Array2, Array3, ArrayView2};
use std::collections::HashMap;

const ENC_PREFIX: &str = "encoder.encoder.";

#[derive(Debug, Clone)]
pub struct ConvEncoderConfig {
    pub audio_channels: usize,
    pub num_filters: usize,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub hidden_size: usize,
    /// HF `upsampling_ratios` (downsample strides applied in reverse order).
    pub upsampling_ratios: Vec<usize>,
    pub num_residual_layers: usize,
    pub residual_kernel_size: usize,
    pub dilation_growth_rate: usize,
    pub compress: usize,
}

impl ConvEncoderConfig {
    pub fn from_speech_tokenizer_dir(dir: &std::path::Path) -> Result<Self> {
        let cfg_path = dir.join("config.json");
        let text = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let enc = v
            .get("encoder_config")
            .context("missing encoder_config in speech_tokenizer config.json")?;
        let get_usize = |k: &str| -> Result<usize> {
            enc.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .with_context(|| format!("missing encoder_config.{k}"))
        };
        let upsampling_ratios = enc
            .get("upsampling_ratios")
            .and_then(|x| x.as_array())
            .context("upsampling_ratios")?
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        Ok(Self {
            audio_channels: get_usize("audio_channels")?,
            num_filters: get_usize("num_filters")?,
            kernel_size: get_usize("kernel_size")?,
            last_kernel_size: get_usize("last_kernel_size")?,
            hidden_size: get_usize("hidden_size")?,
            upsampling_ratios,
            num_residual_layers: get_usize("num_residual_layers")?,
            residual_kernel_size: get_usize("residual_kernel_size")?,
            dilation_growth_rate: get_usize("dilation_growth_rate")?,
            compress: get_usize("compress")?,
        })
    }
}

// -----------------------------------------------------------------------------
// Mimi-faithful causal conv1d.

/// Mimi pad mode for `mimi_causal_conv1d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    /// Zero padding (encoder.encoder convs).
    Constant,
    /// Edge replicate (encoder.downsample).
    Replicate,
}

/// Causal 1D conv with Mimi's asymmetric pad. Pads left by `(k-1)*d - s + 1`
/// (all of `padding_total`) and right by `extra_padding` so the output length
/// equals `ceil(L / stride)` per the SEANet conv.
pub fn mimi_causal_conv1d(
    x: ArrayView2<f32>,
    weight: &Array3<f32>,
    bias: Option<&Array1<f32>>,
    stride: usize,
    dilation: usize,
    pad_mode: PadMode,
) -> Array2<f32> {
    let (out_ch, in_ch, k) = weight.dim();
    debug_assert_eq!(x.dim().0, in_ch);
    let t_in = x.dim().1;
    let effective_k = (k - 1) * dilation + 1;
    let padding_total = effective_k.saturating_sub(stride);

    // Mimi: n_frames = ceil((L - eff_k + padding_total) / stride + 1) - 1
    //              = ceil((L - eff_k + padding_total) / stride)
    // Output length = n_frames + 1.
    let num = t_in as i64 - effective_k as i64 + padding_total as i64;
    let n_frames = if num <= 0 {
        0usize
    } else {
        (num as usize).div_ceil(stride)
    };
    let ideal_len = n_frames * stride + effective_k - padding_total;
    let extra_right = ideal_len.saturating_sub(t_in);

    let pad_left = padding_total;
    let pad_right = extra_right;
    let t_pad = t_in + pad_left + pad_right;

    let mut padded = vec![0f32; in_ch * t_pad];
    for ci in 0..in_ch {
        let row = x.row(ci);
        let base = ci * t_pad;
        match pad_mode {
            PadMode::Constant => {
                for i in 0..t_in {
                    padded[base + pad_left + i] = row[i];
                }
            }
            PadMode::Replicate => {
                let edge_left = row[0];
                let edge_right = row[t_in - 1];
                for i in 0..pad_left {
                    padded[base + i] = edge_left;
                }
                for i in 0..t_in {
                    padded[base + pad_left + i] = row[i];
                }
                for i in 0..pad_right {
                    padded[base + pad_left + t_in + i] = edge_right;
                }
            }
        }
    }
    let t_out = (t_pad - effective_k) / stride + 1;

    // im2col + gemm.
    let patch = in_ch * k;
    let mut col = vec![0f32; t_out * patch];
    for ti in 0..t_out {
        let row_base = ti * patch;
        for ci in 0..in_ch {
            let pad_base = ci * t_pad + ti * stride;
            let dst_base = row_base + ci * k;
            for ki in 0..k {
                col[dst_base + ki] = padded[pad_base + ki * dilation];
            }
        }
    }
    let mut w_flat = vec![0f32; out_ch * patch];
    for oc in 0..out_ch {
        for ci in 0..in_ch {
            for ki in 0..k {
                w_flat[oc * patch + ci * k + ki] = weight[[oc, ci, ki]];
            }
        }
    }

    // col [T_out, patch] @ w_flat.T [patch, out_ch] → [T_out, out_ch].
    use ndarray::ArrayView2 as V;
    let col_a = V::from_shape((t_out, patch), &col).unwrap();
    let w_a = V::from_shape((out_ch, patch), &w_flat).unwrap();
    let out_tc = col_a.dot(&w_a.t());

    // Transpose back to [out_ch, t_out] + bias.
    let mut out = Array2::<f32>::zeros((out_ch, t_out));
    for oc in 0..out_ch {
        let b = bias.map(|b| b[oc]).unwrap_or(0.0);
        for ti in 0..t_out {
            out[[oc, ti]] = out_tc[[ti, oc]] + b;
        }
    }
    out
}

#[inline]
fn elu(x: f32) -> f32 {
    if x >= 0.0 { x } else { x.exp() - 1.0 }
}

fn elu_inplace(a: &mut Array2<f32>) {
    for v in a.iter_mut() {
        *v = elu(*v);
    }
}

// -----------------------------------------------------------------------------
// Layers.

#[derive(Debug, Clone)]
struct Conv1dLayer {
    weight: Array3<f32>,
    bias: Option<Array1<f32>>,
    stride: usize,
    dilation: usize,
    pad_mode: PadMode,
}

impl Conv1dLayer {
    fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        mimi_causal_conv1d(
            x,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.dilation,
            self.pad_mode,
        )
    }
}

#[derive(Debug, Clone)]
struct ResnetBlock {
    /// `block.1` — k=residual_kernel_size, dilation=d
    conv_a: Conv1dLayer,
    /// `block.3` — k=1, dilation=1
    conv_b: Conv1dLayer,
}

impl ResnetBlock {
    fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let residual = x.to_owned();
        let mut h = x.to_owned();
        elu_inplace(&mut h);
        let h = self.conv_a.forward(h.view());
        let mut h = h;
        elu_inplace(&mut h);
        let h = self.conv_b.forward(h.view());
        let mut h = h;
        ensure_eq_shape(&h, &residual);
        for ((c, t), v) in residual.indexed_iter() {
            h[[c, t]] += *v;
        }
        h
    }
}

fn ensure_eq_shape(a: &Array2<f32>, b: &Array2<f32>) {
    debug_assert_eq!(a.dim(), b.dim(), "shape mismatch in resnet residual");
}

#[derive(Debug, Clone)]
enum EncoderLayer {
    Conv(Conv1dLayer),
    Res(ResnetBlock),
    Elu,
}

impl EncoderLayer {
    fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        match self {
            EncoderLayer::Conv(c) => c.forward(x),
            EncoderLayer::Res(r) => r.forward(x),
            EncoderLayer::Elu => {
                let mut out = x.to_owned();
                elu_inplace(&mut out);
                out
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MimiConvEncoder {
    pub cfg: ConvEncoderConfig,
    layers: Vec<EncoderLayer>,
}

impl MimiConvEncoder {
    /// Forward the conv encoder. Input is `[audio_channels, T_in]`, output is
    /// `[hidden_size, T_out]` with `T_out ≈ ceil(T_in / prod(strides))`.
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let mut h = x.to_owned();
        for layer in &self.layers {
            h = layer.forward(h.view());
        }
        h
    }

    /// Forward, returning the per-layer intermediates in addition to the final
    /// output. Used by the parity test.
    pub fn forward_with_intermediates(
        &self,
        x: ArrayView2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let mut outs = Vec::with_capacity(self.layers.len());
        let mut h = x.to_owned();
        for layer in &self.layers {
            h = layer.forward(h.view());
            outs.push(h.clone());
        }
        (h, outs)
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

// -----------------------------------------------------------------------------
// Weight loader.

fn take_conv(
    raw: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    dilation: usize,
) -> Result<Conv1dLayer> {
    take_conv_with(
        raw,
        prefix,
        in_c,
        out_c,
        k,
        stride,
        dilation,
        true,
        PadMode::Constant,
    )
}

fn take_conv_with(
    raw: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    has_bias: bool,
    pad_mode: PadMode,
) -> Result<Conv1dLayer> {
    let w_key = format!("{prefix}.conv.weight");
    let (w_data, w_shape) = raw
        .remove(&w_key)
        .with_context(|| format!("missing tensor {w_key}"))?;
    ensure!(
        w_shape == vec![out_c, in_c, k],
        "{w_key} shape {:?} != [{out_c}, {in_c}, {k}]",
        w_shape
    );
    let weight = Array3::from_shape_vec((out_c, in_c, k), w_data).context("conv weight reshape")?;
    let bias = if has_bias {
        let b_key = format!("{prefix}.conv.bias");
        let (b_data, b_shape) = raw
            .remove(&b_key)
            .with_context(|| format!("missing tensor {b_key}"))?;
        ensure!(
            b_shape == vec![out_c],
            "{b_key} shape {:?} != [{out_c}]",
            b_shape
        );
        Some(Array1::from_vec(b_data))
    } else {
        None
    };
    Ok(Conv1dLayer {
        weight,
        bias,
        stride,
        dilation,
        pad_mode,
    })
}

pub fn build_conv_encoder(
    cfg: &ConvEncoderConfig,
    raw: HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<MimiConvEncoder> {
    // Strip prefix from keys before passing to take_conv.
    let mut local: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        if let Some(rest) = k.strip_prefix(ENC_PREFIX) {
            local.insert(rest.to_string(), v);
        }
    }

    let mut layers: Vec<EncoderLayer> = Vec::new();
    let mut layer_idx = 0usize;

    // 0: init conv
    layers.push(EncoderLayer::Conv(take_conv(
        &mut local,
        &format!("layers.{layer_idx}"),
        cfg.audio_channels,
        cfg.num_filters,
        cfg.kernel_size,
        1,
        1,
    )?));
    layer_idx += 1;

    // For each ratio in reversed(upsampling_ratios): residual + ELU + downsample
    let mut scaling = 1usize;
    let ratios_rev: Vec<usize> = cfg.upsampling_ratios.iter().rev().copied().collect();
    for ratio in &ratios_rev {
        let current_scale = scaling * cfg.num_filters;
        for j in 0..cfg.num_residual_layers {
            let d = cfg.dilation_growth_rate.pow(j as u32);
            let block = ResnetBlock {
                conv_a: take_conv(
                    &mut local,
                    &format!("layers.{layer_idx}.block.1"),
                    current_scale,
                    current_scale / cfg.compress,
                    cfg.residual_kernel_size,
                    1,
                    d,
                )?,
                conv_b: take_conv(
                    &mut local,
                    &format!("layers.{layer_idx}.block.3"),
                    current_scale / cfg.compress,
                    current_scale,
                    1,
                    1,
                    1,
                )?,
            };
            layers.push(EncoderLayer::Res(block));
            layer_idx += 1;
        }
        // ELU activation
        layers.push(EncoderLayer::Elu);
        layer_idx += 1;
        // Downsample conv: kernel=2*ratio, stride=ratio, current_scale → current_scale*2
        layers.push(EncoderLayer::Conv(take_conv(
            &mut local,
            &format!("layers.{layer_idx}"),
            current_scale,
            current_scale * 2,
            ratio * 2,
            *ratio,
            1,
        )?));
        layer_idx += 1;
        scaling *= 2;
    }

    // Final ELU + conv
    layers.push(EncoderLayer::Elu);
    layer_idx += 1;
    layers.push(EncoderLayer::Conv(take_conv(
        &mut local,
        &format!("layers.{layer_idx}"),
        scaling * cfg.num_filters,
        cfg.hidden_size,
        cfg.last_kernel_size,
        1,
        1,
    )?));
    layer_idx += 1;
    let _ = layer_idx;

    if !local.is_empty() {
        let leftover: Vec<&String> = local.keys().take(6).collect();
        bail!(
            "{} unused encoder tensors (first: {:?})",
            local.len(),
            leftover
        );
    }
    Ok(MimiConvEncoder {
        cfg: cfg.clone(),
        layers,
    })
}

/// `encoder.downsample` — stride-2 conv with kernel=4, replicate pad, no bias.
/// Reduces the post-transformer hidden from `[C, T]` to `[C, ceil(T/2)]`.
#[derive(Debug, Clone)]
pub struct MimiDownsample {
    conv: Conv1dLayer,
}

impl MimiDownsample {
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        self.conv.forward(x)
    }

    pub fn open(tok_dir: &std::path::Path, hidden_size: usize) -> Result<Self> {
        let ckpt = rlx_core::safetensors_checkpoint::SafetensorsCheckpoint::open(tok_dir)?;
        let key = "encoder.downsample.conv.weight".to_string();
        let want: std::collections::HashSet<String> = [key.clone()].into_iter().collect();
        let mut wm = ckpt.load_selected(&want)?;
        let (data, shape) = wm.take(&key)?;
        ensure!(
            shape == vec![hidden_size, hidden_size, 4],
            "downsample shape {:?} != [{hidden_size}, {hidden_size}, 4]",
            shape
        );
        let weight = Array3::from_shape_vec((hidden_size, hidden_size, 4), data)?;
        Ok(Self {
            conv: Conv1dLayer {
                weight,
                bias: None,
                stride: 2,
                dilation: 1,
                pad_mode: PadMode::Replicate,
            },
        })
    }
}

/// Build encoder from a `Qwen3-TTS-Base/speech_tokenizer/` directory.
pub fn open_conv_encoder(tok_dir: &std::path::Path) -> Result<MimiConvEncoder> {
    let cfg = ConvEncoderConfig::from_speech_tokenizer_dir(tok_dir)?;
    let ckpt = rlx_core::safetensors_checkpoint::SafetensorsCheckpoint::open(tok_dir)?;
    let want: std::collections::HashSet<String> = ckpt
        .keys()
        .filter(|k| k.starts_with(ENC_PREFIX))
        .map(str::to_string)
        .collect();
    if want.is_empty() {
        bail!("no encoder.encoder.* tensors under {}", tok_dir.display());
    }
    let mut wm = ckpt.load_selected(&want)?;
    let mut raw: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::with_capacity(want.len());
    for k in want.iter() {
        let (data, shape) = wm.take(k)?;
        raw.insert(k.clone(), (data, shape));
    }
    build_conv_encoder(&cfg, raw)
}
