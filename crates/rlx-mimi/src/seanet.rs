use crate::config::MimiConfig;
use crate::conv::{PadMode, conv_transpose1d, elu_inplace, mimi_causal_conv1d};
use anyhow::{Context, Result, bail, ensure};
use ndarray::{Array1, Array2, Array3, ArrayView2};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct Conv1dLayer {
    pub(crate) weight: Array3<f32>,
    pub(crate) bias: Option<Array1<f32>>,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
    pub(crate) pad_mode: PadMode,
}

impl Conv1dLayer {
    pub(crate) fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
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
pub(crate) struct ResnetBlock {
    pub(crate) conv_a: Conv1dLayer,
    pub(crate) conv_b: Conv1dLayer,
}

impl ResnetBlock {
    pub(crate) fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let residual = x.to_owned();
        let mut h = x.to_owned();
        elu_inplace(&mut h);
        let h = self.conv_a.forward(h.view());
        let mut h = h;
        elu_inplace(&mut h);
        let h = self.conv_b.forward(h.view());
        let mut h = h;
        for ((c, t), v) in residual.indexed_iter() {
            h[[c, t]] += *v;
        }
        h
    }
}

#[derive(Debug, Clone)]
pub(crate) enum SeanetLayer {
    Conv(Conv1dLayer),
    TransConv {
        weight: Array3<f32>,
        bias: Option<Array1<f32>>,
        stride: usize,
        trim_right_ratio: f32,
    },
    Res(ResnetBlock),
    Elu,
}

impl SeanetLayer {
    fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        match self {
            SeanetLayer::Conv(c) => c.forward(x),
            SeanetLayer::TransConv {
                weight,
                bias,
                stride,
                trim_right_ratio,
            } => conv_transpose1d(x, weight, bias.as_ref(), *stride, *trim_right_ratio),
            SeanetLayer::Res(r) => r.forward(x),
            SeanetLayer::Elu => {
                let mut out = x.to_owned();
                elu_inplace(&mut out);
                out
            }
        }
    }
}

pub struct SeanetEncoder {
    pub(crate) layers: Vec<SeanetLayer>,
}

impl SeanetEncoder {
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let mut h = x.to_owned();
        for layer in &self.layers {
            h = layer.forward(h.view());
        }
        h
    }
}

pub struct SeanetDecoder {
    pub(crate) layers: Vec<SeanetLayer>,
}

impl SeanetDecoder {
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let mut h = x.to_owned();
        for layer in &self.layers {
            h = layer.forward(h.view());
        }
        h
    }
}

fn take_conv(
    raw: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    with_bias: bool,
    pad_mode: PadMode,
) -> Result<Conv1dLayer> {
    let w_key = format!("{prefix}.conv.weight");
    let (w_data, w_shape) = raw
        .remove(&w_key)
        .with_context(|| format!("missing {w_key}"))?;
    ensure!(
        w_shape == vec![out_c, in_c, k],
        "{w_key} shape {:?} != [{out_c}, {in_c}, {k}]",
        w_shape
    );
    let weight = Array3::from_shape_vec((out_c, in_c, k), w_data)?;
    let bias = if with_bias {
        let b_key = format!("{prefix}.conv.bias");
        let (b_data, b_shape) = raw
            .remove(&b_key)
            .with_context(|| format!("missing {b_key}"))?;
        ensure!(b_shape == vec![out_c], "{b_key} shape {b_shape:?}");
        Some(Array1::from_vec(b_data))
    } else {
        raw.remove(&format!("{prefix}.conv.bias"));
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

fn take_trans_conv(
    raw: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    k: usize,
    _stride: usize,
    with_bias: bool,
) -> Result<(Array3<f32>, Option<Array1<f32>>)> {
    let w_key = format!("{prefix}.conv.weight");
    let (w_data, w_shape) = raw
        .remove(&w_key)
        .with_context(|| format!("missing {w_key}"))?;
    ensure!(
        w_shape == vec![in_c, out_c, k],
        "{w_key} shape {:?} != [{in_c}, {out_c}, {k}]",
        w_shape
    );
    let weight = Array3::from_shape_vec((in_c, out_c, k), w_data)?;
    let bias = if with_bias {
        let b_key = format!("{prefix}.conv.bias");
        let (b_data, b_shape) = raw
            .remove(&b_key)
            .with_context(|| format!("missing {b_key}"))?;
        ensure!(b_shape == vec![out_c], "{b_key} shape {b_shape:?}");
        Some(Array1::from_vec(b_data))
    } else {
        raw.remove(&format!("{prefix}.conv.bias"));
        None
    };
    Ok((weight, bias))
}

pub fn build_encoder(
    cfg: &MimiConfig,
    raw: HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<SeanetEncoder> {
    let mut local: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for (k, v) in raw {
        if let Some(rest) = k.strip_prefix("encoder.") {
            local.insert(rest.to_string(), v);
        }
    }
    let mut layers = Vec::new();
    let mut layer_idx = 0usize;
    layers.push(SeanetLayer::Conv(take_conv(
        &mut local,
        &format!("layers.{layer_idx}"),
        cfg.audio_channels,
        cfg.num_filters,
        cfg.kernel_size,
        1,
        1,
        true,
        PadMode::Constant,
    )?));
    layer_idx += 1;

    let mut scaling = 1usize;
    for ratio in cfg.upsampling_ratios.iter().rev() {
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
                    true,
                    PadMode::Constant,
                )?,
                conv_b: take_conv(
                    &mut local,
                    &format!("layers.{layer_idx}.block.3"),
                    current_scale / cfg.compress,
                    current_scale,
                    1,
                    1,
                    1,
                    true,
                    PadMode::Constant,
                )?,
            };
            layers.push(SeanetLayer::Res(block));
            layer_idx += 1;
        }
        layers.push(SeanetLayer::Elu);
        layer_idx += 1;
        layers.push(SeanetLayer::Conv(take_conv(
            &mut local,
            &format!("layers.{layer_idx}"),
            current_scale,
            current_scale * 2,
            ratio * 2,
            *ratio,
            1,
            true,
            PadMode::Constant,
        )?));
        layer_idx += 1;
        scaling *= 2;
    }
    layers.push(SeanetLayer::Elu);
    layer_idx += 1;
    layers.push(SeanetLayer::Conv(take_conv(
        &mut local,
        &format!("layers.{layer_idx}"),
        scaling * cfg.num_filters,
        cfg.hidden_size,
        cfg.last_kernel_size,
        1,
        1,
        true,
        PadMode::Constant,
    )?));
    if !local.is_empty() {
        bail!(
            "unused encoder tensors: {:?}",
            local.keys().take(4).collect::<Vec<_>>()
        );
    }
    Ok(SeanetEncoder { layers })
}

pub fn build_decoder(
    cfg: &MimiConfig,
    raw: HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<SeanetDecoder> {
    let mut local: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for (k, v) in raw {
        if let Some(rest) = k.strip_prefix("decoder.") {
            local.insert(rest.to_string(), v);
        }
    }
    let mut layers = Vec::new();
    let mut layer_idx = 0usize;
    let mut scaling = 1usize << cfg.upsampling_ratios.len();
    layers.push(SeanetLayer::Conv(take_conv(
        &mut local,
        &format!("layers.{layer_idx}"),
        cfg.hidden_size,
        scaling * cfg.num_filters,
        cfg.kernel_size,
        1,
        1,
        true,
        PadMode::Constant,
    )?));
    layer_idx += 1;

    for ratio in &cfg.upsampling_ratios {
        let current_scale = scaling * cfg.num_filters;
        layers.push(SeanetLayer::Elu);
        layer_idx += 1;
        let (w, b) = take_trans_conv(
            &mut local,
            &format!("layers.{layer_idx}"),
            current_scale,
            current_scale / 2,
            ratio * 2,
            *ratio,
            true,
        )?;
        layers.push(SeanetLayer::TransConv {
            weight: w,
            bias: b,
            stride: *ratio,
            trim_right_ratio: cfg.trim_right_ratio,
        });
        layer_idx += 1;
        for j in 0..cfg.num_residual_layers {
            let d = cfg.dilation_growth_rate.pow(j as u32);
            let block = ResnetBlock {
                conv_a: take_conv(
                    &mut local,
                    &format!("layers.{layer_idx}.block.1"),
                    current_scale / 2,
                    current_scale / (2 * cfg.compress),
                    cfg.residual_kernel_size,
                    1,
                    d,
                    true,
                    PadMode::Constant,
                )?,
                conv_b: take_conv(
                    &mut local,
                    &format!("layers.{layer_idx}.block.3"),
                    current_scale / (2 * cfg.compress),
                    current_scale / 2,
                    1,
                    1,
                    1,
                    true,
                    PadMode::Constant,
                )?,
            };
            layers.push(SeanetLayer::Res(block));
            layer_idx += 1;
        }
        scaling /= 2;
    }
    layers.push(SeanetLayer::Elu);
    layer_idx += 1;
    layers.push(SeanetLayer::Conv(take_conv(
        &mut local,
        &format!("layers.{layer_idx}"),
        cfg.num_filters,
        cfg.audio_channels,
        cfg.last_kernel_size,
        1,
        1,
        true,
        PadMode::Constant,
    )?));
    if !local.is_empty() {
        bail!(
            "unused decoder tensors: {:?}",
            local.keys().take(4).collect::<Vec<_>>()
        );
    }
    Ok(SeanetDecoder { layers })
}

pub struct FrameRateDownsample {
    pub(crate) conv: Conv1dLayer,
}

impl FrameRateDownsample {
    pub fn from_weights(
        cfg: &MimiConfig,
        raw: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Result<Self> {
        let k = cfg.frame_rate_kernel();
        let w_key = "downsample.conv.weight".to_string();
        let (w_data, w_shape) = raw
            .remove(&w_key)
            .with_context(|| format!("missing {w_key}"))?;
        ensure!(
            w_shape == vec![cfg.hidden_size, cfg.hidden_size, k],
            "{w_key} shape {w_shape:?}"
        );
        raw.remove("downsample.conv.bias");
        Ok(Self {
            conv: Conv1dLayer {
                weight: Array3::from_shape_vec((cfg.hidden_size, cfg.hidden_size, k), w_data)?,
                bias: None,
                stride: 2,
                dilation: 1,
                pad_mode: PadMode::Replicate,
            },
        })
    }

    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        self.conv.forward(x)
    }
}

pub struct FrameRateUpsample {
    pub(crate) weight: Array3<f32>,
    pub(crate) stride: usize,
    pub(crate) trim_right_ratio: f32,
}

impl FrameRateUpsample {
    pub fn from_weights(
        cfg: &MimiConfig,
        raw: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Result<Self> {
        let k = cfg.frame_rate_kernel();
        let w_key = "upsample.conv.weight".to_string();
        let (w_data, w_shape) = raw
            .remove(&w_key)
            .with_context(|| format!("missing {w_key}"))?;
        ensure!(
            w_shape == vec![cfg.hidden_size, 1, k],
            "{w_key} shape {w_shape:?}"
        );
        raw.remove("upsample.conv.bias");
        Ok(Self {
            weight: Array3::from_shape_vec((cfg.hidden_size, 1, k), w_data)?,
            stride: 2,
            trim_right_ratio: cfg.trim_right_ratio,
        })
    }

    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        crate::conv::grouped_conv_transpose1d(x, &self.weight, self.stride, self.trim_right_ratio)
    }
}
