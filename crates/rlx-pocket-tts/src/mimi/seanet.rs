// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! SEANet decoder — convolutional inverse for the Mimi codec.
//!
//! With `ratios=[6,5,4]`, `n_filters=64`, `dimension=512`, `n_residual_layers=1`,
//! the layer indices in the safetensors are:
//! ```text
//! model.0  Conv1d(512 → 512, k=7)
//! model.2  ConvTranspose1d(512 → 256, k=12, stride=6)
//! model.3  ResnetBlock(256)  — block.1 Conv1d(256→128 k=3), block.3 Conv1d(128→256 k=1)
//! model.5  ConvTranspose1d(256 → 128, k=10, stride=5)
//! model.6  ResnetBlock(128)
//! model.8  ConvTranspose1d(128 → 64,  k=8,  stride=4)
//! model.9  ResnetBlock(64)
//! model.11 Conv1d(64 → 1, k=3)
//! ```
//!
//! ELUs at the even / pre-conv positions are bias-free and weightless.

use anyhow::Result;
use ndarray::{Array1, Array2, Array3};

use crate::config::MimiConfig;
use crate::ops::{PadMode, causal_conv_transpose1d, causal_conv1d, elu_inplace};
use crate::weights::WeightFile;

#[derive(Debug, Clone)]
struct Conv1dLayer {
    weight: Array3<f32>,
    bias: Option<Array1<f32>>,
    stride: usize,
    dilation: usize,
    pad_mode: PadMode,
}

impl Conv1dLayer {
    fn forward(&self, x: ndarray::ArrayView2<f32>) -> Array2<f32> {
        causal_conv1d(
            x,
            self.weight.view(),
            self.bias.as_ref().map(|b| b.view()),
            self.stride,
            self.dilation,
            self.pad_mode,
            1,
        )
    }
}

#[derive(Debug, Clone)]
struct TransConv1dLayer {
    weight: Array3<f32>,
    bias: Option<Array1<f32>>,
    stride: usize,
}

impl TransConv1dLayer {
    fn forward(&self, x: ndarray::ArrayView2<f32>) -> Array2<f32> {
        causal_conv_transpose1d(
            x,
            self.weight.view(),
            self.bias.as_ref().map(|b| b.view()),
            self.stride,
            1,
        )
    }
}

#[derive(Debug, Clone)]
struct ResnetBlock {
    /// block[1] = Conv1d(dim → dim/compress, k=residual_kernel_size, dilation=base^j)
    conv_a: Conv1dLayer,
    /// block[3] = Conv1d(dim/compress → dim, k=1)
    conv_b: Conv1dLayer,
}

impl ResnetBlock {
    fn forward(&self, x: ndarray::ArrayView2<f32>) -> Array2<f32> {
        let mut h = x.to_owned();
        elu_inplace(h.as_slice_mut().unwrap());
        let h = self.conv_a.forward(h.view());
        let mut h = h;
        elu_inplace(h.as_slice_mut().unwrap());
        let h = self.conv_b.forward(h.view());
        let mut out = x.to_owned();
        // Residual: out += h. (Shapes must match — both [dim, T].)
        let (c, t) = out.dim();
        debug_assert_eq!(h.dim(), (c, t));
        for ci in 0..c {
            for ti in 0..t {
                out[[ci, ti]] += h[[ci, ti]];
            }
        }
        out
    }
}

enum Layer {
    Conv(Conv1dLayer),
    TransConv(TransConv1dLayer),
    Resnet(ResnetBlock),
    /// ELU activation (in place). Stored as a stand-in for clarity.
    Elu,
}

pub struct SeanetDecoder {
    layers: Vec<Layer>,
}

impl SeanetDecoder {
    pub fn load(wf: &WeightFile, prefix: &str, cfg: &MimiConfig) -> Result<Self> {
        let pad = PadMode::Constant; // pocket_tts english config sets pad_mode="constant"
        let n_filters = cfg.n_filters;
        let dimension = cfg.outer_dim;
        let n_res_layers = 1usize;

        let mut layers: Vec<Layer> = Vec::new();
        let mut idx;

        // model.0 — Conv1d(dimension → mult*n_filters, k=kernel_size)
        let mult_init: usize = 1 << cfg.ratios.len();
        let conv0_w = wf.get_3d(&format!("{prefix}.model.0.conv.weight"))?;
        let conv0_b = wf.opt_1d(&format!("{prefix}.model.0.conv.bias"))?;
        layers.push(Layer::Conv(Conv1dLayer {
            weight: conv0_w,
            bias: conv0_b,
            stride: 1,
            dilation: 1,
            pad_mode: pad,
        }));
        idx = 1usize;
        debug_assert_eq!(mult_init * n_filters, dimension);

        for &ratio in &cfg.ratios {
            layers.push(Layer::Elu);
            idx += 1;

            let tw = wf.get_3d(&format!("{prefix}.model.{idx}.convtr.weight"))?;
            let tb = wf.opt_1d(&format!("{prefix}.model.{idx}.convtr.bias"))?;
            layers.push(Layer::TransConv(TransConv1dLayer {
                weight: tw,
                bias: tb,
                stride: ratio,
            }));
            idx += 1;

            for _ in 0..n_res_layers {
                let bp = format!("{prefix}.model.{idx}.block");
                let aw = wf.get_3d(&format!("{bp}.1.conv.weight"))?;
                let ab = wf.opt_1d(&format!("{bp}.1.conv.bias"))?;
                let bw = wf.get_3d(&format!("{bp}.3.conv.weight"))?;
                let bb = wf.opt_1d(&format!("{bp}.3.conv.bias"))?;
                layers.push(Layer::Resnet(ResnetBlock {
                    conv_a: Conv1dLayer {
                        weight: aw,
                        bias: ab,
                        stride: 1,
                        dilation: 1,
                        pad_mode: pad,
                    },
                    conv_b: Conv1dLayer {
                        weight: bw,
                        bias: bb,
                        stride: 1,
                        dilation: 1,
                        pad_mode: pad,
                    },
                }));
                idx += 1;
            }
        }

        layers.push(Layer::Elu);
        idx += 1;

        // Final Conv1d(n_filters → channels=1, k=last_kernel_size)
        let fw = wf.get_3d(&format!("{prefix}.model.{idx}.conv.weight"))?;
        let fb = wf.opt_1d(&format!("{prefix}.model.{idx}.conv.bias"))?;
        layers.push(Layer::Conv(Conv1dLayer {
            weight: fw,
            bias: fb,
            stride: 1,
            dilation: 1,
            pad_mode: pad,
        }));

        Ok(Self { layers })
    }

    /// `x: [dimension, T]` (channels × time) → `[1, T_audio]`.
    pub fn forward(&self, x: ndarray::ArrayView2<f32>) -> Array2<f32> {
        let mut h = x.to_owned();
        for layer in &self.layers {
            match layer {
                Layer::Conv(c) => h = c.forward(h.view()),
                Layer::TransConv(c) => h = c.forward(h.view()),
                Layer::Resnet(r) => h = r.forward(h.view()),
                Layer::Elu => elu_inplace(h.as_slice_mut().unwrap()),
            }
        }
        h
    }
}
