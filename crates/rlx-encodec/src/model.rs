// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// EnCodec weights (flat row-major f32) loaded from a folded-weight_norm
// safetensors export keyed by the HuggingFace `EncodecModel` names.

use crate::config::EncodecConfig;
use anyhow::{Context, Result};
use safetensors::SafeTensors;

#[derive(Clone)]
pub struct ConvW {
    pub weight: Vec<f32>, // [c_out, c_in, k]
    pub bias: Vec<f32>,   // [c_out]
    pub c_out: usize,
    pub c_in: usize,
    pub k: usize,
    pub stride: usize,
    pub dilation: usize,
}

#[derive(Clone)]
pub struct ResnetW {
    pub conv1: ConvW,    // dim → hidden, k=residual_kernel
    pub conv2: ConvW,    // hidden → dim, k=1
    pub shortcut: ConvW, // dim → dim, k=1
}

#[derive(Clone)]
pub struct EncStageW {
    pub resnet: ResnetW,
    pub downsample: ConvW, // dim → 2*dim, k=2*ratio, stride=ratio
}

#[derive(Clone)]
pub struct LstmLayerW {
    pub w_ih: Vec<f32>, // [4H, in]
    pub w_hh: Vec<f32>, // [4H, H]
    pub b_ih: Vec<f32>, // [4H]
    pub b_hh: Vec<f32>, // [4H]
}

#[derive(Clone)]
pub struct LstmW {
    pub layers: Vec<LstmLayerW>,
    pub dim: usize,
}

#[derive(Clone)]
pub struct EncoderW {
    pub stem: ConvW,
    pub stages: Vec<EncStageW>,
    pub lstm: LstmW,
    pub final_conv: ConvW, // lstm_dim → hidden, k=last_kernel
}

/// One decoder upsampling stage: ELU → transposed conv → resnet.
#[derive(Clone)]
pub struct DecStageW {
    pub transpose: ConvW, // weight [c_in, c_out, k] (ConvTranspose1d layout)
    pub resnet: ResnetW,
}

#[derive(Clone)]
pub struct DecoderW {
    pub conv0: ConvW, // hidden → lstm_dim, k=kernel
    pub lstm: LstmW,
    pub stages: Vec<DecStageW>,
    pub final_conv: ConvW, // num_filters → audio_channels, k=last_kernel
}

#[derive(Clone)]
pub struct EncodecWeights {
    pub config: EncodecConfig,
    pub encoder: EncoderW,
    pub decoder: Option<DecoderW>,
    pub codebooks: Vec<Vec<f32>>, // each [codebook_size, codebook_dim]
}

fn t_f32(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    use safetensors::tensor::Dtype;
    let t = st
        .tensor(name)
        .with_context(|| format!("missing tensor {name}"))?;
    let raw = t.data();
    Ok(match t.dtype() {
        Dtype::F32 => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        dt => anyhow::bail!("tensor {name}: unsupported dtype {dt:?}"),
    })
}

fn conv(
    st: &SafeTensors<'_>,
    prefix: &str,
    c_out: usize,
    c_in: usize,
    k: usize,
    stride: usize,
    dilation: usize,
) -> Result<ConvW> {
    Ok(ConvW {
        weight: t_f32(st, &format!("{prefix}.weight"))?,
        bias: t_f32(st, &format!("{prefix}.bias"))?,
        c_out,
        c_in,
        k,
        stride,
        dilation,
    })
}

impl EncodecWeights {
    pub fn from_safetensors(bytes: &[u8], config: EncodecConfig) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes).context("parse encodec safetensors")?;
        let nf = config.num_filters;
        let rk = config.residual_kernel_size;
        let ratios = config.encoder_ratios();

        let stem = conv(
            &st,
            "encoder.layers.0.conv",
            nf,
            config.audio_channels,
            config.kernel_size,
            1,
            1,
        )?;

        let mut stages = Vec::new();
        let mut dim = nf;
        for (i, &ratio) in ratios.iter().enumerate() {
            let base = 1 + 3 * i;
            let hidden = dim / config.compress;
            let resnet = ResnetW {
                conv1: conv(
                    &st,
                    &format!("encoder.layers.{base}.block.1.conv"),
                    hidden,
                    dim,
                    rk,
                    1,
                    1,
                )?,
                conv2: conv(
                    &st,
                    &format!("encoder.layers.{base}.block.3.conv"),
                    dim,
                    hidden,
                    1,
                    1,
                    1,
                )?,
                shortcut: conv(
                    &st,
                    &format!("encoder.layers.{base}.shortcut.conv"),
                    dim,
                    dim,
                    1,
                    1,
                    1,
                )?,
            };
            let downsample = conv(
                &st,
                &format!("encoder.layers.{}.conv", base + 2),
                dim * 2,
                dim,
                2 * ratio,
                ratio,
                1,
            )?;
            stages.push(EncStageW { resnet, downsample });
            dim *= 2;
        }

        let lstm_idx = 1 + 3 * ratios.len();
        let lstm = load_lstm(
            &st,
            &format!("encoder.layers.{lstm_idx}.lstm"),
            dim,
            config.num_lstm_layers,
        )?;
        let final_conv = conv(
            &st,
            &format!("encoder.layers.{}.conv", lstm_idx + 2),
            config.hidden_size,
            dim,
            config.last_kernel_size,
            1,
            1,
        )?;

        // codebooks: keep loading until a layer is missing.
        let mut codebooks = Vec::new();
        let mut i = 0;
        while let Ok(embed) = t_f32(&st, &format!("quantizer.layers.{i}.codebook.embed")) {
            codebooks.push(embed);
            i += 1;
        }

        let decoder = if st.tensor("decoder.layers.0.conv.weight").is_ok() {
            Some(load_decoder(&st, &config)?)
        } else {
            None
        };

        Ok(Self {
            encoder: EncoderW {
                stem,
                stages,
                lstm,
                final_conv,
            },
            decoder,
            codebooks,
            config,
        })
    }
}

fn load_decoder(st: &SafeTensors<'_>, config: &EncodecConfig) -> Result<DecoderW> {
    let rk = config.residual_kernel_size;
    let lstm_dim = config.lstm_dim();
    let conv0 = conv(
        st,
        "decoder.layers.0.conv",
        lstm_dim,
        config.hidden_size,
        config.kernel_size,
        1,
        1,
    )?;
    let lstm = load_lstm(
        st,
        "decoder.layers.1.lstm",
        lstm_dim,
        config.num_lstm_layers,
    )?;

    let mut stages = Vec::new();
    let mut dim = lstm_dim;
    for (i, &ratio) in config.upsampling_ratios.iter().enumerate() {
        let out_dim = dim / 2;
        let tbase = 3 + 3 * i; // transpose layer index
        let rbase = 4 + 3 * i; // resnet layer index
        let transpose = conv(
            st,
            &format!("decoder.layers.{tbase}.conv"),
            out_dim,
            dim,
            2 * ratio,
            ratio,
            1,
        )?;
        let hidden = out_dim / config.compress;
        let resnet = ResnetW {
            conv1: conv(
                st,
                &format!("decoder.layers.{rbase}.block.1.conv"),
                hidden,
                out_dim,
                rk,
                1,
                1,
            )?,
            conv2: conv(
                st,
                &format!("decoder.layers.{rbase}.block.3.conv"),
                out_dim,
                hidden,
                1,
                1,
                1,
            )?,
            shortcut: conv(
                st,
                &format!("decoder.layers.{rbase}.shortcut.conv"),
                out_dim,
                out_dim,
                1,
                1,
                1,
            )?,
        };
        stages.push(DecStageW { transpose, resnet });
        dim = out_dim;
    }
    let final_idx = 2 + 3 * config.upsampling_ratios.len() + 1; // after last resnet + ELU
    let final_conv = conv(
        st,
        &format!("decoder.layers.{final_idx}.conv"),
        config.audio_channels,
        dim,
        config.last_kernel_size,
        1,
        1,
    )?;
    Ok(DecoderW {
        conv0,
        lstm,
        stages,
        final_conv,
    })
}

fn load_lstm(st: &SafeTensors<'_>, prefix: &str, dim: usize, n_layers: usize) -> Result<LstmW> {
    let mut layers = Vec::new();
    for l in 0..n_layers {
        layers.push(LstmLayerW {
            w_ih: t_f32(st, &format!("{prefix}.weight_ih_l{l}"))?,
            w_hh: t_f32(st, &format!("{prefix}.weight_hh_l{l}"))?,
            b_ih: t_f32(st, &format!("{prefix}.bias_ih_l{l}"))?,
            b_hh: t_f32(st, &format!("{prefix}.bias_hh_l{l}"))?,
        });
    }
    Ok(LstmW { layers, dim })
}
