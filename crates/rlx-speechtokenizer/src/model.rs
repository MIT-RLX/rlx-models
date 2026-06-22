// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SpeechTokenizer weights (flat row-major f32) from a folded-weight_norm
// safetensors export (encodec-library key nesting `*.conv.conv.*`).

use crate::config::SpeechTokenizerConfig;
use anyhow::{Context, Result};
use safetensors::SafeTensors;

#[derive(Clone)]
pub struct ConvW {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub c_out: usize,
    pub c_in: usize,
    pub k: usize,
    pub stride: usize,
    pub dilation: usize,
}

#[derive(Clone)]
pub struct ResnetW {
    pub conv1: ConvW,
    pub conv2: ConvW,
    pub shortcut: ConvW,
}

#[derive(Clone)]
pub struct EncStageW {
    pub resnet: ResnetW,
    pub downsample: ConvW,
}

#[derive(Clone)]
pub struct LstmLayerW {
    pub w_ih: Vec<f32>,
    pub w_hh: Vec<f32>,
    pub b_ih: Vec<f32>,
    pub b_hh: Vec<f32>,
}

#[derive(Clone)]
pub struct BiLstmW {
    pub fwd: Vec<LstmLayerW>,
    pub rev: Vec<LstmLayerW>,
}

#[derive(Clone)]
pub struct EncoderW {
    pub stem: ConvW,
    pub stages: Vec<EncStageW>,
    pub lstm: BiLstmW,
    pub final_conv: ConvW, // 2*dim → dim
}

#[derive(Clone)]
pub struct DecStageW {
    pub transpose: ConvW, // [c_in, c_out, k]
    pub resnet: ResnetW,
}

#[derive(Clone)]
pub struct DecoderW {
    pub conv0: ConvW,
    pub lstm: Vec<LstmLayerW>, // unidirectional
    pub stages: Vec<DecStageW>,
    pub final_conv: ConvW,
}

#[derive(Clone)]
pub struct StWeights {
    pub config: SpeechTokenizerConfig,
    pub encoder: EncoderW,
    pub decoder: DecoderW,
    pub codebooks: Vec<Vec<f32>>, // each [codebook_size, dim]
}

fn t_f32(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    use safetensors::tensor::Dtype;
    let t = st.tensor(name).with_context(|| format!("missing {name}"))?;
    let raw = t.data();
    match t.dtype() {
        Dtype::F32 => Ok(raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        dt => anyhow::bail!("{name}: unsupported dtype {dt:?}"),
    }
}

fn conv(
    st: &SafeTensors<'_>,
    prefix: &str,
    inner: &str,
    c_out: usize,
    c_in: usize,
    k: usize,
    stride: usize,
    dilation: usize,
) -> Result<ConvW> {
    Ok(ConvW {
        weight: t_f32(st, &format!("{prefix}.{inner}.{inner}.weight"))?,
        bias: t_f32(st, &format!("{prefix}.{inner}.{inner}.bias"))?,
        c_out,
        c_in,
        k,
        stride,
        dilation,
    })
}

fn lstm_layer(st: &SafeTensors<'_>, prefix: &str, l: usize, rev: bool) -> Result<LstmLayerW> {
    let s = if rev { "_reverse" } else { "" };
    Ok(LstmLayerW {
        w_ih: t_f32(st, &format!("{prefix}.weight_ih_l{l}{s}"))?,
        w_hh: t_f32(st, &format!("{prefix}.weight_hh_l{l}{s}"))?,
        b_ih: t_f32(st, &format!("{prefix}.bias_ih_l{l}{s}"))?,
        b_hh: t_f32(st, &format!("{prefix}.bias_hh_l{l}{s}"))?,
    })
}

impl StWeights {
    pub fn from_safetensors(bytes: &[u8], config: SpeechTokenizerConfig) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes).context("parse speechtokenizer safetensors")?;
        let nf = config.n_filters;
        let rk = config.residual_kernel_size;
        let dim = config.dimension;

        // ---- encoder (reversed ratios) ----
        let stem = conv(
            &st,
            "encoder.model.0",
            "conv",
            nf,
            config.audio_channels,
            config.kernel_size,
            1,
            1,
        )?;
        let mut stages = Vec::new();
        let mut d = nf;
        for (i, &ratio) in config.encoder_ratios().iter().enumerate() {
            let base = 1 + 3 * i;
            let hidden = d / config.compress;
            let resnet = ResnetW {
                conv1: conv(
                    &st,
                    &format!("encoder.model.{base}.block.1"),
                    "conv",
                    hidden,
                    d,
                    rk,
                    1,
                    1,
                )?,
                conv2: conv(
                    &st,
                    &format!("encoder.model.{base}.block.3"),
                    "conv",
                    d,
                    hidden,
                    1,
                    1,
                    1,
                )?,
                shortcut: conv(
                    &st,
                    &format!("encoder.model.{base}.shortcut"),
                    "conv",
                    d,
                    d,
                    1,
                    1,
                    1,
                )?,
            };
            let downsample = conv(
                &st,
                &format!("encoder.model.{}", base + 2),
                "conv",
                d * 2,
                d,
                2 * ratio,
                ratio,
                1,
            )?;
            stages.push(EncStageW { resnet, downsample });
            d *= 2;
        }
        let lstm_idx = 1 + 3 * config.encoder_ratios().len();
        let lp = format!("encoder.model.{lstm_idx}.lstm");
        let lstm = BiLstmW {
            fwd: (0..config.lstm_layers)
                .map(|l| lstm_layer(&st, &lp, l, false))
                .collect::<Result<_>>()?,
            rev: (0..config.lstm_layers)
                .map(|l| lstm_layer(&st, &lp, l, true))
                .collect::<Result<_>>()?,
        };
        let final_conv = conv(
            &st,
            &format!("encoder.model.{}", lstm_idx + 2),
            "conv",
            dim,
            2 * dim,
            config.last_kernel_size,
            1,
            1,
        )?;
        let encoder = EncoderW {
            stem,
            stages,
            lstm,
            final_conv,
        };

        // ---- decoder (forward ratios) ----
        let conv0 = conv(
            &st,
            "decoder.model.0",
            "conv",
            dim,
            dim,
            config.kernel_size,
            1,
            1,
        )?;
        let dlp = "decoder.model.1.lstm";
        let dlstm: Vec<LstmLayerW> = (0..config.lstm_layers)
            .map(|l| lstm_layer(&st, dlp, l, false))
            .collect::<Result<_>>()?;
        let mut dstages = Vec::new();
        let mut dd = dim;
        for (i, &ratio) in config.ratios.iter().enumerate() {
            let out_dim = dd / 2;
            let tbase = 3 + 3 * i;
            let rbase = 4 + 3 * i;
            let transpose = conv(
                &st,
                &format!("decoder.model.{tbase}"),
                "convtr",
                out_dim,
                dd,
                2 * ratio,
                ratio,
                1,
            )?;
            let hidden = out_dim / config.compress;
            let resnet = ResnetW {
                conv1: conv(
                    &st,
                    &format!("decoder.model.{rbase}.block.1"),
                    "conv",
                    hidden,
                    out_dim,
                    rk,
                    1,
                    1,
                )?,
                conv2: conv(
                    &st,
                    &format!("decoder.model.{rbase}.block.3"),
                    "conv",
                    out_dim,
                    hidden,
                    1,
                    1,
                    1,
                )?,
                shortcut: conv(
                    &st,
                    &format!("decoder.model.{rbase}.shortcut"),
                    "conv",
                    out_dim,
                    out_dim,
                    1,
                    1,
                    1,
                )?,
            };
            dstages.push(DecStageW { transpose, resnet });
            dd = out_dim;
        }
        let dfinal_idx = 2 + 3 * config.ratios.len() + 1;
        let final_conv_d = conv(
            &st,
            &format!("decoder.model.{dfinal_idx}"),
            "conv",
            config.audio_channels,
            dd,
            config.last_kernel_size,
            1,
            1,
        )?;
        let decoder = DecoderW {
            conv0,
            lstm: dlstm,
            stages: dstages,
            final_conv: final_conv_d,
        };

        // ---- codebooks ----
        let mut codebooks = Vec::new();
        let mut i = 0;
        while let Ok(embed) = t_f32(&st, &format!("quantizer.vq.layers.{i}._codebook.embed")) {
            codebooks.push(embed);
            i += 1;
        }

        Ok(Self {
            encoder,
            decoder,
            codebooks,
            config,
        })
    }
}
