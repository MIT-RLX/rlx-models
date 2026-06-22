// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// FACodecDecoder (amphion/naturalspeech3_facodec) weights: a HiFi-GAN/BigVGAN
// generator with anti-aliased SnakeBeta activations, plus timbre AdaIN
// conditioning. Weight-norm is folded into plain `.weight` in the fixture.

use anyhow::{Context, Result};
use safetensors::SafeTensors;

pub const IN_CH: usize = 256; // latent emb channels (= vq_dim)
pub const INIT_CH: usize = 1024; // upsample_initial_channel
pub const UP_RATIOS: [usize; 4] = [5, 5, 4, 2];
pub const LN_EPS: f32 = 1e-5; // timbre_norm (nn.LayerNorm default)

/// Plain 1-D conv weights (`[c_out, c_in, k]` row-major) + bias.
#[derive(Clone)]
pub struct ConvW {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub c_out: usize,
    pub c_in: usize,
    pub k: usize,
}

/// Transposed conv weights (`[c_in, c_out, k]`, PyTorch layout) + bias + stride.
#[derive(Clone)]
pub struct TConvW {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
    pub stride: usize,
}

/// A SnakeBeta activation's stored (pre-exp) parameters.
#[derive(Clone)]
pub struct SnakeW {
    pub alpha: Vec<f32>,
    pub beta: Vec<f32>,
}

/// `x + conv1x1(snake(conv_k7_dil(snake(x))))`.
#[derive(Clone)]
pub struct ResidualUnitW {
    pub act0: SnakeW,
    pub conv1: ConvW, // k7, dilation d
    pub dilation: usize,
    pub act1: SnakeW,
    pub conv3: ConvW, // k1
}

/// `snake → convT(stride) → 3× ResidualUnit`.
#[derive(Clone)]
pub struct DecoderBlockW {
    pub act0: SnakeW,
    pub convt: TConvW,
    pub units: Vec<ResidualUnitW>,
}

#[derive(Clone)]
pub struct FacodecWeights {
    pub conv0: ConvW, // 256 -> 1024, k7
    pub blocks: Vec<DecoderBlockW>,
    pub act_final: SnakeW,  // on 64
    pub conv_final: ConvW,  // 64 -> 1, k7
    pub timbre_w: Vec<f32>, // [512, 256]
    pub timbre_b: Vec<f32>, // [512]
    pub filter: Vec<f32>,   // shared anti-alias FIR [12]
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
        dt => anyhow::bail!("{name}: {dt:?}"),
    }
}

fn conv(st: &SafeTensors<'_>, p: &str, c_out: usize, c_in: usize, k: usize) -> Result<ConvW> {
    Ok(ConvW {
        weight: t_f32(st, &format!("{p}.weight"))?,
        bias: t_f32(st, &format!("{p}.bias"))?,
        c_out,
        c_in,
        k,
    })
}

fn snake(st: &SafeTensors<'_>, p: &str) -> Result<SnakeW> {
    Ok(SnakeW {
        alpha: t_f32(st, &format!("{p}.alpha"))?,
        beta: t_f32(st, &format!("{p}.beta"))?,
    })
}

fn residual_unit(
    st: &SafeTensors<'_>,
    p: &str,
    dim: usize,
    dilation: usize,
) -> Result<ResidualUnitW> {
    Ok(ResidualUnitW {
        act0: snake(st, &format!("{p}.block.0.act"))?,
        conv1: conv(st, &format!("{p}.block.1"), dim, dim, 7)?,
        dilation,
        act1: snake(st, &format!("{p}.block.2.act"))?,
        conv3: conv(st, &format!("{p}.block.3"), dim, dim, 1)?,
    })
}

impl FacodecWeights {
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes).context("parse facodec decoder")?;
        let conv0 = conv(&st, "model.0", INIT_CH, IN_CH, 7)?;

        let mut blocks = Vec::new();
        for (i, &stride) in UP_RATIOS.iter().enumerate() {
            let p = format!("model.{}", i + 1);
            let c_in = INIT_CH >> i;
            let c_out = INIT_CH >> (i + 1);
            let act0 = snake(&st, &format!("{p}.block.0.act"))?;
            let convt = TConvW {
                weight: t_f32(&st, &format!("{p}.block.1.weight"))?,
                bias: t_f32(&st, &format!("{p}.block.1.bias"))?,
                c_in,
                c_out,
                k: 2 * stride,
                stride,
            };
            let units = [(2usize, 1usize), (3, 3), (4, 9)]
                .iter()
                .map(|&(bi, dil)| residual_unit(&st, &format!("{p}.block.{bi}"), c_out, dil))
                .collect::<Result<Vec<_>>>()?;
            blocks.push(DecoderBlockW { act0, convt, units });
        }

        Ok(Self {
            conv0,
            blocks,
            act_final: snake(&st, "model.5.act")?,
            conv_final: conv(&st, "model.6", 1, 64, 7)?,
            timbre_w: t_f32(&st, "timbre_linear.weight")?,
            timbre_b: t_f32(&st, "timbre_linear.bias")?,
            filter: t_f32(&st, "_filter")?,
        })
    }

    /// timbre AdaIN: `style = W·spk + b` → `gamma = style[:256]`, `beta = style[256:]`.
    pub fn timbre_affine(&self, spk: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut style = self.timbre_b.clone(); // [512]
        for o in 0..2 * IN_CH {
            let mut acc = style[o];
            let row = &self.timbre_w[o * IN_CH..(o + 1) * IN_CH];
            for (w, s) in row.iter().zip(spk) {
                acc += w * s;
            }
            style[o] = acc;
        }
        (style[..IN_CH].to_vec(), style[IN_CH..].to_vec())
    }
}
