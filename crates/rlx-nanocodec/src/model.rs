// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// NVIDIA NanoCodec (nvidia/nemo-nano-codec-22khz-*) decode weights: a Group-FSQ
// dequantizer (pure host arithmetic) + a CausalHiFiGAN decoder. Weight-norm is
// folded into plain `.weight` in the fixture.

use anyhow::{Context, Result};
use safetensors::SafeTensors;

pub const INPUT_DIM: usize = 16; // encoded_dim (= num_groups * codebook_dim_per_group)
pub const BASE_CHANNELS: usize = 864;
pub const UP_RATES: [usize; 5] = [7, 7, 6, 3, 2];
pub const SAMPLES_PER_FRAME: usize = 1764; // = prod(UP_RATES)
pub const RES_KERNELS: [usize; 3] = [3, 7, 11];
pub const RES_DILATIONS: [usize; 3] = [1, 3, 5];

// Group-FSQ: 4 groups, 4 dims/group, levels [9,8,8,7].
pub const NUM_GROUPS: usize = 4;
pub const LEVELS: [i64; 4] = [9, 8, 8, 7];

/// Plain 1-D conv weights (`[c_out, c_in/groups, k]` row-major) + bias.
#[derive(Clone)]
pub struct ConvW {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub c_out: usize,
    pub c_in: usize,
    pub k: usize,
}

/// `x + skip_conv(half_snake(input_conv(half_snake(x))))`. Causal convs.
#[derive(Clone)]
pub struct ResidualBlockW {
    pub act0: Vec<f32>, // half_snake alpha (c/2)
    pub input_conv: ConvW,
    pub dilation: usize,
    pub act1: Vec<f32>,
    pub skip_conv: ConvW,
}

/// One up-sampling stage: half_snake → grouped causal transposed conv → res layer.
#[derive(Clone)]
pub struct StageW {
    pub act: Vec<f32>,       // half_snake alpha before upsample (in_ch/2)
    pub up_weight: Vec<f32>, // transposed conv `[c_in, 1, k]`
    pub up_bias: Vec<f32>,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
    pub stride: usize,
    /// `res_layer[ks][dil]` residual blocks (3 kernels × 3 dilations).
    pub res: Vec<Vec<ResidualBlockW>>,
}

#[derive(Clone)]
pub struct NanoWeights {
    pub pre_conv: ConvW, // 16 -> 864, k7 causal
    pub stages: Vec<StageW>,
    pub post_act: Vec<f32>, // half_snake alpha (27/2)
    pub post_conv: ConvW,   // 27 -> 1, k3 causal
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

impl NanoWeights {
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes).context("parse nanocodec decoder")?;
        let pre_conv = conv(
            &st,
            "audio_decoder.pre_conv.conv",
            BASE_CHANNELS,
            INPUT_DIM,
            7,
        )?;

        let mut stages = Vec::new();
        let mut in_ch = BASE_CHANNELS;
        for (i, &stride) in UP_RATES.iter().enumerate() {
            let out_ch = in_ch / 2;
            let k = 2 * stride;
            let act = t_f32(
                &st,
                &format!("audio_decoder.activations.{i}.activation.snake_act.alpha"),
            )?;
            let up = format!("audio_decoder.up_sample_conv_layers.{i}.conv");
            // grouped transposed conv: groups = out_ch, weight [c_in, 1, k]
            let up_weight = t_f32(&st, &format!("{up}.weight"))?;
            let up_bias = t_f32(&st, &format!("{up}.bias"))?;

            let mut res = Vec::new();
            for (ks_i, &kk) in RES_KERNELS.iter().enumerate() {
                let mut blocks = Vec::new();
                for (di, &dil) in RES_DILATIONS.iter().enumerate() {
                    let bp =
                        format!("audio_decoder.res_layers.{i}.res_blocks.{ks_i}.res_blocks.{di}");
                    blocks.push(ResidualBlockW {
                        act0: t_f32(
                            &st,
                            &format!("{bp}.input_activation.activation.snake_act.alpha"),
                        )?,
                        input_conv: conv(
                            &st,
                            &format!("{bp}.input_conv.conv"),
                            out_ch,
                            out_ch,
                            kk,
                        )?,
                        dilation: dil,
                        act1: t_f32(
                            &st,
                            &format!("{bp}.skip_activation.activation.snake_act.alpha"),
                        )?,
                        skip_conv: conv(&st, &format!("{bp}.skip_conv.conv"), out_ch, out_ch, kk)?,
                    });
                }
                res.push(blocks);
            }
            stages.push(StageW {
                act,
                up_weight,
                up_bias,
                c_in: in_ch,
                c_out: out_ch,
                k,
                stride,
                res,
            });
            in_ch = out_ch;
        }

        Ok(Self {
            pre_conv,
            stages,
            post_act: t_f32(
                &st,
                "audio_decoder.post_activation.activation.snake_act.alpha",
            )?,
            post_conv: conv(&st, "audio_decoder.post_conv.conv", 1, in_ch, 3)?,
        })
    }
}

/// Group-FSQ dequant (host): codes `[NUM_GROUPS][T]` → latent `[INPUT_DIM, T]`
/// channel-major. `code = (((idx / base_d) % levels_d) - levels_d/2) / (levels_d/2)`.
pub fn fsq_decode(codes: &[Vec<i64>], t: usize) -> Vec<f32> {
    let mut base = [1i64; 4];
    for d in 1..4 {
        base[d] = base[d - 1] * LEVELS[d - 1];
    }
    let mut latent = vec![0f32; INPUT_DIM * t];
    for (g, group) in codes.iter().enumerate() {
        for d in 0..4 {
            let half = LEVELS[d] / 2;
            let ch = g * 4 + d;
            for ti in 0..t {
                let nonneg = (group[ti] / base[d]) % LEVELS[d];
                latent[ch * t + ti] = (nonneg - half) as f32 / half as f32;
            }
        }
    }
    latent
}
