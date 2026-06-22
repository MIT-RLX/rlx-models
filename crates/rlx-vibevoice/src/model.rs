// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// VibeVoice acoustic σ-VAE tokenizer decoder (microsoft/VibeVoice-1.5B) weights:
// a ConvNeXt-style causal upsampler (depthwise-conv mixer + FFN + RMSNorm +
// layer-scale). No weight-norm (plain convs).

use anyhow::{Context, Result};
use safetensors::SafeTensors;

pub const DIM_IN: usize = 64; // vae_dim
pub const EPS: f32 = 1e-5; // RMSNorm eps
/// Stage dims (after stem, then after each upsample): n_filters·2^k.
pub const STAGE_DIMS: [usize; 7] = [2048, 1024, 512, 256, 128, 64, 32];
pub const DEPTHS: [usize; 7] = [8, 3, 3, 3, 3, 3, 3];
pub const UP_STRIDES: [usize; 6] = [8, 5, 5, 4, 2, 2];
pub const UP_KERNELS: [usize; 6] = [16, 10, 10, 8, 4, 4];

/// Plain 1-D conv weights (`[c_out, c_in/groups, k]`) + bias.
#[derive(Clone)]
pub struct ConvW {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub c_out: usize,
    pub c_in: usize,
    pub k: usize,
}

/// Transposed conv weights (`[c_in, c_out, k]`, PyTorch layout) + bias.
#[derive(Clone)]
pub struct TConvW {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
    pub stride: usize,
}

/// One ConvNeXt block: RMSNorm → depthwise conv mixer → layer-scale → +res;
/// RMSNorm → FFN(Linear→GELU→Linear) → layer-scale → +res.
#[derive(Clone)]
pub struct BlockW {
    pub norm_w: Vec<f32>,
    pub mixer: ConvW, // depthwise k7, groups = dim
    pub gamma: Vec<f32>,
    pub ffn_norm_w: Vec<f32>,
    pub l1_w: Vec<f32>, // [4·dim, dim]
    pub l1_b: Vec<f32>,
    pub l2_w: Vec<f32>, // [dim, 4·dim]
    pub l2_b: Vec<f32>,
    pub ffn_gamma: Vec<f32>,
    pub dim: usize,
}

#[derive(Clone)]
pub struct VibeWeights {
    pub stem: ConvW,      // 64 -> 2048, k7 causal
    pub ups: Vec<TConvW>, // 6 transposed convs
    pub stages: Vec<Vec<BlockW>>,
    pub head: ConvW, // 32 -> 1, k7 causal
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
        Dtype::BF16 => Ok(raw
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect()),
        dt => anyhow::bail!("{name}: {dt:?}"),
    }
}

impl VibeWeights {
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes).context("parse vibevoice decoder")?;

        let stem = ConvW {
            weight: t_f32(&st, "upsample_layers.0.0.conv.conv.weight")?,
            bias: t_f32(&st, "upsample_layers.0.0.conv.conv.bias")?,
            c_out: STAGE_DIMS[0],
            c_in: DIM_IN,
            k: 7,
        };

        let mut ups = Vec::new();
        for i in 0..6 {
            let p = format!("upsample_layers.{}.0.convtr.convtr", i + 1);
            ups.push(TConvW {
                weight: t_f32(&st, &format!("{p}.weight"))?,
                bias: t_f32(&st, &format!("{p}.bias"))?,
                c_in: STAGE_DIMS[i],
                c_out: STAGE_DIMS[i + 1],
                k: UP_KERNELS[i],
                stride: UP_STRIDES[i],
            });
        }

        let mut stages = Vec::new();
        for (i, &depth) in DEPTHS.iter().enumerate() {
            let dim = STAGE_DIMS[i];
            let mut blocks = Vec::new();
            for j in 0..depth {
                let p = format!("stages.{i}.{j}");
                blocks.push(BlockW {
                    norm_w: t_f32(&st, &format!("{p}.norm.weight"))?,
                    mixer: ConvW {
                        weight: t_f32(&st, &format!("{p}.mixer.conv.conv.conv.weight"))?,
                        bias: t_f32(&st, &format!("{p}.mixer.conv.conv.conv.bias"))?,
                        c_out: dim,
                        c_in: dim,
                        k: 7,
                    },
                    gamma: t_f32(&st, &format!("{p}.gamma"))?,
                    ffn_norm_w: t_f32(&st, &format!("{p}.ffn_norm.weight"))?,
                    l1_w: t_f32(&st, &format!("{p}.ffn.linear1.weight"))?,
                    l1_b: t_f32(&st, &format!("{p}.ffn.linear1.bias"))?,
                    l2_w: t_f32(&st, &format!("{p}.ffn.linear2.weight"))?,
                    l2_b: t_f32(&st, &format!("{p}.ffn.linear2.bias"))?,
                    ffn_gamma: t_f32(&st, &format!("{p}.ffn_gamma"))?,
                    dim,
                });
            }
            stages.push(blocks);
        }

        let head = ConvW {
            weight: t_f32(&st, "head.conv.conv.weight")?,
            bias: t_f32(&st, "head.conv.conv.bias")?,
            c_out: 1,
            c_in: STAGE_DIMS[6],
            k: 7,
        };

        Ok(Self {
            stem,
            ups,
            stages,
            head,
        })
    }
}
