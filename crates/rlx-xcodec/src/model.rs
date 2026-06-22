// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// XCodec2 CodecDecoderVocos weights (RoFormer-Vocos backbone + ISTFT head).

use anyhow::{Context, Result};
use safetensors::SafeTensors;

pub const DIM: usize = 1024;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = 64;
pub const N_FFT: usize = 1280;
pub const HOP: usize = 320;
pub const GN_GROUPS: usize = 32;
pub const N_LAYERS: usize = 12;
pub const EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 10_000.0;

#[derive(Clone)]
pub struct ConvW {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub c_out: usize,
    pub c_in: usize,
    pub k: usize,
}

#[derive(Clone)]
pub struct ResnetW {
    pub norm1_w: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub conv1: ConvW,
    pub norm2_w: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub conv2: ConvW,
}

#[derive(Clone)]
pub struct TransformerW {
    pub att_norm: Vec<f32>, // RMSNorm weight
    pub q_w: Vec<f32>,      // [DIM, DIM] (RoPE-permuted)
    pub k_w: Vec<f32>,      // [DIM, DIM] (RoPE-permuted)
    pub v_w: Vec<f32>,      // [DIM, DIM]
    pub o_w: Vec<f32>,      // [DIM, DIM]
    pub ffn_norm: Vec<f32>,
    pub fc1: Vec<f32>, // [4*DIM, DIM]
    pub fc2: Vec<f32>, // [DIM, 4*DIM]
}

#[derive(Clone)]
pub struct XcodecWeights {
    pub embed: ConvW,
    pub prior: Vec<ResnetW>,
    pub transformers: Vec<TransformerW>,
    pub post: Vec<ResnetW>,
    pub final_ln_w: Vec<f32>,
    pub final_ln_b: Vec<f32>,
    pub out_w: Vec<f32>, // [1282, DIM]
    pub out_b: Vec<f32>,
    pub window: Vec<f32>,
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

fn resnet(st: &SafeTensors<'_>, p: &str) -> Result<ResnetW> {
    Ok(ResnetW {
        norm1_w: t_f32(st, &format!("{p}.norm1.weight"))?,
        norm1_b: t_f32(st, &format!("{p}.norm1.bias"))?,
        conv1: conv(st, &format!("{p}.conv1"), DIM, DIM, 3)?,
        norm2_w: t_f32(st, &format!("{p}.norm2.weight"))?,
        norm2_b: t_f32(st, &format!("{p}.norm2.bias"))?,
        conv2: conv(st, &format!("{p}.conv2"), DIM, DIM, 3)?,
    })
}

impl XcodecWeights {
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes).context("parse xcodec decoder")?;
        let embed = conv(&st, "backbone.embed", DIM, DIM, 7)?;
        let prior = vec![
            resnet(&st, "backbone.prior_net.0")?,
            resnet(&st, "backbone.prior_net.1")?,
        ];
        let post = vec![
            resnet(&st, "backbone.post_net.0")?,
            resnet(&st, "backbone.post_net.1")?,
        ];
        let mut transformers = Vec::new();
        for i in 0..N_LAYERS {
            let p = format!("backbone.transformers.{i}");
            let c_attn = t_f32(&st, &format!("{p}.att.c_attn.weight"))?; // [3*DIM, DIM]
            let q = c_attn[0..DIM * DIM].to_vec();
            let k = c_attn[DIM * DIM..2 * DIM * DIM].to_vec();
            let v = c_attn[2 * DIM * DIM..3 * DIM * DIM].to_vec();
            transformers.push(TransformerW {
                att_norm: t_f32(&st, &format!("{p}.att_norm.weight"))?,
                // XCodec applies RoPE with seq=head index (constant over time), so
                // the same rotation hits q and k and cancels in qᵀk → drop RoPE.
                q_w: q,
                k_w: k,
                v_w: v,
                o_w: t_f32(&st, &format!("{p}.att.c_proj.weight"))?,
                ffn_norm: t_f32(&st, &format!("{p}.ffn_norm.weight"))?,
                fc1: t_f32(&st, &format!("{p}.mlp.fc1.weight"))?,
                fc2: t_f32(&st, &format!("{p}.mlp.fc2.weight"))?,
            });
        }
        Ok(Self {
            embed,
            prior,
            transformers,
            post,
            final_ln_w: t_f32(&st, "backbone.final_layer_norm.weight")?,
            final_ln_b: t_f32(&st, "backbone.final_layer_norm.bias")?,
            out_w: t_f32(&st, "head.out.weight")?,
            out_b: t_f32(&st, "head.out.bias")?,
            window: t_f32(&st, "head.istft.window")?,
        })
    }
}
