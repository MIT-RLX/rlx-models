// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// WavTokenizer Vocos decoder weights (folded-weight_norm safetensors).

use anyhow::{Context, Result};
use safetensors::SafeTensors;

pub const DIM: usize = 768;
pub const INPUT_CH: usize = 512;
pub const GN_GROUPS: usize = 32;
pub const N_FFT: usize = 1280;
pub const HOP: usize = 320;
pub const INTERMEDIATE: usize = 2304;
pub const NUM_CONVNEXT: usize = 12;
pub const EPS: f32 = 1e-6;
pub const LN_EPS: f32 = 1e-6;

#[derive(Clone)]
pub struct ConvW {
    pub weight: Vec<f32>, // [c_out, c_in/groups, k]
    pub bias: Vec<f32>,
    pub c_out: usize,
    pub c_in: usize,
    pub k: usize,
    pub groups: usize,
}

#[derive(Clone)]
pub struct ResnetBlockW {
    pub norm1_w: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub conv1: ConvW,
    pub norm2_w: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub conv2: ConvW,
}

#[derive(Clone)]
pub struct AttnBlockW {
    pub norm_w: Vec<f32>,
    pub norm_b: Vec<f32>,
    pub q: ConvW,
    pub k: ConvW,
    pub v: ConvW,
    pub proj_out: ConvW,
}

/// AdaLayerNorm reduced to a fixed affine (bandwidth_id == 0 → embedding row 0).
#[derive(Clone)]
pub struct AdaLnW {
    pub scale: Vec<f32>, // [dim]
    pub shift: Vec<f32>, // [dim]
}

#[derive(Clone)]
pub struct ConvNextW {
    pub dwconv: ConvW, // depthwise [dim,1,7]
    pub norm: AdaLnW,
    pub pwconv1_w: Vec<f32>, // [intermediate, dim]
    pub pwconv1_b: Vec<f32>,
    pub pwconv2_w: Vec<f32>, // [dim, intermediate]
    pub pwconv2_b: Vec<f32>,
    pub gamma: Vec<f32>, // [dim]
}

#[derive(Clone)]
pub struct BackboneW {
    pub embed: ConvW, // [dim, input, 7]
    pub resnets: [ResnetBlockW; 4],
    pub attn: AttnBlockW,
    pub posnet_gn_w: Vec<f32>,
    pub posnet_gn_b: Vec<f32>,
    pub norm: AdaLnW,
    pub convnext: Vec<ConvNextW>,
    pub final_ln_w: Vec<f32>,
    pub final_ln_b: Vec<f32>,
}

#[derive(Clone)]
pub struct HeadW {
    pub out_w: Vec<f32>, // [1282, dim]
    pub out_b: Vec<f32>,
    pub window: Vec<f32>, // [n_fft]
}

#[derive(Clone)]
pub struct WavtokWeights {
    pub backbone: BackboneW,
    pub head: HeadW,
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
        dt => anyhow::bail!("{name}: dtype {dt:?}"),
    }
}

fn conv(
    st: &SafeTensors<'_>,
    prefix: &str,
    c_out: usize,
    c_in: usize,
    k: usize,
    groups: usize,
) -> Result<ConvW> {
    Ok(ConvW {
        weight: t_f32(st, &format!("{prefix}.weight"))?,
        bias: t_f32(st, &format!("{prefix}.bias"))?,
        c_out,
        c_in,
        k,
        groups,
    })
}

/// Take embedding row 0 (`bandwidth_id == 0`) of an `[n_emb, dim]` table.
fn ada(st: &SafeTensors<'_>, prefix: &str) -> Result<AdaLnW> {
    let s = t_f32(st, &format!("{prefix}.scale.weight"))?;
    let h = t_f32(st, &format!("{prefix}.shift.weight"))?;
    Ok(AdaLnW {
        scale: s[0..DIM].to_vec(),
        shift: h[0..DIM].to_vec(),
    })
}

fn resnet(st: &SafeTensors<'_>, p: &str) -> Result<ResnetBlockW> {
    Ok(ResnetBlockW {
        norm1_w: t_f32(st, &format!("{p}.norm1.weight"))?,
        norm1_b: t_f32(st, &format!("{p}.norm1.bias"))?,
        conv1: conv(st, &format!("{p}.conv1"), DIM, DIM, 3, 1)?,
        norm2_w: t_f32(st, &format!("{p}.norm2.weight"))?,
        norm2_b: t_f32(st, &format!("{p}.norm2.bias"))?,
        conv2: conv(st, &format!("{p}.conv2"), DIM, DIM, 3, 1)?,
    })
}

impl WavtokWeights {
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes).context("parse wavtokenizer safetensors")?;
        let embed = conv(&st, "backbone.embed", DIM, INPUT_CH, 7, 1)?;
        let resnets = [
            resnet(&st, "backbone.pos_net.0")?,
            resnet(&st, "backbone.pos_net.1")?,
            resnet(&st, "backbone.pos_net.3")?,
            resnet(&st, "backbone.pos_net.4")?,
        ];
        let attn = AttnBlockW {
            norm_w: t_f32(&st, "backbone.pos_net.2.norm.weight")?,
            norm_b: t_f32(&st, "backbone.pos_net.2.norm.bias")?,
            q: conv(&st, "backbone.pos_net.2.q", DIM, DIM, 1, 1)?,
            k: conv(&st, "backbone.pos_net.2.k", DIM, DIM, 1, 1)?,
            v: conv(&st, "backbone.pos_net.2.v", DIM, DIM, 1, 1)?,
            proj_out: conv(&st, "backbone.pos_net.2.proj_out", DIM, DIM, 1, 1)?,
        };
        let mut convnext = Vec::new();
        for i in 0..NUM_CONVNEXT {
            let p = format!("backbone.convnext.{i}");
            convnext.push(ConvNextW {
                dwconv: conv(&st, &format!("{p}.dwconv"), DIM, DIM, 7, DIM)?,
                norm: ada(&st, &format!("{p}.norm"))?,
                pwconv1_w: t_f32(&st, &format!("{p}.pwconv1.weight"))?,
                pwconv1_b: t_f32(&st, &format!("{p}.pwconv1.bias"))?,
                pwconv2_w: t_f32(&st, &format!("{p}.pwconv2.weight"))?,
                pwconv2_b: t_f32(&st, &format!("{p}.pwconv2.bias"))?,
                gamma: t_f32(&st, &format!("{p}.gamma"))?,
            });
        }
        let backbone = BackboneW {
            embed,
            resnets,
            attn,
            posnet_gn_w: t_f32(&st, "backbone.pos_net.5.weight")?,
            posnet_gn_b: t_f32(&st, "backbone.pos_net.5.bias")?,
            norm: ada(&st, "backbone.norm")?,
            convnext,
            final_ln_w: t_f32(&st, "backbone.final_layer_norm.weight")?,
            final_ln_b: t_f32(&st, "backbone.final_layer_norm.bias")?,
        };
        let head = HeadW {
            out_w: t_f32(&st, "head.out.weight")?,
            out_b: t_f32(&st, "head.out.bias")?,
            window: t_f32(&st, "head.istft.window")?,
        };
        Ok(Self { backbone, head })
    }
}
