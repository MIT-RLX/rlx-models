// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// GGUF weight loading for the VAE encoders. Every tensor is dequantized to f32
// via rlx-gguf (I8_S conv/linear weights → f32 through the reader added to the
// framework; F32 biases/norms pass through). GGML shapes are innermost-first,
// so a conv weight `[k, in, out]` and a linear weight `[in, out]` both come out
// as out-major (torch) flat data — the layout `rlx_core::audio_ops_ir` wants.

use anyhow::{Context, Result};
use rlx_gguf::GgufFile;

use crate::config::STAGE_DEPTHS;

/// Plain 1-D conv weights (`[c_out, c_in/groups, k]`) + bias.
#[derive(Clone, Debug)]
pub struct ConvW {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub c_out: usize,
    pub c_in: usize,
    pub k: usize,
}

/// One ConvNeXt block (RMSNorm → depthwise conv → layer-scale → +res;
/// RMSNorm → FFN → layer-scale → +res).
#[derive(Clone, Debug)]
pub struct BlockW {
    pub norm_w: Vec<f32>,
    pub mixer: ConvW, // depthwise, groups = dim
    pub gamma: Vec<f32>,
    pub ffn_norm_w: Vec<f32>,
    pub l1_w: Vec<f32>, // [4·dim, dim]
    pub l1_b: Vec<f32>,
    pub l2_w: Vec<f32>, // [dim, 4·dim]
    pub l2_b: Vec<f32>,
    pub ffn_gamma: Vec<f32>,
    pub dim: usize,
}

/// SpeechConnector (fc1 → RMSNorm → fc2), projecting the VAE latent to the LM
/// hidden size.
#[derive(Clone, Debug)]
pub struct ConnectorW {
    pub fc1_w: Vec<f32>, // [connector_dim, vae_dim]
    pub fc1_b: Vec<f32>,
    pub norm_w: Vec<f32>, // [connector_dim]
    pub fc2_w: Vec<f32>,  // [connector_dim, connector_dim]
    pub fc2_b: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
}

/// One ConvNeXt VAE encoder (acoustic or semantic).
#[derive(Clone, Debug)]
pub struct VaeEncoderWeights {
    pub downsamples: Vec<ConvW>, // 7 (index 0 = stem, stride 1)
    pub stages: Vec<Vec<BlockW>>,
    pub head: ConvW,
    pub connector: ConnectorW,
    pub vae_dim: usize,
    pub connector_dim: usize,
}

/// Fetch a tensor as f32 with its GGML shape (innermost-first).
fn t(g: &GgufFile, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    g.dequant_f32(name)
        .with_context(|| format!("VAE tensor `{name}`"))
}

/// Load a plain conv (shape `[k, in, out]`).
fn conv(g: &GgufFile, wname: &str, bname: &str) -> Result<ConvW> {
    let (weight, shape) = t(g, wname)?;
    let (bias, _) = t(g, bname)?;
    anyhow::ensure!(
        shape.len() == 3,
        "{wname}: expected 3-D conv, got {shape:?}"
    );
    Ok(ConvW {
        weight,
        bias,
        k: shape[0],
        c_in: shape[1],
        c_out: shape[2],
    })
}

/// Load a linear (shape `[in, out]`), returning `(weight[out,in], bias, in, out)`.
fn linear(g: &GgufFile, wname: &str, bname: &str) -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
    let (weight, shape) = t(g, wname)?;
    let (bias, _) = t(g, bname)?;
    anyhow::ensure!(
        shape.len() == 2,
        "{wname}: expected 2-D linear, got {shape:?}"
    );
    Ok((weight, bias, shape[0], shape[1]))
}

impl VaeEncoderWeights {
    /// Load one encoder (`prefix` = "acoustic" or "semantic") from a VAE GGUF.
    pub fn load(g: &GgufFile, prefix: &str) -> Result<Self> {
        // 7 downsample convs.
        let mut downsamples = Vec::with_capacity(7);
        for i in 0..7 {
            downsamples.push(conv(
                g,
                &format!("{prefix}.downsample_layers.{i}.0.conv.conv.weight"),
                &format!("{prefix}.downsample_layers.{i}.0.conv.conv.bias"),
            )?);
        }

        // Stages of ConvNeXt blocks.
        let mut stages = Vec::with_capacity(7);
        for (s, &depth) in STAGE_DEPTHS.iter().enumerate() {
            let mut blocks = Vec::with_capacity(depth);
            for b in 0..depth {
                let p = format!("{prefix}.stages.{s}.{b}");
                let mixer = conv(
                    g,
                    &format!("{p}.mixer.conv.conv.conv.weight"),
                    &format!("{p}.mixer.conv.conv.conv.bias"),
                )?;
                let dim = mixer.c_out;
                let (norm_w, _) = t(g, &format!("{p}.norm.weight"))?;
                let (gamma, _) = t(g, &format!("{p}.gamma"))?;
                let (ffn_norm_w, _) = t(g, &format!("{p}.ffn_norm.weight"))?;
                let (l1_w, l1_b, _, _) = linear(
                    g,
                    &format!("{p}.ffn.linear1.weight"),
                    &format!("{p}.ffn.linear1.bias"),
                )?;
                let (l2_w, l2_b, _, _) = linear(
                    g,
                    &format!("{p}.ffn.linear2.weight"),
                    &format!("{p}.ffn.linear2.bias"),
                )?;
                let (ffn_gamma, _) = t(g, &format!("{p}.ffn_gamma"))?;
                blocks.push(BlockW {
                    norm_w,
                    mixer,
                    gamma,
                    ffn_norm_w,
                    l1_w,
                    l1_b,
                    l2_w,
                    l2_b,
                    ffn_gamma,
                    dim,
                });
            }
            stages.push(blocks);
        }

        let head = conv(
            g,
            &format!("{prefix}.head.conv.conv.weight"),
            &format!("{prefix}.head.conv.conv.bias"),
        )?;
        let vae_dim = head.c_out;

        // Connector: fc1 → norm → fc2.
        let (fc1_w, fc1_b, in_dim, out_dim) = linear(
            g,
            &format!("{prefix}_connector.fc1.weight"),
            &format!("{prefix}_connector.fc1.bias"),
        )?;
        let (norm_w, _) = t(g, &format!("{prefix}_connector.norm.weight"))?;
        let (fc2_w, fc2_b, _, connector_dim) = linear(
            g,
            &format!("{prefix}_connector.fc2.weight"),
            &format!("{prefix}_connector.fc2.bias"),
        )?;

        Ok(Self {
            downsamples,
            stages,
            head,
            connector: ConnectorW {
                fc1_w,
                fc1_b,
                norm_w,
                fc2_w,
                fc2_b,
                in_dim,
                out_dim,
            },
            vae_dim,
            connector_dim,
        })
    }
}

/// Load both encoders from the VAE GGUF at `path`.
pub fn load_vae(path: &std::path::Path) -> Result<(VaeEncoderWeights, VaeEncoderWeights)> {
    let g = GgufFile::from_path(path).with_context(|| format!("open VAE gguf {path:?}"))?;
    let acoustic = VaeEncoderWeights::load(&g, "acoustic")?;
    let semantic = VaeEncoderWeights::load(&g, "semantic")?;
    Ok((acoustic, semantic))
}

/// Load one encoder + its SpeechConnector from a safetensors
/// [`rlx_core::weight_map::WeightMap`].
///
/// Expected keys (torch layout):
/// - `{enc_prefix}.downsample_layers.{i}.0.conv.conv.{weight,bias}`
/// - `{enc_prefix}.stages.{s}.{b}.…`
/// - `{enc_prefix}.head.conv.conv.{weight,bias}`
/// - `{conn_prefix}.fc{1,2}.{weight,bias}`, `{conn_prefix}.norm.weight`
pub fn load_vae_from_map(
    wm: &mut rlx_core::weight_map::WeightMap,
    enc_prefix: &str,
    conn_prefix: &str,
) -> Result<VaeEncoderWeights> {
    fn take_f32(
        wm: &mut rlx_core::weight_map::WeightMap,
        name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        wm.take(name)
            .with_context(|| format!("VAE tensor `{name}`"))
    }

    fn conv_st(
        wm: &mut rlx_core::weight_map::WeightMap,
        wname: &str,
        bname: &str,
    ) -> Result<ConvW> {
        let (weight, shape) = take_f32(wm, wname)?;
        let (bias, _) = take_f32(wm, bname)?;
        // Torch Conv1d: [c_out, c_in/groups, k]
        anyhow::ensure!(
            shape.len() == 3,
            "{wname}: expected 3-D conv, got {shape:?}"
        );
        Ok(ConvW {
            weight,
            bias,
            c_out: shape[0],
            c_in: shape[1],
            k: shape[2],
        })
    }

    fn linear_st(
        wm: &mut rlx_core::weight_map::WeightMap,
        wname: &str,
        bname: &str,
    ) -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
        let (weight, shape) = take_f32(wm, wname)?;
        let (bias, _) = take_f32(wm, bname)?;
        // Torch Linear: [out, in]
        anyhow::ensure!(
            shape.len() == 2,
            "{wname}: expected 2-D linear, got {shape:?}"
        );
        Ok((weight, bias, shape[1], shape[0]))
    }

    let mut downsamples = Vec::with_capacity(7);
    for i in 0..7 {
        downsamples.push(conv_st(
            wm,
            &format!("{enc_prefix}.downsample_layers.{i}.0.conv.conv.weight"),
            &format!("{enc_prefix}.downsample_layers.{i}.0.conv.conv.bias"),
        )?);
    }

    let mut stages = Vec::with_capacity(7);
    for (s, &depth) in STAGE_DEPTHS.iter().enumerate() {
        let mut blocks = Vec::with_capacity(depth);
        for b in 0..depth {
            let p = format!("{enc_prefix}.stages.{s}.{b}");
            let mixer = conv_st(
                wm,
                &format!("{p}.mixer.conv.conv.conv.weight"),
                &format!("{p}.mixer.conv.conv.conv.bias"),
            )?;
            let dim = mixer.c_out;
            let (norm_w, _) = take_f32(wm, &format!("{p}.norm.weight"))?;
            let (gamma, _) = take_f32(wm, &format!("{p}.gamma"))?;
            let (ffn_norm_w, _) = take_f32(wm, &format!("{p}.ffn_norm.weight"))?;
            let (l1_w, l1_b, _, _) = linear_st(
                wm,
                &format!("{p}.ffn.linear1.weight"),
                &format!("{p}.ffn.linear1.bias"),
            )?;
            let (l2_w, l2_b, _, _) = linear_st(
                wm,
                &format!("{p}.ffn.linear2.weight"),
                &format!("{p}.ffn.linear2.bias"),
            )?;
            let (ffn_gamma, _) = take_f32(wm, &format!("{p}.ffn_gamma"))?;
            blocks.push(BlockW {
                norm_w,
                mixer,
                gamma,
                ffn_norm_w,
                l1_w,
                l1_b,
                l2_w,
                l2_b,
                ffn_gamma,
                dim,
            });
        }
        stages.push(blocks);
    }

    let head = conv_st(
        wm,
        &format!("{enc_prefix}.head.conv.conv.weight"),
        &format!("{enc_prefix}.head.conv.conv.bias"),
    )?;
    let vae_dim = head.c_out;

    let (fc1_w, fc1_b, in_dim, out_dim) = linear_st(
        wm,
        &format!("{conn_prefix}.fc1.weight"),
        &format!("{conn_prefix}.fc1.bias"),
    )?;
    let (norm_w, _) = take_f32(wm, &format!("{conn_prefix}.norm.weight"))?;
    let (fc2_w, fc2_b, _, connector_dim) = linear_st(
        wm,
        &format!("{conn_prefix}.fc2.weight"),
        &format!("{conn_prefix}.fc2.bias"),
    )?;

    Ok(VaeEncoderWeights {
        downsamples,
        stages,
        head,
        connector: ConnectorW {
            fc1_w,
            fc1_b,
            norm_w,
            fc2_w,
            fc2_b,
            in_dim,
            out_dim,
        },
        vae_dim,
        connector_dim,
    })
}
