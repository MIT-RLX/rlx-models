// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// NeuCodec encoder eager forward — CodecEnc (BigCodec) + SemanticEncoder + fc_prior + FSQ.

#[cfg(feature = "w2v-bert")]
use std::path::PathBuf;

#[cfg(feature = "w2v-bert")]
use anyhow::Context;
use anyhow::{Result, bail};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, s};
use safetensors::SafeTensors;

use super::eager::{
    ENCODER_SAMPLES_PER_TOKEN, as1d, as2d, as3d, fsq_encode, linear, load_f32, shape_of,
};

const SNAKE_EPS: f32 = 1e-9;

#[derive(Clone)]
pub(crate) struct SnakeActW {
    pub(crate) alpha: Array1<f32>,
    pub(crate) beta: Array1<f32>,
}

#[derive(Clone)]
pub(crate) struct ResidualUnitW {
    pub(crate) act0: SnakeActW,
    pub(crate) conv1_w: Array3<f32>,
    pub(crate) conv1_b: Array1<f32>,
    pub(crate) conv1_dilation: usize,
    pub(crate) act1: SnakeActW,
    pub(crate) conv2_w: Array3<f32>,
    pub(crate) conv2_b: Array1<f32>,
    pub(crate) upsample_filter: Array1<f32>,
    pub(crate) downsample_filter: Array1<f32>,
}

#[derive(Clone)]
pub(crate) struct EncoderBlockW {
    pub(crate) units: Vec<ResidualUnitW>,
    pub(crate) act_down: SnakeActW,
    pub(crate) down_w: Array3<f32>,
    pub(crate) down_b: Array1<f32>,
    pub(crate) down_stride: usize,
    pub(crate) upsample_filter: Array1<f32>,
    pub(crate) downsample_filter: Array1<f32>,
}

#[derive(Clone)]
pub(crate) struct CodecEncW {
    pub(crate) stem_w: Array3<f32>,
    pub(crate) stem_b: Array1<f32>,
    pub(crate) blocks: Vec<EncoderBlockW>,
    pub(crate) final_act: SnakeActW,
    pub(crate) final_upsample_filter: Array1<f32>,
    pub(crate) final_downsample_filter: Array1<f32>,
    pub(crate) final_w: Array3<f32>,
    pub(crate) final_b: Array1<f32>,
}

#[derive(Clone)]
pub(crate) struct SemanticEncW {
    pub(crate) initial_w: Array3<f32>,
    pub(crate) res1_w: Array3<f32>,
    pub(crate) res1_b: Array1<f32>,
    pub(crate) res3_w: Array3<f32>,
    pub(crate) res3_b: Array1<f32>,
    pub(crate) final_w: Array3<f32>,
}

/// Loaded NeuCodec encoder tensors (CodecEnc + semantic head + FSQ project_in).
#[derive(Clone)]
pub(crate) struct EncoderWeights {
    pub(crate) codec: CodecEncW,
    pub(crate) semantic: SemanticEncW,
    pub(crate) fsq_proj_in_w: Array2<f32>,
    pub(crate) fsq_proj_in_b: Array1<f32>,
    pub(crate) fc_prior_w: Array2<f32>,
    pub(crate) fc_prior_b: Array1<f32>,
    pub(crate) codec_enc_tensors: usize,
    pub(crate) semantic_enc_tensors: usize,
    pub(crate) codec_enc_strides: [usize; 5],
    pub(crate) semantic_w2v_layer: usize,
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// SnakeBeta with `alpha_logscale=True`: `x + sin(e^a · x)² / (e^b + ε)`.
fn snake_beta(x: ArrayView2<f32>, act: &SnakeActW) -> Array2<f32> {
    let (c, t) = (x.shape()[0], x.shape()[1]);
    assert_eq!(act.alpha.len(), c);
    let mut out = Array2::<f32>::zeros((c, t));
    for ci in 0..c {
        let a = act.alpha[ci].exp();
        let inv_b = 1.0 / (act.beta[ci].exp() + SNAKE_EPS);
        for ti in 0..t {
            let v = x[[ci, ti]];
            let s = (a * v).sin();
            out[[ci, ti]] = v + inv_b * s * s;
        }
    }
    out
}

fn pad_replicate(x: ArrayView2<f32>, left: usize, right: usize) -> Array2<f32> {
    let (c, t) = (x.shape()[0], x.shape()[1]);
    let out_t = t + left + right;
    let mut out = Array2::<f32>::zeros((c, out_t));
    for ci in 0..c {
        for ti in 0..out_t {
            let src = ti as isize - left as isize;
            let src = src.clamp(0, t as isize - 1) as usize;
            out[[ci, ti]] = x[[ci, src]];
        }
    }
    out
}

fn conv1d(
    x: ArrayView2<f32>,
    w: ArrayView3<f32>,
    b: Option<ArrayView1<f32>>,
    pad: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Array2<f32> {
    let (c_in, t_in) = (x.shape()[0], x.shape()[1]);
    let (c_out, _, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
    assert_eq!(c_in % groups, 0);
    assert_eq!(c_out % groups, 0);
    let c_in_per_g = c_in / groups;
    let c_out_per_g = c_out / groups;
    let dil = dilation.max(1);
    let t_out = (t_in + 2 * pad).saturating_sub(dil * (k - 1) + 1) / stride + 1;

    let mut out = Array2::<f32>::zeros((c_out, t_out));
    for g in 0..groups {
        let c_in_start = g * c_in_per_g;
        let c_out_start = g * c_out_per_g;
        for oc in 0..c_out_per_g {
            let out_c = c_out_start + oc;
            for ti in 0..t_out {
                let mut acc = 0.0f32;
                for ic in 0..c_in_per_g {
                    let in_c = c_in_start + ic;
                    for ki in 0..k {
                        let src = ti * stride + ki * dil;
                        if src >= pad && src < t_in + pad {
                            acc += x[[in_c, src - pad]] * w[[out_c, ic, ki]];
                        }
                    }
                }
                out[[out_c, ti]] = acc;
            }
        }
    }
    if let Some(b) = b {
        out += &b.view().insert_axis(Axis(1));
    }
    out
}

fn conv_transpose1d_depthwise(
    x: ArrayView2<f32>,
    filter: ArrayView1<f32>,
    stride: usize,
    trim_left: usize,
    trim_right: usize,
    gain: f32,
) -> Array2<f32> {
    let (c, t_in) = (x.shape()[0], x.shape()[1]);
    let k = filter.len();
    // PyTorch ConvTranspose1d: (T-1)*stride + k when padding=0
    let t_up = (t_in.saturating_sub(1)) * stride + k;
    let mut up = Array2::<f32>::zeros((c, t_up));
    for ci in 0..c {
        for ti in 0..t_in {
            for ki in 0..k {
                let to = ti * stride + ki;
                if to < t_up {
                    up[[ci, to]] += x[[ci, ti]] * filter[ki] * gain;
                }
            }
        }
    }
    let out_t = t_up.saturating_sub(trim_left + trim_right);
    let mut out = Array2::<f32>::zeros((c, out_t));
    for ci in 0..c {
        for ti in 0..out_t {
            out[[ci, ti]] = up[[ci, ti + trim_left]];
        }
    }
    out
}

fn downsample1d_depthwise(
    x: ArrayView2<f32>,
    filter: ArrayView1<f32>,
    stride: usize,
) -> Array2<f32> {
    let pad_left = 5usize;
    let pad_right = 6usize;
    let padded = pad_replicate(x, pad_left, pad_right);
    let (c, _) = (x.shape()[0], x.shape()[1]);
    let k = filter.len();
    let mut w = Array3::<f32>::zeros((c, 1, k));
    for ci in 0..c {
        for ki in 0..k {
            w[[ci, 0, ki]] = filter[ki];
        }
    }
    conv1d(padded.view(), w.view(), None, 0, stride, 1, c)
}

/// Anti-aliased SnakeBeta (BigVGAN `Activation1d`): 2× up → snake → 2× down.
fn aa_snake(
    x: ArrayView2<f32>,
    act: &SnakeActW,
    up_filter: ArrayView1<f32>,
    down_filter: ArrayView1<f32>,
) -> Array2<f32> {
    let padded = pad_replicate(x, 5, 5);
    let up = conv_transpose1d_depthwise(padded.view(), up_filter, 2, 15, 15, 2.0);
    let s = snake_beta(up.view(), act);
    downsample1d_depthwise(s.view(), down_filter, 2)
}

fn residual_unit(x: ArrayView2<f32>, w: &ResidualUnitW) -> Array2<f32> {
    let k = w.conv1_w.shape()[2];
    let pad = ((k - 1) * w.conv1_dilation) / 2;
    let mut h = aa_snake(
        x,
        &w.act0,
        w.upsample_filter.view(),
        w.downsample_filter.view(),
    );
    h = conv1d(
        h.view(),
        w.conv1_w.view(),
        Some(w.conv1_b.view()),
        pad,
        1,
        w.conv1_dilation,
        1,
    );
    h = aa_snake(
        h.view(),
        &w.act1,
        w.upsample_filter.view(),
        w.downsample_filter.view(),
    );
    h = conv1d(
        h.view(),
        w.conv2_w.view(),
        Some(w.conv2_b.view()),
        0,
        1,
        1,
        1,
    );
    &x + &h
}

fn encoder_block(x: ArrayView2<f32>, w: &EncoderBlockW) -> Array2<f32> {
    let mut h = x.to_owned();
    for unit in &w.units {
        h = residual_unit(h.view(), unit);
    }
    h = aa_snake(
        h.view(),
        &w.act_down,
        w.upsample_filter.view(),
        w.downsample_filter.view(),
    );
    let stride = w.down_stride;
    let pad = stride / 2 + stride % 2;
    conv1d(
        h.view(),
        w.down_w.view(),
        Some(w.down_b.view()),
        pad,
        stride,
        1,
        1,
    )
}

fn codec_enc_forward(pcm: &[f32], w: &CodecEncW) -> Array2<f32> {
    let mut x = Array2::<f32>::from_shape_fn((1, pcm.len()), |(_, ti)| pcm[ti]);
    x = conv1d(x.view(), w.stem_w.view(), Some(w.stem_b.view()), 3, 1, 1, 1);
    for block in &w.blocks {
        x = encoder_block(x.view(), block);
    }
    x = aa_snake(
        x.view(),
        &w.final_act,
        w.final_upsample_filter.view(),
        w.final_downsample_filter.view(),
    );
    x = conv1d(
        x.view(),
        w.final_w.view(),
        Some(w.final_b.view()),
        1,
        1,
        1,
        1,
    );
    // [C, T] → [T, C]
    let (c, t) = (x.shape()[0], x.shape()[1]);
    let mut out = Array2::<f32>::zeros((t, c));
    for ti in 0..t {
        for ci in 0..c {
            out[[ti, ci]] = x[[ci, ti]];
        }
    }
    out
}

fn semantic_enc_forward(x: ArrayView2<f32>, w: &SemanticEncW) -> Array2<f32> {
    // Input x: [T, 1024] → conv expects [C, T]
    let (t, c) = (x.shape()[0], x.shape()[1]);
    let mut ct = Array2::<f32>::zeros((c, t));
    for ti in 0..t {
        for ci in 0..c {
            ct[[ci, ti]] = x[[ti, ci]];
        }
    }
    let pad = (w.initial_w.shape()[2] - 1) / 2;
    let mut h = conv1d(ct.view(), w.initial_w.view(), None, pad, 1, 1, 1);
    let skip = h.clone();
    h.mapv_inplace(relu);
    h = conv1d(
        h.view(),
        w.res1_w.view(),
        Some(w.res1_b.view()),
        pad,
        1,
        1,
        1,
    );
    h.mapv_inplace(relu);
    h = conv1d(
        h.view(),
        w.res3_w.view(),
        Some(w.res3_b.view()),
        pad,
        1,
        1,
        1,
    );
    h = &h + &skip;
    h = conv1d(h.view(), w.final_w.view(), None, pad, 1, 1, 1);
    let (c, t) = (h.shape()[0], h.shape()[1]);
    let mut out = Array2::<f32>::zeros((t, c));
    for ti in 0..t {
        for ci in 0..c {
            out[[ti, ci]] = h[[ci, ti]];
        }
    }
    out
}

fn pad_pcm_to_token_grid(pcm: &[f32]) -> Vec<f32> {
    let hop = ENCODER_SAMPLES_PER_TOKEN;
    let rem = pcm.len() % hop;
    if rem == 0 {
        return pcm.to_vec();
    }
    let mut out = pcm.to_vec();
    out.resize(pcm.len() + (hop - rem), 0.0);
    out
}

fn fuse_and_fsq(
    semantic: ArrayView2<f32>,
    acoustic: ArrayView2<f32>,
    w: &EncoderWeights,
) -> Result<Vec<i32>> {
    let t_sem = semantic.shape()[0];
    let t_ac = acoustic.shape()[0];
    let t = t_sem.min(t_ac);
    if t == 0 {
        bail!("encoder produced zero-length latent (semantic={t_sem}, acoustic={t_ac})");
    }
    let d = semantic.shape()[1] + acoustic.shape()[1];
    if d != w.fc_prior_w.shape()[1] {
        bail!(
            "concat dim {d} != fc_prior in_dim {}",
            w.fc_prior_w.shape()[1]
        );
    }
    let mut concat = Array2::<f32>::zeros((t, d));
    for ti in 0..t {
        let d_sem = semantic.shape()[1];
        concat
            .slice_mut(s![ti, 0..d_sem])
            .assign(&semantic.slice(s![ti, ..]));
        concat
            .slice_mut(s![ti, d_sem..])
            .assign(&acoustic.slice(s![ti, ..]));
    }
    let fused = linear(
        concat.view(),
        w.fc_prior_w.view(),
        Some(w.fc_prior_b.view()),
    );
    Ok(fsq_encode(
        fused.view(),
        w.fsq_proj_in_w.view(),
        w.fsq_proj_in_b.view(),
    ))
}

pub(crate) fn encode_forward(
    pcm_16k: &[f32],
    w: &EncoderWeights,
    semantic_features: ArrayView2<f32>,
) -> Result<Vec<i32>> {
    let pcm = pad_pcm_to_token_grid(pcm_16k);
    let acoustic = codec_enc_forward(&pcm, &w.codec);
    let semantic = semantic_enc_forward(semantic_features, &w.semantic);
    fuse_and_fsq(semantic.view(), acoustic.view(), w)
}

fn load_filter(st: &SafeTensors<'_>, key: &str) -> Result<Array1<f32>> {
    let shape = shape_of(st, key)?;
    let n = shape.iter().product();
    Ok(as1d(load_f32(st, key)?, n))
}

fn load_snake(st: &SafeTensors<'_>, prefix: &str) -> Result<SnakeActW> {
    let alpha = load_f32(st, &format!("{prefix}.alpha"))?;
    let beta = load_f32(st, &format!("{prefix}.beta"))?;
    let (n_a, n_b) = (alpha.len(), beta.len());
    Ok(SnakeActW {
        alpha: as1d(alpha, n_a),
        beta: as1d(beta, n_b),
    })
}

fn load_snake_from_keys(
    st: &SafeTensors<'_>,
    alpha_key: &str,
    beta_key: &str,
) -> Result<SnakeActW> {
    let alpha = load_f32(st, alpha_key)?;
    let beta = load_f32(st, beta_key)?;
    let (n_a, n_b) = (alpha.len(), beta.len());
    Ok(SnakeActW {
        alpha: as1d(alpha, n_a),
        beta: as1d(beta, n_b),
    })
}

fn load_residual_unit(
    st: &SafeTensors<'_>,
    prefix: &str,
    dilation: usize,
) -> Result<ResidualUnitW> {
    let up_key = format!("{prefix}.block.0.upsample.filter");
    let down_key = format!("{prefix}.block.0.downsample.lowpass.filter");
    Ok(ResidualUnitW {
        act0: load_snake(st, &format!("{prefix}.block.0.act"))?,
        conv1_w: {
            let shape = shape_of(st, &format!("{prefix}.block.1.weight"))?;
            as3d(
                load_f32(st, &format!("{prefix}.block.1.weight"))?,
                shape[0],
                shape[1],
                shape[2],
            )
        },
        conv1_b: {
            let shape = shape_of(st, &format!("{prefix}.block.1.bias"))?;
            as1d(load_f32(st, &format!("{prefix}.block.1.bias"))?, shape[0])
        },
        conv1_dilation: dilation,
        act1: load_snake(st, &format!("{prefix}.block.2.act"))?,
        conv2_w: {
            let shape = shape_of(st, &format!("{prefix}.block.3.weight"))?;
            as3d(
                load_f32(st, &format!("{prefix}.block.3.weight"))?,
                shape[0],
                shape[1],
                shape[2],
            )
        },
        conv2_b: {
            let shape = shape_of(st, &format!("{prefix}.block.3.bias"))?;
            as1d(load_f32(st, &format!("{prefix}.block.3.bias"))?, shape[0])
        },
        upsample_filter: load_filter(st, &up_key)?,
        downsample_filter: load_filter(st, &down_key)?,
    })
}

fn load_encoder_block(
    st: &SafeTensors<'_>,
    block_idx: usize,
    stride: usize,
) -> Result<EncoderBlockW> {
    let prefix = format!("CodecEnc.conv_blocks.{block_idx}");
    let dilations = [1usize, 3, 9];
    let units = dilations
        .iter()
        .enumerate()
        .map(|(ui, &d)| load_residual_unit(st, &format!("{prefix}.block.{ui}"), d))
        .collect::<Result<Vec<_>>>()?;
    let down_shape = shape_of(st, &format!("{prefix}.block.4.weight"))?;
    Ok(EncoderBlockW {
        units,
        act_down: load_snake(st, &format!("{prefix}.block.3.act"))?,
        down_w: as3d(
            load_f32(st, &format!("{prefix}.block.4.weight"))?,
            down_shape[0],
            down_shape[1],
            down_shape[2],
        ),
        down_b: as1d(
            load_f32(st, &format!("{prefix}.block.4.bias"))?,
            down_shape[0],
        ),
        down_stride: stride,
        upsample_filter: load_filter(st, &format!("{prefix}.block.3.upsample.filter"))?,
        downsample_filter: load_filter(st, &format!("{prefix}.block.3.downsample.lowpass.filter"))?,
    })
}

fn load_codec_enc(st: &SafeTensors<'_>, strides: [usize; 5]) -> Result<CodecEncW> {
    let stem_shape = shape_of(st, "CodecEnc.conv_blocks.0.weight")?;
    let mut blocks = Vec::with_capacity(strides.len());
    for (i, &stride) in strides.iter().enumerate() {
        blocks.push(load_encoder_block(st, i + 1, stride)?);
    }
    let final_shape = shape_of(st, "CodecEnc.conv_final_block.1.weight")?;
    Ok(CodecEncW {
        stem_w: as3d(
            load_f32(st, "CodecEnc.conv_blocks.0.weight")?,
            stem_shape[0],
            stem_shape[1],
            stem_shape[2],
        ),
        stem_b: as1d(load_f32(st, "CodecEnc.conv_blocks.0.bias")?, stem_shape[0]),
        blocks,
        final_act: load_snake_from_keys(
            st,
            "CodecEnc.conv_final_block.0.act.alpha",
            "CodecEnc.conv_final_block.0.act.beta",
        )?,
        final_upsample_filter: load_filter(st, "CodecEnc.conv_final_block.0.upsample.filter")?,
        final_downsample_filter: load_filter(
            st,
            "CodecEnc.conv_final_block.0.downsample.lowpass.filter",
        )?,
        final_w: as3d(
            load_f32(st, "CodecEnc.conv_final_block.1.weight")?,
            final_shape[0],
            final_shape[1],
            final_shape[2],
        ),
        final_b: as1d(
            load_f32(st, "CodecEnc.conv_final_block.1.bias")?,
            final_shape[0],
        ),
    })
}

fn load_semantic_enc(st: &SafeTensors<'_>) -> Result<SemanticEncW> {
    let init_shape = shape_of(st, "SemanticEncoder_module.initial_conv.weight")?;
    let r1_shape = shape_of(st, "SemanticEncoder_module.residual_blocks.1.weight")?;
    let r3_shape = shape_of(st, "SemanticEncoder_module.residual_blocks.3.weight")?;
    let fin_shape = shape_of(st, "SemanticEncoder_module.final_conv.weight")?;
    Ok(SemanticEncW {
        initial_w: as3d(
            load_f32(st, "SemanticEncoder_module.initial_conv.weight")?,
            init_shape[0],
            init_shape[1],
            init_shape[2],
        ),
        res1_w: as3d(
            load_f32(st, "SemanticEncoder_module.residual_blocks.1.weight")?,
            r1_shape[0],
            r1_shape[1],
            r1_shape[2],
        ),
        res1_b: as1d(
            load_f32(st, "SemanticEncoder_module.residual_blocks.1.bias")?,
            r1_shape[0],
        ),
        res3_w: as3d(
            load_f32(st, "SemanticEncoder_module.residual_blocks.3.weight")?,
            r3_shape[0],
            r3_shape[1],
            r3_shape[2],
        ),
        res3_b: as1d(
            load_f32(st, "SemanticEncoder_module.residual_blocks.3.bias")?,
            r3_shape[0],
        ),
        final_w: as3d(
            load_f32(st, "SemanticEncoder_module.final_conv.weight")?,
            fin_shape[0],
            fin_shape[1],
            fin_shape[2],
        ),
    })
}

fn parse_usize_meta(
    meta: &Option<std::collections::HashMap<String, String>>,
    key: &str,
) -> Option<usize> {
    meta.as_ref()
        .and_then(|m| m.get(key))
        .and_then(|s| s.parse().ok())
}

pub(crate) fn load_encoder_weights(
    st: &SafeTensors<'_>,
    user_meta: &Option<std::collections::HashMap<String, String>>,
) -> Result<EncoderWeights> {
    let names: Vec<&str> = st.names();
    let codec_enc_tensors = names.iter().filter(|k| k.starts_with("CodecEnc.")).count();
    let semantic_enc_tensors = names
        .iter()
        .filter(|k| k.starts_with("SemanticEncoder_module."))
        .count();

    if codec_enc_tensors == 0 {
        bail!("No CodecEnc.* tensors found — run scripts/export_neucodec_encoder.py");
    }
    if semantic_enc_tensors == 0 {
        bail!("No SemanticEncoder_module.* tensors found in encoder weights");
    }

    let fsq_shape = shape_of(st, "generator.quantizer.project_in.weight")?;
    let fsq_in_dim = fsq_shape[1];
    let fsq_out_dim = fsq_shape[0];
    if fsq_out_dim != 8 {
        bail!("expected project_in out_dim=8, got {fsq_out_dim}");
    }

    let codec_enc_strides = user_meta
        .as_ref()
        .and_then(|m| m.get("codec_enc_strides"))
        .and_then(|s| {
            let v: Vec<usize> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            (v.len() == 5).then(|| [v[0], v[1], v[2], v[3], v[4]])
        })
        .unwrap_or([2, 2, 4, 4, 5]);

    let semantic_w2v_layer = parse_usize_meta(user_meta, "semantic_w2v_layer").unwrap_or(16);

    Ok(EncoderWeights {
        codec: load_codec_enc(st, codec_enc_strides)?,
        semantic: load_semantic_enc(st)?,
        fsq_proj_in_w: as2d(
            load_f32(st, "generator.quantizer.project_in.weight")?,
            fsq_out_dim,
            fsq_in_dim,
        ),
        fsq_proj_in_b: as1d(
            load_f32(st, "generator.quantizer.project_in.bias")?,
            fsq_out_dim,
        ),
        fc_prior_w: {
            let prior_shape = shape_of(st, "fc_prior.weight")?;
            as2d(
                load_f32(st, "fc_prior.weight")?,
                prior_shape[0],
                prior_shape[1],
            )
        },
        fc_prior_b: as1d(
            load_f32(st, "fc_prior.bias")?,
            shape_of(st, "fc_prior.weight")?[0],
        ),
        codec_enc_tensors,
        semantic_enc_tensors,
        codec_enc_strides,
        semantic_w2v_layer,
    })
}

// ─── Wav2Vec2-BERT semantic branch (optional) ───────────────────────────────

#[cfg(feature = "w2v-bert")]
pub(crate) struct W2vSemanticRunner {
    runner: rlx_wav2vec2_bert::Wav2Vec2BertRunner,
    hidden_layers: usize,
}

#[cfg(feature = "w2v-bert")]
impl W2vSemanticRunner {
    pub(crate) fn try_from_env(layer: usize) -> Result<Option<Self>> {
        let dir = match std::env::var("RLX_W2V_BERT_DIR")
            .ok()
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
        {
            Some(d) if d.join("config.json").exists() => d,
            _ => return Ok(None),
        };
        let weights = dir.join("model.safetensors");
        if !weights.exists() {
            bail!(
                "RLX_W2V_BERT_DIR={} missing model.safetensors",
                dir.display()
            );
        }
        let mut cfg = rlx_wav2vec2_bert::Wav2Vec2BertConfig::from_file(&dir.join("config.json"))
            .with_context(|| format!("reading {}", dir.join("config.json").display()))?;
        if layer > cfg.num_hidden_layers {
            bail!(
                "semantic_w2v_layer={layer} exceeds model num_hidden_layers={}",
                cfg.num_hidden_layers
            );
        }
        cfg.num_hidden_layers = layer;
        let runner = rlx_wav2vec2_bert::Wav2Vec2BertRunner::builder()
            .weights(&weights)
            .config_path(dir.join("config.json"))
            .preprocessor_config_path(dir.join("preprocessor_config.json"))
            .config(cfg.clone())
            .batch(1)
            .build()
            .context("build Wav2Vec2-BERT runner for NeuCodec semantic tap")?;
        Ok(Some(Self {
            runner,
            hidden_layers: layer,
        }))
    }

    pub(crate) fn encode_pcm(&mut self, pcm: &[f32]) -> Result<Array2<f32>> {
        let mel = self.runner.extract_log_mel(pcm);
        let hidden = self
            .runner
            .encode_features(&mel.features, Some(&mel.attention_mask))?;
        let h = self.runner.config().hidden_size;
        let frames = mel.num_frames;
        if hidden.len() != frames * h {
            bail!(
                "w2v hidden len {} != frames*hidden {} (frames={frames}, hidden={h})",
                hidden.len(),
                frames * h
            );
        }
        let token_len = acoustic_token_len(pcm.len());
        let valid_frames = mel
            .attention_mask
            .iter()
            .take_while(|&&value| value != 0.0)
            .count();
        let copy_frames = token_len.min(valid_frames);
        let mut out = Array2::<f32>::zeros((token_len, h));
        for fi in 0..copy_frames {
            for di in 0..h {
                out[[fi, di]] = hidden[fi * h + di];
            }
        }
        Ok(out)
    }
}

pub(crate) fn stub_semantic_features(acoustic_len: usize, hidden: usize) -> Array2<f32> {
    Array2::<f32>::zeros((acoustic_len, hidden))
}

pub(crate) fn acoustic_token_len(pcm_len: usize) -> usize {
    let hop = ENCODER_SAMPLES_PER_TOKEN;
    pcm_len.div_ceil(hop)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_pcm_to_token_grid_rounds_up() {
        let pcm = vec![0.0; 100];
        let padded = pad_pcm_to_token_grid(&pcm);
        assert_eq!(padded.len() % ENCODER_SAMPLES_PER_TOKEN, 0);
        assert!(padded.len() >= pcm.len());
    }
}
