// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Codec encoder + decoder as `rlx_ir::Graph` (trainable encoder, frozen decoder).

use anyhow::{Result, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_voxtral_tts::config::CodecArgs;

pub const CODEC_NORM_EPS: f32 = 1e-2;

#[derive(Debug, Clone)]
pub struct ParamSlot {
    pub name: String,
    pub param: NodeId,
    pub grad: Option<NodeId>,
    pub trainable: bool,
    /// Flat parameter length (`Shape` product) for host buffer validation.
    pub num_elems: usize,
}

#[derive(Debug, Clone)]
pub struct CodecGraphLayout {
    pub n_patches: usize,
    pub patch_size: usize,
    pub latent_t: usize,
    pub wav_t: usize,
    pub encoder_times: Vec<usize>,
}

impl CodecGraphLayout {
    pub fn new(cfg: &CodecArgs, n_patches: usize) -> Self {
        let input_k = cfg.patch_proj_kernel_size;
        let input_pad = input_k.saturating_sub(1);
        let mut t = conv1d_output_time(n_patches, input_k, 1, input_pad);
        let mut times = vec![t];
        let kernels = cfg.encoder_convs_kernels();
        let strides = cfg.encoder_convs_strides();
        let lens = cfg.encoder_transformer_lengths();
        for (stage, _) in lens.iter().enumerate() {
            let k = kernels[stage];
            let st = strides[stage];
            if k != 1 || st != 1 || stage + 1 == lens.len() {
                let pad_left = k.saturating_sub(st);
                t = conv1d_output_time(t, k, st, pad_left);
                times.push(t);
            }
        }
        let latent_t = *times.last().unwrap_or(&t);
        let mut dec_t = latent_t;
        for st in cfg.decoder_convs_strides().iter().skip(1) {
            dec_t *= st;
        }
        let wav_t = conv1d_output_time(
            dec_t,
            cfg.patch_proj_kernel_size,
            1,
            cfg.patch_proj_kernel_size.saturating_sub(1),
        );
        Self {
            n_patches,
            patch_size: cfg.pretransform_patch_size,
            latent_t,
            wav_t,
            encoder_times: times,
        }
    }
}

#[derive(Debug)]
pub struct CodecForwardGraph {
    pub graph: Graph,
    pub layout: CodecGraphLayout,
    pub params: Vec<ParamSlot>,
    pub audio_in: NodeId,
    pub target_wav: Option<NodeId>,
    pub recon_wav: NodeId,
    pub latent: NodeId,
    pub quantized: NodeId,
}

fn scalar(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    )
}

fn i64s(v: usize) -> i64 {
    v as i64
}

pub fn build_codec_forward_graph(
    cfg: &CodecArgs,
    layout: &CodecGraphLayout,
) -> Result<CodecForwardGraph> {
    build_codec_forward_graph_inner(cfg, layout, true)
}

pub fn build_codec_recon_graph(
    cfg: &CodecArgs,
    layout: &CodecGraphLayout,
) -> Result<CodecForwardGraph> {
    build_codec_forward_graph_inner(cfg, layout, false)
}

fn build_codec_forward_graph_inner(
    cfg: &CodecArgs,
    layout: &CodecGraphLayout,
    include_target_input: bool,
) -> Result<CodecForwardGraph> {
    let f = DType::F32;
    let mut g = Graph::new("voxtral_codec");
    let mut params = Vec::new();

    let c_in = layout.patch_size;
    let t_in = layout.n_patches;
    let audio_in = g.input("audio", Shape::new(&[c_in, t_in], f));
    let target_wav = if include_target_input {
        Some(g.input(
            "target_wav",
            Shape::new(&[layout.patch_size, layout.wav_t], f),
        ))
    } else {
        None
    };

    let mut x = audio_in;
    let mut time_idx = 0usize;
    let input_k = cfg.patch_proj_kernel_size;
    let input_pad = input_k.saturating_sub(1);
    x = add_conv_block(
        &mut g,
        &mut params,
        x,
        "input_proj",
        1,
        input_pad,
        true,
        cfg.dim,
        c_in,
        input_k,
        t_in,
    )?;

    let enc_kernels = cfg.encoder_convs_kernels();
    let enc_strides = cfg.encoder_convs_strides();
    let enc_lens = cfg.encoder_transformer_lengths();
    ensure!(
        enc_kernels.len() == enc_strides.len() && enc_kernels.len() == enc_lens.len(),
        "encoder conv config mismatch"
    );

    let mut window = cfg.attn_sliding_window_size;
    let mut block_idx = 0usize;
    for (stage, n_layers) in enc_lens.iter().enumerate() {
        let t = layout.encoder_times.get(time_idx).copied().unwrap_or(t_in);
        x = add_transformer_stack(
            &mut g,
            &mut params,
            x,
            cfg,
            &format!("encoder_blocks.{block_idx}"),
            *n_layers,
            window,
            t,
            true,
        )?;
        block_idx += 1;

        let k = enc_kernels[stage];
        let st = enc_strides[stage];
        let is_last = stage + 1 == enc_lens.len();
        if k != 1 || st != 1 || is_last {
            time_idx += 1;
            let out_ch = if is_last {
                cfg.semantic_dim + cfg.acoustic_dim
            } else {
                cfg.dim
            };
            x = add_conv_block(
                &mut g,
                &mut params,
                x,
                &format!("encoder_blocks.{block_idx}"),
                st,
                k.saturating_sub(st),
                true,
                out_ch,
                cfg.dim,
                k,
                t,
            )?;
            block_idx += 1;
            if st > 1 {
                window = (window / 2).max(1);
            }
        }
    }

    let d_sem = cfg.semantic_dim;
    let d_ac = cfg.acoustic_dim;
    let sem_part = g.narrow_(x, 0, 0, d_sem);
    let ac_part = g.narrow_(x, 0, d_sem, d_ac);
    let ac_scale = scalar(&mut g, 0.3);
    let ac_scaled = g.mul(ac_part, ac_scale);
    let ac_tanh = g.tanh(ac_scaled);
    let latent = g.concat_(vec![sem_part, ac_tanh], 0);

    let codebook = g.param(
        "quantizer.semantic_codebook.embedding",
        Shape::new(&[cfg.semantic_codebook_size, d_sem], f),
    );
    params.push(ParamSlot {
        name: "quantizer.semantic_codebook.embedding".into(),
        param: codebook,
        grad: None,
        trainable: false,
        num_elems: cfg.semantic_codebook_size * d_sem,
    });

    let sem_t = g.transpose_(sem_part, vec![1, 0]);
    let codebook_t = g.transpose_(codebook, vec![1, 0]);
    let dist = g.mm(sem_t, codebook_t);
    let probs = g.sm(dist, -1);
    let sem_q = g.mm(probs, codebook);
    let sem_c = g.transpose_(sem_q, vec![1, 0]);
    let quantized = g.concat_(vec![sem_c, ac_tanh], 0);

    let mut dec = quantized;
    let dec_kernels = cfg.decoder_convs_kernels();
    let dec_strides = cfg.decoder_convs_strides();
    let dec_lens = cfg.decoder_transformer_lengths();

    dec = add_conv_block(
        &mut g,
        &mut params,
        dec,
        "decoder_blocks.0",
        dec_strides[0],
        dec_kernels[0] - dec_strides[0],
        false,
        cfg.dim,
        d_sem + d_ac,
        dec_kernels[0],
        layout.latent_t,
    )?;

    let mut block_idx = 1usize;
    let mut window = cfg.attn_sliding_window_size;
    let mut dec_t = layout.latent_t;
    for (stage, n_layers) in dec_lens.iter().enumerate() {
        dec = add_transformer_stack(
            &mut g,
            &mut params,
            dec,
            cfg,
            &format!("decoder_blocks.{block_idx}"),
            *n_layers,
            window,
            dec_t,
            false,
        )?;
        block_idx += 1;
        if stage + 1 < dec_lens.len() {
            let k = dec_kernels[stage + 1];
            let st = dec_strides[stage + 1];
            let total_pad = k - st;
            dec_t *= st;
            dec = add_conv_transpose_block(
                &mut g,
                &mut params,
                dec,
                &format!("decoder_blocks.{block_idx}"),
                st,
                total_pad - (total_pad / 2),
                total_pad / 2,
                cfg.dim,
                k,
                dec_t / st,
                dec_t,
            )?;
            if st > 1 {
                window *= 2;
            }
            block_idx += 1;
        }
    }

    let k_out = cfg.patch_proj_kernel_size;
    let recon = add_conv_block(
        &mut g,
        &mut params,
        dec,
        "output_proj",
        1,
        k_out - 1,
        false,
        layout.patch_size,
        cfg.dim,
        k_out,
        dec_t,
    )?;

    g.set_outputs(vec![recon]);
    Ok(CodecForwardGraph {
        graph: g,
        layout: layout.clone(),
        params,
        audio_in,
        target_wav,
        recon_wav: recon,
        latent,
        quantized,
    })
}

fn add_conv_block(
    g: &mut Graph,
    params: &mut Vec<ParamSlot>,
    x: NodeId,
    prefix: &str,
    stride: usize,
    pad_left: usize,
    trainable: bool,
    out_ch: usize,
    in_ch: usize,
    k: usize,
    t_in: usize,
) -> Result<NodeId> {
    let f = DType::F32;
    let w = g.param(
        format!("{prefix}.conv.weight"),
        Shape::new(&[out_ch, in_ch, k], f),
    );
    params.push(ParamSlot {
        name: format!("{prefix}.conv.weight"),
        param: w,
        grad: None,
        trainable,
        num_elems: out_ch * in_ch * k,
    });
    conv1d(g, x, w, out_ch, in_ch, k, t_in, stride, pad_left, None)
}

pub fn conv1d_output_time(t_in: usize, k: usize, stride: usize, pad_left: usize) -> usize {
    let t_pad = t_in + pad_left;
    t_pad.saturating_sub(k) / stride + 1
}

fn add_conv_transpose_block(
    g: &mut Graph,
    params: &mut Vec<ParamSlot>,
    x: NodeId,
    prefix: &str,
    stride: usize,
    trim_left: usize,
    _trim_right: usize,
    channels: usize,
    k: usize,
    t_in: usize,
    t_out: usize,
) -> Result<NodeId> {
    let f = DType::F32;
    let w = g.param(
        format!("{prefix}.conv.weight"),
        Shape::new(&[channels, channels, k], f),
    );
    params.push(ParamSlot {
        name: format!("{prefix}.conv.weight"),
        param: w,
        grad: None,
        trainable: false,
        num_elems: channels * channels * k,
    });
    conv_transpose1d(
        g,
        x,
        w,
        channels,
        k,
        stride,
        trim_left,
        _trim_right,
        t_in,
        t_out,
    )
}

fn conv1d(
    g: &mut Graph,
    x: NodeId,
    w: NodeId,
    out_ch: usize,
    in_ch: usize,
    k: usize,
    t_in: usize,
    stride: usize,
    pad_left: usize,
    t_out_override: Option<usize>,
) -> Result<NodeId> {
    let t_out = t_out_override.unwrap_or_else(|| conv1d_output_time(t_in, k, stride, pad_left));
    let t_pad = t_in + pad_left;
    let pad = if pad_left > 0 {
        let left = g.narrow_(x, 1, 0, 1);
        let mut parts = vec![];
        for _ in 0..pad_left {
            parts.push(left);
        }
        parts.push(x);
        g.concat_(parts, 1)
    } else {
        x
    };
    // NCHW `[1, C, T, 1]` (time in H). Using `[1, C, 1, T]` made MLX NHWC
    // `(1, 1, T, C)` collide between input_proj and output_proj activations.
    let x4 = g.reshape_(pad, vec![1, i64s(in_ch), i64s(t_pad), 1]);
    let w4 = g.reshape_(w, vec![i64s(out_ch), i64s(in_ch), i64s(k), 1]);
    let y4 = g.conv2d(x4, w4, [k, 1], [stride, 1], [0, 0], [1, 1], 1);
    Ok(g.reshape_(y4, vec![i64s(out_ch), i64s(t_out)]))
}

fn upsample1d_zero_insert(
    g: &mut Graph,
    x: NodeId,
    channels: usize,
    t_in: usize,
    stride: usize,
) -> NodeId {
    let zero = scalar(g, 0.0);
    let col = g.narrow_(x, 1, 0, 1);
    let zero_col = g.mul(col, zero);
    let z1 = g.reshape_(zero_col, vec![i64s(channels), 1, 1]);
    let mut parts = Vec::with_capacity(t_in + t_in.saturating_sub(1) * stride.saturating_sub(1));
    for ti in 0..t_in {
        let slice = g.narrow_(x, 1, ti, 1);
        parts.push(g.reshape_(slice, vec![i64s(channels), 1, 1]));
        if ti + 1 < t_in {
            for _ in 1..stride {
                parts.push(z1);
            }
        }
    }
    let stacked = g.concat_(parts, 2);
    let t_up = t_in.saturating_sub(1) * stride + 1;
    g.reshape_(stacked, vec![i64s(channels), i64s(t_up)])
}

fn pad_time_right_zeros(g: &mut Graph, x: NodeId, pad: usize) -> Result<NodeId> {
    if pad == 0 {
        return Ok(x);
    }
    let zero = scalar(g, 0.0);
    let col = g.narrow_(x, 1, 0, 1);
    let zero_col = g.mul(col, zero);
    let mut parts = vec![x];
    for _ in 0..pad {
        parts.push(zero_col);
    }
    Ok(g.concat_(parts, 1))
}

fn conv_transpose1d(
    g: &mut Graph,
    x: NodeId,
    w: NodeId,
    channels: usize,
    k: usize,
    stride: usize,
    trim_left: usize,
    _trim_right: usize,
    t_in: usize,
    t_out: usize,
) -> Result<NodeId> {
    // ConvTranspose2d lacks autodiff + Metal/wgpu lowering; deconv = zero-upsample + conv2d.
    let up = upsample1d_zero_insert(g, x, channels, t_in, stride);
    let t_up = t_in.saturating_sub(1) * stride + 1;
    let w_perm = g.transpose_(w, vec![1, 0, 2]);
    let y = conv1d(g, up, w_perm, channels, channels, k, t_up, 1, 0, None)?;
    let pre_trim = t_up + k - 1;
    let got = conv1d_output_time(t_up, k, 1, 0);
    let y = if got < pre_trim {
        pad_time_right_zeros(g, y, pre_trim - got)?
    } else {
        y
    };
    Ok(g.narrow_(y, 1, trim_left, t_out))
}

fn add_transformer_stack(
    g: &mut Graph,
    params: &mut Vec<ParamSlot>,
    x: NodeId,
    cfg: &CodecArgs,
    prefix: &str,
    n_layers: usize,
    window: usize,
    t: usize,
    trainable: bool,
) -> Result<NodeId> {
    let mut h = x;
    for li in 0..n_layers {
        h = add_codec_layer(
            g,
            params,
            h,
            cfg,
            &format!("{prefix}.layers.{li}"),
            window,
            t,
            trainable,
        )?;
    }
    Ok(h)
}

fn add_codec_layer(
    g: &mut Graph,
    params: &mut Vec<ParamSlot>,
    x: NodeId,
    cfg: &CodecArgs,
    prefix: &str,
    _window: usize,
    t: usize,
    trainable: bool,
) -> Result<NodeId> {
    let d = cfg.dim;
    let hd = cfg.hidden_dim;
    let h = cfg.n_heads;
    let kv = cfg.n_kv_heads;
    let dh = cfg.head_dim;

    let attn_norm = param_vec(
        g,
        params,
        &format!("{prefix}.attention_norm.weight"),
        d,
        trainable,
    );
    let ffn_norm = param_vec(
        g,
        params,
        &format!("{prefix}.ffn_norm.weight"),
        d,
        trainable,
    );
    // HF/vLLM weights use per-channel RMSNorm for the full projection width (`dim`),
    // not per-head (`head_dim`). Apply RMSNorm before reshaping into heads.
    let q_norm = param_vec(
        g,
        params,
        &format!("{prefix}.attention.q_norm.weight"),
        d,
        trainable,
    );
    let k_norm = param_vec(
        g,
        params,
        &format!("{prefix}.attention.k_norm.weight"),
        d,
        trainable,
    );
    let attn_scale = param_vec(
        g,
        params,
        &format!("{prefix}.attention_scale"),
        d,
        trainable,
    );
    let ffn_scale = param_vec(g, params, &format!("{prefix}.ffn_scale"), d, trainable);

    let wq = param_mat(
        g,
        params,
        &format!("{prefix}.attention.wq.weight"),
        d,
        d,
        trainable,
    );
    let wk = param_mat(
        g,
        params,
        &format!("{prefix}.attention.wk.weight"),
        d,
        kv * dh,
        trainable,
    );
    let wv = param_mat(
        g,
        params,
        &format!("{prefix}.attention.wv.weight"),
        d,
        kv * dh,
        trainable,
    );
    let wo = param_mat(
        g,
        params,
        &format!("{prefix}.attention.wo.weight"),
        d,
        d,
        trainable,
    );
    let w1 = param_mat(
        g,
        params,
        &format!("{prefix}.feed_forward.w1.weight"),
        d,
        hd,
        trainable,
    );
    let w2 = param_mat(
        g,
        params,
        &format!("{prefix}.feed_forward.w2.weight"),
        hd,
        d,
        trainable,
    );
    let w3 = param_mat(
        g,
        params,
        &format!("{prefix}.feed_forward.w3.weight"),
        d,
        hd,
        trainable,
    );

    let beta = param_vec(g, params, &format!("{prefix}.__beta"), d, false);
    let x_tc = transpose_ct_to_tc(g, x);
    let hn = g.rms_norm(x_tc, attn_norm, beta, CODEC_NORM_EPS);
    let q = g.mm(hn, wq);
    let k = g.mm(hn, wk);
    let v = g.mm(hn, wv);
    let qn = g.rms_norm(q, q_norm, beta, CODEC_NORM_EPS);
    let kn = g.rms_norm(k, k_norm, beta, CODEC_NORM_EPS);

    let q4 = g.reshape_(qn, vec![1, i64s(t), i64s(h), i64s(dh)]);
    let k4 = g.reshape_(kn, vec![1, i64s(t), i64s(kv), i64s(dh)]);
    let v4 = g.reshape_(v, vec![1, i64s(t), i64s(kv), i64s(dh)]);
    let attn_s = rlx_ir::shape::attention_shape(g.shape(q4));
    let attn = g.attention_kind(q4, k4, v4, h, dh, MaskKind::Causal, attn_s);
    let attn2 = g.reshape_(attn, vec![i64s(t), i64s(h * dh)]);
    let attn_out = g.mm(attn2, wo);
    let attn_scaled = g.mul(attn_out, attn_scale);
    let x_t = transpose_ct_to_tc(g, x);
    let residual1 = g.add(x_t, attn_scaled);
    let h2 = g.rms_norm(residual1, ffn_norm, beta, CODEC_NORM_EPS);
    let ff1 = g.mm(h2, w1);
    let ff3 = g.mm(h2, w3);
    let silu1 = g.silu(ff1);
    let gated = g.mul(silu1, ff3);
    let ff = g.mm(gated, w2);
    let ff_scaled = g.mul(ff, ffn_scale);
    let out_t = g.add(residual1, ff_scaled);
    Ok(transpose_tc_to_ct(g, out_t))
}

fn param_vec(
    g: &mut Graph,
    params: &mut Vec<ParamSlot>,
    name: &str,
    n: usize,
    trainable: bool,
) -> NodeId {
    let id = g.param(name, Shape::new(&[n], DType::F32));
    params.push(ParamSlot {
        name: name.to_string(),
        param: id,
        grad: None,
        trainable,
        num_elems: n,
    });
    id
}

fn param_mat(
    g: &mut Graph,
    params: &mut Vec<ParamSlot>,
    name: &str,
    rows: usize,
    cols: usize,
    trainable: bool,
) -> NodeId {
    let id = g.param(name, Shape::new(&[rows, cols], DType::F32));
    params.push(ParamSlot {
        name: name.to_string(),
        param: id,
        grad: None,
        trainable,
        num_elems: rows * cols,
    });
    id
}

fn transpose_ct_to_tc(g: &mut Graph, x: NodeId) -> NodeId {
    g.transpose_(x, vec![1, 0])
}

fn transpose_tc_to_ct(g: &mut Graph, x: NodeId) -> NodeId {
    g.transpose_(x, vec![1, 0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_codec_args() -> CodecArgs {
        CodecArgs {
            channels: 1,
            sampling_rate: 24000,
            pretransform_patch_size: 240,
            patch_proj_kernel_size: 7,
            semantic_codebook_size: 8192,
            semantic_dim: 256,
            acoustic_codebook_size: 21,
            acoustic_dim: 36,
            dim: 1024,
            hidden_dim: 4096,
            head_dim: 128,
            n_heads: 8,
            n_kv_heads: 8,
            attn_sliding_window_size: 16,
            encoder_transformer_lengths_str: "2,2,2,2".into(),
            encoder_convs_kernels_str: "4,4,4,3".into(),
            encoder_convs_strides_str: "2,2,2,1".into(),
            decoder_transformer_lengths_str: "2,2,2,2".into(),
            decoder_convs_kernels_str: "3,4,4,4".into(),
            decoder_convs_strides_str: "1,2,2,2".into(),
        }
    }

    #[test]
    fn codec_graph_builds() {
        let cfg = sample_codec_args();
        let layout = CodecGraphLayout::new(&cfg, 32);
        assert_eq!(layout.latent_t, crate::config::latent_frames(&cfg, 32));
        let fwd = build_codec_forward_graph(&cfg, &layout).expect("build");
        assert!(fwd.params.iter().any(|p| p.trainable));
    }
}
