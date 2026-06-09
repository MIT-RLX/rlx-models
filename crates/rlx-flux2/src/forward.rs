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

//! Native CPU forward for the FLUX.2 transformer denoiser.

use super::config::Flux2Config;
use super::layers::{
    ada_layer_norm_continuous, double_stream_mod, dual_attention, feed_forward, gate_mul,
    layer_norm_no_affine, linear_no_bias, modulate, modulate_scale_shift, parallel_attention,
    single_stream_mod, time_guidance_embed,
};
use super::rope::flux2_pos_embed;
use super::weights::Flux2Weights;
use anyhow::{Result, ensure};

/// Inputs for one transformer forward (noise prediction).
pub struct Flux2ForwardInput<'a> {
    /// Image latents `[batch, img_seq, in_channels]`.
    pub hidden_states: &'a [f32],
    /// Text encoder states `[batch, txt_seq, joint_attention_dim]`.
    pub encoder_hidden_states: &'a [f32],
    /// Per-batch timestep (sigma); multiplied by 1000 inside the model.
    pub timestep: &'a [f32],
    /// Optional second time for flow-map embedding (average with `timestep`).
    pub timestep_target: Option<&'a [f32]>,
    /// Per-batch guidance scale; multiplied by 1000 when guidance embeds are enabled.
    pub guidance: Option<&'a [f32]>,
    /// Position ids `[img_seq + txt_seq, 4]` (concatenated txt then img along axis 0).
    pub img_ids: &'a [f32],
    pub txt_ids: &'a [f32],
    pub batch: usize,
    pub img_seq: usize,
    pub txt_seq: usize,
}

/// Run the FLUX.2 transformer and return noise prediction
/// `[batch, img_seq, patch_size² * out_channels]`.
pub fn flux2_transformer_forward(
    weights: &Flux2Weights,
    cfg: &Flux2Config,
    input: Flux2ForwardInput<'_>,
) -> Result<Vec<f32>> {
    let dim = cfg.inner_dim();
    let heads = cfg.num_attention_heads;
    let head_dim = cfg.attention_head_dim;
    let eps = cfg.eps as f32;
    let rope_dim: usize = cfg.axes_dims_rope.iter().sum();
    let b = input.batch;
    let img_seq = input.img_seq;
    let txt_seq = input.txt_seq;
    ensure!(input.hidden_states.len() == b * img_seq * cfg.in_channels);
    ensure!(input.encoder_hidden_states.len() == b * txt_seq * cfg.joint_attention_dim);
    ensure!(input.timestep.len() == b);

    let t_scaled: Vec<f32> = input.timestep.iter().map(|t| t * 1000.0).collect();
    let g_scaled = input
        .guidance
        .map(|g| g.iter().map(|x| x * 1000.0).collect::<Vec<_>>());
    let tg_tgt = weights
        .time_guidance_target
        .as_ref()
        .unwrap_or(&weights.time_guidance);
    let temb = if let Some(t_tgt) = input.timestep_target {
        let tgt_scaled: Vec<f32> = t_tgt.iter().map(|t| t * 1000.0).collect();
        super::layers::time_guidance_embed_dual(
            &t_scaled,
            &tgt_scaled,
            g_scaled.as_deref(),
            &weights.time_guidance,
            tg_tgt,
            dim,
        )?
    } else {
        time_guidance_embed(&t_scaled, g_scaled.as_deref(), &weights.time_guidance, dim)?
    };

    let mod_img = double_stream_mod(&temb, b, dim, &weights.double_mod_img.linear)?;
    let mod_txt = double_stream_mod(&temb, b, dim, &weights.double_mod_txt.linear)?;
    let single_mod = single_stream_mod(&temb, b, dim, &weights.single_mod.linear)?;

    let mut hidden = linear_no_bias(input.hidden_states, b * img_seq, &weights.x_embedder)?;
    let mut encoder = linear_no_bias(
        input.encoder_hidden_states,
        b * txt_seq,
        &weights.context_embedder,
    )?;

    let n_axes = 4usize;
    let total_seq = txt_seq + img_seq;
    let mut ids = vec![0.0f32; total_seq * n_axes];
    for t in 0..txt_seq {
        for a in 0..n_axes {
            ids[t * n_axes + a] = input.txt_ids[t * n_axes + a];
        }
    }
    for t in 0..img_seq {
        for a in 0..n_axes {
            ids[(txt_seq + t) * n_axes + a] = input.img_ids[t * n_axes + a];
        }
    }
    let (cos, sin) = flux2_pos_embed(cfg, &ids, total_seq, n_axes);

    for block in &weights.transformer_blocks {
        let (img_msa, img_mlp) = &mod_img;
        let (txt_msa, txt_mlp) = &mod_txt;

        let n1 = layer_norm_no_affine(&hidden, dim, eps)?;
        let n1 = modulate(&n1, &img_msa.0, &img_msa.1, dim, b, img_seq);
        let nc = layer_norm_no_affine(&encoder, dim, eps)?;
        let nc = modulate(&nc, &txt_msa.0, &txt_msa.1, dim, b, txt_seq);

        let (enc_attn, img_attn) = dual_attention(
            &block.attn,
            &n1,
            &nc,
            b,
            img_seq,
            txt_seq,
            heads,
            head_dim,
            dim,
            &cos,
            &sin,
            rope_dim,
        )?;
        hidden = add_residual(&hidden, &gate_mul(&img_attn, &img_msa.2, dim, b, img_seq));
        encoder = add_residual(&encoder, &gate_mul(&enc_attn, &txt_msa.2, dim, b, txt_seq));

        let n2 = layer_norm_no_affine(&hidden, dim, eps)?;
        let n2 = modulate_scale_shift(&n2, &img_mlp.1, &img_mlp.0, dim, b, img_seq);
        let ff = feed_forward(&block.ff, &n2, b * img_seq, dim)?;
        hidden = add_residual(&hidden, &gate_mul(&ff, &img_mlp.2, dim, b, img_seq));

        let nc2 = layer_norm_no_affine(&encoder, dim, eps)?;
        let nc2 = modulate_scale_shift(&nc2, &txt_mlp.1, &txt_mlp.0, dim, b, txt_seq);
        let ffc = feed_forward(&block.ff_context, &nc2, b * txt_seq, dim)?;
        encoder = add_residual(&encoder, &gate_mul(&ffc, &txt_mlp.2, dim, b, txt_seq));
    }

    let mut concat = vec![0.0f32; b * (txt_seq + img_seq) * dim];
    for bi in 0..b {
        concat[bi * (txt_seq + img_seq) * dim..bi * (txt_seq + img_seq) * dim + txt_seq * dim]
            .copy_from_slice(&encoder[bi * txt_seq * dim..(bi + 1) * txt_seq * dim]);
        concat
            [bi * (txt_seq + img_seq) * dim + txt_seq * dim..(bi + 1) * (txt_seq + img_seq) * dim]
            .copy_from_slice(&hidden[bi * img_seq * dim..(bi + 1) * img_seq * dim]);
    }

    let mlp_hidden = (dim as f64 * cfg.mlp_ratio) as usize;
    let mut stream = concat;
    for block in &weights.single_transformer_blocks {
        let n = layer_norm_no_affine(&stream, dim, eps)?;
        let n = modulate(&n, &single_mod.0, &single_mod.1, dim, b, txt_seq + img_seq);
        let attn = parallel_attention(
            &block.attn,
            &n,
            b,
            txt_seq + img_seq,
            heads,
            head_dim,
            dim,
            mlp_hidden,
            &cos,
            &sin,
            rope_dim,
        )?;
        stream = add_residual(
            &stream,
            &gate_mul(&attn, &single_mod.2, dim, b, txt_seq + img_seq),
        );
    }

    let mut hidden = vec![0.0f32; b * img_seq * dim];
    for bi in 0..b {
        hidden[bi * img_seq * dim..(bi + 1) * img_seq * dim].copy_from_slice(
            &stream[bi * (txt_seq + img_seq) * dim + txt_seq * dim
                ..(bi + 1) * (txt_seq + img_seq) * dim],
        );
    }

    let normed = ada_layer_norm_continuous(
        &hidden,
        &temb,
        b,
        img_seq,
        dim,
        &weights.norm_out.linear,
        eps,
    )?;
    linear_no_bias(&normed, b * img_seq, &weights.proj_out)
}

fn add_residual(base: &[f32], delta: &[f32]) -> Vec<f32> {
    base.iter().zip(delta.iter()).map(|(a, d)| a + d).collect()
}
