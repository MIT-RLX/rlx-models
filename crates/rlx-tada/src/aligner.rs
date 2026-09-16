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

//! The forced aligner: `facebook/wav2vec2-large` with a CTC head over the
//! **Llama vocabulary** (`tada.modules.aligner.Aligner`).
//!
//! The unusual part is the head: instead of ~32 characters it emits 128 256
//! logits, one per Llama BPE token, so a text token can be scored directly
//! against a 50 Hz frame with no grapheme-to-token bridge in between. That is
//! what lets [`crate::align`]'s dynamic program assign one frame per token and
//! makes the whole model "dual aligned".
//!
//! Standard post-norm wav2vec2 otherwise: a 7-layer strided conv front end
//! (320× downsampling, so 16 kHz audio lands exactly on the 50 Hz grid), a
//! grouped positional convolution, and 24 encoder layers.

use crate::builder::{Ctx, F32, compile, lower};
use crate::config::AlignerConfig;
use crate::weights::{TensorStore, fuse_weight_norm_dim2};
use anyhow::{Context, Result, bail};
use ndarray::ArrayView3;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::Op;
use rlx_ir::{HirGraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::sync::Arc;

/// Key prefix for one encoder layer; every shape is derivable from the config.
struct EncoderLayer {
    prefix: String,
}

/// The aligner, held as names and shapes rather than weights.
///
/// This stack is 447 M parameters — 1.8 GB as f32 — and every one of them is
/// interned into the graph at build time anyway. Keeping a second copy on the
/// heap for the life of the model is pure overhead, so weights are read from
/// the checkpoint on demand, exactly like the backbone and the codec.
pub struct Aligner {
    store: Arc<TensorStore>,
    /// `(key, shape, stride)` per conv layer.
    convs: Vec<(String, Vec<usize>, usize)>,
    /// Positional conv shape; its weight-norm fusion happens at build time.
    pos_conv_shape: Vec<usize>,
    layers: Vec<EncoderLayer>,
    /// `[vocab, hidden]` — the CTC head, 131 M parameters on its own.
    lm_head_out: usize,
    cfg: AlignerConfig,
}

const W2V: &str = "encoder.wav2vec2";

impl Aligner {
    pub fn load(store: Arc<TensorStore>, cfg: AlignerConfig) -> Result<Self> {
        let mut convs = Vec::with_capacity(cfg.conv_layers.len());
        for (i, &(_, _, stride)) in cfg.conv_layers.iter().enumerate() {
            let key = format!("{W2V}.feature_extractor.conv_layers.{i}.conv.weight");
            convs.push((key.clone(), store.shape(&key)?.to_vec(), stride));
        }
        let v_key = format!("{W2V}.encoder.pos_conv_embed.conv.parametrizations.weight.original1");
        let pos_conv_shape = store.shape(&v_key)?.to_vec();
        if pos_conv_shape.len() != 3 {
            bail!("pos_conv kernel has shape {pos_conv_shape:?}, expected 3-D");
        }
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| EncoderLayer {
                prefix: format!("{W2V}.encoder.layers.{i}"),
            })
            .collect();
        let lm_head_out = store.shape("encoder.lm_head.weight")?[0];
        Ok(Self {
            store,
            convs,
            pos_conv_shape,
            layers,
            lm_head_out,
            cfg,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.lm_head_out
    }

    /// Frames the conv front end emits for `samples` of 16 kHz audio.
    pub fn frames_for(&self, samples: usize) -> usize {
        let mut t = samples;
        for (_, shape, stride) in &self.convs {
            let stride = *stride;
            let k = shape[2];
            if t < k {
                return 0;
            }
            t = (t - k) / stride + 1;
        }
        t
    }

    /// Compile a CTC graph for a fixed input length.
    ///
    /// Input `"pcm"` is `[1, 1, samples, 1]`; the output is `[frames, vocab]`
    /// logits, which [`crate::align::align_tokens`] consumes directly.
    pub fn compile(&self, device: Device, samples: usize) -> Result<CompiledGraph> {
        let frames = self.frames_for(samples);
        if frames == 0 {
            bail!("{samples} samples is too short for the aligner's conv front end");
        }
        let mut hir = HirModule::new("tada_aligner");
        let mut g = HirMut::new(&mut hir);
        let mut ctx = Ctx::new(&mut g);

        let x = ctx.g.input("pcm", Shape::new(&[1, 1, samples, 1], F32));
        let h = self.build_features(&mut ctx, x, samples)?;
        let h = self.build_encoder(&mut ctx, h, frames)?;
        let logits = ctx.store_linear(
            &self.store,
            h,
            "encoder.lm_head",
            self.lm_head_out,
            self.cfg.hidden_size,
        )?;
        let logits = ctx
            .g
            .reshape_(logits, vec![frames as i64, self.lm_head_out as i64]);

        let params = ctx.into_params();
        hir.set_outputs(vec![logits]);
        Ok(compile(device, lower(hir, "aligner")?, params))
    }

    /// Fuse the positional convolution's weight-norm pair.
    ///
    /// Weight-normed along **dim 2** (the kernel axis), unlike every other
    /// weight-normed conv in TADA — see [`fuse_weight_norm_dim2`].
    fn fuse_pos_conv(&self) -> Result<Vec<f32>> {
        let g = self.store.get(&format!(
            "{W2V}.encoder.pos_conv_embed.conv.parametrizations.weight.original0"
        ))?;
        let v = self.store.get(&format!(
            "{W2V}.encoder.pos_conv_embed.conv.parametrizations.weight.original1"
        ))?;
        let sh = &self.pos_conv_shape;
        Ok(fuse_weight_norm_dim2(
            &g,
            ArrayView3::from_shape((sh[0], sh[1], sh[2]), &v)?,
        ))
    }

    /// 7 strided convolutions with GELU, and a per-channel group norm on the
    /// first. Returns `[1, frames, conv_dim]`.
    fn build_features(&self, ctx: &mut Ctx, x: HirNodeId, samples: usize) -> Result<HirNodeId> {
        let mut cur = x;
        let mut c_in = 1usize;
        let mut t = samples;
        for (i, (key, shape, stride)) in self.convs.iter().enumerate() {
            let (c_out, w_in, k) = (shape[0], shape[1], shape[2]);
            if w_in != c_in {
                bail!("conv {i}: kernel expects {w_in} input channels, got {c_in}");
            }
            let t_out = (t - k) / stride + 1;
            let wp = ctx.param_keyed_try(&format!("conv{i}"), &[c_out, c_in, k, 1], || {
                self.store.get(key)
            })?;
            cur = ctx.g.add_node(
                Op::Conv {
                    kernel_size: vec![k, 1],
                    stride: vec![*stride, 1],
                    padding: vec![0, 0],
                    dilation: vec![1, 1],
                    groups: 1,
                },
                vec![cur, wp],
                Shape::new(&[1, c_out, t_out, 1], F32),
            );
            if i == 0 {
                // `feat_extract_norm="group"` with num_groups == channels, i.e.
                // each channel normalized over time on its own.
                let (gp, bp) = ctx.store_norm(
                    &self.store,
                    &format!("{W2V}.feature_extractor.conv_layers.0.layer_norm"),
                    c_out,
                )?;
                cur = ctx.g.group_norm(cur, gp, bp, c_out, 1e-5);
            }
            cur = ctx.g.gelu(cur);
            c_in = c_out;
            t = t_out;
        }
        // `[1, C, T, 1]` → `[1, T, C]`
        let nct = ctx.g.reshape_(cur, vec![1, c_in as i64, t as i64]);
        Ok(ctx.g.transpose_(nct, vec![0, 2, 1]))
    }

    /// Feature projection, positional convolution, then 24 post-norm layers.
    fn build_encoder(&self, ctx: &mut Ctx, x: HirNodeId, frames: usize) -> Result<HirNodeId> {
        let d = self.cfg.hidden_size;
        let cd = self.cfg.conv_dim;
        let (pg, pb) = ctx.store_norm(
            &self.store,
            &format!("{W2V}.feature_projection.layer_norm"),
            cd,
        )?;
        let normed = ctx.g.ln(x, pg, pb, self.cfg.layer_norm_eps);
        let mut h = ctx.store_linear(
            &self.store,
            normed,
            &format!("{W2V}.feature_projection.projection"),
            d,
            cd,
        )?;

        // Positional conv: grouped, symmetric-padded, then the trailing sample
        // is dropped because the kernel width is even (`Wav2Vec2SamePadLayer`).
        let k = self.pos_conv_shape[2];
        let groups = self.cfg.num_conv_pos_embedding_groups;
        let pad = k / 2;
        let hc = ctx.g.transpose_(h, vec![0, 2, 1]);
        let hc = ctx.g.reshape_(hc, vec![1, d as i64, frames as i64, 1]);
        let zl = ctx.param(vec![0.0; d * pad], &[1, d, pad, 1]);
        let zr = ctx.param(vec![0.0; d * pad], &[1, d, pad, 1]);
        let padded = ctx.g.concat_(vec![zl, hc, zr], 2);
        let t_pad = frames + 2 * pad;
        let t_out = t_pad - k + 1;
        let wp = ctx.param_keyed_try(
            "pos_conv",
            &[
                self.pos_conv_shape[0],
                self.pos_conv_shape[1],
                self.pos_conv_shape[2],
                1,
            ],
            || self.fuse_pos_conv(),
        )?;
        let mut pos = ctx.g.add_node(
            Op::Conv {
                kernel_size: vec![k, 1],
                stride: vec![1, 1],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups,
            },
            vec![padded, wp],
            Shape::new(&[1, d, t_out, 1], F32),
        );
        let bp = ctx.param_keyed_try("pos_conv.b", &[1, d, 1, 1], || {
            self.store
                .get(&format!("{W2V}.encoder.pos_conv_embed.conv.bias"))
        })?;
        let be = ctx.g.expand_(bp, vec![1, d as i64, t_out as i64, 1]);
        pos = ctx.g.add(pos, be);
        // Even kernel → one extra output step; upstream removes it from the end.
        let pos = if t_out > frames {
            ctx.g.narrow_(pos, 2, 0, frames)
        } else {
            pos
        };
        let pos = ctx.g.gelu(pos);
        let pos = ctx.g.reshape_(pos, vec![1, d as i64, frames as i64]);
        let pos = ctx.g.transpose_(pos, vec![0, 2, 1]);
        h = ctx.g.add(h, pos);
        let (eg, eb) = ctx.store_norm(&self.store, &format!("{W2V}.encoder.layer_norm"), d)?;
        h = ctx.g.ln(h, eg, eb, self.cfg.layer_norm_eps);

        let heads = self.cfg.num_attention_heads;
        let hd = self.cfg.head_dim();
        for layer in &self.layers {
            let lp = &layer.prefix;
            let proj_of = |ctx: &mut Ctx, name: &str, x| {
                ctx.store_linear(&self.store, x, &format!("{lp}.attention.{name}"), d, d)
            };
            let q = proj_of(ctx, "q_proj", h)?;
            let kk = proj_of(ctx, "k_proj", h)?;
            let v = proj_of(ctx, "v_proj", h)?;
            let q = ctx
                .g
                .reshape_(q, vec![1, frames as i64, heads as i64, hd as i64]);
            let kk = ctx
                .g
                .reshape_(kk, vec![1, frames as i64, heads as i64, hd as i64]);
            let v = ctx
                .g
                .reshape_(v, vec![1, frames as i64, heads as i64, hd as i64]);
            let attn = ctx.g.attention_kind(
                q,
                kk,
                v,
                heads,
                hd,
                rlx_ir::op::MaskKind::None,
                Shape::new(&[1, frames, heads, hd], F32),
            );
            let attn = ctx.g.reshape_(attn, vec![1, frames as i64, d as i64]);
            let proj = proj_of(ctx, "out_proj", attn)?;
            let res = ctx.g.add(h, proj);
            let (ag, ab) = ctx.store_norm(&self.store, &format!("{lp}.layer_norm"), d)?;
            h = ctx.g.ln(res, ag, ab, self.cfg.layer_norm_eps);

            let ff = self.cfg.intermediate_size;
            let inner = ctx.store_linear(
                &self.store,
                h,
                &format!("{lp}.feed_forward.intermediate_dense"),
                ff,
                d,
            )?;
            let inner = ctx.g.gelu(inner);
            let inner = ctx.store_linear(
                &self.store,
                inner,
                &format!("{lp}.feed_forward.output_dense"),
                d,
                ff,
            )?;
            let res = ctx.g.add(h, inner);
            let (fg, fb) = ctx.store_norm(&self.store, &format!("{lp}.final_layer_norm"), d)?;
            h = ctx.g.ln(res, fg, fb, self.cfg.layer_norm_eps);
        }
        Ok(h)
    }

    /// Run the CTC head over 16 kHz `pcm`, returning `[frames, vocab]` logits.
    pub fn logits(&self, device: Device, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        let frames = self.frames_for(pcm.len());
        let mut g = self
            .compile(device, pcm.len())
            .context("compile aligner graph")?;
        let out = g
            .run(&[("pcm", pcm)])
            .into_iter()
            .next()
            .context("aligner produced no output")?;
        Ok((out, frames))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_front_end_downsamples_by_320() {
        let a = AlignerConfig::default();
        // Mirror `frames_for` without a loaded model.
        let frames = |samples: usize| {
            let mut t = samples;
            for &(_, k, s) in &a.conv_layers {
                t = (t - k) / s + 1;
            }
            t
        };
        // One second of 16 kHz audio → 49 frames (320× with edge loss), which
        // is the 50 Hz grid the codec and the aligner have to agree on.
        assert_eq!(frames(16_000), 49);
        assert_eq!(frames(32_000), 99);
        assert_eq!(a.downsample(), 320);
    }
}
