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

//! The TADA codec: 24 kHz waveform ⇄ 50 Hz, 512-wide continuous latents.
//!
//! Both halves are a DAC convolutional stack bolted to a
//! [`crate::local_attn::LocalAttnStack`]. The convolutions are
//! DAC's verbatim — same block layout, same weight-norm parameterization, same
//! Snake activations — so they are built with `rlx-dac`'s graph builders rather
//! than re-derived here, which also means they inherit that crate's
//! cross-backend validation.
//!
//! Conv stack and transformer run as two compiled graphs with a host hop
//! between them. The hop moves `[1024, frames]` floats (a few hundred kB for a
//! typical prompt) and buys a clean split: the conv graph is specialized on PCM
//! length while the transformer graph is specialized on frame count and needs a
//! host-built segment mask anyway.

use crate::builder::{Ctx, F32, NamedTensors};
use crate::config::{DecoderConfig, EncoderConfig};
use crate::local_attn::LocalAttnStack;
use crate::mask::{decoder_segment_bias, encoder_segment_bias};
use crate::weights::{self, Linear, TensorStore};
use anyhow::{Context, Result, bail};
use ndarray::Array2;
use rlx_dac::graph::CodecGraph;
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{HirGraphExt, Shape};
use rlx_runtime::Device;
use std::sync::Arc;

/// Compile a codec transformer graph through the on-disk LIR cache.
///
/// Keyed on frame count, so a repeated utterance of the same length skips
/// lowering. The DAC conv stacks around it go through `rlx-dac`'s own builder
/// and are not cached here.
fn compile_cached(
    device: Device,
    key: &str,
    hir: HirModule,
    params: NamedTensors,
) -> Result<rlx_runtime::CompiledGraph> {
    let cache = crate::backbone::aot_cache("codec");
    let mut g = cache
        .compile_hir_cached(key, device, hir, &rlx_runtime::CompileOptions::default())
        .map_err(|e| anyhow::anyhow!("compile {key}: {e}"))?;
    for (name, data) in &params {
        g.set_param(name, data);
    }
    g.finalize_params();
    Ok(g)
}

/// Zero samples appended to the reference audio before encoding. Upstream pads
/// by exactly this much (`F.pad(audio, (0, 960))`), which buys two extra
/// frames of encoder context past the end of the signal.
const ENCODER_TAIL_PAD: usize = 960;

/// Waveform → per-token acoustic latents.
pub struct CodecEncoder {
    store: Arc<TensorStore>,
    stack: LocalAttnStack,
    hidden_linear: Linear,
    /// `Embedding(2, hidden_dim)`, indexed by the frame's token mask bit.
    pos_emb: Vec<f32>,
    cfg: EncoderConfig,
}

impl CodecEncoder {
    pub fn load(store: Arc<TensorStore>, cfg: EncoderConfig) -> Result<Self> {
        let stack = LocalAttnStack::load(
            store.clone(),
            "local_attention_encoder",
            cfg.num_attn_layers,
            cfg.num_attn_heads,
            1e-5,
        )
        .context("codec encoder: local attention stack")?;
        let pos_emb = store.get("pos_emb.weight")?;
        if pos_emb.len() != 2 * cfg.hidden_dim {
            bail!(
                "pos_emb.weight has {} values, expected 2 × {}",
                pos_emb.len(),
                cfg.hidden_dim
            );
        }
        Ok(Self {
            hidden_linear: store.linear("hidden_linear.weight")?,
            store,
            stack,
            pos_emb,
            cfg,
        })
    }

    /// Fuse the DAC conv stack out of the mapping. ~10 M params for the
    /// encoder, ~530 M for the decoder — materialized for the length of one
    /// graph build and then dropped, rather than held for the model's life.
    fn wav_layers(&self) -> Result<rlx_dac::layers::Encoder> {
        weights::load_wav_encoder(&self.store, "wav_encoder", &self.cfg.strides)
    }

    /// Frames the conv stack will emit for `pcm_len` input samples.
    ///
    /// Derived by building the graph rather than by re-deriving the
    /// convolution arithmetic, so it cannot drift from what actually runs.
    pub fn frames_for(&self, device: Device, pcm_len: usize) -> Result<usize> {
        let g = CodecGraph::encoder(device, &self.wav_layers()?, pcm_len + ENCODER_TAIL_PAD)?;
        Ok(g.out_dims().1)
    }

    /// Encode 24 kHz mono `pcm` into `[frames, embed_dim]` latents.
    ///
    /// `token_mask` marks which frames a text token landed on; it is padded
    /// with zeros out to the encoder's frame count. The mask drives both the
    /// learned position embedding and the segment attention bias.
    pub fn encode(&self, device: Device, pcm: &[f32], token_mask: &[u8]) -> Result<Array2<f32>> {
        if pcm.is_empty() {
            bail!("codec encoder: empty reference audio");
        }
        let mut padded_pcm = Vec::with_capacity(pcm.len() + ENCODER_TAIL_PAD);
        padded_pcm.extend_from_slice(pcm);
        padded_pcm.resize(pcm.len() + ENCODER_TAIL_PAD, 0.0);

        let mut conv = CodecGraph::encoder(device, &self.wav_layers()?, padded_pcm.len())
            .context("compile codec encoder conv stack")?;
        let (channels, frames) = conv.out_dims();
        if channels != self.cfg.hidden_dim {
            bail!(
                "codec encoder conv emitted {channels} channels, config says {}",
                self.cfg.hidden_dim
            );
        }
        if token_mask.len() > frames {
            bail!(
                "token mask covers {} frames but the encoder produced {frames}",
                token_mask.len()
            );
        }
        // `[hidden, frames]` channel-major out of the conv graph.
        let enc = conv
            .run(&padded_pcm)
            .context("run codec encoder conv stack")?;

        let mut mask = token_mask.to_vec();
        mask.resize(frames, 0);

        let mut transformer = self.compile_transformer(device, &mask, frames)?;
        // Transpose to `[frames, hidden]` and fold in the position embedding
        // host-side — it is a two-row table, so a gather would cost more graph
        // than it saves.
        let mut x = vec![0f32; frames * self.cfg.hidden_dim];
        for t in 0..frames {
            let row = &mut x[t * self.cfg.hidden_dim..(t + 1) * self.cfg.hidden_dim];
            let emb = &self.pos_emb[mask[t] as usize * self.cfg.hidden_dim..];
            for (c, slot) in row.iter_mut().enumerate() {
                *slot = enc[[c, t]] + emb[c];
            }
        }
        let out = transformer
            .run(&[("x", &x)])
            .into_iter()
            .next()
            .context("codec encoder transformer produced no output")?;
        Array2::from_shape_vec((frames, self.cfg.embed_dim), out)
            .context("codec encoder output reshape")
    }

    fn compile_transformer(
        &self,
        device: Device,
        mask: &[u8],
        frames: usize,
    ) -> Result<rlx_runtime::CompiledGraph> {
        let mut hir = HirModule::new("tada_codec_encoder");
        let mut g = HirMut::new(&mut hir);
        let mut ctx = Ctx::new(&mut g);
        let h = self.cfg.hidden_dim;
        let x = ctx.g.input("x", Shape::new(&[1, frames, h], F32));

        let heads = self.cfg.num_attn_heads;
        let bias2d = encoder_segment_bias(mask);
        let bias = ctx.param(bias2d, &[1, 1, frames, frames]);
        let bias = ctx
            .g
            .expand_(bias, vec![1, heads as i64, frames as i64, frames as i64]);

        let y = self.stack.build(&mut ctx, x, bias, frames)?;
        let y = ctx.linear(
            y,
            &self.hidden_linear.weight,
            self.cfg.embed_dim,
            h,
            self.hidden_linear.bias.as_deref(),
        );
        let params = ctx.into_params();
        hir.set_outputs(vec![y]);
        compile_cached(device, &format!("codec_enc_f{frames}_h{h}"), hir, params)
    }
}

/// Per-frame latents → 24 kHz waveform.
pub struct CodecDecoder {
    store: Arc<TensorStore>,
    prefix: String,
    proj: Linear,
    stack: LocalAttnStack,
    cfg: DecoderConfig,
}

impl CodecDecoder {
    /// Load from `{prefix}decoder_proj.weight` &c. The standalone
    /// `tada-codec/decoder` checkpoint uses an empty prefix; the copy bundled
    /// in `tada-1b` lives under `_decoder.`.
    pub fn load(store: Arc<TensorStore>, prefix: &str, cfg: DecoderConfig) -> Result<Self> {
        let stack = LocalAttnStack::load(
            store.clone(),
            &format!("{prefix}local_attention_decoder"),
            cfg.num_attn_layers,
            cfg.num_attn_heads,
            1e-5,
        )
        .context("codec decoder: local attention stack")?;
        Ok(Self {
            proj: store.linear(&format!("{prefix}decoder_proj.weight"))?,
            store,
            prefix: prefix.to_string(),
            stack,
            cfg,
        })
    }

    /// See [`CodecEncoder::wav_layers`] — the 530 M-parameter DAC stack is
    /// fused from the mapping per call and dropped once its graph owns it.
    fn wav_layers(&self) -> Result<rlx_dac::layers::Decoder> {
        weights::load_wav_decoder(
            &self.store,
            &format!("{}wav_decoder", self.prefix),
            &self.cfg.strides,
        )
    }

    /// Decode `[frames, embed_dim]` latents to mono 24 kHz PCM.
    ///
    /// Frames whose latent is exactly zero are the silence inserted between
    /// tokens; upstream derives the segment mask from that same test
    /// (`norm(x, dim=-1) != 0`), so it is reproduced here rather than passed in.
    pub fn decode(&self, device: Device, latents: &Array2<f32>) -> Result<Vec<f32>> {
        let (frames, width) = latents.dim();
        if width != self.cfg.embed_dim {
            bail!(
                "codec decoder: latents are {width} wide, config says {}",
                self.cfg.embed_dim
            );
        }
        if frames == 0 {
            return Ok(Vec::new());
        }
        let mask: Vec<u8> = (0..frames)
            .map(|t| u8::from(latents.row(t).iter().any(|v| *v != 0.0)))
            .collect();

        let mut transformer = self.compile_transformer(device, &mask, frames)?;
        let flat: Vec<f32> = latents.iter().copied().collect();
        let decoded = transformer
            .run(&[("z", &flat)])
            .into_iter()
            .next()
            .context("codec decoder transformer produced no output")?;

        // `[frames, hidden]` → `[hidden, frames]` for the conv stack.
        let h = self.cfg.hidden_dim;
        let mut chan_major = vec![0f32; h * frames];
        for t in 0..frames {
            for c in 0..h {
                chan_major[c * frames + t] = decoded[t * h + c];
            }
        }
        let mut conv = CodecGraph::decoder(device, &self.wav_layers()?, h, frames)
            .context("compile codec decoder conv stack")?;
        let wav = conv
            .run(&chan_major)
            .context("run codec decoder conv stack")?;
        Ok(wav.iter().copied().collect())
    }

    fn compile_transformer(
        &self,
        device: Device,
        mask: &[u8],
        frames: usize,
    ) -> Result<rlx_runtime::CompiledGraph> {
        let mut hir = HirModule::new("tada_codec_decoder");
        let mut g = HirMut::new(&mut hir);
        let mut ctx = Ctx::new(&mut g);
        let e = self.cfg.embed_dim;
        let h = self.cfg.hidden_dim;
        let z = ctx.g.input("z", Shape::new(&[1, frames, e], F32));
        let x = ctx.linear(z, &self.proj.weight, h, e, self.proj.bias.as_deref());

        let heads = self.cfg.num_attn_heads;
        let bias2d = decoder_segment_bias(mask);
        let bias = ctx.param(bias2d, &[1, 1, frames, frames]);
        let bias = ctx
            .g
            .expand_(bias, vec![1, heads as i64, frames as i64, frames as i64]);

        let y = self.stack.build(&mut ctx, x, bias, frames)?;
        let params = ctx.into_params();
        hir.set_outputs(vec![y]);
        compile_cached(
            device,
            &format!("codec_dec_f{frames}_e{e}_h{h}"),
            hir,
            params,
        )
    }
}

/// Expand per-token latents onto the frame grid, inserting `gap - 1` silent
/// frames before each token and one final run of `gaps[n]` after the last.
///
/// This is `TadaForCausalLM._decode_wav`'s layout step. The gaps are what the
/// diffusion head predicted as `time_before`, so this is where the model's
/// duration decisions become actual silence. Upstream truncates `gaps` to
/// `n + 1` and then indexes it unconditionally, which raises if it is shorter;
/// that becomes an error here rather than a short waveform, because a silently
/// truncated utterance is indistinguishable from a correct one downstream.
pub fn expand_to_frames(latents: &Array2<f32>, gaps: &[u32], width: usize) -> Result<Array2<f32>> {
    let n = latents.nrows();
    if gaps.len() < n + 1 {
        bail!(
            "need {} frame gaps for {n} latents, got {}",
            n + 1,
            gaps.len()
        );
    }
    let total: usize = (0..n).map(|i| gaps[i].max(1) as usize).sum::<usize>() + gaps[n] as usize;
    let mut out = Array2::<f32>::zeros((total, width));
    let mut row = 0usize;
    for i in 0..n {
        row += gaps[i].saturating_sub(1) as usize;
        for c in 0..width {
            out[[row, c]] = latents[[i, c]];
        }
        row += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_places_each_latent_after_its_silence() {
        let latents = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        // First token: 3 frames before → 2 silent rows then the latent.
        // Second token: 1 frame before → no silence.
        // Trailing gap 2 → two silent rows at the end.
        let out = expand_to_frames(&latents, &[3, 1, 2], 2).unwrap();
        assert_eq!(out.nrows(), 2 + 1 + 1 + 2);
        assert_eq!(out.row(2).to_vec(), vec![1.0, 2.0]);
        assert_eq!(out.row(3).to_vec(), vec![3.0, 4.0]);
        assert!(out.row(0).iter().all(|&v| v == 0.0));
        assert!(out.row(5).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn a_zero_gap_still_reserves_the_token_row() {
        let latents = Array2::from_shape_vec((1, 2), vec![5.0, 6.0]).unwrap();
        let out = expand_to_frames(&latents, &[0, 0], 2).unwrap();
        assert_eq!(out.nrows(), 1);
        assert_eq!(out.row(0).to_vec(), vec![5.0, 6.0]);
    }

    #[test]
    fn total_length_is_the_sum_of_the_gaps() {
        let latents = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let gaps = [4u32, 2, 7, 3];
        let out = expand_to_frames(&latents, &gaps, 1).unwrap();
        assert_eq!(out.nrows(), 4 + 2 + 7 + 3);
        // Each latent lands on the last row of its own gap.
        assert_eq!(out[[3, 0]], 1.0);
        assert_eq!(out[[5, 0]], 2.0);
        assert_eq!(out[[12, 0]], 3.0);
    }

    #[test]
    fn too_few_gaps_is_an_error_not_a_short_waveform() {
        let latents = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(expand_to_frames(&latents, &[2, 1], 1).is_err());
    }
}
