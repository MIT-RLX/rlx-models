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

//! Voxtral codec **decoder** (vLLM-Omni `VoxtralTTSAudioTokenizer` decode path).

use crate::codec::layers::{
    CodecConvBlock, CodecTransformer, compute_semantic_embedding, load_codec_layer, rescale_fsq,
    run_conv, run_transformer, take_conv, take_conv_transpose, take1d, take2d,
};
use crate::config::CodecArgs;
use crate::tokens::{AUDIO_TOKEN_OFFSET, split_voxtral_frames};
use anyhow::{Result, ensure};
use ndarray::Array2;
use std::collections::HashMap;

pub struct CodecDecoder {
    cfg: CodecArgs,
    semantic_embedding: Array2<f32>,
    blocks: Vec<DecoderBlock>,
    output_weight: ndarray::Array3<f32>,
}

enum DecoderBlock {
    Conv(CodecConvBlock),
    Transformer(CodecTransformer),
}

impl CodecDecoder {
    pub fn from_tensors(
        prefix: &str,
        tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        cfg: &CodecArgs,
    ) -> Result<Self> {
        let sem_sum = take2d(
            tensors,
            &format!("{prefix}quantizer.semantic_codebook.embedding_sum"),
        )?;
        let sem_usage = take1d(
            tensors,
            &format!("{prefix}quantizer.semantic_codebook.cluster_usage"),
        )?;
        let semantic_embedding = compute_semantic_embedding(&sem_sum, &sem_usage);

        let dec_kernels = cfg.decoder_convs_kernels();
        let dec_strides = cfg.decoder_convs_strides();
        let dec_lens = cfg.decoder_transformer_lengths();
        ensure!(
            dec_kernels.len() == dec_strides.len(),
            "decoder conv config mismatch"
        );

        let mut blocks = Vec::new();
        let mut window = cfg.attn_sliding_window_size;
        let mut block_idx = 0usize;

        blocks.push(DecoderBlock::Conv(CodecConvBlock::Forward {
            weight: take_conv(tensors, &format!("{prefix}decoder_blocks.{block_idx}"))?,
            stride: dec_strides[0],
            pad_left: dec_kernels[0] - dec_strides[0],
        }));
        block_idx += 1;

        for (stage, n_layers) in dec_lens.iter().enumerate() {
            blocks.push(DecoderBlock::Transformer(CodecTransformer {
                window,
                layers: (0..*n_layers)
                    .map(|li| {
                        load_codec_layer(
                            tensors,
                            &format!("{prefix}decoder_blocks.{block_idx}.layers.{li}"),
                            cfg,
                        )
                    })
                    .collect::<Result<_>>()?,
            }));
            block_idx += 1;

            if stage + 1 < dec_lens.len() {
                let k = dec_kernels[stage + 1];
                let st = dec_strides[stage + 1];
                let total_pad = k - st;
                blocks.push(DecoderBlock::Conv(CodecConvBlock::Transpose {
                    weight: take_conv_transpose(
                        tensors,
                        &format!("{prefix}decoder_blocks.{block_idx}"),
                    )?,
                    stride: st,
                    trim_left: total_pad - (total_pad / 2),
                    trim_right: total_pad / 2,
                }));
                if st > 1 {
                    window *= 2;
                }
                block_idx += 1;
            }
        }

        let output_weight = take_conv(tensors, &format!("{prefix}output_proj"))?;
        Ok(Self {
            cfg: cfg.clone(),
            semantic_embedding,
            blocks,
            output_weight,
        })
    }

    /// Decode `[n_frames, 37]` vLLM-layout codes (semantic raw, acoustic +2 offset).
    pub fn decode_codes(&self, codes: &[u32], n_frames: usize) -> Result<Vec<f32>> {
        ensure!(
            codes.len() == n_frames * 37,
            "expected {}*37 codes",
            n_frames
        );
        let (semantic, acoustic, actual) = split_voxtral_frames(codes, n_frames);
        if actual == 0 {
            return Ok(Vec::new());
        }
        let latent = self.quantizer_decode(&semantic, &acoustic, actual)?;
        self.forward_decoder(&latent)
    }

    fn quantizer_decode(
        &self,
        semantic: &[usize],
        acoustic: &[u32],
        n_frames: usize,
    ) -> Result<Array2<f32>> {
        let d_sem = self.cfg.semantic_dim;
        let d_ac = self.cfg.acoustic_dim;
        let mut out = Array2::<f32>::zeros((d_sem + d_ac, n_frames));
        for fi in 0..n_frames {
            let sid = semantic[fi];
            ensure!(
                sid < self.semantic_embedding.dim().0,
                "semantic id {sid} oob"
            );
            for di in 0..d_sem {
                out[[di, fi]] = self.semantic_embedding[[sid, di]];
            }
            for ai in 0..36 {
                let level = acoustic[fi * 36 + ai];
                let v = rescale_fsq(level, self.cfg.acoustic_codebook_size);
                out[[d_sem + ai, fi]] = v;
            }
        }
        Ok(out)
    }

    fn forward_decoder(&self, emb: &Array2<f32>) -> Result<Vec<f32>> {
        let mut x = emb.to_owned();
        for block in &self.blocks {
            match block {
                DecoderBlock::Conv(conv) => {
                    x = run_conv(&x, conv);
                }
                DecoderBlock::Transformer(tr) => {
                    x = run_transformer(&x, tr)?;
                }
            }
        }
        let k = self.output_weight.shape()[2];
        let pad_left = k - 1;
        let wav = crate::math::conv1d(x.view(), self.output_weight.view(), 1, pad_left);
        let (c, t) = wav.dim();
        let mut pcm = Vec::with_capacity(c * t);
        for ti in 0..t {
            for ci in 0..c {
                pcm.push(wav[[ci, ti]]);
            }
        }
        Ok(pcm)
    }
}

#[allow(dead_code)]
pub fn apply_audio_offset(codes: &mut [u32]) {
    for c in codes.iter_mut() {
        *c += AUDIO_TOKEN_OFFSET;
    }
}
