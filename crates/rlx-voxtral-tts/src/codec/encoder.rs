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

//! Voxtral codec **encoder** — reference audio → discrete codes → voice embeddings.

use crate::backbone::embed::EmbeddingTables;
use crate::codec::layers::{
    CodecConvBlock, CodecTransformer, compute_semantic_embedding, load_codec_layer, run_conv,
    run_transformer, take_conv,
};
use crate::codec::layout::{EncoderExecBlock, encoder_execution_plan};
use crate::config::CodecArgs;
use crate::tokens::AUDIO_TOKEN_OFFSET;
use crate::voice::VoiceEmbedding;
use anyhow::{Context, Result, bail, ensure};
use ndarray::Array2;
use std::collections::HashMap;

enum EncoderBlock {
    Transformer(CodecTransformer),
    Conv(CodecConvBlock),
}

pub struct CodecEncoder {
    cfg: CodecArgs,
    patch_size: usize,
    input_weight: ndarray::Array3<f32>,
    input_stride: usize,
    input_pad_left: usize,
    blocks: Vec<EncoderBlock>,
    semantic_embedding: Array2<f32>,
}

impl CodecEncoder {
    pub fn from_tensors(
        prefix: &str,
        tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        cfg: &CodecArgs,
    ) -> Result<Self> {
        if !has_encoder_tensors(prefix, tensors) {
            bail!(
                "codec encoder weights missing under {prefix}input_proj / encoder_blocks.\n\
                 Reference-audio cloning needs a checkpoint that includes trained encoder weights."
            );
        }
        let sem_sum = crate::codec::layers::take2d(
            tensors,
            &format!("{prefix}quantizer.semantic_codebook.embedding_sum"),
        )?;
        let sem_usage = crate::codec::layers::take1d(
            tensors,
            &format!("{prefix}quantizer.semantic_codebook.cluster_usage"),
        )?;
        let semantic_embedding = compute_semantic_embedding(&sem_sum, &sem_usage);

        let input_weight = take_conv(tensors, &format!("{prefix}input_proj"))?;
        let input_kernel = input_weight.shape()[2];
        let input_stride = 1;
        let input_pad_left = input_kernel.saturating_sub(input_stride);

        let mut blocks = Vec::new();
        for step in encoder_execution_plan(cfg)? {
            match step {
                EncoderExecBlock::Transformer {
                    block_idx,
                    n_layers,
                    window,
                } => {
                    blocks.push(EncoderBlock::Transformer(CodecTransformer {
                        window,
                        layers: (0..n_layers)
                            .map(|li| {
                                load_codec_layer(
                                    tensors,
                                    &format!("{prefix}encoder_blocks.{block_idx}.layers.{li}"),
                                    cfg,
                                )
                            })
                            .collect::<Result<_>>()?,
                    }));
                }
                EncoderExecBlock::ConvDown {
                    block_idx,
                    kernel,
                    stride,
                    in_dim,
                    out_dim,
                } => {
                    let weight =
                        take_conv(tensors, &format!("{prefix}encoder_blocks.{block_idx}"))?;
                    let (oc, ic, _) = (weight.shape()[0], weight.shape()[1], weight.shape()[2]);
                    ensure!(
                        oc == out_dim && ic == in_dim,
                        "encoder_blocks.{block_idx} conv shape [{oc}, {ic}, _] \
                         != expected [{out_dim}, {in_dim}, {kernel}]"
                    );
                    blocks.push(EncoderBlock::Conv(CodecConvBlock::Forward {
                        weight,
                        stride,
                        pad_left: kernel.saturating_sub(stride),
                    }));
                }
            }
        }

        Ok(Self {
            cfg: cfg.clone(),
            patch_size: cfg.pretransform_patch_size,
            input_weight,
            input_stride,
            input_pad_left,
            blocks,
            semantic_embedding,
        })
    }

    /// Encode mono PCM at model sample rate into LLM voice rows `[n_frames, hidden]`.
    pub fn encode_pcm_to_voice_embedding(
        &self,
        pcm: &[f32],
        embed: &EmbeddingTables,
        name: &str,
    ) -> Result<VoiceEmbedding> {
        ensure!(!pcm.is_empty(), "reference audio is empty");
        let latent = self.forward_encoder(pcm)?;
        let (semantic, acoustic) = self.quantizer_encode(&latent)?;
        let hidden = embed.hidden_size();
        let n_frames = semantic.len();
        let mut data = Vec::with_capacity(n_frames * hidden);
        for fi in 0..n_frames {
            let mut frame = vec![0u32; 37];
            frame[0] = semantic[fi] as u32 + AUDIO_TOKEN_OFFSET;
            for ai in 0..36 {
                frame[1 + ai] = acoustic[fi * 36 + ai] + AUDIO_TOKEN_OFFSET;
            }
            let row = embed.embed_audio_frame(&frame);
            data.extend(row.iter().copied());
        }
        Ok(VoiceEmbedding {
            name: name.to_string(),
            data,
            n_tokens: n_frames,
            hidden,
        })
    }

    fn forward_encoder(&self, pcm: &[f32]) -> Result<Array2<f32>> {
        let patch = self.patch_size;
        let mut samples = pcm.to_vec();
        let rem = samples.len() % patch;
        if rem != 0 {
            samples.extend(std::iter::repeat_n(0f32, patch - rem));
        }
        let n_patches = samples.len() / patch;
        let mut x = Array2::<f32>::zeros((patch, n_patches));
        for pi in 0..n_patches {
            for ic in 0..patch {
                x[[ic, pi]] = samples[pi * patch + ic];
            }
        }
        x = crate::math::conv1d(
            x.view(),
            self.input_weight.view(),
            self.input_stride,
            self.input_pad_left,
        );

        for block in &self.blocks {
            match block {
                EncoderBlock::Conv(conv) => {
                    x = run_conv(&x, conv);
                }
                EncoderBlock::Transformer(tr) => {
                    x = run_transformer(&x, tr)?;
                }
            }
        }

        let latent_dim = self.cfg.latent_dim();
        let (d, _t) = x.dim();
        ensure!(
            d == latent_dim,
            "encoder output channels {d} != latent_dim {latent_dim}"
        );
        Ok(x.slice(ndarray::s![..latent_dim, ..]).to_owned())
    }

    fn quantizer_encode(&self, latent: &Array2<f32>) -> Result<(Vec<usize>, Vec<u32>)> {
        let d_sem = self.cfg.semantic_dim;
        let (_, n_frames) = latent.dim();
        let levels = self.cfg.acoustic_codebook_size;
        let mut semantic = Vec::with_capacity(n_frames);
        let mut acoustic = Vec::with_capacity(n_frames * 36);
        for fi in 0..n_frames {
            let mut best_id = 0usize;
            let mut best_dist = f32::INFINITY;
            for cid in 0..self.semantic_embedding.dim().0 {
                let mut dist = 0f32;
                for di in 0..d_sem {
                    let diff = latent[[di, fi]] - self.semantic_embedding[[cid, di]];
                    dist += diff * diff;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_id = cid;
                }
            }
            semantic.push(best_id);
            for ai in 0..36 {
                let v = latent[[d_sem + ai, fi]].tanh();
                let scaled = ((v + 1.0) * 0.5) * (levels as f32 - 1.0);
                let code = scaled.round().clamp(0.0, levels as f32 - 1.0) as u32;
                acoustic.push(code);
            }
        }
        Ok((semantic, acoustic))
    }
}

pub fn has_encoder_tensors(
    prefix: &str,
    tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> bool {
    tensors.contains_key(&format!(
        "{prefix}input_proj.conv.parametrizations.weight.original1"
    )) || tensors.contains_key(&format!("{prefix}input_proj.conv.weight"))
}

pub fn has_encoder_weights(keys: &std::collections::HashSet<String>, prefix: &str) -> bool {
    keys.iter()
        .any(|k| k.starts_with(&format!("{prefix}input_proj")))
        && keys
            .iter()
            .any(|k| k.starts_with(&format!("{prefix}encoder_blocks")))
}

pub fn load_mono_wav(path: &std::path::Path, target_rate: u32) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open wav {}", path.display()))?;
    let spec = reader.spec();
    ensure!(
        spec.channels == 1,
        "reference wav must be mono (got {} channels)",
        spec.channels
    );
    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map_err(|e| anyhow::anyhow!("{e}")))
            .collect(),
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| {
                s.map(|v| v as f32 / i32::MAX as f32)
                    .map_err(|e| anyhow::anyhow!("{e}"))
            })
            .collect(),
    };
    let mut pcm = samples?;
    if spec.sample_rate != target_rate {
        pcm = resample_linear(&pcm, spec.sample_rate, target_rate);
    }
    Ok(pcm)
}

fn resample_linear(pcm: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || pcm.is_empty() {
        return pcm.to_vec();
    }
    let out_len = ((pcm.len() as u64 * to_rate as u64) / from_rate as u64) as usize;
    let mut out = Vec::with_capacity(out_len.max(1));
    for i in 0..out_len {
        let src = (i as f64 * from_rate as f64) / to_rate as f64;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(pcm.len() - 1);
        let frac = (src - i0 as f64) as f32;
        out.push(pcm[i0] * (1.0 - frac) + pcm[i1] * frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_downsample_halves_length() {
        let pcm = vec![0.0, 1.0, 0.0, -1.0];
        let out = resample_linear(&pcm, 24000, 12000);
        assert_eq!(out.len(), 2);
    }
}
